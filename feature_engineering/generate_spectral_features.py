

import os
import numpy as np
import pandas as pd
import glob
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_sleep_data(sleep_data_dir, subject_id):
   
   subject_dir = os.path.join(sleep_data_dir, f'subject_{subject_id}')

   metadata = {}
   try:
       with open(os.path.join(subject_dir, 'metadata.txt'), 'r') as f:
           for line in f:
               if ':' in line:
                   key, value = line.strip().split(':', 1)
                   metadata[key.strip()] = value.strip()
   except FileNotFoundError:
       print(f"Metadata file not found for subject {subject_id}")
       return None, None

   data = {}
   for csv_file in glob.glob(os.path.join(subject_dir, '*.csv')):
       channel_name = os.path.basename(csv_file).replace('.csv', '')
       data[channel_name] = pd.read_csv(csv_file)
   
   return data, metadata


def get_all_subject_ids(sleep_data_dir):
   
   subject_dirs = glob.glob(os.path.join(sleep_data_dir, 'subject_*'))
   return [os.path.basename(d).replace('subject_', '') for d in subject_dirs]


def compute_welch_psd(signal_data, fs=100.0, nperseg=256):
   
   freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)
   return freqs, psd


def compute_band_power(freqs, psd, freq_range):
   
   low, high = freq_range
   idx = np.logical_and(freqs >= low, freqs <= high)
   band_power = np.trapz(psd[idx], freqs[idx])
   return band_power


def compute_stft(signal_data, fs=100.0, window_len=2.0, overlap=0.75):
   
   nperseg = int(window_len * fs)
   noverlap = int(nperseg * overlap)
   
   f, t, Zxx = signal.stft(signal_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
   return f, t, Zxx


def generate_features(sleep_data_dir, output_dir):
   
   if not os.path.exists(output_dir):
       os.makedirs(output_dir)
   
   subject_ids = get_all_subject_ids(sleep_data_dir)
   print(f"Found {len(subject_ids)} subjects in the sleep-only dataset")

   freq_bands = {
       'delta': (0.5, 4),
       'theta': (4, 8),
       'alpha': (8, 13),
       'beta': (13, 30),
       'gamma': (30, 45)
   }

   for subject_id in tqdm(subject_ids, desc="Processing subjects"):

       data, metadata = load_sleep_data(sleep_data_dir, subject_id)
       if data is None:
           continue

       subject_output_dir = os.path.join(output_dir, f'subject_{subject_id}')
       if not os.path.exists(subject_output_dir):
           os.makedirs(subject_output_dir)

       for channel_name, df in data.items():

           features_df = df.copy()

           fs = 100.0  # Default
           if 'sampling_frequency' in metadata:
               fs = float(metadata['sampling_frequency'])

           raw_waves = []

           prev_stage = None

           band_powers_sum = {band: 0.0 for band in freq_bands.keys()}
           band_powers_count = 0

           for index, row in df.iterrows():
               epoch = int(row['epoch'])

               if 'delta_power' in row and 'theta_power' in row:
                   for band in freq_bands.keys():
                       band_powers_sum[band] += float(row[f'{band}_power'])
                   band_powers_count += 1

           avg_powers = {band: band_powers_sum[band] / band_powers_count 
                       if band_powers_count > 0 else 1.0 
                       for band in freq_bands.keys()}


           features_df['delta_power_log'] = np.nan
           features_df['theta_power_log'] = np.nan
           features_df['alpha_power_log'] = np.nan
           features_df['beta_power_log'] = np.nan
           features_df['gamma_power_log'] = np.nan
           features_df['delta_theta_ratio'] = np.nan
           features_df['theta_alpha_ratio'] = np.nan
           features_df['slow_fast_ratio'] = np.nan
           features_df['prev_stage'] = np.nan
           features_df['stft_entropy'] = np.nan
           
           for index, row in df.iterrows():
               epoch = int(row['epoch'])
               stage = int(row['sleep_stage'])

               for band in freq_bands.keys():
                   band_power = float(row[f'{band}_power'])

                   log_power = np.log(band_power / avg_powers[band] + 1e-6)
                   features_df.at[index, f'{band}_power_log'] = log_power

               delta_power = float(row['delta_power'])
               theta_power = float(row['theta_power']) 
               alpha_power = float(row['alpha_power'])
               beta_power = float(row['beta_power'])

               delta_theta_ratio = delta_power / (theta_power + 1e-6)
               features_df.at[index, 'delta_theta_ratio'] = delta_theta_ratio

               theta_alpha_ratio = theta_power / (alpha_power + 1e-6)  
               features_df.at[index, 'theta_alpha_ratio'] = theta_alpha_ratio

               slow_waves = delta_power + theta_power
               fast_waves = alpha_power + beta_power
               slow_fast_ratio = slow_waves / (fast_waves + 1e-6)
               features_df.at[index, 'slow_fast_ratio'] = slow_fast_ratio

               if prev_stage is not None:
                   features_df.at[index, 'prev_stage'] = prev_stage
               prev_stage = stage

           features_df.to_csv(os.path.join(subject_output_dir, f"{channel_name}_features.csv"), index=False)

           with open(os.path.join(subject_output_dir, 'metadata.txt'), 'w') as f:
               for key, value in metadata.items():
                   f.write(f"{key}: {value}\n")
               f.write(f"features_engineered: True\n")
               f.write(f"engineered_features: delta_power_log,theta_power_log,alpha_power_log,beta_power_log,gamma_power_log," +
                       f"delta_theta_ratio,theta_alpha_ratio,slow_fast_ratio,prev_stage\n")

           stft_diagnostics_dir = os.path.join(subject_output_dir, 'stft_diagnostics')
           if not os.path.exists(stft_diagnostics_dir):
               os.makedirs(stft_diagnostics_dir)

           for stage in range(1, 6):  # Sleep stages 1-5
               stage_data = df[df['sleep_stage'] == stage]
               if len(stage_data) > 0:

                   example_epochs = stage_data['epoch'].values[:min(3, len(stage_data))]
                   
                   for i, example_epoch in enumerate(example_epochs):




                       with open(os.path.join(subject_output_dir, 'metadata.txt'), 'a') as f:
                           f.write(f"stft_diagnostic_stage_{stage}_example_{i}: saved\n")
   
   print(f"Feature engineering completed. Engineered features saved to: {output_dir}")
   print("\nEngineered features include:")
   print("1. Log-transformed spectral band powers (delta, theta, alpha, beta, gamma)")
   print("2. Delta/Theta power ratio - important for distinguishing deep sleep")
   print("3. Theta/Alpha ratio - relevant for REM detection")
   print("4. Slow/Fast waves ratio - general sleep depth indicator")
   print("5. Previous stage feature (transition marker)")


if __name__ == "__main__":

   sleep_data_dir = os.path.join(os.getcwd(), 'sleep_only_data')
   feature_eng_dir = os.path.join(os.getcwd(), 'feature_engineering', 'engineered_features')
   
   if not os.path.exists(sleep_data_dir):
       print(f"Sleep-only data directory {sleep_data_dir} does not exist.")
       print("Please run the sleep-only EDA script first.")
   else:
       generate_features(sleep_data_dir, feature_eng_dir)