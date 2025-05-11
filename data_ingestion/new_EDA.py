import os
import pandas as pd
import numpy as np
import glob
from scipy import stats
from collections import Counter
from sklearn.decomposition import FastICA

def load_subject_data(base_dir, subject_id):
    
    subject_dir = os.path.join(base_dir, f'subject_{subject_id}')

    metadata = {}
    with open(os.path.join(subject_dir, 'metadata.txt'), 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                metadata[key.strip()] = value.strip()

    data = {}
    for csv_file in glob.glob(os.path.join(subject_dir, '*.csv')):
        channel_name = os.path.basename(csv_file).replace('.csv', '')
        data[channel_name] = pd.read_csv(csv_file)
    
    return data, metadata

def get_all_subject_ids(base_dir):
    
    subject_dirs = glob.glob(os.path.join(base_dir, 'subject_*'))
    return [os.path.basename(d).replace('subject_', '') for d in subject_dirs]

def run_sleep_only_eda(base_dir, output_dir):
    print("Running Focused Sleep-Only EEG Data Analysis...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    subject_ids = get_all_subject_ids(base_dir)
    print(f"Found {len(subject_ids)} subjects")

    sleep_stage_names = {
        1: 'Stage 1 (N1)',
        2: 'Stage 2 (N2)', 
        3: 'Stage 3 (N3)',
        4: 'Stage 4 (N4)',
        5: 'REM (R)'
    }



    print("\n=== 1. SLEEP-ONLY DATA STRUCTURE AND DISTRIBUTION ===")

    total_epochs = 0
    sleep_epochs = 0
    sleep_stage_counts = {stage: 0 for stage in sleep_stage_names.keys()}

    for subject_id in subject_ids:
        try:
            data, metadata = load_subject_data(base_dir, subject_id)

            subject_output_dir = os.path.join(output_dir, f'subject_{subject_id}')
            if not os.path.exists(subject_output_dir):
                os.makedirs(subject_output_dir)

            for channel_name, channel_df in data.items():

                total_epochs += len(channel_df)

                sleep_only_df = channel_df[channel_df['sleep_stage'] > 0]
                sleep_epochs += len(sleep_only_df)

                for stage in sleep_stage_names.keys():
                    sleep_stage_counts[stage] += len(sleep_only_df[sleep_only_df['sleep_stage'] == stage])

                sleep_only_df.to_csv(os.path.join(subject_output_dir, f"{channel_name}.csv"), index=False)

            with open(os.path.join(subject_output_dir, 'metadata.txt'), 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write(f"sleep_only: True\n")
                f.write(f"original_epochs: {sum(len(df) for df in data.values())}\n")
                f.write(f"sleep_only_epochs: {sum(len(df[df['sleep_stage'] > 0]) for df in data.values())}\n")
                
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            continue
    
    print("\nOverall Sleep Stage Distribution (across all subjects):")
    print(f"Total epochs: {total_epochs}")
    print(f"Sleep-only epochs: {sleep_epochs} ({sleep_epochs/total_epochs*100:.2f}% of total)")
    
    for stage, count in sorted(sleep_stage_counts.items()):
        print(f"  Stage {stage} ({sleep_stage_names[stage]}): {count} epochs ({count/sleep_epochs*100:.2f}% of sleep)")



    print("\n=== 2. SPECTRAL PROFILES ACROSS SLEEP STAGES ===")

    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    stage_band_powers = {stage: {band: [] for band in freq_bands.keys()} for stage in sleep_stage_names.keys()}

    subject_variation = {stage: {band: [] for band in freq_bands.keys()} for stage in sleep_stage_names.keys()}
    
    for subject_id in subject_ids[:20]:  # Use a subset of subjects for this analysis
        try:
            data, _ = load_subject_data(base_dir, subject_id)
            
            for channel_name in data.keys():
                df = data[channel_name]

                sleep_df = df[df['sleep_stage'] > 0]
                
                for stage in sleep_stage_names.keys():
                    stage_data = sleep_df[sleep_df['sleep_stage'] == stage]
                    
                    if len(stage_data) > 0:

                        subject_stage_powers = {band: stage_data[f'{band}_power'].mean() 
                                               for band in freq_bands.keys()}

                        for band in freq_bands.keys():
                            if not np.isnan(subject_stage_powers[band]):
                                subject_variation[stage][band].append(subject_stage_powers[band])

                        for band in freq_bands.keys():
                            stage_band_powers[stage][band].extend(stage_data[f'{band}_power'].tolist())
        
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            continue

    print("\nAverage Spectral Power by Sleep Stage and Frequency Band:")
    for stage in sleep_stage_names.keys():
        print(f"\n{sleep_stage_names[stage]}:")
        for band, band_range in freq_bands.items():
            if stage_band_powers[stage][band]:
                avg_power = np.mean(stage_band_powers[stage][band])
                std_power = np.std(stage_band_powers[stage][band])
                print(f"  {band.capitalize()} band ({band_range[0]}-{band_range[1]} Hz): {avg_power:.4f} ± {std_power:.4f}")

    print("\nSubject Variation in Spectral Power:")
    for stage in sleep_stage_names.keys():
        print(f"\n{sleep_stage_names[stage]}:")
        for band in freq_bands.keys():
            if subject_variation[stage][band]:
                subj_std = np.std(subject_variation[stage][band])
                subj_mean = np.mean(subject_variation[stage][band])
                cv = subj_std / subj_mean * 100 if subj_mean > 0 else 0
                print(f"  {band.capitalize()} band: Mean = {subj_mean:.4f}, Std Dev = {subj_std:.4f}, CV = {cv:.2f}%")

    print("\nBiological Plausibility Check:")

    if 3 in stage_band_powers and 4 in stage_band_powers:
        delta_by_stage = {stage: np.mean(stage_band_powers[stage]['delta']) 
                         for stage in sleep_stage_names.keys() if stage_band_powers[stage]['delta']}
        max_delta_stage = max(delta_by_stage, key=delta_by_stage.get)
        
        print(f"Delta power should be highest in N3 (stage 3) or N4 (stage 4): {'✓' if max_delta_stage in [3, 4] else '✗'}")
        print(f"  Highest delta power is in stage {max_delta_stage} ({sleep_stage_names.get(max_delta_stage, 'Unknown')})")

        print("  Delta power by stage:")
        for stage, power in sorted(delta_by_stage.items(), key=lambda x: x[1], reverse=True):
            print(f"    Stage {stage} ({sleep_stage_names[stage]}): {power:.4f}")

    if 5 in stage_band_powers and 1 in stage_band_powers:
        theta_by_stage = {stage: np.mean(stage_band_powers[stage]['theta']) 
                         for stage in sleep_stage_names.keys() if stage_band_powers[stage]['theta']}

        sorted_theta = sorted(theta_by_stage.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTheta power should be prominent in REM (stage 5) and N1 (stage 1):")

        top_theta_stages = [stage for stage, _ in sorted_theta[:2]]
        if 5 in top_theta_stages or 1 in top_theta_stages:
            print(f"  ✓ Theta power is prominent in expected stages")
        else:
            print(f"  ✗ Theta power is not prominent in expected stages")
            
        for i, (stage, power) in enumerate(sorted_theta):
            print(f"    #{i+1} Theta power: Stage {stage} ({sleep_stage_names[stage]}) - {power:.4f}")



    print("\n=== 3. DIMENSIONALITY REDUCTION (ICA) ON SLEEP-ONLY DATA ===")

    all_features = []
    all_labels = []
    
    for subject_id in subject_ids[:15]:  # Use subset for computational efficiency
        try:
            data, _ = load_subject_data(base_dir, subject_id)
            
            for channel_name in data.keys():
                df = data[channel_name]

                sleep_df = df[df['sleep_stage'] > 0]

                features = sleep_df[['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']].values
                labels = sleep_df['sleep_stage'].values

                all_features.append(features)
                all_labels.append(labels)
        
        except Exception as e:
            print(f"Error collecting data for ICA: {e}")
            continue
    
    if all_features:

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        print(f"Performing ICA on {len(X)} sleep-only epochs with 5 features per epoch")
        
        try:

            ica = FastICA(n_components=2, random_state=42)
            X_ica = ica.fit_transform(X)

            ica_by_stage = {}
            for stage in sleep_stage_names.keys():
                stage_mask = (y == stage)
                if np.sum(stage_mask) > 0:
                    ica_by_stage[stage] = {
                        'mean_comp1': np.mean(X_ica[stage_mask, 0]),
                        'mean_comp2': np.mean(X_ica[stage_mask, 1]),
                        'std_comp1': np.std(X_ica[stage_mask, 0]),
                        'std_comp2': np.std(X_ica[stage_mask, 1])
                    }
            
            print("\nICA Component Statistics by Sleep Stage:")
            for stage in sorted(ica_by_stage.keys()):
                print(f"\n{sleep_stage_names[stage]}:")
                print(f"  Component 1: {ica_by_stage[stage]['mean_comp1']:.4f} ± {ica_by_stage[stage]['std_comp1']:.4f}")
                print(f"  Component 2: {ica_by_stage[stage]['mean_comp2']:.4f} ± {ica_by_stage[stage]['std_comp2']:.4f}")

            feature_names = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
            
            print("\nFeature Contributions to ICA Components:")
            print("\nComponent 1:")
            for i, feature in enumerate(feature_names):
                print(f"  {feature}: {abs(ica.mixing_[i, 0]):.4f}")
            
            print("\nComponent 2:")
            for i, feature in enumerate(feature_names):
                print(f"  {feature}: {abs(ica.mixing_[i, 1]):.4f}")

            np.save(os.path.join(output_dir, 'ica_mixing.npy'), ica.mixing_)
            
            print("\nICA mixing matrix saved to output directory for future use")
            
        except Exception as e:
            print(f"Error in ICA analysis: {e}")
    
    print("\nSleep-Only analysis completed. Sleep-only data saved to:", output_dir)

if __name__ == "__main__":

    base_dir = os.path.join(os.getcwd(), 'processed_eeg_data')
    output_dir = os.path.join(os.getcwd(), 'sleep_only_data')
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Please run the data processing script first.")
    else:
        run_sleep_only_eda(base_dir, output_dir)