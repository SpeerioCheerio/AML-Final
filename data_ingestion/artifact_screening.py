import os
import numpy as np
from scipy import signal
from scipy import stats
from tqdm import tqdm

def screen_artifacts(input_dir, output_dir=None, exclude_threshold=5.0):
    
    if output_dir is None:
        output_dir = input_dir + '_clean'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {input_dir}...")
    X_raw = np.load(os.path.join(input_dir, 'X_eeg_signals.npy'))
    y_labels = np.load(os.path.join(input_dir, 'y_sleep_stages.npy'))
    subject_ids = np.load(os.path.join(input_dir, 'subject_ids.npy'))
    recording_ids = np.load(os.path.join(input_dir, 'recording_ids.npy'))
    metadata = np.load(os.path.join(input_dir, 'metadata.npy'), allow_pickle=True).item()

    sfreq = metadata['sampling_rate']
    
    print(f"Data loaded. Shape: {X_raw.shape}")

    clean_epochs = []
    clean_labels = []
    clean_subjects = []
    clean_recordings = []

    stats_dict = {
        'total_epochs': len(X_raw),
        'excluded_epochs': 0,
        'flatline_exclusions': 0,
        'amplitude_exclusions': 0,
        'spectral_exclusions': 0,
        'recordings_checked': {},
        'excluded_recordings': set()
    }

    print("Checking recordings for systemic issues...")
    unique_recordings = np.unique(recording_ids)
    
    for recording in tqdm(unique_recordings):
        recording_mask = recording_ids == recording
        recording_epochs = X_raw[recording_mask]

        all_recording_data = recording_epochs.reshape(-1, recording_epochs.shape[-1])

        flatline_percentage = check_recording_flatline(all_recording_data)

        amplitude_outlier_percentage = check_recording_amplitude_outliers(all_recording_data)

        has_powerline_interference, noise_ratio = check_recording_spectral_noise(all_recording_data, sfreq)

        stats_dict['recordings_checked'][recording] = {
            'flatline_percentage': flatline_percentage,
            'amplitude_outlier_percentage': amplitude_outlier_percentage,
            'has_powerline_interference': has_powerline_interference,
            'noise_ratio': noise_ratio,
            'excluded': False
        }

        exclude_recording = False
        
        if flatline_percentage > exclude_threshold:
            exclude_recording = True
            stats_dict['recordings_checked'][recording]['excluded'] = True
            stats_dict['recordings_checked'][recording]['exclusion_reason'] = f"Excessive flatline: {flatline_percentage:.2f}%"
            stats_dict['excluded_recordings'].add(recording)
            stats_dict['flatline_exclusions'] += np.sum(recording_mask)
            print(f"Excluding recording {recording}: Excessive flatline ({flatline_percentage:.2f}%)")
        
        if amplitude_outlier_percentage > exclude_threshold * 2:  # More lenient for amplitude outliers
            exclude_recording = True
            stats_dict['recordings_checked'][recording]['excluded'] = True
            stats_dict['recordings_checked'][recording]['exclusion_reason'] = f"Excessive amplitude outliers: {amplitude_outlier_percentage:.2f}%"
            stats_dict['excluded_recordings'].add(recording)
            stats_dict['amplitude_exclusions'] += np.sum(recording_mask)
            print(f"Excluding recording {recording}: Excessive amplitude outliers ({amplitude_outlier_percentage:.2f}%)")
        
        if has_powerline_interference and noise_ratio > 0.3:  # High noise ratio
            exclude_recording = True
            stats_dict['recordings_checked'][recording]['excluded'] = True
            stats_dict['recordings_checked'][recording]['exclusion_reason'] = f"Significant powerline interference: {noise_ratio:.2f}"
            stats_dict['excluded_recordings'].add(recording)
            stats_dict['spectral_exclusions'] += np.sum(recording_mask)
            print(f"Excluding recording {recording}: Significant powerline interference (ratio: {noise_ratio:.2f})")

    print("\nProcessing individual epochs...")
    for i in tqdm(range(len(X_raw))):
        recording = recording_ids[i]
        epoch_data = X_raw[i]

        if recording in stats_dict['excluded_recordings']:
            continue

        has_artifact = check_epoch_for_artifacts(epoch_data, sfreq)
        
        if has_artifact:
            stats_dict['excluded_epochs'] += 1
            continue

        clean_epochs.append(epoch_data)
        clean_labels.append(y_labels[i])
        clean_subjects.append(subject_ids[i])
        clean_recordings.append(recording_ids[i])

    X_clean = np.array(clean_epochs)
    y_clean = np.array(clean_labels)
    clean_subject_ids = np.array(clean_subjects)
    clean_recording_ids = np.array(clean_recordings)

    stats_dict['clean_epochs'] = len(X_clean)
    stats_dict['excluded_epoch_percentage'] = (stats_dict['excluded_epochs'] / stats_dict['total_epochs']) * 100

    print(f"Saving cleaned data to {output_dir}...")
    np.save(os.path.join(output_dir, 'X_eeg_signals.npy'), X_clean)
    np.save(os.path.join(output_dir, 'y_sleep_stages.npy'), y_clean)
    np.save(os.path.join(output_dir, 'subject_ids.npy'), clean_subject_ids)
    np.save(os.path.join(output_dir, 'recording_ids.npy'), clean_recording_ids)

    metadata['artifact_screening'] = {
        'original_epochs': stats_dict['total_epochs'],
        'clean_epochs': stats_dict['clean_epochs'],
        'excluded_epochs': stats_dict['excluded_epochs'],
        'excluded_recordings': list(stats_dict['excluded_recordings']),
        'exclusion_stats': {
            'flatline_exclusions': stats_dict['flatline_exclusions'],
            'amplitude_exclusions': stats_dict['amplitude_exclusions'],
            'spectral_exclusions': stats_dict['spectral_exclusions']
        }
    }
    
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)

    with open(os.path.join(output_dir, 'artifact_screening_report.txt'), 'w') as f:
        f.write("Artifact Screening Report\n")
        f.write("=======================\n\n")
        f.write(f"Total epochs analyzed: {stats_dict['total_epochs']}\n")
        f.write(f"Epochs excluded: {stats_dict['excluded_epochs']} ({stats_dict['excluded_epoch_percentage']:.2f}%)\n")
        f.write(f"Epochs retained: {stats_dict['clean_epochs']}\n\n")
        
        f.write("Exclusion breakdown:\n")
        f.write(f"  Flatline exclusions: {stats_dict['flatline_exclusions']} epochs\n")
        f.write(f"  Amplitude outlier exclusions: {stats_dict['amplitude_exclusions']} epochs\n")
        f.write(f"  Spectral noise exclusions: {stats_dict['spectral_exclusions']} epochs\n\n")
        
        f.write("Excluded recordings:\n")
        for recording in stats_dict['excluded_recordings']:
            reason = stats_dict['recordings_checked'][recording].get('exclusion_reason', 'Unknown reason')
            f.write(f"  {recording}: {reason}\n")
    
    print("\nArtifact screening complete!")
    print(f"Original dataset: {stats_dict['total_epochs']} epochs")
    print(f"Cleaned dataset: {stats_dict['clean_epochs']} epochs ({stats_dict['clean_epochs']/stats_dict['total_epochs']*100:.2f}%)")
    print(f"Excluded {len(stats_dict['excluded_recordings'])} recordings and {stats_dict['excluded_epochs']} individual epochs")
    
    return stats_dict

def check_recording_flatline(data):
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_channels, n_samples = data.shape
    
    flatline_samples = 0
    
    for ch in range(n_channels):
        diff = np.diff(data[ch, :])
        
        window_size = 100  # 1 second at 100 Hz
        for i in range(0, n_samples - window_size, window_size):
            if np.all(np.abs(diff[i:i+window_size-1]) < 1e-6):  # Effectively zero change
                flatline_samples += window_size
    
    flatline_percentage = (flatline_samples / (n_channels * n_samples)) * 100
    
    return flatline_percentage

def check_recording_amplitude_outliers(data):
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_channels, n_samples = data.shape
    
    outlier_samples = 0
    
    for ch in range(n_channels):
        ch_data = data[ch, :]

        median = np.median(ch_data)
        mad = stats.median_abs_deviation(ch_data)

        lower_bound = median - 5 * mad
        upper_bound = median + 5 * mad

        extreme_lower = -500  # µV
        extreme_upper = 500   # µV

        outliers = np.sum(
            (ch_data < lower_bound) | 
            (ch_data > upper_bound) | 
            (ch_data < extreme_lower) | 
            (ch_data > extreme_upper)
        )
        
        outlier_samples += outliers

    outlier_percentage = (outlier_samples / (n_channels * n_samples)) * 100
    
    return outlier_percentage

def check_recording_spectral_noise(data, sfreq):
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_channels, n_samples = data.shape
    
    has_powerline_interference = False
    total_noise_ratio = 0
    
    for ch in range(n_channels):
        ch_data = data[ch, :]

        f, Pxx = signal.welch(ch_data, fs=sfreq, nperseg=min(2048, n_samples))


        powerline_mask_50 = (f >= 49) & (f <= 51)
        powerline_mask_60 = (f >= 59) & (f <= 61)

        highfreq_mask = (f > 30) & (~powerline_mask_50) & (~powerline_mask_60)

        total_power = np.sum(Pxx)
        powerline_power_50 = np.sum(Pxx[powerline_mask_50])
        powerline_power_60 = np.sum(Pxx[powerline_mask_60])
        highfreq_power = np.sum(Pxx[highfreq_mask])

        powerline_ratio_50 = powerline_power_50 / total_power if total_power > 0 else 0
        powerline_ratio_60 = powerline_power_60 / total_power if total_power > 0 else 0
        
        if powerline_ratio_50 > 0.1 or powerline_ratio_60 > 0.1:  # More than 10% of power
            has_powerline_interference = True

        noise_ratio = (powerline_power_50 + powerline_power_60 + highfreq_power) / total_power if total_power > 0 else 0
        total_noise_ratio += noise_ratio

    avg_noise_ratio = total_noise_ratio / n_channels
    
    return has_powerline_interference, avg_noise_ratio

def check_epoch_for_artifacts(epoch_data, sfreq):

    flatline_percentage = check_recording_flatline(epoch_data)
    if flatline_percentage > 10.0:  # More than 10% flatline in this epoch
        return True

    amplitude_outlier_percentage = check_recording_amplitude_outliers(epoch_data)
    if amplitude_outlier_percentage > 15.0:  # More than 15% outliers in this epoch
        return True

    _, noise_ratio = check_recording_spectral_noise(epoch_data, sfreq)
    if noise_ratio > 0.25:  # More than 25% of power in noise bands
        return True
    
    return False

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description='Screen artifacts in Sleep-EDF dataset')
    parser.add_argument('--input_dir', type=str, default='sleep_edf_npy',
                        help='Directory containing the raw EEG signal NPY files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the cleaned data (default: input_dir + "_clean")')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='Threshold percentage for excluding recordings (default: 5.0%%)')
    
    args = parser.parse_args()

    screen_artifacts(args.input_dir, args.output_dir, args.threshold)