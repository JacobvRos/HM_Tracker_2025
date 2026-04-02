import os
import numpy as np
from pathlib import Path
import argparse
import shutil
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
import probeinterface as pi
import gc
from scipy.signal import welch

try:
    from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
except ImportError:
    print("Warning: 'readTrodesExtractedDataFile3.py' not found.")


def select_cleanest_channels(rec_lfp, n_best=3, segment_duration_s=60):
    """
    Rank channels by 'cleanness' for sleep scoring.
    
    Cleanness metric: ratio of in-band power (0.5-30 Hz) to high-frequency noise (>50 Hz).
    Channels with high SNR in the sleep-relevant bands score best.
    
    Uses a subset of data (first segment_duration_s seconds) to keep it fast.
    """
    fs = rec_lfp.get_sampling_frequency()
    n_samples = min(int(segment_duration_s * fs), rec_lfp.get_num_samples())
    traces = rec_lfp.get_traces(start_frame=0, end_frame=n_samples)
    num_channels = traces.shape[1]

    scores = np.zeros(num_channels)
    for ch in range(num_channels):
        f, psd = welch(traces[:, ch], fs=fs, nperseg=int(2 * fs))
        sleep_band = np.mean(psd[(f >= 0.5) & (f <= 30)])
        noise_band = np.mean(psd[(f > 50)])
        # avoid division by zero
        scores[ch] = sleep_band / (noise_band + 1e-12)

    best_idx = np.argsort(scores)[::-1][:n_best]
    print(f"Cleanest {n_best} channels (by sleep-band SNR): {best_idx}")
    print(f"  Scores: {scores[best_idx]}")
    return best_idx, scores


def select_emg_channel(rec, fs_orig, segment_duration_s=60):
    """
    Pick the channel with the highest power in the EMG band (20-200 Hz)
    from the raw (pre-LFP-filter) recording — i.e. the channel that 
    carries the most muscle artifact.
    """
    n_samples = min(int(segment_duration_s * fs_orig), rec.get_num_samples())
    traces = rec.get_traces(start_frame=0, end_frame=n_samples)
    num_channels = traces.shape[1]

    emg_power = np.zeros(num_channels)
    for ch in range(num_channels):
        f, psd = welch(traces[:, ch], fs=fs_orig, nperseg=int(2 * fs_orig))
        emg_power[ch] = np.mean(psd[(f >= 20) & (f <= 200)])

    emg_ch = np.argmax(emg_power)
    print(f"Selected EMG channel: {emg_ch} (highest 20-200 Hz power)")
    return emg_ch


def extract_lfp_and_sort(file_path, output_parent, target_fs=1000.0):
    file_path_obj = Path(file_path)
    file_stem = file_path_obj.stem
    output_dir = Path(output_parent) / f"{file_stem}_LFP_Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. LOAD DATA
    print(f"\n--- Loading {file_stem} ---")
    raw = readTrodesExtractedDataFile(str(file_path_obj))
    traces = raw['data']['voltage']
    fs_orig = 30000.0
    num_channels = traces.shape[1]
    print(f"Detected {num_channels} channels.")

    # 2. CREATE RECORDING
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs_orig)

    # 3. DYNAMIC PROBE
    print("Configuring probe geometry...")
    num_channels = rec.get_num_channels()
    probe = pi.Probe(ndim=2, si_units='um')
    positions = np.zeros((num_channels, 2))
    positions[:, 1] = np.arange(num_channels) * 20
    probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 5})
    probe.set_device_channel_indices(np.arange(num_channels))
    rec = rec.set_probe(probe)

    # --- STEP A: LFP EXTRACTION ---
    print("Processing LFP (1-450 Hz)...")
    rec_filtered = spre.bandpass_filter(rec, freq_min=1.0, freq_max=450.0)

    decimation_factor = int(fs_orig / target_fs)
    rec_lfp = spre.decimate(rec_filtered, decimation_factor=decimation_factor)

    print("Extracting LFP traces in parallel...")
    job_kwargs = dict(n_jobs=1, chunk_duration='1s', progress_bar=True)

    temp_cache_dir = output_dir / "temp_si_cache"
    if temp_cache_dir.exists():
        shutil.rmtree(temp_cache_dir)

    rec_lfp_cached = rec_lfp.save(folder=temp_cache_dir, format="binary", **job_kwargs)

    # --- STEP B: SELECT CLEANEST CHANNELS ---
    print("\n--- Selecting cleanest EEG channels ---")
    best_ch_idx, scores = select_cleanest_channels(rec_lfp_cached, n_best=3)
    np.save(output_dir / "cleanest_channel_indices.npy", best_ch_idx)
    np.save(output_dir / "channel_snr_scores.npy", scores)
    print(f"Saved cleanest channel indices: {best_ch_idx}")

    # --- STEP C: EMG EXTRACTION ---
    # Select from the raw (unfiltered) recording so we keep high-freq EMG content
    print("\n--- Extracting EMG channel ---")
    emg_ch = select_emg_channel(rec, fs_orig)
    np.save(output_dir / "emg_channel_index.npy", np.array([emg_ch]))

    # Filter the selected channel for EMG: 10-100 Hz, then decimate
    rec_emg_single = rec.select_channels([rec.channel_ids[emg_ch]])
    rec_emg_filtered = spre.bandpass_filter(rec_emg_single, freq_min=10.0, freq_max=100.0)
    rec_emg_dec = spre.decimate(rec_emg_filtered, decimation_factor=decimation_factor)

    temp_emg_cache = output_dir / "temp_emg_cache"
    if temp_emg_cache.exists():
        shutil.rmtree(temp_emg_cache)
    rec_emg_cached = rec_emg_dec.save(folder=temp_emg_cache, format="binary", **job_kwargs)

    emg_data = rec_emg_cached.get_traces()
    np.save(output_dir / "emg_data.npy", emg_data.astype('float32'))
    print(f"Saved EMG data from channel {emg_ch}")

    # --- STEP D: SAVE LFP ---
    print("\nWriting processed LFP to Numpy arrays...")
    lfp_data = rec_lfp_cached.get_traces()
    lfp_timestamps = np.arange(lfp_data.shape[0]) / (fs_orig / decimation_factor)

    np.save(output_dir / "lfp_data.npy", lfp_data.astype('float32'))
    np.save(output_dir / "lfp_timestamps.npy", lfp_timestamps.astype('float64'))
    np.save(output_dir / "lfp_channels.npy", np.arange(num_channels))

    # --- CLEANUP ---
    del rec_lfp_cached, rec_emg_cached
    gc.collect()

    for cache_dir in [temp_cache_dir, temp_emg_cache]:
        try:
            shutil.rmtree(cache_dir)
        except PermissionError:
            print(f"Warning: Windows locked {cache_dir}. Delete manually later.")
            
    lfp_for_awake = rec_lfp_cached.get_traces()
    emg_for_awake = rec_emg_cached.get_traces()
    actual_fs = fs_orig / decimation_factor

    awakeness, emg_rms, theta_delta = compute_awakeness(
        lfp_for_awake, emg_for_awake, actual_fs, best_ch_idx
    )

    np.save(output_dir / "awakeness.npy", awakeness.astype('float32'))
    np.save(output_dir / "emg_rms.npy", emg_rms.astype('float32'))
    np.save(output_dir / "theta_delta_ratio.npy", theta_delta.astype('float32'))

    print(f"\n=== All files saved to {output_dir} ===")
    print(f"  lfp_data.npy           — all channels, 1-450 Hz")
    print(f"  cleanest_channel_indices.npy — top 3 EEG channels: {best_ch_idx}")
    print(f"  channel_snr_scores.npy — SNR scores for all channels")
    print(f"  emg_data.npy           — EMG from channel {emg_ch}, 10-100 Hz")
    print(f"  emg_channel_index.npy  — EMG channel index")

def compute_awakeness(lfp_data, emg_data, fs, best_ch_idx, epoch_s=1):
    """
    Compute a per-second awakeness score from EEG + EMG.
    
    High values = likely awake
    Low values  = likely NREM
    Intermediate with low EMG = likely REM
    """
    from scipy.signal import welch
    from scipy.stats import zscore
    
    n_seconds = int(lfp_data.shape[0] / fs)
    n_per_epoch = int(fs * epoch_s)
    
    # Use the best EEG channel
    eeg = lfp_data[:, best_ch_idx[0]]
    emg = emg_data[:, 0]
    
    emg_power = np.zeros(n_seconds)
    theta_delta = np.zeros(n_seconds)
    
    for i in range(n_seconds):
        start = i * n_per_epoch
        end_ = start + n_per_epoch
        
        # EMG RMS
        emg_seg = emg[start:end_]
        emg_power[i] = np.sqrt(np.mean(emg_seg ** 2))
        
        # Theta/Delta ratio from EEG
        eeg_seg = eeg[start:end_]
        f, psd = welch(eeg_seg, fs=fs, nperseg=min(n_per_epoch, int(2 * fs)))
        delta = np.mean(psd[(f >= 0.5) & (f <= 4)])
        theta = np.mean(psd[(f >= 5) & (f <= 9)])
        theta_delta[i] = theta / (delta + 1e-12)
    
    # Z-score both, then combine
    emg_z = zscore(emg_power)
    td_z = zscore(theta_delta)
    
    # Awakeness = weighted sum (EMG dominates for wake vs sleep,
    # theta/delta helps separate REM from wake)
    awakeness = 0.6 * emg_z + 0.4 * td_z
    
    return awakeness, emg_power, theta_delta
def run_pipeline(input_folder, output_folder):
    base_path = Path(input_folder)
    dat_files = list(base_path.glob("**/*.raw/*_group0.dat"))

    for dat_file in dat_files:
        try:
            extract_lfp_and_sort(dat_file, output_folder)
        except Exception as e:
            print(f"Error processing {dat_file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    run_pipeline(args.input_folder, args.output_folder)