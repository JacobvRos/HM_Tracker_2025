import os
import numpy as np
from pathlib import Path
import argparse
import shutil
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
import probeinterface as pi

try:
    from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
except ImportError:
    print("Warning: 'readTrodesExtractedDataFile3.py' not found.")

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

    # 3. DYNAMIC PROBE (Fixes the 128 vs 76 error)
    # We create a simple linear probe with the detected number of channels
    probe = pi.Probe(ndim=2, si_units='um')
    positions = np.zeros((num_channels, 2))
    positions[:, 1] = np.arange(num_channels) * 20  # 20um spacing
    probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 5})
    probe.set_device_channel_indices(np.arange(num_channels))
    rec = rec.set_probe(probe)

    # --- STEP A: LFP EXTRACTION ---
    print("Processing LFP (1-450 Hz)...")
    rec_filtered = spre.bandpass_filter(rec, freq_min=1.0, freq_max=450.0)
    
    # Use decimate to avoid "Non integer" errors
    decimation_factor = int(fs_orig / target_fs) 
    rec_lfp = spre.decimate(rec_filtered, decimation_factor=decimation_factor)
    
    print("Extracting LFP traces to Numpy...")
    lfp_data = rec_lfp.get_traces()
    lfp_timestamps = np.arange(lfp_data.shape[0]) / (fs_orig / decimation_factor)
    
    np.save(output_dir / "lfp_data.npy", lfp_data.astype('float32'))
    np.save(output_dir / "lfp_timestamps.npy", lfp_timestamps.astype('float64'))
    np.save(output_dir / "lfp_channels.npy", np.arange(num_channels))
    print(f"LFP files saved to {output_dir}")

    # --- STEP B: SPIKE SORTING (Optional/Integration) ---
    # If you want to continue to sorting with this dynamic probe, 
    # you can insert your Mountainsort4 logic here using 'rec'.

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