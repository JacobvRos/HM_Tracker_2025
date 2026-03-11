import os
import numpy as np
from pathlib import Path
import argparse
import shutil
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
import probeinterface as pi
import gc


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
    print("Configuring probe geometry...")
    
    # Dynamically get the number of channels from the recording
    num_channels = rec.get_num_channels()
    
    # Create the probe based on the actual channel count
    probe = pi.Probe(ndim=2, si_units='um')
    positions = np.zeros((num_channels, 2))
    positions[:, 1] = np.arange(num_channels) * 20  # 20um spacing
    probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 5})
    
    # This is the crucial line: it generates indices from 0 to 75 (for 76 channels)
    probe.set_device_channel_indices(np.arange(num_channels))
    
    # Attach the dynamic probe
    rec = rec.set_probe(probe)

    # --- STEP A: LFP EXTRACTION ---
    print("Processing LFP (1-450 Hz)...")
    rec_filtered = spre.bandpass_filter(rec, freq_min=1.0, freq_max=450.0)
    
    # Use decimate to avoid "Non integer" errors
    decimation_factor = int(fs_orig / target_fs) 
    rec_lfp = spre.decimate(rec_filtered, decimation_factor=decimation_factor)
    
    print("Extracting LFP traces in parallel (this will be much faster)...")
    
    # Define multiprocessing parameters (uses all available cores, 1-second chunks)
    # Set n_jobs to 1 to avoid Windows pickling errors with in-memory arrays.
    # The 1-second chunking will still keep memory usage low and prevent freezing.
    job_kwargs = dict(n_jobs=1, chunk_duration='1s', progress_bar=True)
    
    # Create a temporary directory for the parallel cache
    temp_cache_dir = output_dir / "temp_si_cache"
    if temp_cache_dir.exists():
        shutil.rmtree(temp_cache_dir)
        
    # Force parallel evaluation by saving to a temporary binary format
    rec_lfp_cached = rec_lfp.save(folder=temp_cache_dir, format="binary", **job_kwargs)
    
    print("Writing processed LFP to Numpy arrays...")
    # Because it's already processed and cached, get_traces() is now nearly instant
    lfp_data = rec_lfp_cached.get_traces()
    lfp_timestamps = np.arange(lfp_data.shape[0]) / (fs_orig / decimation_factor)
    
    np.save(output_dir / "lfp_data.npy", lfp_data.astype('float32'))
    np.save(output_dir / "lfp_timestamps.npy", lfp_timestamps.astype('float64'))
    np.save(output_dir / "lfp_channels.npy", np.arange(num_channels))
    
    # 1. Delete the cached object so SpikeInterface closes the file handle
    del rec_lfp_cached
    # 2. Force Python's garbage collector to clean up the memory map
    gc.collect()
    
    # 3. Clean up the temporary cache safely
    try:
        shutil.rmtree(temp_cache_dir)
    except PermissionError:
        print(f"Warning: Windows locked {temp_cache_dir}. You can delete this manually later.")
        
    print(f"LFP files saved to {output_dir}")

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