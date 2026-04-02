import gc
import shutil
import argparse
import numpy as np
from pathlib import Path
from scipy.signal import welch
from scipy.stats import zscore
from tqdm import tqdm
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
import probeinterface as pi

try:
    from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
except ImportError:
    print("Warning: 'readTrodesExtractedDataFile3.py' not found.")


# ──────────────────────────────────────────────────────────────────────────────
# LOAD RAW FILE — metadata via readTrodesExtractedDataFile,
#                 data via BinaryRecordingExtractor (no deepcopy)
# ──────────────────────────────────────────────────────────────────────────────

def load_recording(file_path, output_dir, voltage_scale_default=0.195):
    """
    1. Call readTrodesExtractedDataFile to get traces + metadata.
    2. If traces is already a memmap  → point BinaryRecordingExtractor at the
       same file (zero extra disk/RAM).
    3. If traces is a plain ndarray   → write it once to a temp binary file
       in chunks, free the ndarray, then use BinaryRecordingExtractor.
    4. Either way, set_probe() only deepcopies tiny path metadata — no OOM.
    """
    tqdm.write("  Loading with readTrodesExtractedDataFile …")
    raw          = readTrodesExtractedDataFile(str(file_path))
    traces       = raw['data']['voltage']
    num_channels = traces.shape[1]
    fs_orig      = float(raw.get('samplingRate', 30000))
    voltage_scale = float(raw.get('voltageScaling', voltage_scale_default))

    tqdm.write(f"  ✓ {num_channels} channels  |  {fs_orig} Hz  "
               f"|  scale={voltage_scale} µV/count  "
               f"|  shape={traces.shape}  dtype={traces.dtype}")

    if isinstance(traces, np.memmap):
        # ── Fast path: traces already lives on disk ──────────────────────────
        tqdm.write("  ✓ Traces are a memmap — using file directly "
                   "(no copy needed)")
        rec = si.BinaryRecordingExtractor(
            file_paths         = [traces.filename],
            sampling_frequency = fs_orig,
            num_channels       = num_channels,
            dtype              = str(traces.dtype),
            time_axis          = 0,
            file_offset        = traces.offset,
            gain_to_uV         = voltage_scale,
            offset_to_uV       = 0.0,
        )
        del traces, raw
        gc.collect()

    else:
        # ── Fallback: plain ndarray — write to temp binary in chunks ---------
        # (only hit on small files that fit in RAM; for >64 GB files
        #  readTrodesExtractedDataFile will have returned a memmap above)
        temp_raw_bin  = output_dir / "temp_raw.raw"
        traces_dtype  = str(traces.dtype)   # save before del
        traces_shape  = traces.shape

        if temp_raw_bin.exists():
            tqdm.write(f"  ✓ Temp binary already exists — skipping write")
            del traces, raw
            gc.collect()
        else:
            tqdm.write("  Traces are a plain ndarray — writing temp binary …")
            n_samples  = traces_shape[0]
            chunk_size = 100_000

            with tqdm(total=int(np.ceil(n_samples / chunk_size)),
                      unit='chunk', desc="  Writing temp raw", leave=False) as pb:
                fp = np.memmap(temp_raw_bin, dtype=traces_dtype, mode='w+',
                               shape=traces_shape)
                for start in range(0, n_samples, chunk_size):
                    end = min(start + chunk_size, n_samples)
                    fp[start:end] = traces[start:end]
                    pb.update(1)
                del fp

            del traces, raw
            gc.collect()
            tqdm.write(f"  ✓ Temp binary written to {temp_raw_bin}")

        rec = si.BinaryRecordingExtractor(
            file_paths         = [str(temp_raw_bin)],
            sampling_frequency = fs_orig,
            num_channels       = num_channels,
            dtype              = traces_dtype,
            time_axis          = 0,
            file_offset        = 0,
            gain_to_uV         = voltage_scale,
            offset_to_uV       = 0.0,
        )

    return rec, num_channels, fs_orig, voltage_scale


# ──────────────────────────────────────────────────────────────────────────────
# MEMORY-SAFE I/O HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_binary_memmap(cache_dir, n_channels, dtype='float32'):
    """Memory-map the flat .raw written by SI's .save(format='binary')."""
    cache_dir = Path(cache_dir)
    raw_files = list(cache_dir.glob("*.raw"))
    if not raw_files:
        raise FileNotFoundError(f"No .raw file found in {cache_dir}")
    bin_file  = raw_files[0]
    itemsize  = np.dtype(dtype).itemsize
    n_samples = bin_file.stat().st_size // (itemsize * n_channels)
    return np.memmap(bin_file, dtype=dtype, mode='r',
                     shape=(n_samples, n_channels))


def save_in_chunks(src, out_path, chunk_size=100_000, dtype='float32',
                   desc="  Saving"):
    """Write array-like to .npy in fixed-size row chunks."""
    n_samples, n_channels = src.shape
    out = np.lib.format.open_memmap(
        out_path, mode='w+', dtype=dtype, shape=(n_samples, n_channels)
    )
    n_chunks = int(np.ceil(n_samples / chunk_size))
    with tqdm(total=n_chunks, unit='chunk', desc=desc, leave=False) as pb:
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            out[start:end] = src[start:end].astype(dtype)
            pb.update(1)
    del out


# ──────────────────────────────────────────────────────────────────────────────
# CHANNEL SELECTION
# ──────────────────────────────────────────────────────────────────────────────

def select_cleanest_channels(rec_lfp, n_best=3, segment_duration_s=60):
    """Sleep-band SNR: mean PSD(0.5–30 Hz) / mean PSD(>50 Hz)."""
    fs        = rec_lfp.get_sampling_frequency()
    n_samples = min(int(segment_duration_s * fs), rec_lfp.get_num_samples())
    traces    = rec_lfp.get_traces(start_frame=0, end_frame=n_samples)
    n_ch      = traces.shape[1]

    scores = np.zeros(n_ch)
    with tqdm(total=n_ch, unit='ch', desc="  Scoring EEG channels",
              leave=False) as pb:
        for ch in range(n_ch):
            f, psd     = welch(traces[:, ch], fs=fs, nperseg=int(2 * fs))
            sleep_band = np.mean(psd[(f >= 0.5) & (f <= 30)])
            noise_band = np.mean(psd[f > 50])
            scores[ch] = sleep_band / (noise_band + 1e-12)
            pb.update(1)

    best_idx = np.argsort(scores)[::-1][:n_best]
    tqdm.write(f"  ✓ Cleanest {n_best} channels: {best_idx}  "
               f"(scores: {scores[best_idx].round(2)})")
    return best_idx, scores


def select_emg_channel(rec, fs_orig, segment_duration_s=60):
    """Channel with highest power in the EMG band (20–200 Hz)."""
    n_samples = min(int(segment_duration_s * fs_orig), rec.get_num_samples())
    traces    = rec.get_traces(start_frame=0, end_frame=n_samples)
    n_ch      = traces.shape[1]

    emg_power = np.zeros(n_ch)
    with tqdm(total=n_ch, unit='ch', desc="  Scoring EMG channels",
              leave=False) as pb:
        for ch in range(n_ch):
            f, psd        = welch(traces[:, ch], fs=fs_orig,
                                  nperseg=int(2 * fs_orig))
            emg_power[ch] = np.mean(psd[(f >= 20) & (f <= 200)])
            pb.update(1)

    emg_ch = int(np.argmax(emg_power))
    tqdm.write(f"  ✓ EMG channel: {emg_ch}  "
               f"(power: {emg_power[emg_ch]:.4f})")
    return emg_ch


# ──────────────────────────────────────────────────────────────────────────────
# AWAKENESS
# ──────────────────────────────────────────────────────────────────────────────

def compute_awakeness(lfp_path, emg_path, fs, best_ch_idx, epoch_s=1):
    """
    Per-second awakeness score from mmap'd .npy files — one epoch in RAM
    at a time.   High → awake  |  Low → NREM  |  Mid + low EMG → REM
    """
    lfp = np.load(lfp_path, mmap_mode='r')
    emg = np.load(emg_path, mmap_mode='r')

    n_samples   = lfp.shape[0]
    n_per_epoch = int(fs * epoch_s)
    n_seconds   = n_samples // n_per_epoch
    eeg_ch      = int(best_ch_idx[0])

    emg_power   = np.zeros(n_seconds, dtype='float32')
    theta_delta = np.zeros(n_seconds, dtype='float32')

    with tqdm(total=n_seconds, unit='s', desc="  Awakeness epochs",
              leave=False) as pb:
        for i in range(n_seconds):
            s = i * n_per_epoch
            e = s + n_per_epoch
            emg_seg = np.array(emg[s:e, 0],      dtype='float64')
            eeg_seg = np.array(lfp[s:e, eeg_ch], dtype='float64')

            emg_power[i] = np.sqrt(np.mean(emg_seg ** 2))

            f, psd         = welch(eeg_seg, fs=fs,
                                   nperseg=min(n_per_epoch, int(2 * fs)))
            delta          = np.mean(psd[(f >= 0.5) & (f <= 4)])
            theta          = np.mean(psd[(f >= 5)   & (f <= 9)])
            theta_delta[i] = theta / (delta + 1e-12)
            pb.update(1)

    emg_z     = zscore(emg_power)
    td_z      = zscore(theta_delta)
    return (0.6 * emg_z + 0.4 * td_z).astype('float32'), emg_power, theta_delta


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

STEPS = [
    "Load file",
    "Build recording + probe",
    "Stream LFP to cache",
    "Select EEG channels",
    "Stream EMG to cache",
    "Write LFP .npy",
    "Write EMG .npy",
    "Compute awakeness",
]


def extract_lfp_and_sort(file_path, output_parent, target_fs=1000.0):
    file_path_obj = Path(file_path)
    file_stem     = file_path_obj.stem
    output_dir    = Path(output_parent) / f"{file_stem}_LFP_Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    job_kwargs = dict(n_jobs=1, chunk_duration='30s', progress_bar=True)

    fmt = ("{l_bar}{bar}| {n_fmt}/{total_fmt} steps  "
           "[{elapsed}<{remaining}]")

    with tqdm(total=len(STEPS), unit='step', desc=f"[{STEPS[0]}]",
              bar_format=fmt) as sp:

        def advance(i):
            sp.set_description(f"[{STEPS[i]}]")
            sp.update(1)

        # ── 1. LOAD FILE ─────────────────────────────────────────────────────
        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"▶  {file_path_obj.name}")
        tqdm.write("Step 1/8 — Loading file")

        rec, num_channels, fs_orig, voltage_scale = \
            load_recording(file_path_obj, output_dir)

        decimation_factor = int(fs_orig / target_fs)
        actual_fs         = fs_orig / decimation_factor
        advance(1)

        # ── 2. PROBE GEOMETRY ────────────────────────────────────────────────
        tqdm.write("Step 2/8 — Attaching probe geometry")
        probe = pi.Probe(ndim=2, si_units='um')
        positions       = np.zeros((num_channels, 2))
        positions[:, 1] = np.arange(num_channels) * 20
        probe.set_contacts(positions=positions, shapes='circle',
                           shape_params={'radius': 5})
        probe.set_device_channel_indices(np.arange(num_channels))
        rec = rec.set_probe(probe)   # deepcopy is now tiny ✓
        tqdm.write("  ✓ Probe attached")
        advance(2)

        # ── 3. LFP CACHE ─────────────────────────────────────────────────────
        tqdm.write("Step 3/8 — Streaming LFP (1–450 Hz → decimate to 1 kHz)")
        rec_filtered = spre.bandpass_filter(rec, freq_min=1.0, freq_max=450.0)
        rec_lfp      = spre.resample(rec_filtered,
                                     resample_rate=int(target_fs))
        temp_lfp_cache = output_dir / "temp_si_cache"
        if temp_lfp_cache.exists():
            shutil.rmtree(temp_lfp_cache)
        rec_lfp.save(folder=temp_lfp_cache, format="binary", **job_kwargs)
        tqdm.write("  ✓ LFP cache written")
        advance(3)

        # ── 4. CLEANEST EEG CHANNELS ─────────────────────────────────────────
        tqdm.write("Step 4/8 — Selecting cleanest EEG channels")
        rec_lfp_cached      = si.load_extractor(temp_lfp_cache)
        best_ch_idx, scores = select_cleanest_channels(rec_lfp_cached, n_best=3)
        np.save(output_dir / "cleanest_channel_indices.npy", best_ch_idx)
        np.save(output_dir / "channel_snr_scores.npy", scores)
        advance(4)

        # ── 5. EMG CACHE ─────────────────────────────────────────────────────
        tqdm.write("Step 5/8 — Streaming EMG (10–100 Hz → decimate to 1 kHz)")
        emg_ch = select_emg_channel(rec, fs_orig)
        np.save(output_dir / "emg_channel_index.npy", np.array([emg_ch]))

        rec_emg_filtered = spre.bandpass_filter(
            rec.select_channels([rec.channel_ids[emg_ch]]),
            freq_min=10.0, freq_max=100.0)
        rec_emg_dec = spre.decimate(rec_emg_filtered,
                                    decimation_factor=decimation_factor)
        temp_emg_cache = output_dir / "temp_emg_cache"
        if temp_emg_cache.exists():
            shutil.rmtree(temp_emg_cache)
        rec_emg_dec.save(folder=temp_emg_cache, format="binary", **job_kwargs)
        tqdm.write("  ✓ EMG cache written")
        advance(5)

        # ── 6. WRITE LFP .npy ────────────────────────────────────────────────
        tqdm.write("Step 6/8 — Writing LFP .npy (chunked)")
        lfp_out  = output_dir / "lfp_data.npy"
        lfp_mmap = load_binary_memmap(temp_lfp_cache, num_channels)
        save_in_chunks(lfp_mmap, lfp_out, desc="  Writing lfp_data.npy")
        timestamps = (np.arange(lfp_mmap.shape[0]) / actual_fs).astype('float64')
        np.save(output_dir / "lfp_timestamps.npy", timestamps)
        np.save(output_dir / "lfp_channels.npy", np.arange(num_channels))
        tqdm.write(f"  ✓ lfp_data.npy  {lfp_mmap.shape}")
        del lfp_mmap
        advance(6)

        # ── 7. WRITE EMG .npy ────────────────────────────────────────────────
        tqdm.write("Step 7/8 — Writing EMG .npy (chunked)")
        emg_out  = output_dir / "emg_data.npy"
        emg_mmap = load_binary_memmap(temp_emg_cache, 1)
        save_in_chunks(emg_mmap, emg_out, desc="  Writing emg_data.npy")
        tqdm.write(f"  ✓ emg_data.npy  {emg_mmap.shape}")
        del emg_mmap
        gc.collect()

        for cache_dir in [temp_lfp_cache, temp_emg_cache]:
            try:
                shutil.rmtree(cache_dir)
            except PermissionError:
                tqdm.write(f"  Warning: could not delete {cache_dir} "
                           "(Windows lock). Delete manually.")
        advance(7)

        # ── 8. AWAKENESS ─────────────────────────────────────────────────────
        tqdm.write("Step 8/8 — Computing awakeness score (epoch-by-epoch)")
        awakeness, emg_rms, theta_delta = compute_awakeness(
            lfp_out, emg_out, actual_fs, best_ch_idx
        )
        np.save(output_dir / "awakeness.npy",          awakeness)
        np.save(output_dir / "emg_rms.npy",            emg_rms.astype('float32'))
        np.save(output_dir / "theta_delta_ratio.npy",   theta_delta.astype('float32'))
        advance(8)   # completes the bar

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"✅  All outputs → {output_dir}")
    tqdm.write(f"   lfp_data.npy                 {num_channels} ch, 1-450 Hz @ {actual_fs} Hz")
    tqdm.write(f"   lfp_timestamps.npy           time axis (s)")
    tqdm.write(f"   lfp_channels.npy             channel indices")
    tqdm.write(f"   cleanest_channel_indices.npy top-3 EEG ch: {best_ch_idx}")
    tqdm.write(f"   channel_snr_scores.npy       SNR scores (all channels)")
    tqdm.write(f"   emg_data.npy                 channel {emg_ch}, 10-100 Hz")
    tqdm.write(f"   emg_channel_index.npy        EMG channel index")
    tqdm.write(f"   awakeness.npy                per-second awakeness score")
    tqdm.write(f"   emg_rms.npy                  per-second EMG RMS")
    tqdm.write(f"   theta_delta_ratio.npy        per-second θ/δ ratio")
    tqdm.write(f"{'='*60}")


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_folder, output_folder):
    base_path = Path(input_folder)
    dat_files = list(base_path.glob("**/*.raw/*_group0.dat"))

    if not dat_files:
        print(f"No matching .dat files found under {input_folder}")
        return

    print(f"\nFound {len(dat_files)} file(s) to process.\n")
    fmt = ("{l_bar}{bar}| {n_fmt}/{total_fmt} files  "
           "[{elapsed}<{remaining}]")

    with tqdm(total=len(dat_files), unit='file',
              desc="Overall progress", bar_format=fmt) as fp:
        for dat_file in dat_files:
            try:
                extract_lfp_and_sort(dat_file, output_folder)
            except Exception as e:
                import traceback
                tqdm.write(f"\n❌  ERROR — {dat_file.name}: {e}")
                traceback.print_exc()
            finally:
                fp.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory-safe LFP extraction for large Trodes recordings."
    )
    parser.add_argument('--input_folder',  required=True,
                        help="Root folder containing *_group0.dat files.")
    parser.add_argument('--output_folder', required=True,
                        help="Destination for all output files.")
    args = parser.parse_args()
    run_pipeline(args.input_folder, args.output_folder)