import gc
import re
import argparse
import numpy as np
from pathlib import Path
from scipy.signal import welch
from scipy.stats import zscore
from tqdm import tqdm
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
import probeinterface as pi


# ──────────────────────────────────────────────────────────────────────────────
# PARSE TRODES HEADER + MEMMAP  (zero-RAM replacement for readTrodesExtracted)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_trodes_fields(fieldstr):
    """Identical dtype parser from readTrodesExtractedDataFile3."""
    sep = re.split(r'\s', re.sub(r"\>\<|\>|\<", ' ', fieldstr).strip())
    typearr = []
    for i in range(0, len(sep), 2):
        fieldname = sep[i]
        repeats = 1
        ftype = 'uint32'
        if '*' in sep[i + 1]:
            parts = re.split(r'\*', sep[i + 1])
            ftype   = parts[parts[0].isdigit()]
            repeats = int(parts[parts[1].isdigit()])
        else:
            ftype = sep[i + 1]
        fieldtype = getattr(np, ftype)
        typearr.append((str(fieldname), fieldtype, repeats))
    return np.dtype(typearr)


def open_trodes_memmap(file_path):
    """
    Parse the Trodes header, then return:
      - metadata dict  (samplingRate, voltageScaling, …)
      - np.memmap of the structured binary data (zero RAM)
    """
    file_path = str(file_path)
    fields_text = {}

    with open(file_path, 'rb') as f:
        first_line = f.readline().decode('ascii').strip()
        if first_line != '<Start settings>':
            raise ValueError("Settings format not supported")

        for line in f:
            line = line.decode('ascii').strip()
            if line == '<End settings>':
                break
            key, val = line.split(': ', 1)
            fields_text[key.lower()] = val

        header_end = f.tell()                       # byte offset of data start

    dt       = _parse_trodes_fields(fields_text['fields'])
    data_mm  = np.memmap(file_path, dtype=dt, mode='r', offset=header_end)

    return fields_text, data_mm


# ──────────────────────────────────────────────────────────────────────────────
# FILTER + DECIMATE A SINGLE 1-D CHANNEL ARRAY
# ──────────────────────────────────────────────────────────────────────────────

def process_channel(ch_1d, fs_orig, voltage_scale,
                    freq_min, freq_max, decimation_factor):
    """
    Bandpass → decimate one channel.
    ch_1d  : 1-D array (n_samples,)  — may be a memmap slice
    Returns: 1-D float32 (n_lfp_samples,)
    """
    # Copy into a contiguous float32 column  (only ONE channel in RAM)
    ch_2d = np.array(ch_1d, dtype='float32')[:, np.newaxis]

    rec = si.NumpyRecording(
        traces_list=[ch_2d],
        sampling_frequency=fs_orig,
    )
    rec.set_channel_gains(voltage_scale)
    rec.set_channel_offsets(0.0)

    probe = pi.Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=np.array([[0.0, 0.0]]),
                       shapes='circle', shape_params={'radius': 5})
    probe.set_device_channel_indices(np.array([0]))
    rec = rec.set_probe(probe)

    rec_filt = spre.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max)
    rec_dec  = spre.decimate(rec_filt, decimation_factor=decimation_factor)

    out = rec_dec.get_traces()[:, 0].astype('float32')

    del ch_2d, rec, rec_filt, rec_dec
    gc.collect()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CHANNEL SELECTION  (reads from the memmap — no extra RAM)
# ──────────────────────────────────────────────────────────────────────────────

def select_emg_channel(voltage_mm, fs_orig, num_channels,
                       segment_duration_s=60):
    """Channel with highest power in the EMG band (20–200 Hz)."""
    n = min(int(segment_duration_s * fs_orig), voltage_mm.shape[0])
    emg_power = np.zeros(num_channels)

    with tqdm(total=num_channels, unit='ch', desc="  Scoring EMG channels",
              leave=False) as pb:
        for ch in range(num_channels):
            seg = np.array(voltage_mm[:n, ch], dtype='float64')
            f, psd        = welch(seg, fs=fs_orig, nperseg=int(2 * fs_orig))
            emg_power[ch] = np.mean(psd[(f >= 20) & (f <= 200)])
            del seg
            pb.update(1)

    emg_ch = int(np.argmax(emg_power))
    tqdm.write(f"  ✓ EMG channel: {emg_ch}  "
               f"(power: {emg_power[emg_ch]:.4f})")
    return emg_ch


def select_cleanest_channels(lfp_path, fs, n_best=3, segment_duration_s=60):
    """Sleep-band SNR from saved LFP .npy (mmap'd)."""
    lfp  = np.load(lfp_path, mmap_mode='r')
    n    = min(int(segment_duration_s * fs), lfp.shape[0])
    n_ch = lfp.shape[1]

    scores = np.zeros(n_ch)
    with tqdm(total=n_ch, unit='ch', desc="  Scoring EEG channels",
              leave=False) as pb:
        for ch in range(n_ch):
            seg        = np.array(lfp[:n, ch], dtype='float64')
            f, psd     = welch(seg, fs=fs, nperseg=int(2 * fs))
            sleep_band = np.mean(psd[(f >= 0.5) & (f <= 30)])
            noise_band = np.mean(psd[f > 50])
            scores[ch] = sleep_band / (noise_band + 1e-12)
            pb.update(1)

    del lfp
    best_idx = np.argsort(scores)[::-1][:n_best]
    tqdm.write(f"  ✓ Cleanest {n_best} channels: {best_idx}  "
               f"(scores: {scores[best_idx].round(2)})")
    return best_idx, scores


# ──────────────────────────────────────────────────────────────────────────────
# AWAKENESS
# ──────────────────────────────────────────────────────────────────────────────

def compute_awakeness(lfp_path, emg_path, fs, best_ch_idx, n_lfp_samples,
                      epoch_s=1):
    lfp = np.load(lfp_path, mmap_mode='r')
    emg = np.load(emg_path, mmap_mode='r')

    n_samples   = lfp.shape[0]
    n_per_epoch = int(fs * epoch_s)
    n_epochs    = n_samples // n_per_epoch
    eeg_ch      = int(best_ch_idx[0])

    emg_power   = np.zeros(n_epochs, dtype='float32')
    theta_delta = np.zeros(n_epochs, dtype='float32')

    with tqdm(total=n_epochs, unit='s', desc="  Awakeness epochs",
              leave=False) as pb:
        for i in range(n_epochs):
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

    del lfp, emg

    emg_z = zscore(emg_power)
    td_z  = zscore(theta_delta)
    awakeness_epochs = (0.6 * emg_z + 0.4 * td_z).astype('float32')

    tqdm.write(f"  Upsampling awakeness from {n_epochs} epochs → "
               f"{n_lfp_samples} samples")
    epoch_times  = (np.arange(n_epochs) + 0.5) * n_per_epoch
    sample_times = np.arange(n_lfp_samples, dtype='float64')
    awakeness    = np.interp(sample_times, epoch_times,
                             awakeness_epochs).astype('float32')
    emg_power_up   = np.interp(sample_times, epoch_times,
                               emg_power).astype('float32')
    theta_delta_up = np.interp(sample_times, epoch_times,
                               theta_delta).astype('float32')

    return awakeness, emg_power_up, theta_delta_up


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

STEPS = [
    "Open file (memmap)",
    "Select EMG channel",
    "Build LFP (channel by channel)",
    "Build EMG",
    "Select EEG channels",
    "Compute awakeness",
]


def extract_lfp_and_sort(file_path, output_parent, target_fs=1000.0):
    file_path_obj = Path(file_path)
    file_stem     = file_path_obj.stem
    output_dir    = Path(output_parent) / f"{file_stem}_LFP_Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = ("{l_bar}{bar}| {n_fmt}/{total_fmt} steps  "
           "[{elapsed}<{remaining}]")

    with tqdm(total=len(STEPS), unit='step', desc=f"[{STEPS[0]}]",
              bar_format=fmt) as sp:

        def advance(i):
            if i < len(STEPS):
                sp.set_description(f"[{STEPS[i]}]")
            sp.update(1)

        # ── 1. MEMMAP THE ORIGINAL .dat  (zero RAM) ─────────────────────────
        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"▶  {file_path_obj.name}")
        tqdm.write("Step 1/6 — Memory-mapping raw file (no RAM load)")

        meta, data_mm = open_trodes_memmap(file_path_obj)

        # The structured dtype has a 'voltage' field with shape (n_ch,)
        voltage_mm    = data_mm['voltage']              # view, still mmap'd
        n_samples_raw = voltage_mm.shape[0]
        num_channels  = voltage_mm.shape[1]
        fs_orig       = float(meta.get('samplingrate', 30000))
        voltage_scale = float(meta.get('voltagescaling', 0.195))

        decimation_factor = int(fs_orig / target_fs)
        actual_fs         = fs_orig / decimation_factor

        tqdm.write(f"  ✓ {n_samples_raw} samples × {num_channels} ch  "
                   f"@ {fs_orig} Hz → decimate ×{decimation_factor}")
        advance(1)

        # ── 2. PICK EMG CHANNEL (reads slices from the memmap) ───────────────
        tqdm.write("Step 2/6 — Selecting EMG channel")
        emg_ch = select_emg_channel(voltage_mm, fs_orig, num_channels)
        np.save(output_dir / "emg_channel_index.npy", np.array([emg_ch]))
        advance(2)

        # ── 3. LFP: process each channel straight from the memmap ────────────
        tqdm.write("Step 3/6 — Building LFP channel-by-channel "
                   "(1–450 Hz → 1 kHz)")

        # First channel → learn output length
        lfp_ch0 = process_channel(
            voltage_mm[:, 0], fs_orig, voltage_scale,
            freq_min=1.0, freq_max=450.0,
            decimation_factor=decimation_factor,
        )
        n_lfp_samples = lfp_ch0.shape[0]
        tqdm.write(f"  Output length: {n_lfp_samples} samples/ch")

        # Allocate output memmap (decimated size — much smaller than raw)
        lfp_tmp = output_dir / "_lfp_tmp.dat"
        lfp_mm  = np.memmap(lfp_tmp, dtype='float32', mode='w+',
                            shape=(n_lfp_samples, num_channels))
        lfp_mm[:, 0] = lfp_ch0
        lfp_mm.flush()
        del lfp_ch0

        for ch in tqdm(range(1, num_channels), desc="  LFP channels",
                       unit='ch', leave=False):
            lfp_mm[:, ch] = process_channel(
                voltage_mm[:, ch], fs_orig, voltage_scale,
                freq_min=1.0, freq_max=450.0,
                decimation_factor=decimation_factor,
            )
            lfp_mm.flush()

        lfp_out = output_dir / "lfp_data.npy"
        np.save(lfp_out, np.array(lfp_mm))
        del lfp_mm
        lfp_tmp.unlink(missing_ok=True)

        timestamps = (np.arange(n_lfp_samples) / actual_fs).astype('float64')
        np.save(output_dir / "lfp_timestamps.npy", timestamps)
        np.save(output_dir / "lfp_channels.npy",   np.arange(num_channels))
        tqdm.write(f"  ✓ lfp_data.npy  ({n_lfp_samples}, {num_channels})")
        advance(3)

        # ── 4. EMG: single channel straight from memmap ─────────────────────
        tqdm.write("Step 4/6 — Building EMG (10–100 Hz → 1 kHz)")
        emg_1d = process_channel(
            voltage_mm[:, emg_ch], fs_orig, voltage_scale,
            freq_min=10.0, freq_max=100.0,
            decimation_factor=decimation_factor,
        )
        emg_out = output_dir / "emg_data.npy"
        np.save(emg_out, emg_1d[:, np.newaxis])
        tqdm.write(f"  ✓ emg_data.npy  ({emg_1d.shape[0]}, 1)")
        del emg_1d

        # Release the memmap of the original file
        del voltage_mm, data_mm
        gc.collect()
        advance(4)

        # ── 5. CLEANEST EEG CHANNELS ────────────────────────────────────────
        tqdm.write("Step 5/6 — Selecting cleanest EEG channels")
        best_ch_idx, scores = select_cleanest_channels(lfp_out, actual_fs,
                                                       n_best=3)
        np.save(output_dir / "cleanest_channel_indices.npy", best_ch_idx)
        np.save(output_dir / "channel_snr_scores.npy", scores)
        advance(5)

        # ── 6. AWAKENESS ────────────────────────────────────────────────────
        tqdm.write("Step 6/6 — Computing awakeness score")
        awakeness, emg_rms, theta_delta = compute_awakeness(
            lfp_out, emg_out, actual_fs, best_ch_idx,
            n_lfp_samples=n_lfp_samples,
        )
        np.save(output_dir / "awakeness.npy",         awakeness)
        np.save(output_dir / "emg_rms.npy",           emg_rms)
        np.save(output_dir / "theta_delta_ratio.npy",  theta_delta)
        advance(6)

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"✅  All outputs → {output_dir}")
    tqdm.write(f"   lfp_data.npy                 {num_channels} ch, 1-450 Hz @ {actual_fs} Hz")
    tqdm.write(f"   lfp_timestamps.npy           time axis (s)")
    tqdm.write(f"   lfp_channels.npy             channel indices")
    tqdm.write(f"   cleanest_channel_indices.npy top-3 EEG ch: {best_ch_idx}")
    tqdm.write(f"   channel_snr_scores.npy       SNR scores (all channels)")
    tqdm.write(f"   emg_data.npy                 channel {emg_ch}, 10-100 Hz")
    tqdm.write(f"   emg_channel_index.npy        EMG channel index")
    tqdm.write(f"   awakeness.npy                per-sample ({n_lfp_samples})")
    tqdm.write(f"   emg_rms.npy                  per-sample ({n_lfp_samples})")
    tqdm.write(f"   theta_delta_ratio.npy        per-sample ({n_lfp_samples})")
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