#!/usr/bin/env python3
"""Generate real-world waterfall diagrams and audio from the Panoradio HF dataset.

Reads the Panoradio radio signal classification dataset (complex IQ samples)
and produces for each of 18 radio modes:
  1. Waterfall spectrogram image (PNG) — narrow, matching presentation layout
  2. SNR comparison image — same mode at 25/15/5/-5 dB side by side
  3. Power spectrum plot (PNG)
  4. WAV audio file (48 kHz) — playable in browser
"""

import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import wavfile
from scipy.signal import resample_poly

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../panoradio'))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'assets')

NPY_FILE = os.path.join(DATASET_DIR, 'dataset_hf_radio.npy')
CSV_FILE = os.path.join(DATASET_DIR, 'dataset_panoradio_hf_tags.csv')

# === Parameters ===
FS = 6000           # Dataset sample rate (Hz)
FS_OUT = 48000      # Output audio sample rate
N_CONCAT = 25       # Samples to concatenate (~8.5s of audio)
SAMPLES_PER_BLOCK = 2048

# === Custom waterfall colormap (dark → blue → cyan → yellow → white) ===
CMAP = LinearSegmentedColormap.from_list('wf', [
    (0.01, 0.01, 0.05),
    (0.03, 0.05, 0.20),
    (0.03, 0.25, 0.50),
    (0.15, 0.60, 0.70),
    (0.85, 0.85, 0.25),
    (1.0, 1.0, 0.85),
], N=256)

# === Mode metadata ===
# fc    = audio center frequency for output
# bw    = display bandwidth in waterfall (Hz around signal center)
# nfft  = FFT size — smaller = better time resolution (keying, FSK transitions)
#                     larger  = better frequency resolution (narrow tones)
# align = frequency alignment strategy: 'peak' (default), 'centroid', or 'none'
MODES = {
    # CW: must resolve ~60ms dits → need NFFT≤128 (21ms frames)
    'morse':         {'label': 'CW (Morse)',      'fc': 700,  'bw': 500,  'nfft': 128},
    # PSK: ultra-narrow single tone, need freq resolution to show thin line
    'psk31':         {'label': 'PSK31',            'fc': 1000, 'bw': 300,  'nfft': 512},
    'psk63':         {'label': 'PSK63',            'fc': 1000, 'bw': 400,  'nfft': 256},
    'qpsk31':        {'label': 'QPSK31',           'fc': 1000, 'bw': 300,  'nfft': 512},
    # RTTY: need to resolve two tones AND the FSK transitions
    'rtty45_170':    {'label': 'RTTY 45/170',      'fc': 1500, 'bw': 600,  'nfft': 256},
    'rtty50_170':    {'label': 'RTTY 50/170',      'fc': 1500, 'bw': 600,  'nfft': 256},
    'rtty100_850':   {'label': 'RTTY 100/850',     'fc': 1500, 'bw': 1600, 'nfft': 128},
    # Olivia: need freq resolution to see individual tones hopping
    'olivia8_250':   {'label': 'Olivia 8/250',     'fc': 1500, 'bw': 600,  'nfft': 512},
    'olivia16_500':  {'label': 'Olivia 16/500',    'fc': 1500, 'bw': 1000, 'nfft': 256},
    'olivia16_1000': {'label': 'Olivia 16/1000',   'fc': 1500, 'bw': 1600, 'nfft': 256},
    'olivia32_1000': {'label': 'Olivia 32/1000',   'fc': 1500, 'bw': 1600, 'nfft': 256},
    # DominoEX: ~93ms symbols, need moderate time resolution
    'dominoex11':    {'label': 'DominoEX 11',      'fc': 1500, 'bw': 500,  'nfft': 256},
    # MT63: wide OFDM band, moderate resolution is fine
    'mt63_1000':     {'label': 'MT63/1000',        'fc': 1500, 'bw': 1600, 'nfft': 256},
    # NAVTEX: 100 baud FSK, need time resolution for fast transitions
    'navtex':        {'label': 'NAVTEX',           'fc': 1000, 'bw': 700,  'nfft': 128},
    # Voice/analog: wideband, use centroid alignment for AM carrier
    'usb':           {'label': 'USB (Voice)',       'fc': 1500, 'bw': 4000, 'nfft': 256, 'align': 'centroid'},
    'lsb':           {'label': 'LSB (Voice)',       'fc': 1500, 'bw': 4000, 'nfft': 256, 'align': 'centroid'},
    'am':            {'label': 'AM',               'fc': 1500, 'bw': 4000, 'nfft': 256, 'align': 'centroid'},
    'fax':           {'label': 'Fax',              'fc': 1900, 'bw': 1600, 'nfft': 256},
}


def load_metadata():
    """Load CSV metadata and return as structured arrays."""
    indices, modes, snrs = [], [], []
    with open(CSV_FILE) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                indices.append(int(parts[0].strip()))
                modes.append(parts[1].strip())
                snrs.append(int(parts[2].strip()))
    return np.array(indices), np.array(modes), np.array(snrs)


def estimate_freq_offset(iq_block, method='peak'):
    """Estimate the frequency offset of an IQ block.

    method='peak': find the single strongest frequency (good for CW, PSK, FSK)
    method='centroid': power-weighted centroid (good for wideband/AM)
    """
    fft = np.fft.fft(iq_block)
    power = np.abs(fft) ** 2
    freqs = np.fft.fftfreq(len(iq_block), 1.0 / FS)

    if method == 'centroid':
        # Use power-weighted centroid (wraps correctly for complex FFT)
        # Weight by power above the median to ignore noise floor
        threshold = np.percentile(power, 75)
        mask = power > threshold
        if np.sum(mask) > 0:
            return np.average(freqs[mask], weights=power[mask])
        # Fallback to peak
        method = 'peak'

    # Peak method: smooth to avoid noise spikes
    kernel_size = max(3, len(power) // 100)
    kernel = np.ones(kernel_size) / kernel_size
    power_smooth = np.convolve(power, kernel, mode='same')
    peak_idx = np.argmax(power_smooth)
    return freqs[peak_idx]


def align_and_concat(data, indices, mode_key=None):
    """Frequency-align IQ samples and concatenate with crossfading.

    Each sample in the Panoradio dataset has a random frequency offset of ±250 Hz.
    We estimate and remove this offset so that concatenated samples are coherent.
    Crossfading at block boundaries smooths amplitude/phase discontinuities.
    """
    info = MODES.get(mode_key, {})
    align_method = info.get('align', 'peak')

    blocks = []
    ref_freq = None

    for idx in indices:
        block = np.array(data[idx], dtype=np.complex128)

        # Estimate frequency offset
        offset = estimate_freq_offset(block, method=align_method)

        # Use first sample's offset as reference
        if ref_freq is None:
            ref_freq = offset

        # Shift to align with reference
        shift = ref_freq - offset
        if abs(shift) > 1.0:
            t = np.arange(len(block)) / FS
            block = block * np.exp(1j * 2 * np.pi * shift * t)

        blocks.append(block)

    # Crossfade concatenation to smooth block boundaries
    if len(blocks) <= 1:
        return (np.concatenate(blocks) if blocks else np.array([])), ref_freq or 0.0

    fade_len = 64  # ~10ms crossfade at 6 kHz
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = 1 - fade_in

    result = blocks[0].copy()
    for block in blocks[1:]:
        # Crossfade: blend end of result with start of next block
        result[-fade_len:] = result[-fade_len:] * fade_out + block[:fade_len] * fade_in
        result = np.concatenate([result, block[fade_len:]])

    return result, ref_freq


def complex_spectrogram(iq, nfft=256, noverlap_frac=0.9):
    """Compute spectrogram of complex IQ data (full bandwidth, -fs/2 to +fs/2)."""
    noverlap = int(nfft * noverlap_frac)
    hop = nfft - noverlap
    n_frames = (len(iq) - nfft) // hop
    window = np.hanning(nfft)

    spec = np.zeros((nfft, n_frames))
    for i in range(n_frames):
        start = i * hop
        frame = iq[start:start + nfft] * window
        fft_result = np.fft.fftshift(np.fft.fft(frame))
        spec[:, i] = np.abs(fft_result) ** 2

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / FS))
    times = np.arange(n_frames) * hop / FS
    return freqs, times, spec


def iq_to_audio(iq, center_freq):
    """Convert complex IQ to real audio by mixing up to center_freq."""
    t = np.arange(len(iq)) / FS
    audio = np.real(iq * np.exp(1j * 2 * np.pi * center_freq * t))
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85
    return audio


def generate_waterfall(mode_key, iq, ref_freq, out_path):
    """Generate narrow waterfall spectrogram image for a mode slide."""
    info = MODES.get(mode_key, {'bw': 1000, 'nfft': 256})
    nfft = info.get('nfft', 256)
    bw = info['bw']

    freqs, times, spec = complex_spectrogram(iq, nfft=nfft)

    # Convert to dB
    spec_db = 10 * np.log10(spec + 1e-12)

    # Normalize with good contrast
    vmin = np.percentile(spec_db, 15)
    vmax = np.percentile(spec_db, 99.5)
    spec_norm = np.clip((spec_db - vmin) / (vmax - vmin + 1e-6), 0, 1)

    # Crop frequency axis around the signal (centered on ref_freq)
    half_bw = bw / 2
    freq_mask = (freqs >= ref_freq - half_bw) & (freqs <= ref_freq + half_bw)

    if np.sum(freq_mask) < 15:
        # Fallback: show full bandwidth
        freq_mask = np.ones(len(freqs), dtype=bool)

    spec_crop = spec_norm[freq_mask, :]

    # Create figure — 220x280 at 2x for retina
    fig, ax = plt.subplots(figsize=(2.2, 2.8), dpi=200)
    fig.patch.set_facecolor('#0a0a14')

    # Display: frequency on x-axis, time on y-axis (downward)
    ax.imshow(spec_crop.T, aspect='auto', origin='upper', cmap=CMAP,
              interpolation='bilinear')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0,
                facecolor='#0a0a14', dpi=200)
    plt.close(fig)


def generate_snr_comparison(mode_key, all_idx, all_modes, all_snrs, data, out_path):
    """Generate side-by-side SNR comparison image."""
    snr_levels = [25, 15, 5, -5]
    n_per_snr = 10
    info = MODES.get(mode_key, {'bw': 1000, 'nfft': 256})
    nfft = info.get('nfft', 256)
    bw = info['bw']

    fig, axes = plt.subplots(1, len(snr_levels), figsize=(8, 3), dpi=150)
    fig.patch.set_facecolor('#0a0a14')

    # Use a consistent dB range across all SNR panels
    global_vmin, global_vmax = None, None

    panels = []
    for snr in snr_levels:
        mask = (all_modes == mode_key) & (all_snrs == snr)
        indices = all_idx[mask][:n_per_snr]
        if len(indices) == 0:
            panels.append(None)
            continue

        iq, ref_freq = align_and_concat(data, indices, mode_key)
        freqs, times, spec = complex_spectrogram(iq, nfft=nfft)
        spec_db = 10 * np.log10(spec + 1e-12)

        half_bw = bw / 2
        freq_mask = (freqs >= ref_freq - half_bw) & (freqs <= ref_freq + half_bw)
        if np.sum(freq_mask) < 10:
            freq_mask = np.ones(len(freqs), dtype=bool)

        cropped = spec_db[freq_mask, :]

        if global_vmin is None:
            global_vmin = np.percentile(cropped, 10)
            global_vmax = np.percentile(cropped, 99.5)

        panels.append(cropped)

    for col, (snr, panel) in enumerate(zip(snr_levels, panels)):
        if panel is None:
            axes[col].axis('off')
            continue

        norm = np.clip((panel - global_vmin) / (global_vmax - global_vmin + 1e-6), 0, 1)
        axes[col].imshow(norm.T, aspect='auto', origin='upper', cmap=CMAP,
                         interpolation='bilinear')
        axes[col].set_title(f'{snr} dB SNR', color='white', fontsize=10, pad=4)
        axes[col].axis('off')

    plt.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.88, bottom=0.02)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05,
                facecolor='#0a0a14', dpi=150)
    plt.close(fig)


def generate_spectrum(mode_key, iq, out_path):
    """Generate averaged power spectrum plot."""
    nfft = 1024
    n_frames = len(iq) // nfft
    psd = np.zeros(nfft)
    window = np.hanning(nfft)
    for i in range(n_frames):
        frame = iq[i * nfft:(i + 1) * nfft] * window
        psd += np.abs(np.fft.fftshift(np.fft.fft(frame))) ** 2
    psd /= max(n_frames, 1)
    psd_db = 10 * np.log10(psd + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / FS))

    fig, ax = plt.subplots(figsize=(4, 1.8), dpi=150)
    fig.patch.set_facecolor('#0a0a14')
    ax.set_facecolor('#0a0a14')
    ax.plot(freqs, psd_db, color='#55bbff', linewidth=0.8)
    ax.fill_between(freqs, psd_db, psd_db.min(), alpha=0.15, color='#55bbff')
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_xlabel('Frequency offset (Hz)', color='#888', fontsize=8)
    ax.set_ylabel('Power (dB)', color='#888', fontsize=8)
    ax.tick_params(colors='#666', labelsize=7)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#444')
    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05,
                facecolor='#0a0a14', dpi=150)
    plt.close(fig)


def main():
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_DIR}")

    for subdir in ['waterfalls', 'audio', 'snr', 'spectra']:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

    print("Loading metadata...")
    all_idx, all_modes, all_snrs = load_metadata()
    unique_modes = sorted(set(all_modes))
    print(f"Found {len(unique_modes)} modes: {', '.join(unique_modes)}")

    print("Memory-mapping dataset...")
    data = np.load(NPY_FILE, mmap_mode='r')
    print(f"Dataset shape: {data.shape}, dtype: {data.dtype}")

    for mode_key in unique_modes:
        info = MODES.get(mode_key, {'label': mode_key, 'fc': 1500, 'bw': 1000, 'nfft': 256})
        print(f"\n{'='*50}")
        print(f"Processing: {info['label']} ({mode_key})")

        # Get high-SNR samples
        mask = (all_modes == mode_key) & (all_snrs == 25)
        indices = all_idx[mask][:N_CONCAT]
        if len(indices) == 0:
            print(f"  WARNING: No samples found for {mode_key}")
            continue

        # Frequency-align and concatenate
        iq, ref_freq = align_and_concat(data, indices, mode_key)
        print(f"  Aligned {len(indices)} samples → {len(iq)} IQ ({len(iq)/FS:.1f}s), ref_freq={ref_freq:.0f} Hz")

        # 1. Waterfall image
        wf_path = os.path.join(OUTPUT_DIR, 'waterfalls', f'{mode_key}.png')
        print(f"  Waterfall  → {os.path.basename(wf_path)}")
        generate_waterfall(mode_key, iq, ref_freq, wf_path)

        # 2. Power spectrum
        spec_path = os.path.join(OUTPUT_DIR, 'spectra', f'{mode_key}.png')
        print(f"  Spectrum   → {os.path.basename(spec_path)}")
        generate_spectrum(mode_key, iq, spec_path)

        # 3. SNR comparison
        snr_path = os.path.join(OUTPUT_DIR, 'snr', f'{mode_key}.png')
        print(f"  SNR comp   → {os.path.basename(snr_path)}")
        generate_snr_comparison(mode_key, all_idx, all_modes, all_snrs, data, snr_path)

        # 4. Audio file
        audio_path = os.path.join(OUTPUT_DIR, 'audio', f'{mode_key}.wav')
        print(f"  Audio      → {os.path.basename(audio_path)}")
        audio = iq_to_audio(iq, info['fc'])
        audio_48k = resample_poly(audio, FS_OUT // (FS // 2), 2)  # 6000→48000 = ×8
        # Trim to exact length
        expected_len = int(len(audio) * FS_OUT / FS)
        audio_48k = audio_48k[:expected_len]
        audio_int16 = (np.clip(audio_48k, -1, 1) * 32767).astype(np.int16)
        wavfile.write(audio_path, FS_OUT, audio_int16)
        print(f"  Audio: {len(audio_48k)/FS_OUT:.1f}s, {os.path.getsize(audio_path)//1024} KB")

    print(f"\n{'='*50}")
    print("Done! Assets in:", OUTPUT_DIR)
    for subdir in ['waterfalls', 'audio', 'snr', 'spectra']:
        path = os.path.join(OUTPUT_DIR, subdir)
        files = [f for f in os.listdir(path) if not f.startswith('.')]
        total_kb = sum(os.path.getsize(os.path.join(path, f)) for f in files) // 1024
        print(f"  {subdir}/: {len(files)} files, {total_kb} KB")


if __name__ == '__main__':
    main()
