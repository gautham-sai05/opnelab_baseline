import numpy as np
import scipy.signal as signal

def remove_invalid_samples(waveform):
    return np.nan_to_num(waveform)

def remove_dc(waveform):
    return waveform - np.mean(waveform)

def apply_anti_alias(waveform, orig_sr, target_sr):
    if orig_sr <= target_sr:
        return waveform
    nyq = 0.5 * target_sr
    b, a = signal.butter(4, nyq / (0.5 * orig_sr), btype='low')
    return signal.filtfilt(b, a, waveform)

def resample(waveform, orig_sr, target_sr):
    if orig_sr == target_sr:
        return waveform
    num_samples = int(len(waveform) * target_sr / orig_sr)
    return signal.resample(waveform, num_samples)

def preprocess_waveform(waveform, orig_sr, target_sr=1000):
    waveform = remove_invalid_samples(waveform)
    waveform = remove_dc(waveform)
    waveform = apply_anti_alias(waveform, orig_sr, target_sr)
    waveform = resample(waveform, orig_sr, target_sr)
    return waveform

def create_windows(waveform, window_size=1000, hop_size=700):
    windows = []
    for i in range(0, len(waveform) - window_size + 1, hop_size):
        windows.append(waveform[i:i + window_size])
    return np.array(windows)

def standardize_windows(windows):
    means = np.mean(windows, axis=1, keepdims=True)
    stds = np.std(windows, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (windows - means) / stds
