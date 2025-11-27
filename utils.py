import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from config import *  # ADD THIS IMPORT

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def normalize_signal(signal):
    signal = np.array(signal)
    if len(signal) == 0:
        return signal
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

def calculate_snr(signal, fs):
    if len(signal) < 10:
        return 0
        
    N = len(signal)
    freqs = fftfreq(N, 1/fs)
    fft_vals = np.abs(fft(signal))
    
    hr_band = np.where((freqs >= 0.7) & (freqs <= 4.0))
    if len(hr_band[0]) == 0:
        return 0
    signal_power = np.sum(fft_vals[hr_band] ** 2)
    
    noise_band1 = np.where((freqs >= 0.1) & (freqs < 0.7))
    noise_band2 = np.where((freqs > 4.0) & (freqs <= 8.0))
    noise_power = np.sum(fft_vals[noise_band1] ** 2) + np.sum(fft_vals[noise_band2] ** 2)
    
    return 10 * np.log10(signal_power / (noise_power + 1e-10))