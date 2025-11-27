import numpy as np
from scipy.signal import find_peaks
from utils import *
from config import *  # ADD THIS IMPORT

class SignalProcessor:
    def __init__(self):
        self.roi_signals = {}
        self.roi_weights = {}
        self.timestamps = []
        self.bpm_history = []
        self.snr_history = []
        self.current_bpm = 0
        self.confidence = 0
        self.avg_snr = 0
        
    def calculate_roi_quality(self, roi_signals):
        if len(self.timestamps) < 10:
            return [0.5] * len(roi_signals)
        
        fs = len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0]) if self.timestamps[-1] > self.timestamps[0] else 30
        
        quality_scores = []
        for signal in roi_signals:
            if len(signal) < 10:
                quality_scores.append(0.1)
                continue
                
            snr = calculate_snr(signal, fs)
            signal_diff = np.diff(signal)
            stability = 1.0 / (np.std(signal_diff) + 1e-10)
            stability = min(1.0, stability / 100.0)
            
            quality = 0.7 * min(1.0, snr / 10.0) + 0.3 * stability
            quality_scores.append(max(0.1, min(1.0, quality)))
        
        return quality_scores
    
    def update_roi_weights(self, current_rois):
        if len(self.timestamps) > 10:
            roi_qualities = self.calculate_roi_quality([self.roi_signals[i] for i in range(len(current_rois))])
            
            total_quality = sum(roi_qualities)
            if total_quality > 0:
                for i in range(len(current_rois)):
                    self.roi_weights[i] = roi_qualities[i] / total_quality
            else:
                for i, roi_info in enumerate(current_rois):
                    self.roi_weights[i] = roi_info['weight']
    
    def combine_roi_signals(self, current_rois):
        combined_signal = 0
        total_weight = 0
        
        for i in range(len(current_rois)):
            if i in self.roi_weights and len(self.roi_signals[i]) > 0:
                weight = self.roi_weights[i]
                combined_signal += self.roi_signals[i][-1] * weight
                total_weight += weight
        
        if total_weight > 0:
            combined_signal /= total_weight
        
        if 'combined' not in self.roi_signals:
            self.roi_signals['combined'] = []
        self.roi_signals['combined'].append(combined_signal)
        
        return combined_signal
    
    def calculate_heart_rate(self, fs):
        if len(self.roi_signals['combined']) <= 45:
            return None, None, None, None
        
        signal = self.roi_signals['combined']
        normalized = normalize_signal(signal)
        filtered = butter_bandpass_filter(normalized, BANDPASS_LOWCUT, BANDPASS_HIGHCUT, fs, FILTER_ORDER)
        
        snr = calculate_snr(filtered, fs)
        self.snr_history.append(snr)
        if len(self.snr_history) > 5:
            self.snr_history.pop(0)
        self.avg_snr = np.mean(self.snr_history)
        
        N = len(filtered)
        freqs = fftfreq(N, 1/fs)
        fft_values = np.abs(fft(filtered))
        freqs = freqs[:N//2]
        fft_values = fft_values[:N//2]
        
        valid_idx = np.where((freqs >= 0.8) & (freqs <= 3.5))
        if len(valid_idx[0]) == 0:
            return None, None, None, None
            
        dominant_idx = np.argmax(fft_values[valid_idx])
        fft_bpm = freqs[valid_idx][dominant_idx] * 60.0
        
        try:
            peaks, _ = find_peaks(filtered, distance=fs/2.5, height=np.std(filtered)*0.5)
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / fs
                avg_interval = np.mean(peak_intervals)
                peak_bpm = 60.0 / avg_interval
                
                if self.avg_snr > 3:
                    combined_bpm = (fft_bpm + peak_bpm) / 2
                else:
                    combined_bpm = fft_bpm
            else:
                combined_bpm = fft_bpm
        except:
            combined_bpm = fft_bpm
        
        return combined_bpm, filtered, freqs, fft_values
    
    def update_bpm(self, new_bpm):
        if 40 <= new_bpm <= 180:
            self.bpm_history.append(new_bpm)
            if len(self.bpm_history) > 12:
                self.bpm_history.pop(0)
            
            smoothed_bpm = np.median(self.bpm_history)
            self.current_bpm = smoothed_bpm
            self.confidence = min(100, self.avg_snr * 20)
    
    def cleanup_buffers(self):
        while len(self.timestamps) > 1 and (self.timestamps[-1] - self.timestamps[0]) > BUFFER_DURATION:
            self.timestamps.pop(0)
            for key in self.roi_signals:
                if len(self.roi_signals[key]) > 0:
                    self.roi_signals[key].pop(0)