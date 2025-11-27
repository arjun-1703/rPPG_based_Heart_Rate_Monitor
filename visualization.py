import cv2
import numpy as np
import time
from scipy.signal import find_peaks
from config import *

class Visualization:
    def __init__(self):
        self.graph_signals = {
            'raw': [],
            'filtered': [],
            'freqs': [],
            'fft': []
        }
    
    def create_signal_graph(self, signal, width, height, color=(0, 255, 0), title="PPG Signal"):
        graph = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(signal) < 2:
            return graph
        
        signal_normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)
        signal_normalized = signal_normalized * (height - 40) + 20
        
        for i in range(0, width, 50):
            cv2.line(graph, (i, 0), (i, height), (50, 50, 50), 1)
        for i in range(0, height, 50):
            cv2.line(graph, (0, i), (width, i), (50, 50, 50), 1)
        
        points = []
        for i, value in enumerate(signal_normalized):
            x = int(i * width / len(signal_normalized))
            y = height - int(value)
            points.append((x, y))
        
        for i in range(1, len(points)):
            cv2.line(graph, points[i-1], points[i], color, 2)
        
        cv2.putText(graph, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(signal) > 30:
            try:
                peaks, _ = find_peaks(signal, distance=len(signal)//8, prominence=np.std(signal)*0.3)
                for peak in peaks:
                    if peak < len(points):
                        cv2.circle(graph, points[peak], 4, (0, 0, 255), -1)
            except:
                pass
        
        return graph
    
    def create_fft_graph(self, freqs, fft_values, width, height, current_bpm=0):
        graph = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(freqs) < 2 or len(fft_values) < 2:
            return graph
        
        hr_range = (freqs >= 0.5) & (freqs <= 4.0)
        hr_freqs = freqs[hr_range]
        hr_fft = fft_values[hr_range]
        
        if len(hr_freqs) < 2:
            return graph
        
        hr_fft_normalized = (hr_fft - np.min(hr_fft)) / (np.max(hr_fft) - np.min(hr_fft) + 1e-10)
        hr_fft_normalized = hr_fft_normalized * (height - 60) + 30
        
        for i in range(0, width, 50):
            cv2.line(graph, (i, 0), (i, height), (50, 50, 50), 1)
        for i in range(0, height, 50):
            cv2.line(graph, (0, i), (width, i), (50, 50, 50), 1)
        
        points = []
        for i, (freq, value) in enumerate(zip(hr_freqs, hr_fft_normalized)):
            x = int((freq - 0.5) / 3.5 * width)
            y = height - int(value)
            points.append((x, y))
        
        for i in range(1, len(points)):
            cv2.line(graph, points[i-1], points[i], (0, 200, 255), 2)
        
        if len(hr_fft) > 0:
            dominant_idx = np.argmax(hr_fft)
            dominant_freq = hr_freqs[dominant_idx]
            dominant_bpm = dominant_freq * 60
            
            if 0 <= dominant_idx < len(points):
                cv2.circle(graph, points[dominant_idx], 6, (0, 255, 255), -1)
                cv2.putText(graph, f"{dominant_bpm:.0f} BPM", 
                           (points[dominant_idx][0] + 10, points[dominant_idx][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        cv2.putText(graph, "Frequency Spectrum", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(graph, "0.5 Hz", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(graph, "4.0 Hz", (width-50, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return graph
    
    def create_display_frame(self, camera_frame, signal_processor, current_rois, video_processor):
        display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        
        camera_display = cv2.resize(camera_frame, (500, 400))
        display_frame[50:450, 50:550] = camera_display
        
        self._create_control_panel(display_frame, signal_processor, current_rois, video_processor)
        self._create_signal_graphs(display_frame, signal_processor.current_bpm)
        self._add_instructions(display_frame)
        
        return display_frame
    
    def _create_control_panel(self, display_frame, signal_processor, current_rois, video_processor):
        panel_x, panel_y = 600, 50
        
        cv2.rectangle(display_frame, (panel_x, panel_y), (panel_x + 250, panel_y + 400), (40, 40, 40), -1)
        cv2.rectangle(display_frame, (panel_x, panel_y), (panel_x + 250, panel_y + 400), (100, 100, 100), 2)
        
        cv2.putText(display_frame, "ROI WEIGHTS & STATUS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if current_rois and len(current_rois) > 0 and len(signal_processor.timestamps) > 10:
            weight_y = panel_y + 60
            for i, roi_info in enumerate(current_rois):
                if i in signal_processor.roi_weights:
                    weight = signal_processor.roi_weights[i] * 100
                    color = roi_info['color']
                    
                    weight_text = f"{roi_info['name']}: {weight:.1f}%"
                    cv2.putText(display_frame, weight_text, (panel_x + 20, weight_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    bar_width = int(weight * 1.5)
                    cv2.rectangle(display_frame, (panel_x + 150, weight_y - 10),
                                (panel_x + 150 + bar_width, weight_y), color, -1)
                    
                    weight_y += 35
        
        status_y = panel_y + 200
        bpm_color = (0, 255, 0) if signal_processor.confidence > 50 else (0, 165, 255)
        cv2.putText(display_frame, f"HEART RATE: {int(signal_processor.current_bpm)} BPM", 
                   (panel_x + 20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bpm_color, 2)
        
        cv2.putText(display_frame, f"Confidence: {int(signal_processor.confidence)}%", 
                   (panel_x + 20, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"SNR: {signal_processor.avg_snr:.1f} dB", 
                   (panel_x + 20, status_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if signal_processor.confidence > 70:
            quality_text, quality_color = "EXCELLENT SIGNAL", (0, 255, 0)
        elif signal_processor.confidence > 50:
            quality_text, quality_color = "GOOD SIGNAL", (0, 255, 255)
        elif signal_processor.confidence > 30:
            quality_text, quality_color = "FAIR SIGNAL", (0, 165, 255)
        else:
            quality_text, quality_color = "POOR SIGNAL", (0, 0, 255)
            
        cv2.putText(display_frame, quality_text, (panel_x + 20, status_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
        
        # ADD COUNTDOWN TIMER DISPLAY
        current_time = time.time()
        if video_processor.countdown_start is not None and video_processor.countdown_active:
            time_elapsed = current_time - video_processor.countdown_start
            time_remaining = max(0, UPDATE_INTERVAL - time_elapsed)
            countdown_text = f"Next update: {time_remaining:.1f}s"
            cv2.putText(display_frame, countdown_text, (panel_x + 20, status_y + 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _create_signal_graphs(self, display_frame, current_bpm):
        graph_y = 470
        
        if self.graph_signals['raw']:
            raw_graph = self.create_signal_graph(self.graph_signals['raw'], GRAPH_WIDTH, GRAPH_HEIGHT, 
                                              color=(0, 255, 0), title="Raw PPG Signal")
            display_frame[graph_y:graph_y+GRAPH_HEIGHT, 50:50+GRAPH_WIDTH] = raw_graph
        
        if self.graph_signals['filtered']:
            filtered_graph = self.create_signal_graph(self.graph_signals['filtered'], GRAPH_WIDTH, GRAPH_HEIGHT,
                                                   color=(0, 200, 255), title="Filtered PPG Signal")
            display_frame[graph_y:graph_y+GRAPH_HEIGHT, 50+GRAPH_WIDTH+50:50+GRAPH_WIDTH+50+GRAPH_WIDTH] = filtered_graph
        
        if len(self.graph_signals['freqs']) > 0 and len(self.graph_signals['fft']) > 0:
            fft_graph = self.create_fft_graph(self.graph_signals['freqs'], self.graph_signals['fft'], 
                                           GRAPH_WIDTH, GRAPH_HEIGHT, current_bpm)
            if graph_y + GRAPH_HEIGHT * 2 + 20 < DISPLAY_HEIGHT:
                display_frame[graph_y+GRAPH_HEIGHT+20:graph_y+GRAPH_HEIGHT*2+20, 50:50+GRAPH_WIDTH] = fft_graph
    
    def _add_instructions(self, display_frame):
        cv2.putText(display_frame, "Three-ROI Heart Rate Monitor - Press 'q' to quit", 
                   (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, "Green: Forehead | Blue: Cheeks", 
                   (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)