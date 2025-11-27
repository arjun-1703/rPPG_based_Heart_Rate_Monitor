import cv2
import time
from config import *

class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.face_mesh = None
        self.countdown_start = None
        self.countdown_active = False
        self.last_update_time = None
    
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        if not self.cap.isOpened():
            raise Exception("Error: Camera not detected.")
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        return self.face_mesh
    
    def process_frame(self, frame, roi_manager, signal_processor, visualization):
        current_rois = []
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                
                if landmark_points:
                    xs, ys = zip(*landmark_points)
                    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
                    face_width, face_height = x_max - x_min, y_max - y_min
                    
                    current_rois = roi_manager.get_three_main_rois((x_min, y_min, face_width, face_height))
                    current_time = time.time()
                    signal_processor.timestamps.append(current_time)
                    
                    for i, roi_info in enumerate(current_rois):
                        intensity, quality, _ = roi_manager.extract_roi_signal(frame, roi_info)
                        
                        if i not in signal_processor.roi_signals:
                            signal_processor.roi_signals[i] = []
                        signal_processor.roi_signals[i].append(intensity)
                        
                        color = roi_info['color']
                        cv2.rectangle(frame, (roi_info['x1'], roi_info['y1']), 
                                    (roi_info['x2'], roi_info['y2']), color, 2)
                        cv2.putText(frame, roi_info['name'], (roi_info['x1'], roi_info['y1']-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    signal_processor.update_roi_weights(current_rois)
                    combined_signal = signal_processor.combine_roi_signals(current_rois)
                    
                    visualization.graph_signals['raw'].append(combined_signal)
                    if len(visualization.graph_signals['raw']) > 100:
                        visualization.graph_signals['raw'].pop(0)
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    signal_processor.cleanup_buffers()
                    
                    if len(signal_processor.roi_signals['combined']) > 45:
                        elapsed_time = signal_processor.timestamps[-1] - signal_processor.timestamps[0]
                        fs = len(signal_processor.roi_signals['combined']) / elapsed_time if elapsed_time > 0 else 30
                        
                        bpm, filtered, freqs, fft_values = signal_processor.calculate_heart_rate(fs)
                        
                        if bpm is not None:
                            visualization.graph_signals['filtered'] = filtered.tolist()[-100:]
                            visualization.graph_signals['freqs'] = freqs
                            visualization.graph_signals['fft'] = fft_values
                            
                            current_time = time.time()
                            if self.last_update_time is None:
                                signal_processor.update_bpm(bpm)
                                self.last_update_time = current_time
                                self.countdown_start = current_time
                                self.countdown_active = True
                            elif (current_time - self.last_update_time) >= UPDATE_INTERVAL:
                                if signal_processor.avg_snr > 2:
                                    signal_processor.update_bpm(bpm)
                                    self.last_update_time = current_time
                                    self.countdown_start = current_time
                                else:
                                    signal_processor.confidence = max(0, signal_processor.confidence - 10)
        
        return frame, current_rois
    
    def release(self):
        if self.cap:
            self.cap.release()