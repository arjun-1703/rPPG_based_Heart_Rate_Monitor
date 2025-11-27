import cv2
import numpy as np
from config import *  # ADD THIS IMPORT

class ROIManager:
    def __init__(self):
        self.roi_definitions = [] 
               
    def get_three_main_rois(self, face_rect):
        x, y, w, h = face_rect
        rois = []
        
        forehead_roi = {
            'x1': x + int(w * 0.10), 'y1': y + int(h * 0.01),
            'x2': x + int(w * 0.90), 'y2': y + int(h * 0.15),
            'weight': ROI_WEIGHTS['forehead'], 'name': 'Forehead',
            'color': (0, 255, 0), 'type': 'forehead'
        }
        rois.append(forehead_roi)
        
        left_cheek_roi = {
            'x1': x + int(w * 0.05), 'y1': y + int(h * 0.40),
            'x2': x + int(w * 0.35), 'y2': y + int(h * 0.65),
            'weight': ROI_WEIGHTS['left_cheek'], 'name': 'Left Cheek',
            'color': (255, 0, 0), 'type': 'cheek'
        }
        rois.append(left_cheek_roi)
        
        right_cheek_roi = {
            'x1': x + int(w * 0.65), 'y1': y + int(h * 0.40),
            'x2': x + int(w * 0.95), 'y2': y + int(h * 0.65),
            'weight': ROI_WEIGHTS['right_cheek'], 'name': 'Right Cheek',
            'color': (255, 0, 0), 'type': 'cheek'
        }
        rois.append(right_cheek_roi)
        
        self.roi_definitions = rois
        return rois
    
    def adaptive_skin_mask(self, roi):
        if roi.size == 0:
            return None
            
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        lower_ycrcb = np.array(SKIN_LOWER_YCRCB, dtype=np.uint8)
        upper_ycrcb = np.array(SKIN_UPPER_YCRCB, dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        lower_hsv = np.array(SKIN_LOWER_HSV, dtype=np.uint8)
        upper_hsv = np.array(SKIN_UPPER_HSV, dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def extract_roi_signal(self, frame, roi_info):
        x1, y1, x2, y2 = roi_info['x1'], roi_info['y1'], roi_info['x2'], roi_info['y2']
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0, None, None
            
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0, None, None
        
        mask = self.adaptive_skin_mask(roi)
        if mask is None:
            return 0, None, None
        
        skin_roi = cv2.bitwise_and(roi, roi, mask=mask)
        green_channel = skin_roi[:, :, 1]
        
        if np.any(mask > 0):
            skin_pixels = green_channel[mask > 0]
            if len(skin_pixels) > 0:
                mean_intensity = np.mean(skin_pixels)
                signal_quality = np.std(skin_pixels) / (mean_intensity + 1e-10)
            else:
                mean_intensity = 0
                signal_quality = 1.0
        else:
            mean_intensity = np.mean(green_channel)
            signal_quality = 1.0
        
        return mean_intensity, signal_quality, roi