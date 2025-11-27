import cv2
import time
from config import *
from video_processor import VideoProcessor
from roi_manager import ROIManager
from signal_processor import SignalProcessor
from visualization import Visualization

def main():
    print("Press 'q' to quit.")
    print("rPPG based Heart Rate Monitor (Forehead + Cheeks)")
    
    # Initialize all components
    video_processor = VideoProcessor()
    roi_manager = ROIManager()
    signal_processor = SignalProcessor()
    visualization = Visualization()
    
    try:
        face_mesh = video_processor.initialize_camera()
    except Exception as e:
        print(e)
        return
    
    with face_mesh as face_mesh:
        video_processor.face_mesh = face_mesh
        
        while True:
            ret, frame = video_processor.cap.read()
            if not ret:
                break
            
            # Process the frame through the pipeline
            processed_frame, current_rois = video_processor.process_frame(
                frame, roi_manager, signal_processor, visualization
            )
            
            # Create and display the visualization
            display_frame = visualization.create_display_frame(
    processed_frame, signal_processor, current_rois, video_processor
            )
            
            cv2.imshow('Heart Rate Monitor - Three ROI System', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    video_processor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()