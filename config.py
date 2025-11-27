import mediapipe as mp

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Signal processing parameters
BUFFER_DURATION = 15
UPDATE_INTERVAL = 5
BANDPASS_LOWCUT = 0.8
BANDPASS_HIGHCUT = 3.5
FILTER_ORDER = 4

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60

# Display settings
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 700
GRAPH_WIDTH = 550
GRAPH_HEIGHT = 200

# ROI weights (initial)
ROI_WEIGHTS = {
    'forehead': 0.4,
    'left_cheek': 0.3, 
    'right_cheek': 0.3
}

# Skin detection parameters
SKIN_LOWER_YCRCB = [0, 133, 77]
SKIN_UPPER_YCRCB = [255, 173, 127]
SKIN_LOWER_HSV = [0, 30, 60]
SKIN_UPPER_HSV = [25, 150, 255]