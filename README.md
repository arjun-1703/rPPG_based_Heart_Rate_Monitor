# rPPG_based_Heart_Rate_Monitor
Heart Rate Monitor - Contactless heart rate detection using webcam &amp; computer vision. Real-time PPG analysis from facial blood flow. No wearables needed. Python, OpenCV, MediaPipe. Educational/wellness use. Runs on standard webcams
Heart Rate Monitor - Remote PPG using Webcam
A real-time, non-contact heart rate monitoring system that uses computer vision and photoplethysmography (PPG) to detect heart rate through standard webcam footage.

https://img.shields.io/badge/Version-1.0-blue.svg
https://img.shields.io/badge/Python-3.8%252B-green.svg
https://img.shields.io/badge/OpenCV-4.5%252B-orange.svg

🎯 Overview
This project implements a remote photoplethysmography (rPPG) system that can detect heart rate by analyzing subtle color changes in facial skin caused by blood circulation. The system uses multiple facial regions (forehead and cheeks) for robust signal extraction and combines advanced signal processing techniques for accurate heart rate calculation.

✨ Features
Real-time Processing: Live heart rate monitoring at 30 FPS

Multi-ROI Analysis: Simultaneously monitors forehead and both cheeks

Adaptive Signal Quality: Automatically weights better quality signals

Dual-Method Detection: Combines FFT frequency analysis and peak detection

Skin Detection: Robust skin pixel isolation using YCrCb and HSV color spaces

Visual Feedback: Real-time graphs and signal quality indicators

Confidence Scoring: Intelligent confidence estimation for each measurement

🛠️ How It Works
Technical Pipeline
Face Detection: MediaPipe detects facial landmarks and creates bounding boxes

ROI Definition: Three regions of interest (forehead, left cheek, right cheek) are defined

Skin Pixel Extraction: Adaptive skin masks isolate skin pixels using YCrCb and HSV color spaces

Signal Extraction: Green channel intensity is measured from skin pixels over time

Signal Processing:

Normalization and bandpass filtering (0.8-3.5 Hz)

FFT analysis for frequency-domain heart rate detection

Peak detection for time-domain analysis

Signal Fusion: Weighted combination of multiple ROI signals

Result Display: Real-time visualization with confidence metrics

Signal Processing Flow
text
Webcam → Face Detection → ROI Extraction → Skin Masking → Green Channel → 
Normalization → Bandpass Filtering → FFT Analysis + Peak Detection → 
BPM Calculation → Confidence Assessment → Display
📋 Requirements
Hardware
Webcam (standard USB camera)

Modern CPU (Intel i5 or equivalent minimum)

4GB RAM minimum

Software
Python 3.8 or higher

Webcam drivers

🚀 Installation
Method 1: Quick Install
bash
# Clone the repository
git clone https://github.com/yourusername/heart-rate-monitor.git
cd heart-rate-monitor

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
Method 2: Manual Installation
bash
# Create virtual environment (recommended)
python -m venv hr_monitor
source hr_monitor/bin/activate  # On Windows: hr_monitor\Scripts\activate

# Install core dependencies
pip install opencv-python
pip install mediapipe
pip install scipy
pip install numpy

# Run the application
python main.py
📁 Project Structure
text
heart-rate-monitor/
├── main.py                 # Main application entry point
├── config.py              # Configuration parameters and constants
├── video_processor.py     # Camera handling and frame processing
├── roi_manager.py         # ROI definition and skin detection
├── signal_processor.py    # Signal processing and heart rate calculation
├── visualization.py       # GUI and graph rendering
├── utils.py               # Signal processing utilities
└── requirements.txt       # Python dependencies
🎮 Usage
Start the application:

bash
python main.py
Position yourself:

Sit 1-2 feet from the webcam

Ensure good lighting (avoid backlighting)

Keep your face within the camera view

Wait for stabilization:

The system needs 10-15 seconds to collect enough data

Watch the confidence indicator - higher is better

Monitor results:

Real-time BPM display

Signal quality graphs

Confidence percentage

Exit: Press 'q' to quit the application

📊 Understanding the Display
Control Panel
ROI Weights: Shows which facial regions are providing the best signals

Heart Rate: Current BPM with confidence indicator

Signal Quality: SNR and overall signal quality assessment

Countdown Timer: Time until next BPM update

Graphs
Raw PPG Signal: Original green channel intensity

Filtered Signal: Processed signal after bandpass filtering

Frequency Spectrum: FFT analysis showing dominant heart rate frequency

Color Codes
Green ROIs: Forehead region

Blue ROIs: Cheek regions

Red Circles: Detected heartbeats in signal graphs

Yellow Circle: Dominant frequency in spectrum

⚙️ Configuration
Key parameters in config.py:

python
# Signal Processing
BANDPASS_LOWCUT = 0.8    # Minimum frequency (48 BPM)
BANDPASS_HIGHCUT = 3.5   # Maximum frequency (210 BPM)
BUFFER_DURATION = 15     # Seconds of data to keep
UPDATE_INTERVAL = 5      # Seconds between BPM updates

# ROI Weights
ROI_WEIGHTS = {
    'forehead': 0.4,     # Forehead weight (usually most stable)
    'left_cheek': 0.3,   # Left cheek weight
    'right_cheek': 0.3   # Right cheek weight
}
🔧 Troubleshooting
Common Issues
Poor Signal Quality

Ensure good, even lighting

Avoid facial movements

Remove glasses if possible

No Face Detection

Check camera permissions

Ensure adequate lighting

Position face centrally

Inaccurate Readings

Remain still during measurement

Ensure consistent lighting

Wait for confidence to stabilize

Performance Tips
Close other camera applications

Use consistent artificial lighting

Position camera at eye level

Avoid sudden head movements

📈 Accuracy Notes
Best Case: ±2-3 BPM accuracy under ideal conditions

Typical: ±5 BPM in normal room conditions

Factors Affecting Accuracy: Lighting, movement, skin tone, camera quality

Validation: Compare with chest strap monitors for best validation

🧪 Technical Details
Signal Processing
Sampling Rate: 30 Hz (webcam frame rate)

Filter Type: 4th order Butterworth bandpass

Frequency Analysis: FFT with Hanning window

Peak Detection: Scipy find_peaks with adaptive thresholds

Computer Vision
Face Detection: MediaPipe Face Mesh (468 landmarks)

Skin Detection: Combined YCrCb and HSV color spaces

ROI Tracking: Dynamic adjustment based on face movement

🤝 Contributing
We welcome contributions! Please feel free to submit pull requests or open issues for:

Algorithm improvements

Performance optimizations

Additional features

Bug fixes

Documentation improvements

Development Setup
bash
git clone https://github.com/yourusername/heart-rate-monitor.git
cd heart-rate-monitor
pip install -r requirements.txt
# Make your changes and test with python main.py
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
MediaPipe: For robust real-time face detection

OpenCV: For computer vision infrastructure

SciPy: For signal processing algorithms

Research: Based on principles from remote PPG literature

📚 References
Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation.

Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light.

Lewandowska, M., et al. (2011). Measuring pulse rate with a webcam—a non-contact method for evaluating cardiac activity.

🐛 Known Limitations
Performance decreases with significant movement

Accuracy affected by poor lighting conditions

May struggle with very dark or very bright skin tones

Not intended for medical diagnosis

Disclaimer: This project is for educational and research purposes only. It is not intended for medical use or diagnosis. Always consult healthcare professionals for medical concerns.
