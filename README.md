# Gesture-Recognition
This project is a real-time hand gesture recognition system using Python, OpenCV, and MediaPipe. It detects 21 key hand landmarks from a webcam feed and classifies gestures such as Fist, Open Palm etc.The system overlays detected landmarks and gesture names directly on the video stream, providing instant feedback.
🚀 Features
Real-time hand tracking using MediaPipe Hands
Landmark drawing on detected hands
Simple gesture classification:
👊 Fist
✋ Open Palm
☝ One
✌ Two

Other combinations (3–4 fingers)

📂 Project Structure
project-root/
│── collect_data.py
│── gesture_model.pkl
│── gesture_tracker.py
│── predict_gesture.py
│── train_model.py
│── dataset/          # (stores collected CSV/landmark files here)


⚙️ Installation

Clone the repository:
git clone https://github.com/your-username/gesture-recognition.git
cd gesture-recognition
Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:
pip install -r requirements.txt

▶️ Usage

Run the script to start gesture recognition:
python gesture_tracker.py
A webcam window will open.
Detected hand landmarks will be drawn on your hand.
Recognized gestures will appear on the screen.
Press Q to quit the application.


📦 Requirements
Python 3.8+
OpenCV
MediaPipe
NumPy
(These are already listed in requirements.txt)

🛠 How it Works
Captures frames from webcam using OpenCV.
Converts each frame to RGB for MediaPipe.
MediaPipe detects hand landmarks.
A simple finger-counting logic classifies gestures.
Displays the result in real-time.


🔮 Future Improvements

Train a custom model for more complex gestures.
Add support for multi-hand detection.
Integrate with applications (e.g., media player control, virtual mouse).
