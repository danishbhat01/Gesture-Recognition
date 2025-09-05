import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model
model = joblib.load("gesture_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

gesture_labels = {
    0: "Wave",
    1: "Thumbs Up",
    2: "Rock",
    3: "Peace",
    4: "fist"
}

cap = cv2.VideoCapture(0)

print("✅ Starting camera... Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera read failed.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Check input size
            if len(landmarks) == model.n_features_in_:
                try:
                    prediction = model.predict([landmarks])[0]
                    gesture_name = gesture_labels.get(prediction, "Unknown")
                    print(f"✅ Detected Gesture: {gesture_name}")

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, gesture_name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                except Exception as e:
                    print("❌ Prediction error:", e)
            else:
                print(f"⚠️ Landmark count mismatch: got {len(landmarks)}, expected {model.n_features_in_}")
    else:
        print("🔍 No hand detected.")

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
