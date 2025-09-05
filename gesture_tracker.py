import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
)
mp_drawing = mp.solutions.drawing_utils


def detect_hands(frame):
    """
    Process the frame to detect hands and return the processed frame and detected gesture.
    """
    # Convert the frame to RGB (MediaPipe needs RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe to detect hand landmarks
    result = hands.process(rgb_frame)

    gesture = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Gesture recognition (finger counting example)
            gesture = classify_gesture(hand_landmarks, frame)

    return frame, gesture


def classify_gesture(hand_landmarks, frame):
    """
    Simple gesture classification: Fist, Open Palm, One, Two, etc.
    """
    h, w, _ = frame.shape
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    finger_tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb: check if tip is to the left of the knuckle (for right hand)
    if landmarks[finger_tips[0]][0] < landmarks[finger_tips[0] - 1][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers: check if tip is above the joint
    for tip in finger_tips[1:]:
        if landmarks[tip][1] < landmarks[tip - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    total_fingers = fingers.count(1)

    if total_fingers == 0:
        return "Fist"
    elif total_fingers == 5:
        return "Open Palm"
    elif total_fingers == 1:
        return "One"
    elif total_fingers == 2:
        return "Two"
    else:
        return f"{total_fingers} Fingers"


def extract_landmarks(result):
    """
    Extract the hand landmarks from the result returned by MediaPipe.
    """
    landmarks = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks


if __name__ == "__main__":
    # Debugging mode: run webcam directly
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, gesture = detect_hands(frame)
        if gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
