# collect_data.py

import cv2
import os
import csv
import mediapipe as mp

# Settings
gesture_name = "gesture26"  # Change this for each gesture
save_path = f"dataset/gesture2/{gesture_name}.csv"
num_samples = 100  # Number of frames to collect

# Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Create dataset directory if it doesn't exist
os.makedirs("dataset", exist_ok=True)

# Open CSV file
with open(save_path, 'w', newline='') as f:
    writer = csv.writer(f)
    cap = cv2.VideoCapture(0)
    count = 0
    print(f"Collecting data for: {gesture_name}")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])  # 63 features
                writer.writerow(data)
                count += 1
                print(f"Sample {count}/{num_samples}")

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Samples Collected: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Collecting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete.")
