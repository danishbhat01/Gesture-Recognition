import os
import numpy as np
import pandas as pd  # We need pandas for CSV loading
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import mediapipe as mp

# Settings
dataset_dir = "dataset"
gesture_map = {
    "gesture1": 0,  # Wave
    "gesture2": 1,  # Thumbs Up
    "gesture3": 2,  # Rock
    "gesture4": 3,  # Peace
    "gesture5": 4   # fist
}

mp_hands = mp.solutions.hands

X, y = [], []

# Load data
for folder_name, label in gesture_map.items():
    folder_path = os.path.join(dataset_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} does not exist!")
        continue
    print(f"✅ Loading from: {folder_path}")
    
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):  # Handle CSV files
            file_path = os.path.join(folder_path, file)
            # Load the CSV file
            data = pd.read_csv(file_path, header=None)
            
            # Check if the data has 100 samples of 63 features each
            if data.shape == (100, 63):
                # Directly add the data as is, no reshaping needed
                X.append(data.values)  # Append all 100 samples from the file
                y.extend([label] * 100)  # Label all samples from the file with the current gesture label
            else:
                print(f"⚠️ Skipped file {file} due to incorrect shape: {data.shape}")

X = np.array(X)
y = np.array(y)

# Flatten the dataset (X contains multiple files, each with 100 samples)
X = X.reshape(-1, 63)

print(f"Total samples: {len(X)}, Feature size: {X[0].shape if len(X) > 0 else 'N/A'}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
acc = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# Save the trained model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved to: gesture_model.pkl")
