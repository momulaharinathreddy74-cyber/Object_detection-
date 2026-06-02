import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "gesture_model.keras")

# Load model
model = load_model(MODEL_PATH)

# Labels (same order as folders)
labels = ['thumbs_up', 'palm', 'peace', 'fist', 'index_up']

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "No Hand Detected"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Get bounding box
            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            # Add margin
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Crop hand
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Resize
            hand_img = cv2.resize(hand_img, (64, 64))

            # Normalize
            img = hand_img / 255.0
            img = np.reshape(img, (1, 64, 64, 3))

            # Predict
            pred = model.predict(img)
            gesture = labels[np.argmax(pred)]

            # Draw bounding box on original frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.putText(frame, gesture, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()