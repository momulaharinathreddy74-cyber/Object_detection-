import cv2

import mediapipe as mp
print(hasattr(mp, "solutions"))
print("successfully imported cv2 and mediapipe")
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Create dataset folder
gesture = "fist"   # change this for each class
path = f"datasets/{gesture}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

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

            # Show crop preview
            cv2.imshow("Hand Crop", hand_img)

            # Save image
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(f"{path}/{count}.jpg", hand_img)
                count += 1
                print("Saved:", count)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()