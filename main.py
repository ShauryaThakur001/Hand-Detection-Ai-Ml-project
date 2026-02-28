import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ---------------------------
# Load Hand Landmarker Model
# ---------------------------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ---------------------------
# Start Camera
# ---------------------------
cap = cv2.VideoCapture(0)

# ---------------------------
# State Variables
# ---------------------------
stable_frames_required = 8
closed_counter = 0
open_counter = 0
prev_confirmed_state = "OPEN"

screenshot_count = 0
last_capture_time = 0
cooldown = 2
animation_time = 0

# ---------------------------
# Main Loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    current_state = "OPEN"

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            h, w, _ = frame.shape
            landmarks = []

            for lm in hand_landmarks:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                landmarks.append((cx, cy))
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Fingertip and lower joint indexes
            tips = [8, 12, 16, 20]
            lower = [6, 10, 14, 18]

            fingers_folded = []

            for tip, low in zip(tips, lower):
                if landmarks[tip][1] > landmarks[low][1]:
                    fingers_folded.append(1)
                else:
                    fingers_folded.append(0)

            # If all 4 fingers folded → CLOSED
            if sum(fingers_folded) == 4:
                current_state = "CLOSED"

    # ---------------------------
    # Stable State Detection
    # ---------------------------
    if current_state == "CLOSED":
        closed_counter += 1
        open_counter = 0
    else:
        open_counter += 1
        closed_counter = 0

    confirmed_state = prev_confirmed_state

    if closed_counter > stable_frames_required:
        confirmed_state = "CLOSED"

    if open_counter > stable_frames_required:
        confirmed_state = "OPEN"

    # ---------------------------
    # CLOSED → OPEN Transition
    # ---------------------------
    if prev_confirmed_state == "CLOSED" and confirmed_state == "OPEN":
        current_time = time.time()

        if current_time - last_capture_time > cooldown:
            screenshot_name = f"screenshot_{screenshot_count}.png"
            cv2.imwrite(screenshot_name, frame)
            print("Screenshot Taken:", screenshot_name)

            screenshot_count += 1
            last_capture_time = current_time
            animation_time = time.time()

    prev_confirmed_state = confirmed_state

    # ---------------------------
    # Shutter Animation
    # ---------------------------
    if time.time() - animation_time < 0.3:
        h, w, _ = frame.shape
        progress = (time.time() - animation_time) / 0.3
        bar_height = int(h * progress / 2)

        cv2.rectangle(frame, (0, 0), (w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), -1)

        cv2.putText(frame, "Screenshot Captured!",
                    (50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3)

    cv2.imshow("Gesture Screenshot", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------
# Cleanup
# ---------------------------
cap.release()
cv2.destroyAllWindows()