import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

model_path = 'hand_landmarker.task'

BaseOptions = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions,
    num_hands=2,
    running_mode=RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)

frame_timestamp_ms = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera failed to open")
else:
    print("Camera opened successfully")

with detector as landmarker:
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        frame_timestamp_ms += 1

        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.hand_landmarks:
            connections = [
                (0,1),(1,2),(2,3),(3,4),      # thumb
                (0,5),(5,6),(6,7),(7,8),      # index
                (5,9),(9,10),(10,11),(11,12), # middle
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),
                (0,17)
            ]

            for hand in result.hand_landmarks:
                points = []
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)

                for c in connections:
                    cv2.line(frame, points[c[0]], points[c[1]], (255,0,0), 2)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()