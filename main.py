import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import HandLandmarksConnections
from mediapipe.tasks.python.vision import FaceLandmarksConnections

import gestures
import assets

# Load stickers
assets.load_all_images()

# Import Google models
hand_model_path = 'hand_landmarker.task'
face_model_path = 'face_landmarker.task'

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_model_path),
    num_hands=2,
    running_mode=RunningMode.VIDEO # Best for webcams
)

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=face_model_path),
    num_faces=1,
    running_mode=RunningMode.VIDEO
)

# Set drawing specifications for a monochromatic theme
PURPLE = (255, 0, 255)

purple_dots = drawing_utils.DrawingSpec(color=PURPLE, thickness=-1, circle_radius=3)
purple_lines_thick = drawing_utils.DrawingSpec(color=PURPLE, thickness=2)
purple_lines_thin = drawing_utils.DrawingSpec(color=PURPLE, thickness=1)

CONTOUR_INDICES = set()
for connection in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
    CONTOUR_INDICES.add(connection.start)
    CONTOUR_INDICES.add(connection.end)

frame_timestamp_ms = 0
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

with vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
     vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:

    peaceGraphic = cv2.imread('./GFProvidedHamsters/Peace.png', cv2.IMREAD_UNCHANGED)

    # Resize to fit
    if peaceGraphic is not None:
        peaceGraphic = cv2.resize(peaceGraphic, (150, 150))
    else:
        print("Warning: Could not load the image file!")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        frame_timestamp_ms += 1

        hand_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        face_result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=purple_dots,
                    connection_drawing_spec=purple_lines_thick
                )
                
                # Compare detected gesture with the keys in the graphics dictionary
                if gestures.detect_gesture(hand_result.hand_landmarks) and gestures.detect_gesture(hand_result.hand_landmarks) in assets.stickers:
                    # Draw the corresponding image
                    frame = assets.overlay_transparent(
                        background=frame, 
                        overlay=assets.stickers[gestures.detect_gesture(hand_result.hand_landmarks)], 
                        x=50, 
                        y=70
                    )

        if face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                
                drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS, 
                    landmark_drawing_spec=None,  
                    connection_drawing_spec=purple_lines_thin
                )
                
                for idx in CONTOUR_INDICES:
                    lm = face_landmarks[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 2, PURPLE, -1)

        cv2.imshow("Monochromatic Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

