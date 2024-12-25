import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from math import hypot

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Pycaw for Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get Volume Range
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Start Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and Convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract Thumb and Index Finger Tip Coordinates
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            h, w, _ = frame.shape
            thumb_tip = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_tip = (int(index_tip.x * w), int(index_tip.y * h))

            # Draw Circles and Line
            cv2.circle(frame, thumb_tip, 10, (255, 0, 0), -1)
            cv2.circle(frame, index_tip, 10, (255, 0, 0), -1)
            cv2.line(frame, thumb_tip, index_tip, (0, 255, 0), 2)

            # Calculate Distance Between Thumb and Index Finger
            distance = hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

            # Map Distance to Volume Level
            vol = np.interp(distance, [20, 150], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Display Volume Level
            cv2.putText(frame, f"Volume: {int(np.interp(distance, [20, 150], [0, 100]))}%", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display Frame
    cv2.imshow("Volume Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
