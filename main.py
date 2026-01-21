import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
import numpy as np
import time
import screeninfo

mouse = Controller()


#screen info
screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot open camera. Check camera index or connection.")
    exit() #No Camera is detected or cannot be accessed


mp_vis = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

#smoothing
prev_x, prev_y = screen_width // 2, screen_height // 2
smoothing = 0.5

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1) #Flip camera so that it works more like a mirror

    # If the frame is not read correctly, 'ret' is False
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(RGB_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_vis.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display the resulting frame in a window named 'Live Camera Feed'
    cv2.imshow('Live Camera Feed', frame)


    # q exits
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break #Sorry Professor, I know you hate goto/break statements

# close windows
camera.release()
cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')



