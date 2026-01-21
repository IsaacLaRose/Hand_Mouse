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

    # If the frame is not read correctly, 'ret' is False
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame in a window named 'Live Camera Feed'
    cv2.imshow('Live Camera Feed', frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # q exits
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break #Sorry Professor, I know you hate goto/break statements

# close windows
camera.release()
cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')



