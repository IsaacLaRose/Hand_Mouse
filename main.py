import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
import numpy as np
import time
import screeninfo
from filterpy.kalman import KalmanFilter

mouse = Controller()

thumb_pressed = False  # track click state

double_click_threshold = 0.3  # seconds
last_click_time = 0


#screen info
screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

camera = cv2.VideoCapture(0)



def is_thumb_folded(hand_landmarks):
    wrist_x = hand_landmarks.landmark[0].x
    thumb_tip_x = hand_landmarks.landmark[4].x
    thumb_mcp_x = hand_landmarks.landmark[2].x

    # Thumb folded if tip is between MCP and wrist horizontally
    return min(wrist_x, thumb_mcp_x) < thumb_tip_x < max(wrist_x, thumb_mcp_x)



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



# Kalman filter for smooth cursor movement
kf_x = KalmanFilter(dim_x=2, dim_z=1)
kf_y = KalmanFilter(dim_x=2, dim_z=1)

for kf in [kf_x, kf_y]:
    kf.x = np.array([0., 0.])   # initial state: position, velocity
    kf.F = np.array([[1., 1.],
                     [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 1000.
    kf.R = 5
    kf.Q = 0.1

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
        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw hand
        mp_vis.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mouse movement (index finger)
        index_finger = hand_landmarks.landmark[8]

        raw_x = index_finger.x * screen_width
        raw_y = index_finger.y * screen_height

        # Kalman filter prediction + update
        kf_x.predict()
        kf_y.predict()
        kf_x.update(raw_x)
        kf_y.update(raw_y)
        smooth_x = int(kf_x.x[0])
        smooth_y = int(kf_y.x[0])

        # Move mouse
        mouse.position = (smooth_x, smooth_y)

        thumb_folded = is_thumb_folded(hand_landmarks)
        current_time = time.time()

        if thumb_folded:
            if not thumb_pressed:
                # Thumb just folded → press mouse
                mouse.press(Button.left)
                thumb_pressed = True

                # Double click if previous release was recent
                if current_time - last_click_time < double_click_threshold:
                    mouse.click(Button.left, 2)
                    print("DoubleClick")

        else:
            if thumb_pressed:
                # Thumb just released → release mouse
                mouse.release(Button.left)
                thumb_pressed = False
                last_click_time = current_time

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



