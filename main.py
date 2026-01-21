import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
import numpy as np
import time
import screeninfo

mouse = Controller()

thumb_pressed = False  # track click state
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



#smoothing
smoothing = 0.5
prev_x, prev_y = mouse.position

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
        mouse_x = int(prev_x + (index_finger.x * screen_width - prev_x) * smoothing)
        mouse_y = int(prev_y + (index_finger.y * screen_height - prev_y) * smoothing)
        mouse.position = (mouse_x, mouse_y)
        prev_x, prev_y = mouse_x, mouse_y

        thumb_folded = is_thumb_folded(hand_landmarks)

        if thumb_folded and not thumb_pressed:
            mouse.press(Button.left)
            thumb_pressed = True
        elif not thumb_folded and thumb_pressed:
            mouse.release(Button.left)
            thumb_pressed = False

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



