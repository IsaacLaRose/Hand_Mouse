# Hand_Mouse

Hand_Mouse is a computer vision project that allows users to control their desktop mouse using hand gestures captured through a webcam.

The application uses **OpenCV** for camera input and **MediaPipe’s Hand Tracking API** to detect and track hand landmarks in real time. Mouse movement is controlled by tracking the tip of the user’s index finger and the use of the **Pynput** library.

A mouse click is triggered by a simple gesture: folding the thumb inward while keeping the other fingers extended (similar to making the number four or the ASL letter **B**). This approach enables intuitive, hands-free mouse interaction without additional hardware.

### Features
- Real-time hand tracking via webcam  
- Cursor movement mapped to index finger tip  
- Gesture-based clicking  
- No external sensors required  

### Technologies Used
- Python  
- OpenCV  
- MediaPipe
- Pynput  

### Potential Improvements
- Adjustable sensitivity and smoothing  
- Additional gestures (right-click, scroll, drag)  
- Multi-monitor support  
- GUI for calibration and settings
