from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import base64
import threading
import time

app = Flask(__name__, template_folder='.')
socketio = SocketIO(app)

# Global variables
camera = None
is_running = False
drawing_thread = None

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Color points
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Color indices
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Initialize canvas
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)

def draw_canvas():
    global camera, is_running, paintWindow
    
    while is_running:
        ret, frame = camera.read()
        if not ret:
            break
            
        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand landmarks
        result = hands.process(framergb)
        
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)
                    landmarks.append([lmx, lmy])
                
                mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)
                
            if len(landmarks) > 8:
                fore_finger = (landmarks[8][0], landmarks[8][1])
                thumb = (landmarks[4][0], landmarks[4][1])
                
                if (thumb[1] - fore_finger[1] < 30):
                    # Drawing mode
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(fore_finger)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(fore_finger)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(fore_finger)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(fore_finger)
                
                elif fore_finger[1] <= 65:
                    # Color selection
                    if 40 <= fore_finger[0] <= 140:  # Clear
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]
                        paintWindow[67:,:,:] = 255
                    elif 160 <= fore_finger[0] <= 255:
                        colorIndex = 0  # Blue
                    elif 275 <= fore_finger[0] <= 370:
                        colorIndex = 1  # Green
                    elif 390 <= fore_finger[0] <= 485:
                        colorIndex = 2  # Red
                    elif 505 <= fore_finger[0] <= 600:
                        colorIndex = 3  # Yellow
        
        # Draw lines
        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k-1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k-1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k-1], points[i][j][k], colors[i], 2)
        
        # Convert frames to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        _, paint_buffer = cv2.imencode('.jpg', paintWindow)
        paint_base64 = base64.b64encode(paint_buffer).decode('utf-8')
        
        # Send frames to client
        socketio.emit('video_feed', {
            'frame': frame_base64,
            'canvas': paint_base64
        })
        
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start')
def handle_start():
    global camera, is_running, drawing_thread
    if not is_running:
        camera = cv2.VideoCapture(0)
        is_running = True
        drawing_thread = threading.Thread(target=draw_canvas)
        drawing_thread.start()

@socketio.on('stop')
def handle_stop():
    global camera, is_running, drawing_thread
    if is_running:
        is_running = False
        if drawing_thread:
            drawing_thread.join()
        if camera:
            camera.release()

@socketio.on('clear')
def handle_clear():
    global paintWindow
    paintWindow = np.zeros((471, 636, 3)) + 255
    paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
    paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)

if __name__ == '__main__':
    socketio.run(app, debug=True) 