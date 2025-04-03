import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize session state
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'paintWindow' not in st.session_state:
    st.session_state.paintWindow = np.zeros((471, 636, 3)) + 255
    st.session_state.paintWindow = cv2.rectangle(st.session_state.paintWindow, (40,1), (140,65), (0,0,0), 2)
    st.session_state.paintWindow = cv2.rectangle(st.session_state.paintWindow, (160,1), (255,65), (255,0,0), 2)
    st.session_state.paintWindow = cv2.rectangle(st.session_state.paintWindow, (275,1), (370,65), (0,255,0), 2)
    st.session_state.paintWindow = cv2.rectangle(st.session_state.paintWindow, (390,1), (485,65), (0,0,255), 2)
    st.session_state.paintWindow = cv2.rectangle(st.session_state.paintWindow, (505,1), (600,65), (0,255,255), 2)

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

# Streamlit UI
st.title("Air Canvas")
st.markdown("""
    ### How to Use:
    1. Click "Start" to begin the air canvas
    2. Use your index finger to draw in the air
    3. Touch your thumb and index finger together to start drawing
    4. Move your hand to the top of the screen to select colors:
        - Blue (leftmost)
        - Green
        - Red
        - Yellow (rightmost)
    5. Click "Clear Canvas" to erase everything
    6. Click "Stop" to stop the camera feed
""")

# Create columns for video and canvas
col1, col2 = st.columns(2)

# Add buttons
start_button = st.button("Start")
stop_button = st.button("Stop")
clear_button = st.button("Clear Canvas")

# Handle button clicks
if start_button:
    st.session_state.camera = cv2.VideoCapture(0)
    st.session_state.is_running = True

if stop_button:
    if st.session_state.camera is not None:
        st.session_state.camera.release()
    st.session_state.is_running = False

if clear_button:
    st.session_state.paintWindow = np.zeros((471, 636, 3)) + 255
    st.session_state.paintWindow = cv2.rectangle(st.session_state.paintWindow, (40,1), (140,65), (0,0,0), 2)
    st.session_state.paintWindow = cv2.rectangle(st.session_state.paintWindow, (160,1), (255,65), (255,0,0), 2)
    st.session_state.paintWindow = cv2.rectangle(st.session_state.paintWindow, (275,1), (370,65), (0,255,0), 2)
    st.session_state.paintWindow = cv2.rectangle(st.session_state.paintWindow, (390,1), (485,65), (0,0,255), 2)
    st.session_state.paintWindow = cv2.rectangle(st.session_state.paintWindow, (505,1), (600,65), (0,255,255), 2)

# Main processing loop
if st.session_state.is_running and st.session_state.camera is not None:
    ret, frame = st.session_state.camera.read()
    if ret:
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
                        st.session_state.paintWindow[67:,:,:] = 255
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
                    cv2.line(st.session_state.paintWindow, points[i][j][k-1], points[i][j][k], colors[i], 2)
        
        # Display frames
        col1.image(frame, channels="BGR", use_column_width=True)
        col2.image(st.session_state.paintWindow, channels="BGR", use_column_width=True)
        
        time.sleep(0.1) 
