from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
from gaze_tracker import process_frame
import mediapipe as mp

app = FastAPI()

# Camera internals
frame_width = 640
frame_height = 480
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array([[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], dtype='float64')
dist_coeffs = np.zeros((4, 1))  # assuming no distortion

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive frame bytes from the client
            frame_bytes = await websocket.receive_bytes()

            # Convert bytes back to image
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Process the frame and get the gaze result
            result = process_frame(img, face_mesh, camera_matrix, dist_coeffs)

            # Send back the gaze status
            await websocket.send_text(result)

    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
