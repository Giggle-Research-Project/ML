from eyeball_tracker import instantiate_faceMesh, is_concentrated

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import cv2
import json
import logging
import asyncio

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()


# Global flag to control streaming (for manual control)
streaming_active = True
websocket_clients = set()  # To track connected WebSocket clients

# WebSocket endpoint to handle concentration status updates
@app.websocket("/ws/concentration_status")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)  # Track the connected client
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    global face_mesh, streaming_active
    streaming_active = True
    face_mesh = instantiate_faceMesh()

    try:
        while True and streaming_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    concentrated = is_concentrated(face_landmarks, frame_width, frame_height)
                    concentration_status = "Concentrated" if concentrated else "Not Concentrated"
                    
                    # Send the concentration status as JSON
                    await websocket.send_text(json.dumps({'status': concentration_status}))
            
            # To simulate real time streaming, add a small delay
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logging.info("Client disconnected")
    finally:
        try:
            websocket_clients.remove(websocket)  # Clean up the client from the set
        except KeyError:
            pass  # Ignore if the client is already removed
        cap.release()
        face_mesh.close()

@app.get("/stop_stream")
async def stop_stream():
    global streaming_active
    streaming_active = False  # Stop the stream
    for websocket in websocket_clients:
        await websocket.send_text(json.dumps({'status': 'Stream has been stopped.'}))
        await websocket.close()
    websocket_clients.clear()
    return {"message": "Stream has been stopped and all clients disconnected."}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
