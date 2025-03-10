from live_emotion import detect_emotion

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

# WebSocket endpoint to handle emotion status updates
@app.websocket("/ws/detect_emotion")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)  # Track the connected client
    cap = cv2.VideoCapture(0)
    streaming_active = True

    try:
        while True and streaming_active:
            ret, frame = cap.read()
            if not ret:
                break

            emotion = detect_emotion(frame)
                    
            # Send the emotion status as JSON
            await websocket.send_text(json.dumps({'emotion': emotion}))
            
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