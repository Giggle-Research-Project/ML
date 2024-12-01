import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

# Initialize FastAPI app
app = FastAPI()

# Load trained model for facial emotion recognition
model_best = load_model('emotion_model.h5')

# Map Classes for the three emotional states
class_names = ['Confused', 'Confused', 'Confused', 'Happy', 'Neutral', 'Sad', 'Confused']

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Frame model for incoming frame data
class Frame(BaseModel):
    frame: str  # Base64-encoded frame

# Preprocessing the frame and predicting emotion
def process_frame(frame_data):
    # Decode base64 image data
    frame_bytes = base64.b64decode(frame_data)
    np_img = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return "No face detected"

    # Only the first detected face
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face to the required input size for the model
        face_image = cv2.resize(face_roi, (224, 224))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension

        # Predict emotion using the model
        predictions = model_best.predict(face_image)

        # Threshold for prediction confidence
        threshold = 0.5
        max_prob = np.max(predictions)
        max_index = np.argmax(predictions)

        # If prediction confidence exceeds the threshold
        if max_prob >= threshold:
            emotion_label = class_names[max_index]
        else: # Default label if low confidence
            emotion_label = "Neutral"  # Or Happy

        return emotion_label

# POST endpoint to process the frame
@app.post("/process_frame/")
async def process_frame_endpoint(frame: Frame):
    try:
        # Extract and process the frame
        emotion = process_frame(frame.frame)
        return {"emotion": emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))