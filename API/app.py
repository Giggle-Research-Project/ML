"""
------ Main API for the Giggle ML Components ------

To run the API, stay in the root directory (API) and execute:

    python -m app

The -m flag ensures that Python treats API.main as a module, recognizing API as a package.
"""

import base64
import sys
import os
import numpy as np
import cv2
import shutil
import json
import tensorflow as tf
import io
import requests
import librosa
import numpy as np
import joblib
import pickle

from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.getcwd()) # set root path (API), other paths should be relative to that.

from function_01.question_generation.resources.constants import DyscalculiaType, Difficulty, Lesson
from function_01.question_generation.question_generator.get_question import get_random_question
from function_03.Speech_to_text.transcribe import transcribe_audio
from function_04.Handwriting_Recognition.handwriting_rec import predict_handwrite
from function_02.eyeball_tracking_system.eyeball_tracker import instantiate_faceMesh, is_concentrated
from function_02.facial_emotion_detection.live_emotion import detect_emotion

# Ensure default encoding is UTF-8
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding='utf-8')

tf.keras.utils.disable_interactive_logging()

app = FastAPI()

# -------------------- Load Models --------------------

""" 
Paths should be realtive to the root dir (API)
"""

# Confidence Model
confidence_model = joblib.load('models/confidence_time_result_model.pkl')

# Gesture Model
gesture_model = load_model('models/crossing_classifier.h5')

# Emotion Model
emotion_model = tf.keras.models.load_model('models/emotion_recognition_model.h5')

# Performance Prediction Model
with open('models/performace_predciton.pkl', 'rb') as file:
    performance_model = pickle.load(file)

# Load Performance Scaler
scaler_path = 'models/scaler.pkl'
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

# Load Performance Model
performance_model_path = 'models/Support_Vector_Regressor.pkl'
with open(performance_model_path, "rb") as file:
    performance_model = pickle.load(file)

# Load F1 Performance Prediction Model
with open('models/F1_performace_predict.pkl', 'rb') as file:
    f1_performance_model = pickle.load(file)

# Initialize the scalers
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()


# -------------------- Confidence Prediction --------------------

class TestData(BaseModel):
    completion_time: float
    test_marks: float

@app.post("/predict_Confidence/")
def predict_confidence(data: TestData):
    # Input validation
    if not (0 <= data.test_marks <= 100):
        raise HTTPException(status_code=400, detail="Test Marks should be between 0 and 100.")
    if not (5 <= data.completion_time <= 30):
        raise HTTPException(status_code=400, detail="Completion Time should be between 5 and 30 minutes.")
    
    # Prepare input data for prediction
    input_data = np.array([[data.completion_time, data.test_marks]])
    predicted_confidence = confidence_model.predict(input_data)

    # Convert numpy.float32 to Python float for serialization
    predicted_confidence = float(predicted_confidence[0])
    
    return {"Predicted Confidence Score": round(predicted_confidence, 2)}


# -------------------- Gesture Prediction --------------------

@app.post("/predict-gesture/")
async def predict_gesture(file: UploadFile = File(...)):
    # Read the uploaded image file
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Preprocess the image
    img = np.array(img)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")
    img = cv2.resize(img, (128, 128))  # Resize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values

    # Predict using the gesture model
    prediction = gesture_model.predict(img)
    predicted_label = np.argmax(prediction)

    # Return the result as JSON
    result = "Crossing" if predicted_label == 1 else "Not Crossing"
    return {"Prediction": result}


# -------------------- Mispronunciation Detection --------------------

API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h"
API_TOKEN = "************************************" 

def transcribe_audio_hf(file_bytes):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, data=file_bytes)
    if response.status_code == 200:
        return response.json().get('text', 'No transcription found.')
    else:
        return f"Error: {response.status_code}, {response.text}"

@app.post("/Mispronunciation_correction/")
async def upload_audio(file: UploadFile = File(...)):
    if file.content_type not in ['audio/wav', 'audio/mpeg', 'audio/mp3']:
        return {"error": "Invalid file type. Only WAV, MP3 are allowed."}
    
    file_content = await file.read()
    transcription = transcribe_audio_hf(file_content)
    return {"transcription": transcription}


# -------------------- Voice Emotion Prediction --------------------

# Label Encoder for emotions
le = LabelEncoder()
le.fit(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])

# Function to extract features from audio
def extract_features(audio_file):
    audio, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def map_emotion_to_confidence(emotion):
    confident_emotions = ['happy', 'surprise', 'neutral']
    not_confident_emotions = ['sad', 'fear', 'angry', 'disgust']
    if emotion in confident_emotions:
        return "confident"
    elif emotion in not_confident_emotions:
        return "not confident"
    return "unknown"

def predict_emotion_from_audio(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    prediction = emotion_model.predict(features)
    predicted_label = np.argmax(prediction, axis=1)
    return le.inverse_transform(predicted_label)[0]

@app.post("/predict-emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    file_location = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_location, "wb") as f:
        f.write(await file.read())

    predicted_emotion = predict_emotion_from_audio(file_location)
    confidence_level = map_emotion_to_confidence(predicted_emotion)

    return {"predicted_emotion": predicted_emotion, "confidence_level": confidence_level}


# -------------------- Performance Prediction --------------------

class InputData(BaseModel):
    test1_marks: float
    test1_time: float
    test2_marks: float
    test2_time: float
    test3_marks: float
    test3_time: float

@app.post("/predict-Performance/")
def predict_performance(data: InputData):
    # Validate input data
    if not all(0 <= x <= 100 for x in [data.test1_marks, data.test2_marks, data.test3_marks]):
        raise HTTPException(status_code=400, detail="Marks should be between 0 and 100.")
    if not all(0 <= x <= 100 for x in [data.test1_time, data.test2_time, data.test3_time]):
        raise HTTPException(status_code=400, detail="Time should be between 0 and 100 minutes.")
    
    # Prepare input data for prediction
    input_data = np.array([[data.test1_marks, data.test1_time, 
                            data.test2_marks, data.test2_time, 
                            data.test3_marks, data.test3_time]])

    # Scale the input using the loaded scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = performance_model.predict(input_data_scaled)
    predicted_score = prediction[0]

    return {"predicted_performance": round(predicted_score, 2)}


# -------------------- F1 Performance Prediction --------------------

class F1InputData(BaseModel):
    child_marks: float
    child_time: float
    parent_marks: float
    parent_time: float

@app.post("/predict-f1-performance/")
async def predict_f1_performance(data: F1InputData):
    # Extract values from the input data
    child_marks = data.child_marks
    child_time = data.child_time
    parent_marks = data.parent_marks
    parent_time = data.parent_time
    
    # Ensure the inputs are within valid ranges
    if not (1 <= child_marks <= 100):
        return {"error": "Child's marks must be between 1 and 100."}
    if not (1 <= child_time <= 30):
        return {"error": "Child's time must be between 1 and 30 minutes."}
    if not (1 <= parent_marks <= 100):
        return {"error": "Parent's marks must be between 1 and 100."}
    if not (1 <= parent_time <= 10):
        return {"error": "Parent's time must be between 1 and 10 minutes."}

    # Create input array for prediction
    input_data = np.array([[child_marks, child_time, parent_marks, parent_time]])

    # Make prediction
    prediction = f1_performance_model.predict(input_data)

    # Return the prediction
    return {"predicted_performance": float(prediction[0])}


# -------------------- Function 03: Speech To Text --------------------

@app.post("/transcribe")
async def transcribe_audio_route(audio_file: UploadFile = File(...)):
    # Save the file temporarily
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(await audio_file.read())
    
    # Call the transcription function
    result = transcribe_audio(temp_file_path)
    
    # Return the result
    if "error" in result:
        return JSONResponse(status_code=500, content=result)
    return result


# -------------------- Function 04: Handwriting Recognition -------------------- 

@app.post("/predict-handwriting/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Create a temporary file to save the uploaded image
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Predict the uploaded image
        predicted_class, prediction_probs = predict_handwrite(temp_file_path)
        
        # Remove the temporary file after prediction
        os.remove(temp_file_path)
        
        # Return the prediction result
        return JSONResponse(content={
            "predicted_class": int(predicted_class),
            #"prediction_probabilities": prediction_probs.tolist()
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    
# -------------------- Function 1: Question Generator -------------------- 

# Define the question request model using Pydantic
class QuestionRequest(BaseModel):
    dyscalculia_type: str
    lesson: str = None # Optional lesson
    difficulty: str
   
@app.post("/generate-question")
def generate_question(request: QuestionRequest):
    try:
        # Convert request data (strings) to corresponding enum types
        dyscalculia_type = DyscalculiaType[request.dyscalculia_type]
        lesson = Lesson[request.lesson] if request.lesson else None
        difficulty = Difficulty[request.difficulty]
                
        generated_question = get_random_question(dyscalculia_type, difficulty, lesson)
        
        # Return the generated question in JSON format
        return {"question": generated_question}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid key: {str(e)}")


# -------------------- WebSocket endpoints --------------------

# Global flag to control streaming (for manual control)
streaming_active = True
websocket_clients = set()  # To track connected WebSocket clients

# Disconnect all WebSocket
@app.get("/stop-stream")
async def stop_stream():
    global streaming_active
    streaming_active = False  # Stop the stream

    for websocket in websocket_clients:
        await websocket.send_text(json.dumps({'status': 'Stream has been stopped.'}))
        await websocket.close()
    websocket_clients.clear()
    return {"message": "Stream has been stopped and all clients disconnected."}


# -------------------- Function 02 - Eyeball Tracker -------------------- 

@app.post("/predict-concentration/")
async def predict_concentration(image: UploadFile):
    # Process the uploaded image file
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_mesh = instantiate_faceMesh()
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        frame_height, frame_width, _ = frame_rgb.shape
        for face_landmarks in results.multi_face_landmarks:
            concentrated = is_concentrated(face_landmarks, frame_width, frame_height)
            score = 0.9 if concentrated else 0.3  # Convert boolean to score
            return {"concentration_score": score}
    
    return {"concentration_score": 0.5}  # Default score

@app.post("/detect-emotion/")
async def detect_emotion_endpoint(image: UploadFile):
    # Process the uploaded image file
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    emotion = detect_emotion(frame)
    return {"emotion": emotion, "confidence": 0.8}  # Add confidence score

@app.websocket("/ws/concentration-status")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    face_mesh = instantiate_faceMesh()
    
    try:
        while True:
            # Receive base64 image from client
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data)

            # Convert bytes to OpenCV image
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get the dimensions from the image
            frame_height, frame_width, _ = frame_rgb.shape
            
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    concentrated = is_concentrated(face_landmarks, frame_width, frame_height)
                    concentration_status = "Concentrated" if concentrated else "Not Concentrated"
                    
                    # Send the concentration status as JSON
                    await websocket.send_text(json.dumps({'status': concentration_status}))
    
    except WebSocketDisconnect:
        print("Client disconnected.") 
    
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.close(code=1001, reason="Error on the server side")
    
    finally:
        print("WebSocket connection closed - concentration-status")
        
        
# -------------------- Function 02 - Face Emotion Detection --------------------
 
@app.websocket("/ws/detect-emotion")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive base64 image from the client
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data)

            # Convert bytes to OpenCV image
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Process the frame to detect emotion
            emotion = detect_emotion(frame)
            
            # Send the emotion status as JSON
            await websocket.send_text(json.dumps({'emotion': emotion}))

    except WebSocketDisconnect:
        print("Client disconnected.") 
    
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.close(code=1001, reason="Error on the server side")
        
    finally:
        print("WebSocket connection closed - detect-emotion")


# -------------------- Run the API --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
