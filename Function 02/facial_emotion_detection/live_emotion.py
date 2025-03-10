import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.layers import DepthwiseConv2D # type: ignore

# Map Classes for the three emotional states
class_names = ["Confused", "Confused", "Confused", "Happy", "Neutral", "Sad", "Confused"]

# Load trained model for facial emotion recognition
model_path = 'function_02/facial_emotion_detection/mobilenet_face_ft.h5'


# --- Fixing model load error ---
# Removing the argument that was deprecated in TF 2.18 (Model was trained on 2.10)
def custom_depthwise_conv2d(**kwargs):
    if 'groups' in kwargs:
        kwargs.pop('groups')  # Remove the problematic argument
    return DepthwiseConv2D(**kwargs)

model_best = load_model(model_path, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})

# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(frame):
    
    # Convert the frame to grayscale for face detection (required for haarcascade) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))
    
    if len(faces) == 0:  # No face detected
        return "No Face Detected"  # Instead of returning None

    # Process each detected face
    for (x, y, w, h) in faces:
        
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face image to the required input size for the model
        face_image = cv2.resize(face_roi, (224, 224))  # Change to 224*224
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # Convert to RGB (if BGR)
        face_image = image.img_to_array(face_image)  # Convert to array
        face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension


        # Predict emotion using the loaded model
        predictions = model_best.predict(face_image)

        # Set a threshold for the prediction confidence
        threshold = 0.5
        
        # Get the maximum prediction probability and its corresponding label index
        max_prob = np.max(predictions)
        max_index = np.argmax(predictions)
        
        # If the maximum probability exceeds the threshold, assign the corresponding label
        if max_prob >= threshold:
            emotion_label = class_names[max_index]
        else:
            emotion_label = "Neutral"  # Default value
            
        return emotion_label
    
    
