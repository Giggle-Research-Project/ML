
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Load the trained model
model = load_model('function_04/Handwriting_Recognition/best_model.h5')

# Function to preprocess a single image
def preprocess_image(img_path, target_size=(28, 28)):
    # Load the image
    img = image.load_img(img_path, target_size=target_size, color_mode="grayscale")
    
    # Convert image to numpy array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Normalize the image (if required for your model)
    img_array = img_array / 255.0  # Scale pixel values to [0, 1]
    
    return img_array

# Function to predict a single image
def predict_handwrite(img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    # Get the model's prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    
    return predicted_class[0], prediction[0]

