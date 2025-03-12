# app.py

from handwriting_rec import predict_handwrite

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil

# Initialize the FastAPI app
app = FastAPI()

# Endpoint to handle file upload and prediction
@app.post("/predict_handwriting/")
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

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
