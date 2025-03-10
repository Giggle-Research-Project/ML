from transcribe import transcribe_audio

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn


# FastAPI instance
app = FastAPI()

class TranscriptionResponse(BaseModel):
    transcription: str


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
    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

