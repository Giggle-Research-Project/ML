import whisper
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model
model_name = "base"  # Replace with the path to your base.pt file if custom
try:
    model = whisper.load_model(model_name, device=device)
    print(f"Model '{model_name}' loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Check if the file exists
def transcribe_audio(audio_file):
    try:
        # Transcribe the audio
        result = model.transcribe(audio_file)
        transcription = result.get('text', "")
        
        # Return transcription as a dictionary
        return {"transcription": transcription}
    
    except Exception as e:
        # If there is an error, return an error message
        return {"error": f"Error during transcription: {str(e)}"}

