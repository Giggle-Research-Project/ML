import librosa
import whisper
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model
model_name = "tiny"
try:
    model = whisper.load_model(model_name, device=device)
    print(f"Model '{model_name}' loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to preprocess audio using librosa
def preprocess_audio(audio_file):
    try:
        # Load audio file using librosa, resampling to 16kHz (required by Whisper)
        audio, sr = librosa.load(audio_file, sr=16000)
        return audio
    
    except Exception as e:
        print(f"Error during audio preprocessing: {e}")
        return None

# Function to transcribe audio after preprocessing
def transcribe_audio(audio_file):
    try:
        # Preprocess the audio file
        audio_data = preprocess_audio(audio_file)
        if audio_data is None:
            return {"error": "Failed to preprocess audio"}

        # Transcribe the audio using Whisper
        result = model.transcribe(audio_data)
        transcription = result.get('text', "")
        # Return transcription as a dictionary
        return {"transcription": transcription}
    
    except Exception as e:
        # If there is an error, return an error message
        return {"error": f"Error during transcription: {str(e)}"}

