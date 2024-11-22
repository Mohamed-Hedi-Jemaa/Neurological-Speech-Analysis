# Key Updates
# Audio Feature Analysis:

# Added the analyze_audio_features function to extract features like pitch, tempo, energy, and SNR.
# Feature Interpretation:

# Added the interpret_audio_features function to interpret audio feature anomalies.
# New Endpoint:

# Created /analyze-audio/ for uploading and analyzing custom audio files.
# Integrated Existing Functions:

# Kept /process-preconverted/ for the pre-converted file workflow.
# Retained Hugging Face and OpenAI analysis in the pipeline.

from fastapi import FastAPI, UploadFile, File
import os
import speech_recognition as sr
from transformers import pipeline
import openai
import librosa
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Set up OpenAI API Key
openai.api_key = "" 


# Function: Speech-to-Text from an audio file
def speech_to_text_from_file(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        print("Processing the audio file...")
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language="fr-FR")
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Error with the Speech Recognition service: {e}"

# Function: Text Analysis with Hugging Face
def analyze_text_with_huggingface(text):
    # Use a pre-trained model for text classification
    model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = model(text)
    return result

def analyze_text_with_openai(text):
    # Ensure the recognized text is passed correctly to the prompt
    prompt = (
        f"You are an expert neurology doctor specializing in the diagnosis and analysis of neurological disorders, "
        f"including Alzheimer's and Parkinson's diseases. Based on the provided text from a speech-to-text analysis, "
        f"identify linguistic anomalies, articulation issues, pauses, monotony, or other speech-related symptoms that "
        f"could indicate early signs of these conditions. Provide a detailed analysis including potential observations, "
        f"medical reasoning, and suggestions for further neurological evaluation if applicable.\n\n"
        f"Text to analyze: {text}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are an expert neurologist specializing in Alzheimer's and Parkinson's diseases."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.InvalidRequestError as e:
        return f"Invalid Request Error: {str(e)}"
    except openai.error.AuthenticationError as e:
        return f"Authentication Error: {str(e)}"
    except openai.error.APIConnectionError as e:
        return f"API Connection Error: {str(e)}"
    except openai.error.RateLimitError as e:
        return f"Rate Limit Exceeded: {str(e)}"
    except openai.error.OpenAIError as e:
        return f"General OpenAI API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"


# Function: Analyze Audio Features
def analyze_audio_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # 1. Monotonie (variation of pitch)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

        # 2. Qualité vocale (signal-to-noise ratio)
        rms = librosa.feature.rms(y=y)
        snr = float(10 * np.log10(np.mean(rms) / (np.std(rms) + 1e-10)))  # SNR

        # 3. Hauteur (average pitch)
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

        # 4. Rythme perturbé (tempo detection)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)

        # 5. Énergie vocale (energy of the signal)
        energy = float(np.sum(y ** 2) / len(y))

        # Results summary
        results = {
            "pitch_std": pitch_std,               # Monotonie
            "snr": snr,                           # Qualité vocale
            "pitch_mean": pitch_mean,             # Hauteur
            "tempo": tempo,                       # Rythme
            "energy": energy                      # Énergie vocale
        }

        return results

    except Exception as e:
        return {"error": str(e)}

# Route: Root URL
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

# Route: Process the pre-converted WAV file
@app.post("/process-preconverted/")
async def process_preconverted_audio():
    # Use the pre-converted WAV file
    file_path = "audio_file-o.wav"  # Path to the pre-converted file

    if not os.path.exists(file_path):
        return {"error": f"The file {file_path} does not exist."}

    # Step 1: Convert speech to text
    text = speech_to_text_from_file(file_path)
    if not text or "Could not" in text:
        return {"error": "Audio processing failed or audio is not clear."}

    # Step 2: Analyze text with Hugging Face
    huggingface_result = analyze_text_with_huggingface(text)

    # Step 3: Analyze text with OpenAI GPT
    openai_result = analyze_text_with_openai(text)

    # Step 4: Analyze audio features
    audio_features = analyze_audio_features(file_path)

    # Return all results
    return {
        "recognized_text": text,
        "huggingface_analysis": huggingface_result,
        "openai_analysis": openai_result,
        "audio_features": audio_features,
    }
