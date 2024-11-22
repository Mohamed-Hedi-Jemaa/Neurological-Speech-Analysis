import os
import wave
import pyaudio
import speech_recognition as sr
from fastapi import FastAPI
from transformers import pipeline
import openai
import librosa
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Set up OpenAI API Key
openai.api_key = ""  # Replace with your OpenAI API key

# Function: Record Real-Time Audio
def record_audio(output_file="real_time_audio.wav", record_seconds=20, sample_rate=44100, chunk_size=1024):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)
    
    print("Recording...")
    frames = []
    for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)
    
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save the audio to a WAV file
    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    return output_file

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

# Function: Analyze Audio Features
def analyze_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
        rms = librosa.feature.rms(y=y)
        snr = float(10 * np.log10(np.mean(rms) / (np.std(rms) + 1e-10)))
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        energy = float(np.sum(y ** 2) / len(y))
        results = {
            "pitch_std": pitch_std,
            "snr": snr,
            "pitch_mean": pitch_mean,
            "tempo": tempo,
            "energy": energy,
        }
        return results
    except Exception as e:
        return {"error": str(e)}

# Function: Analyze Text with OpenAI
def analyze_text_with_openai(text):
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
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert neurologist specializing in Alzheimer's and Parkinson's diseases."},
                {"role": "user", "content": prompt},
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenAI API Error: {str(e)}"

# Route: Root URL
@app.get("/")
def read_root():
    return {"message": "Welcome to the Real-Time Audio Analysis API!"}

# Route: Real-Time Audio Analysis
@app.post("/analyze-real-time/")
def analyze_real_time_audio():
    # Step 1: Record Audio
    recorded_file = record_audio()

    # Step 2: Speech-to-Text
    text = speech_to_text_from_file(recorded_file)

    # Step 3: Analyze Audio Features
    audio_features = analyze_audio_features(recorded_file)

    # Step 4: Analyze Text with OpenAI
    openai_analysis = analyze_text_with_openai(text)

    # Clean up the recorded file
    if os.path.exists(recorded_file):
        os.remove(recorded_file)

    return {
        "recognized_text": text,
        "audio_features": audio_features,
        "openai_analysis": openai_analysis,
    }


