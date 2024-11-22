#to run uvicorn main:app --reload

from fastapi import FastAPI, UploadFile, File
import os
import speech_recognition as sr
from transformers import pipeline
import openai

# Initialize FastAPI
app = FastAPI()

# Set up OpenAI API Key
openai.api_key = ""  # Replace with your OpenAI API key

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

# Function: Text Analysis with OpenAI GPT
def analyze_text_with_openai(text):
    prompt = f"Analyze this text for linguistic anomalies, articulation issues, and pauses: {text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Changed from "gpt-4"
            messages=[
                {"role": "system", "content": "You are an AI assistant specializing in linguistic analysis."},
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

    return {
        "recognized_text": text,
        "huggingface_analysis": huggingface_result,
        "openai_analysis": openai_result,
    }