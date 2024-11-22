# Neurological-Speech-Analysis
# **Real-Time Neurological Speech Analysis**

This project is a FastAPI application designed for analyzing speech data in real time to detect potential neurological symptoms related to **Alzheimer's** and **Parkinson's diseases**. The application includes two main functionalities:

1. **Speech Analysis with Key Updates (`main1.py`)**:
   - Extracts speech-to-text, analyzes linguistic features, and uses Hugging Face and OpenAI APIs for text interpretation.
2. **Real-Time Audio Recording and Analysis (`realtime.py`)**:
   - Records audio in real time, processes it, and integrates speech-to-text, audio feature extraction, and OpenAI analysis.

---

## **Features**

- **Speech-to-Text Conversion**:
  - Converts recorded speech into text using Google's Speech Recognition API.
  
- **Audio Feature Extraction**:
  - Extracts features like pitch variation, energy, tempo, and signal-to-noise ratio using `librosa`.

- **Advanced Text Analysis**:
  - Uses Hugging Face and OpenAI APIs to analyze linguistic anomalies, monotony, pauses, and articulation issues.

- **Real-Time Recording**:
  - Records live audio and processes it for immediate analysis.

---

## **Installation Guide**

### **1. Clone the Repository**

To get started, clone this repository to your local machine:
```bash
git clone https://github.com/your-username/neurological-speech-analysis.git
cd neurological-speech-analysis
```
### **2. Set Up the Virtual Environment**

It is recommended to create a virtual environment to isolate the project dependencies.

#### On Windows:
1. Create the virtual environment:
   ```bash
   python -m venv env
2. Activate the virtual environment:
   ```bash
   .\env\Scripts\activate
Once activated, you will see (env) in your terminal, indicating the environment is active.

On Linux/MacOS:
1. Create the virtual environment:
  ```bash
   python3 -m venv env
```
2. Activate the virtual environment:
   ```bash
   source env/bin/activate
   ```
### **3. Install Required Dependencies
Install the dependencies listed in the requirements.txt file:
  ```bash
pip install -r requirements.txt
```
If requirements.txt is missing, install the following manually:
  ```bash
  pip install fastapi uvicorn speechrecognition librosa numpy transformers openai pyaudio python-dotenv
  ```
## How to Use
### Run the FastAPI Application
Start the server using either main1.py or realtime.py:

### Option 1: For Pre-Recorded Audio Analysis
Run the application:
   ```bash
uvicorn main1:app --reload
  ```
The server will start, and the API will be accessible at:
  ```bash
http://127.0.0.1:8000
  ```
### Option 2: For Real-Time Audio Recording and Analysis
Run the application:
  ```bash
uvicorn realtime:app --reload
  ```
The server will start, and the API will be accessible at:
  ```bash
http://127.0.0.1:8000
  ```
## API Endpoints
### Root URL (GET /)
  Description: Confirms that the application is running.
   ```bash
  Response:
{
  "message": "Welcome to the Real-Time Audio Analysis API!"
}
```
## Pre-Recorded Audio Analysis (POST /process-preconverted/)
Description: Processes a pre-recorded .wav file for speech-to-text conversion, audio feature extraction, and advanced text analysis.
## How to Use:
  Place your .wav file in the root directory of the project.
  Ensure the file is named audio_file-o.wav.
Response Example:
   ```bash
{
  "recognized_text": "elle publie les comptes rendus des conseils municipaux",
  "huggingface_analysis": [
      {"label": "POSITIVE", "score": 0.9349}
  ],
  "openai_analysis": "The text is linguistically correct with no signs of Alzheimer's or Parkinson's symptoms.",
  "audio_features": {
      "pitch_std": 1183.5,
      "snr": -0.67,
      "pitch_mean": 1774.01,
      "tempo": 117.18,
      "energy": 0.0014
  }
}
```
