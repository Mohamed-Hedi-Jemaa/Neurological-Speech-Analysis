from pydub import AudioSegment

def convert_mp3_to_wav(input_file, output_file):
    # Load the MP3 file
    audio = AudioSegment.from_file(input_file, format="mp3")
    # Export the audio to WAV format
    audio.export(output_file, format="wav")
    print(f"Converted {input_file} to {output_file}")

# Example usage
convert_mp3_to_wav("audio_file.mp3", "audio_file-o.wav")
