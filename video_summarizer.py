import os
import subprocess
import speech_recognition as sr
from transformers import pipeline

def transcribe_audio(audio_path):
    # Use SpeechRecognition to transcribe audio
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)

    with audio_file as source:
        audio_data = recognizer.record(source)
    
    try:
        # Use Google Web Speech API for transcription
        transcribed_text = recognizer.recognize_google(audio_data)
        return transcribed_text
    except sr.UnknownValueError:
        return "Google Web Speech API could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Web Speech API; {e}"

def extract_audio(video_path):
    # Extract audio from video and save as WAV
    audio_path = "temp_audio.wav"
    command = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path

def summarize_audio(video_path):
    # Extract audio from the video
    audio_path = extract_audio(video_path)

    # Load the BART summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Transcribe the audio
    transcribed_text = transcribe_audio(audio_path)
    
    # Debugging: Print the transcribed text
    print("Transcribed Text:", transcribed_text)

    # Check if the transcribed text is valid
    if not transcribed_text.strip():  # Ensure there's text to summarize
        return "No valid text found for summarization."

    # Generate the summary
    summary = summarizer(transcribed_text, max_length=150, min_length=30, do_sample=False)

    # Debugging: Print the raw summary output
    print("Raw Summary Output:", summary)

    return summary[0]['summary_text'] if summary else "Summary generation failed."

# Clean up temporary files after processing
def cleanup_temp_files():
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
    if os.path.exists("temp_video.mp4"):
        os.remove("temp_video.mp4")
