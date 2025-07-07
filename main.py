# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "assemblyai",
#     "numpy",
#     "openai",
#     "print-color",
#     "pyautogui",
#     "pyperclip",
#     "pydub",
#     "pynput",
#     "python-dotenv",
#     "sounddevice",
# ]
# ///

import io
import os
import threading
import time

import assemblyai as aai
import numpy as np
import pyautogui
import pyperclip
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from print_color import print
from pydub import AudioSegment
from pynput import keyboard

# Load environment variables from .env file
load_dotenv()

# Recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORDING_SECONDS = 120

HOTKEY = os.getenv("HOTKEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Check if API key is set
if not OPENAI_API_KEY or not HOTKEY:
    print("Error: OPENAI_API_KEY or HOTKEY environment variable not set.")
    exit(1)

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# Recording state class
class RecordingState:
    """Manages the state of audio recording including frames and timer."""
    def __init__(self):
        self.is_recording = False
        self.recorded_frames = []
        self.timer = None


state = RecordingState()


def on_key_press():
    """Toggle recording state and handle transcription when recording stops."""
    # Toggle recording state
    state.is_recording = not state.is_recording

    if state.is_recording:
        print("\x1b[2K\r", end="")
        print("Recording started... Press ", end="")
        print(HOTKEY, end="", color="cyan")
        print(" to stop.", end="\r",)
        state.recorded_frames = []  # Clear previous recordings
        state.timer = threading.Timer(MAX_RECORDING_SECONDS, on_key_press)
        state.timer.start()
    else:
        print("\x1b[2K\r", end="")
        print("Recording stopped. Transcribing...", end="\r")
        # Cancel the timer if manually stopped
        if state.timer and state.timer.is_alive():
            state.timer.cancel()
            state.timer = None

        transcribe_audio()
        print("Ready. Press ", end="")
        print(HOTKEY, end="", color="cyan")
        print(" to start/stop recording.", end="\r")


def get_audio_file():
    """Get the audio file from the recorded frames."""
    # Combine all recorded frames
    audio_data = np.concatenate(state.recorded_frames)

    # Convert to int16 for proper audio format
    audio_int16 = (audio_data * np.iinfo(np.int16).max).astype(np.int16)

    # Create an in-memory file-like object
    byte_io = io.BytesIO()

    # Create a temporary audio segment
    AudioSegment(
        audio_int16.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=CHANNELS,
    ).export(byte_io, format="wav")

    byte_io.seek(0)

    return byte_io


def transcribe_audio():
    """Transcribe the recorded audio."""
    if not state.recorded_frames:
        print("No audio recorded.")
        return

    if os.getenv("TRANSCRIPTION_SERVICE") == "openai":
        result = transcribe_openai()
    else:
        result = transcribe_assemblyai()

    # Check if transcription was successful
    if not result or not result.strip():
        print("Transcription failed or returned empty text.", color="red")
        return

    print("\x1b[2K\r", end="")
    print(f"\"{result}\"", color="green")

    # Save original clipboard content
    try:
        original_clipboard = pyperclip.paste()
    except Exception as e:
        print(f"Error in getting original clipboard: {str(e)}", color="red")
        original_clipboard = ""

    # Copy and paste transcription
    pyperclip.copy(result)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.2)  # wait for content to be pasted

    # Restore original clipboard content
    pyperclip.copy(original_clipboard)


def transcribe_openai():
    """Transcribe the recorded audio using OpenAI."""
    try:
        result = openai_client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=("audio.wav", get_audio_file())
        )
        return result.text
    except Exception as e:
        print(f"Error in OpenAI transcription: {str(e)}", color="red")
        return None


def transcribe_assemblyai():
    """Transcribe the recorded audio using AssemblyAI."""

    config = aai.TranscriptionConfig(
        speech_model="universal",
        language_code="de",
    )

    try:
        transcript = aai.Transcriber().transcribe(get_audio_file(), config)

        if (transcript.status == "error"):
            raise Exception(transcript.error)

        return transcript.text
    except Exception as e:
        print(f"Error in AssemblyAI transcription: {str(e)}", color="red")
        return None


def audio_callback(indata, frames, time_info, status):
    """Callback function to capture audio data."""
    if status:
        print(status, color="red")
    if state.is_recording:
        state.recorded_frames.append(indata.copy())


def setup_hotkey():
    """Set up the keyboard listener"""
    hotkey = keyboard.HotKey(
        keyboard.HotKey.parse(HOTKEY),
        on_key_press
    )

    # Create a listener
    with keyboard.Listener(
        on_press=hotkey.press,
        on_release=hotkey.release,
    ) as listener:
        print("Ready. Press ", end="")
        print(HOTKEY, end="", color="cyan")
        print(" to start/stop recording.", end="\r")
        listener.join()


# Start capturing audio from the microphone in the background
stream = sd.InputStream(
    callback=audio_callback,
    channels=CHANNELS,
    samplerate=SAMPLE_RATE,
)
stream.start()

# Setup and start the keyboard listener
try:
    setup_hotkey()
except KeyboardInterrupt:
    print("\nBye!")
finally:
    if stream.active:
        stream.stop()
        stream.close()
