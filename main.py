# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "sounddevice",
#     "numpy",
#     "openai",
#     "pydub",
#     "pynput",
#     "pyperclip",
#     "pyautogui",
# ]
# ///

import io
import threading
import argparse

import sounddevice as sd
import numpy as np
from openai import OpenAI
from pydub import AudioSegment
from pynput import keyboard
import pyperclip
import pyautogui

# Parse command line arguments
parser = argparse.ArgumentParser(description='Speech to text recording tool')
parser.add_argument(
    '--language',
    '-l',
    type=str,
    default="de",
    help='Language code for transcription (default: de)',
)
args = parser.parse_args()

# Recording parameters
LANGUAGE = args.language
HOTKEY = "<ctrl>+<alt>+;"
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORDING_SECONDS = 120
OPENAI_API_KEY = "sk-proj-1234567890"

# Display selected language
print(f"Using language: {LANGUAGE}")

client = OpenAI(api_key=OPENAI_API_KEY)


# Recording state class
class RecordingState:
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
        print(f"Recording started... Press {HOTKEY} to stop.")
        state.recorded_frames = []  # Clear previous recordings
        state.timer = threading.Timer(MAX_RECORDING_SECONDS, on_key_press)
        state.timer.start()
    else:
        print("Recording stopped. Transcribing...")
        # Cancel the timer if manually stopped
        if state.timer and state.timer.is_alive():
            state.timer.cancel()
            state.timer = None

        transcribe_audio()
        print("Press Ctrl+Alt+; to start recording again.")


def transcribe_audio():
    """Transcribe the recorded audio."""
    try:
        # Combine all recorded frames
        if not state.recorded_frames:
            return

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

        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", byte_io),
            language=LANGUAGE,
            response_format="text",
        )

        # Save original clipboard content
        original_clipboard = pyperclip.paste()

        # Copy and paste transcription
        pyperclip.copy(result)
        pyautogui.hotkey('ctrl', 'v')

        # Restore original clipboard content
        pyperclip.copy(original_clipboard)
    except Exception as e:
        print(f"Error in transcription: {str(e)}")


def audio_callback(indata, frames, time_info, status):
    """Callback function to capture audio data."""
    if status:
        print(status)
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
        print(f"Ready. Press {HOTKEY} to start/stop recording.")
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
    print("\nStopping the program...")
finally:
    if stream.active:
        stream.stop()
        stream.close()
