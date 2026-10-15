"""Voice interface for a smart-home assistant with simulated actions only."""
from __future__ import annotations

import platform

import pyttsx3
import speech_recognition as sr

from src.commands import interpret_command, is_exit_command


def create_speaker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    return engine


def respond(engine, message: str, speak: bool = True) -> None:
    """Print every response; optionally speak it through text-to-speech."""
    print(f"Assistant: {message}")
    if speak:
        engine.say(message)
        engine.runAndWait()


def main() -> None:
    recognizer = sr.Recognizer()
    speaker = create_speaker()
    state = {"demo_light": False}
    print("Smart Home Voice Assistant — simulated commands only")
    print(f"Platform: {platform.system()}")
    respond(speaker, "Hello. Say help to hear the commands I understand.")

    try:
        with sr.Microphone() as source:
            print("Calibrating microphone noise level...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            while True:
                try:
                    print("\nListening...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
                    command = recognizer.recognize_google(audio)
                    print(f"You: {command}")
                    reply = interpret_command(command, state)
                    respond(speaker, reply)
                    if is_exit_command(command):
                        break
                except sr.WaitTimeoutError:
                    print("No speech detected. Try again.")
                except sr.UnknownValueError:
                    respond(speaker, "Sorry, I couldn't understand that.")
                except sr.RequestError as exc:
                    print(f"Speech recognition service error: {exc}")
                    respond(speaker, "Speech recognition is currently unavailable.")
                    break
    except (OSError, AttributeError) as exc:
        print(f"Microphone could not be initialized: {exc}")
        print("Check microphone permissions and PyAudio installation.")


if __name__ == "__main__":
    main()
