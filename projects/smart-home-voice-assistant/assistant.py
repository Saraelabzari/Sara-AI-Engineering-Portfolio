"""Starter voice assistant with simulated smart-home commands.

No real devices are controlled. Press Ctrl+C to exit.
"""
from __future__ import annotations

from datetime import datetime
import platform

import pyttsx3
import speech_recognition as sr


def create_speaker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    return engine


def respond(engine, message: str) -> None:
    print(f"Assistant: {message}")
    engine.say(message)
    engine.runAndWait()


def interpret_command(command: str, state: dict[str, bool]) -> str:
    text = command.lower().strip()
    if not text:
        return "I didn't catch that. Please try again."
    if "help" in text or "what can you do" in text:
        return "You can ask for the time, say turn demo light on or off, or say stop."
    if "time" in text:
        return f"The time is {datetime.now().strftime('%I:%M %p')}."
    if "turn" in text and "light" in text and "on" in text:
        state["demo_light"] = True
        return "The demo light is now on. This is only a simulated action."
    if "turn" in text and "light" in text and "off" in text:
        state["demo_light"] = False
        return "The demo light is now off. This is only a simulated action."
    if text in {"stop", "exit", "quit", "goodbye"}:
        return "Goodbye."
    return "I don't know that command yet. Say help to hear supported commands."


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
                    should_exit = command.lower().strip() in {"stop", "exit", "quit", "goodbye"}
                    reply = interpret_command(command, state)
                    respond(speaker, reply)
                    if should_exit:
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
