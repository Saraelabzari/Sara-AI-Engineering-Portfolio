# Smart Home Voice Assistant

A voice-controlled Python prototype that recognizes a small set of spoken commands, responds using text-to-speech, and simulates a smart-home light state.

> **Status:** Portfolio prototype. It does not control physical devices. Speech recognition uses the default Google backend through SpeechRecognition, which requires internet access and sends audio to that service for recognition.

## Features

- Microphone input with SpeechRecognition
- Text-to-speech responses with pyttsx3
- Reusable allowlist-based command parser in `src/commands.py`
- Demo light on/off state simulation
- Help, time, and exit commands
- Unknown commands do not trigger device actions
- Unit tests for command handling and state changes

## Requirements

- Python 3.10+
- Microphone and speakers
- Internet access for the default Google speech-recognition backend
- PyAudio and the system dependencies needed to access your microphone

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python assistant.py
```

Say “help” for supported commands. Say “stop” or “goodbye” to exit.

## Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Tests are included but have not been executed in this environment. They test command interpretation without requiring a microphone or audio service.

## Supported commands

- “What time is it?”
- “Turn the light on”
- “Turn the light off”
- “Help”
- “Stop”, “exit”, “quit”, or “goodbye”

Only these explicit phrases are recognized. The assistant does not infer arbitrary device commands.

## Safety, privacy, and limitations

The light actions update only an in-memory simulated state. The application does not intentionally save audio, but the Google recognition backend processes captured audio remotely. Avoid speaking private information while testing.

Before connecting real devices, implement authentication, an explicit allowlist of devices and actions, confirmation for consequential operations, secure network communication, audit controls, and a physical/manual override. Never store home-network credentials in source control. Consider an offline recognition backend if sending audio to a remote service is not appropriate.
