# Smart Home Voice Assistant

A local voice-assistant starter that listens for a short command, converts speech to text, matches it against a small set of supported commands, and responds using text-to-speech.

> **Status:** Starter prototype. It does not control real smart-home devices. Commands are simulated in the terminal/app and should be tested locally before extending.

## Features

- Speech recognition with SpeechRecognition
- Text-to-speech responses
- Simple intent matching for time, help, and demo light commands
- Explicit confirmation for simulated state changes
- No audio recording saved by the application

## Requirements

- Python 3.10+
- Microphone and speakers
- Internet may be required by the default Google speech-recognition backend
- For offline speech recognition, replace the backend with an appropriately configured local engine such as Vosk

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python assistant.py
```

## Safety

The included light on/off actions only update a simulated state. Before connecting real devices, implement authentication, explicit confirmation for consequential actions, network security, an allowlist of devices, and a manual override. Never expose device credentials or home-network secrets in source control.
