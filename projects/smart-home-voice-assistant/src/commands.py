"""Safe command parsing for a simulated smart-home voice assistant."""
from __future__ import annotations

from datetime import datetime
import re
from typing import Callable

EXIT_COMMANDS = {"stop", "exit", "quit", "goodbye"}
HELP_RESPONSE = (
    "You can ask for the time, say turn demo light on or off, "
    "or say stop."
)


def normalize_command(command: str) -> str:
    """Normalize whitespace and punctuation for simple intent matching."""
    if not isinstance(command, str):
        raise TypeError("command must be a string.")
    return re.sub(r"\s+", " ", command.strip().lower()).strip(" .,!?")


def is_exit_command(command: str) -> bool:
    """Return True only for a direct exit phrase."""
    return normalize_command(command) in EXIT_COMMANDS


def interpret_command(
    command: str,
    state: dict[str, bool],
    now: Callable[[], datetime] = datetime.now,
) -> str:
    """Interpret a small allowlist of commands; actions only modify simulated state."""
    text = normalize_command(command)
    if not text:
        return "I didn't catch that. Please try again."
    if text in {"help", "what can you do", "show help"}:
        return HELP_RESPONSE
    if text in {"what time is it", "tell me the time", "time", "current time"}:
        return f"The time is {now().strftime('%I:%M %p')}."
    if text in {
        "turn the light on", "turn demo light on", "switch the light on",
        "turn on the light", "turn on demo light",
    }:
        state["demo_light"] = True
        return "The demo light is now on. This is only a simulated action."
    if text in {
        "turn the light off", "turn demo light off", "switch the light off",
        "turn off the light", "turn off demo light",
    }:
        state["demo_light"] = False
        return "The demo light is now off. This is only a simulated action."
    if is_exit_command(text):
        return "Goodbye."
    return "I don't know that command yet. Say help to hear supported commands."
