from datetime import datetime

import pytest

from src.commands import interpret_command, is_exit_command, normalize_command


@pytest.fixture
def state():
    return {"demo_light": False}


def fixed_time():
    return datetime(2026, 10, 15, 14, 5)


def test_turn_demo_light_on_changes_simulated_state(state):
    response = interpret_command("turn the light on", state)
    assert state["demo_light"] is True
    assert "simulated" in response


def test_turn_demo_light_off_changes_simulated_state(state):
    state["demo_light"] = True
    response = interpret_command("turn off the light", state)
    assert state["demo_light"] is False
    assert "simulated" in response


def test_time_command_can_use_injected_clock(state):
    assert interpret_command("what time is it", state, now=fixed_time) == "The time is 02:05 PM."


def test_unknown_command_does_not_change_state(state):
    response = interpret_command("launch a spaceship", state)
    assert "don't know" in response
    assert state == {"demo_light": False}


def test_blank_command_is_handled(state):
    assert "didn't catch" in interpret_command("   ", state)


def test_command_normalization():
    assert normalize_command("  TURN   THE LIGHT ON!! ") == "turn the light on"
    assert is_exit_command(" Goodbye! ")
    assert not is_exit_command("goodbye lights")


def test_unknown_device_or_action_is_not_executed(state):
    interpret_command("unlock the front door", state)
    assert state == {"demo_light": False}
