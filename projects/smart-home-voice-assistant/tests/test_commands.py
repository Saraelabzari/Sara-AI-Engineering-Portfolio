import unittest

from assistant import interpret_command


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.state = {"demo_light": False}

    def test_turn_demo_light_on(self):
        response = interpret_command("turn the light on", self.state)
        self.assertTrue(self.state["demo_light"])
        self.assertIn("simulated", response)

    def test_turn_demo_light_off(self):
        self.state["demo_light"] = True
        interpret_command("turn the light off", self.state)
        self.assertFalse(self.state["demo_light"])

    def test_unknown_command(self):
        response = interpret_command("launch a spaceship", self.state)
        self.assertIn("don't know", response)


if __name__ == "__main__":
    unittest.main()
