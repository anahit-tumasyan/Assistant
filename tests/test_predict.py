import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Predict_command as pc  # noqa: E402
from Run_command import check_command  # noqa: E402

FAKE_APPS = {"google chrome": "Google Chrome", "system settings": "System Settings",
             "music": "Music", "safari": "Safari"}


class FakeLLM:
    def __init__(self, answer=None):
        self.answer = answer
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        if self.answer is None:
            raise AssertionError("the model should not have been asked")
        return self.answer


class CleanCommandTests(unittest.TestCase):
    def test_strips_decoration(self):
        cases = {
            "```bash\nls -la\n```": "ls -la",
            "`pwd`": "pwd",
            "$ ls": "ls",
            "Command: date": "date",
            '"open ~/Downloads"': "open ~/Downloads",
            "ls\nThis lists the files.": "ls",
            "# list files\nls": "ls",
            "": "",
        }
        for raw, expected in cases.items():
            self.assertEqual(pc.clean_command(raw), expected, raw)


@mock.patch.object(pc, "_installed_apps", lambda: FAKE_APPS)
class FixCommandTests(unittest.TestCase):
    def test_open_slips_are_repaired(self):
        cases = {
            "open -a ~/Downloads": "open ~/Downloads",
            "open -a Downloads": "open ~/Downloads",
            "open -a github.com": "open https://github.com",
            "open github.com": "open https://github.com",
            "open -a Chrome": "open -a 'Google Chrome'",
            "open -a Settings": "open -a 'System Settings'",
            "open -a Music": "open -a Music",
            "open -a Safari": "open -a Safari",
            "open https://www.youtube.com": "open https://www.youtube.com",
            "ls -la": "ls -la",
        }
        for raw, expected in cases.items():
            self.assertEqual(pc.fix_command(raw), expected, raw)

    def test_go_to_a_folder_becomes_cd(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "inner"))
            with mock.patch.object(pc, "current_directory", lambda: tmp):
                self.assertEqual(pc.fix_command("ls inner", "go to the inner folder"), "cd inner")
                self.assertEqual(pc.fix_command("ls inner", "what is in inner"), "ls inner")
        self.assertEqual(pc.fix_command("open https://www.google.com", "go to google dot com"),
                         "open https://www.google.com")


class PromptTests(unittest.TestCase):
    def test_every_example_is_an_allowed_command(self):
        for request, command in pc.EXAMPLES:
            ok, reason = check_command(command)
            self.assertTrue(ok, f"example {request!r} -> {command!r}: {reason}")

    def test_most_similar_example_comes_last(self):
        chosen = pc.select_examples("open the pictures folder")
        self.assertEqual(len(chosen), pc.EXAMPLES_PER_REQUEST)
        self.assertTrue(chosen[-1][1].startswith("open"))

    def test_refusals_skip_the_model(self):
        assistant = pc.TerminalAssistant(llm=FakeLLM())
        for request in ["delete everything on my desktop", "please shut down the computer"]:
            command = assistant.predict_command(request)
            self.assertTrue(command.startswith('echo "Sorry'), command)
            self.assertTrue(check_command(command)[0])

    @mock.patch.object(pc, "_installed_apps", lambda: FAKE_APPS)
    def test_model_answer_is_cleaned_and_fixed(self):
        assistant = pc.TerminalAssistant(llm=FakeLLM("```\nopen -a Downloads\n```"))
        self.assertEqual(assistant.predict_command("open downloads"), "open ~/Downloads")

    def test_example_phrases_are_answered_without_guessing(self):
        assistant = pc.TerminalAssistant(llm=FakeLLM())
        self.assertEqual(assistant.predict_command("Open the downloads folder."), "open ~/Downloads")
        self.assertEqual(assistant.predict_command("What is the date today?"), "date")

    def test_empty_request(self):
        self.assertEqual(pc.TerminalAssistant(llm=FakeLLM()).predict_command("  "), "")


@unittest.skipUnless(os.environ.get("ASSISTANT_SLOW_TESTS") and os.path.isdir(pc.DEFAULT_MODEL_PATH),
                     "set ASSISTANT_SLOW_TESTS=1 to run the real Gemma model")
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from emergency_assistant import EmergencyAssistant
        cls.terminal = pc.TerminalAssistant()
        cls.emergency = EmergencyAssistant(llm=cls.terminal.llm)

    def test_simple_requests(self):
        self.assertTrue(self.terminal.predict_command("list the files in this folder").startswith("ls"))
        self.assertEqual(self.terminal.predict_command("open the downloads folder"), "open ~/Downloads")
        self.assertTrue(self.terminal.predict_command("what is the date today").startswith("date"))

    def test_emergency_chat_keeps_history(self):
        first = self.emergency.respond("I'm stuck in an elevator and my arm is bleeding.")
        self.assertTrue(first)
        second = self.emergency.respond("Yes, I called 112.")
        self.assertTrue(second)
        self.assertEqual(len(self.emergency.history), 4)


if __name__ == "__main__":
    unittest.main()
