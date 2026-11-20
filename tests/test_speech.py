import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import speech_module as sm  # noqa: E402

RATE = 16000


def synthesize(text: str) -> bytes:
    """Raw 16 kHz mono audio of the macOS voice saying `text`."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "speech.wav")
        subprocess.run(["say", "-o", path, f"--data-format=LEI16@{RATE}", "--", text], check=True)
        with wave.open(path) as wav:
            return wav.readframes(wav.getnframes())


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(RATE * seconds)


def chunks(pcm: bytes, seconds: float = sm.CHUNK_SECONDS):
    size = int(RATE * seconds) * 2
    for start in range(0, len(pcm), size):
        yield pcm[start:start + size]


def trim_trailing_silence(pcm: bytes) -> bytes:
    samples = memoryview(pcm).cast("h")
    end = len(samples)
    while end > 0 and abs(samples[end - 1]) < 300:
        end -= 1
    return pcm[:end * 2]


class HelperTests(unittest.TestCase):
    def test_pcm_to_wav(self):
        data = sm.pcm_to_wav(silence(0.5), RATE)
        with wave.open(io.BytesIO(data)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, RATE))
            self.assertEqual(wav.getnframes(), RATE // 2)

    def test_yes_no(self):
        self.assertTrue(sm.interpret_yes_no("yes"))
        self.assertTrue(sm.interpret_yes_no("Yeah, sure."))
        self.assertFalse(sm.interpret_yes_no("no"))
        self.assertFalse(sm.interpret_yes_no("do not"))
        self.assertFalse(sm.interpret_yes_no("yes no"))   # unclear -> treat as no
        self.assertIsNone(sm.interpret_yes_no(""))
        self.assertIsNone(sm.interpret_yes_no("banana"))


@unittest.skipUnless(shutil.which("say") and os.path.isdir(sm.DEFAULT_VOSK_MODEL),
                     "needs the macOS 'say' command and the Vosk model")
class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.speech = sm.SpeechAssistant(api_key="")   # offline; never opens the microphone

    def test_recognises_a_command(self):
        text, _ = self.speech.transcribe_stream(chunks(synthesize("list the files on the desktop") + silence(2)))
        self.assertIn("desktop", text)

    def test_stops_about_a_second_after_you_stop_talking(self):
        spoken = synthesize("open the downloads folder")
        _, audio = self.speech.transcribe_stream(chunks(spoken + silence(8)), max_seconds=20,
                                                 silence_after_speech=1.0)
        heard_seconds = len(audio) / (2 * RATE)
        self.assertLess(heard_seconds, len(spoken) / (2 * RATE) + 2.5)

    def test_words_right_at_the_end_are_kept(self):
        spoken = trim_trailing_silence(synthesize("what time is it"))
        text, _ = self.speech.transcribe_stream(chunks(spoken))
        self.assertIn("time", text)

    def test_yes_no_grammar(self):
        grammar = sorted(sm.YES_WORDS | sm.NO_WORDS) + ["[unk]"]
        yes, _ = self.speech.transcribe_stream(chunks(synthesize("yes") + silence(1)), grammar=grammar)
        no, _ = self.speech.transcribe_stream(chunks(synthesize("no, cancel") + silence(1)), grammar=grammar)
        self.assertTrue(sm.interpret_yes_no(yes), yes)
        self.assertFalse(sm.interpret_yes_no(no), no)


if __name__ == "__main__":
    unittest.main()
