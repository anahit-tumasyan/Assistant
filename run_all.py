#!/usr/bin/env python3
"""
run_all.py - hands-free mode: gesture mouse control and voice commands at once.

  main thread : camera window + hand gestures (macOS only allows OpenCV
                windows on the main thread)
  background  : listen -> Gemma 3 turns the words into a command -> run it

The voice status is shown under the gesture bar in the camera window, and
every command and its output are also printed in the terminal.

Commands that could change your system or run code (python, pip, brew,
curl, ...) are only run after you answer "yes" out loud. Everything else on
the allow-list in Run_command.py runs straight away.
"""

import os
import threading
import time

import hand_landmark
from Predict_command import TerminalAssistant
from Run_command import check_command, needs_confirmation, run_command
from speech_module import SpeechAssistant


class Status:
    """A line of text shared between the voice thread and the camera loop."""

    def __init__(self, text=""):
        self._text = text
        self._lock = threading.Lock()

    def set(self, text):
        with self._lock:
            self._text = text

    def get(self):
        with self._lock:
            return self._text


def voice_loop(status: Status) -> None:
    status.set("Voice: loading speech recognition...")
    try:
        speech = SpeechAssistant(api_key=os.environ.get("ELEVENLABS_API_KEY", ""))
        if not speech.can_listen:
            status.set("Voice: off - no Vosk model in Includes/model")
            return
    except Exception as exc:
        status.set(f"Voice: off - speech failed to start ({exc})")
        return

    status.set("Voice: loading Gemma 3...")
    try:
        terminal = TerminalAssistant()
    except Exception as exc:
        status.set(f"Voice: off - could not load Gemma ({exc})")
        return

    while True:
        try:
            status.set("Voice: listening - say a command")
            text = speech.speech_to_text(max_seconds=10)
            if not text:
                continue

            status.set(f'Voice: heard "{text}" - thinking...')
            command = terminal.predict_command(text)
            print(f'\nHeard:   "{text}"\nCommand: {command or "(none)"}')
            if not command:
                status.set("Voice: couldn't turn that into a command")
                time.sleep(2)
                continue

            ok, reason = check_command(command)
            if not ok:
                print(f"Blocked: {reason}")
                status.set(f"Voice: blocked {command} - {reason}")
                time.sleep(3)
                continue

            if needs_confirmation(command):
                status.set(f"Voice: run  {command}  ? say yes or no")
                speech.text_to_speech("Should I run this command? Say yes or no.")
                if not speech.confirm():
                    print("Cancelled.")
                    status.set("Voice: cancelled")
                    time.sleep(1.5)
                    continue

            status.set(f"Voice: running {command}")
            output = run_command(command)
            print(output.rstrip())
            first_line = next((line for line in output.splitlines() if line.strip()), "done")
            status.set(f"Voice: {first_line.strip()}")
            time.sleep(2.5)

        except Exception as exc:
            print(f"Voice error: {exc}")
            status.set(f"Voice: error - {exc}")
            time.sleep(3)


def main() -> None:
    status = Status("Voice: starting...")
    threading.Thread(target=voice_loop, args=(status,), daemon=True).start()
    hand_landmark.run(status_text=status.get, window_title="Assistant  [q = quit]")
    print("Bye.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
