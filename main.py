"""
main.py - talk (or type) to your terminal.

Press Enter and say what you want ("open my downloads folder"), or just type
it. The local Gemma 3 model turns it into a terminal command, shows it to
you, and runs it only after you say yes.
"""

import os

from Predict_command import TerminalAssistant
from Run_command import check_command, current_directory, run_command
from speech_module import SpeechAssistant

QUIT_WORDS = {"quit", "q", "exit"}


def short_path(path: str) -> str:
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path.startswith(home) else path


def load_speech():
    try:
        speech = SpeechAssistant(api_key=os.environ.get("ELEVENLABS_API_KEY", ""))
    except Exception as exc:
        print(f"Voice input is off ({exc}). You can still type requests.")
        return None
    return speech if speech.can_listen else None


def ask_for_request(speech) -> str | None:
    """Return the next request, "" to skip a turn, or None to quit."""
    typed = input(f"[{short_path(current_directory())}] Enter = speak, or type a request: ").strip()
    if typed.lower() in QUIT_WORDS:
        return None
    if typed:
        return typed
    if speech is None:
        print("Voice input is off - type your request instead.\n")
        return ""
    print("Listening... (it stops when you stop talking)")
    try:
        heard = speech.speech_to_text(max_seconds=10)
    except Exception as exc:
        print(f"Microphone problem: {exc}\n")
        return ""
    if not heard:
        print("I didn't catch that. Try again.\n")
        return ""
    print(f'You said: "{heard}"')
    return heard


def main() -> None:
    speech = load_speech()
    print("Loading Gemma 3...")
    terminal = TerminalAssistant()

    print("\n=== Mini Assistant ===")
    print("Press Enter and speak, or type what you want. Type 'quit' to exit.")
    print("Set ELEVENLABS_API_KEY to use ElevenLabs instead of offline Vosk.\n")

    while True:
        request = ask_for_request(speech)
        if request is None:
            break
        if not request:
            continue

        command = terminal.predict_command(request)
        if not command:
            print("Sorry, I couldn't turn that into a command.\n")
            continue
        print(f"Command: {command}")

        ok, reason = check_command(command)
        if not ok:
            print(f"Not running it: {reason}\n")
            continue

        if input("Run it? [y/N]: ").strip().lower() in ("y", "yes"):
            print(run_command(command).rstrip() + "\n")
        else:
            print("Skipped.\n")

    print("Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
