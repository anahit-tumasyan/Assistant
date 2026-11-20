# src/main.py

import os
from speech_module import SpeechAssistant
from Predict_command import TerminalAssistant
from Run_command import run_command


def main():
    # -----------------------------------------------------------------------
    # 1. Speech assistant (ElevenLabs online / Vosk offline)
    # -----------------------------------------------------------------------
    api_key        = os.environ.get("ELEVENLABS_API_KEY", "")
    vosk_model_path = os.path.abspath("Includes/model")

    speech = SpeechAssistant(
        api_key=api_key,
        model_path=vosk_model_path,
    )

    # -----------------------------------------------------------------------
    # 2. Local LLM — Gemma-3-1B for terminal command prediction
    # -----------------------------------------------------------------------
    gemma_model_path = os.path.abspath("Includes/gemma-3-1b-it")
    terminal = TerminalAssistant(
        model_path=gemma_model_path,
        device="mps",   # change to "cpu" or "cuda:0" if needed
    )

    print("=== Mini Assistant Started ===")
    print("Set ELEVENLABS_API_KEY env-var to enable online TTS/STT.")
    print("Press Ctrl-C to quit.\n")

    try:
        while True:
            answer = input("Press Enter to speak a command (or type 'quit' to exit): ").strip().lower()
            if answer in ("quit", "q", "exit"):
                break

            print("\nListening for your command…")
            text = speech.speech_to_text(duration=5)

            if not text or not text.strip():
                print("No speech detected. Please try again.\n")
                continue

            print(f'You said: "{text}"')

            cmd = terminal.predict_command(text)
            print(f"→ Generated command: {cmd}")

            confirm = input("Run this command? [y/N]: ").strip().lower()
            if confirm == "y":
                output = run_command(cmd)
                print(f"Output:\n{output}")
            else:
                print("Skipped.\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
