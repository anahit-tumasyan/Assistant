import shutil
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import scrolledtext

from transformers import AutoTokenizer, pipeline

try:
    import speech_recognition as sr
except ImportError:
    sr = None


SYSTEM_PROMPT = """You are an AI Emergency Response Assistant. Your primary function is to provide immediate, calming, and clear guidance to individuals in distressing and life-threatening situations, such as being trapped under rubble, stuck in a broken elevator, or injured and awaiting rescue. You are the voice of help before professional help arrives.

Your personality must be:
- Calm and Reassuring: Use a steady, empathetic, and supportive tone. Avoid panic-inducing language.
- Clear and Authoritative: Give simple, direct, and actionable instructions. Avoid jargon or complex sentences.
- Methodical: Follow a logical sequence of questions and instructions.
- Appropriately Gentle Humor: Only when the person is stable and not in immediate danger, use a very light, gentle touch to help ease tension.

Your Core Operating Protocol:
1. Acknowledge and Reassure Immediately.
2. Prioritize Professional Emergency Services — always ask first if they've called 911/112.
3. Assess the situation: name, location, injuries, others present, immediate dangers.
4. Provide Calming and Survival Instructions:
   - Breathing exercises (4-4-6 box breathing).
   - Conserve phone battery.
   - Signal for help every 10-15 minutes (sets of three).
   - Stay put unless there is an immediate life-threatening danger.

Handling Injury Images (Visual Triage):
A. IMMEDIATE DISCLAIMER: "IMPORTANT: I am an AI and not a medical doctor. This is not a diagnosis."
B. Ask about bleeding severity.
C. Simple first-aid steps based on severity.
D. CRITICAL DO NOTs: no removing embedded objects, no creams or folk remedies."""


class EmergencyAssistant:
    def __init__(self,
                 model_path: str = "./Includes/gemma-3-1b-it",
                 device: str = "mps"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=self.tokenizer,
            device=device,
            torch_dtype="auto",
        )
        self.history: list[dict] = []  # multi-turn conversation history

    def respond(self, user_text: str) -> str:
        """Generate a response, maintaining full conversation history."""
        self.history.append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.pipe(prompt, max_new_tokens=256, do_sample=True)
        generated = outputs[0]["generated_text"][len(prompt):].strip()

        self.history.append({"role": "assistant", "content": generated})
        return generated


# ---------------------------------------------------------------------------
# TTS helper
# ---------------------------------------------------------------------------

def speak(text: str) -> None:
    """Speak text using the macOS 'say' command, or print as fallback."""
    if not text:
        return
    say_path = shutil.which("say")
    if say_path:
        try:
            subprocess.run([say_path, text], check=False)
        except Exception:
            print(text)
    else:
        print(f"[TTS fallback] {text}")


# ---------------------------------------------------------------------------
# STT helper
# ---------------------------------------------------------------------------

def listen(timeout: int = 10, phrase_time_limit: int = 10) -> str:
    """
    Record one phrase from the microphone and return the transcription.
    Falls back to keyboard input if SpeechRecognition is not installed.
    """
    if sr is None:
        try:
            return input("You: ").strip()
        except EOFError:
            return ""

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening… (speak now)")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        text = recognizer.recognize_google(audio)
        print(f"You (heard): {text}")
        return text
    except Exception:
        try:
            return input("You (type fallback): ").strip()
        except EOFError:
            return ""


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class ChatGUI:
    def __init__(self, assistant: EmergencyAssistant):
        self.assistant = assistant

        self.root = tk.Tk()
        self.root.title("Emergency Assistant (Gemma3)")

        self.chat = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, height=20, width=70, state=tk.DISABLED
        )
        self.chat.grid(row=0, column=0, columnspan=4, padx=8, pady=8, sticky="nsew")

        self.entry = tk.Entry(self.root, width=60)
        self.entry.grid(row=1, column=0, columnspan=3, padx=8, pady=8, sticky="ew")
        self.entry.bind("<Return>", lambda _: self.on_send())

        self.send_btn = tk.Button(self.root, text="Send", command=self.on_send)
        self.send_btn.grid(row=1, column=3, padx=8, pady=8)

        self.mic_btn = tk.Button(self.root, text="🎤 Mic", command=self.on_mic)
        self.mic_btn.grid(row=2, column=0, padx=8, pady=4, sticky="w")

        self.speak_var = tk.BooleanVar(value=True)
        self.speak_chk = tk.Checkbutton(
            self.root, text="Speak responses", variable=self.speak_var
        )
        self.speak_chk.grid(row=2, column=1, padx=8, pady=4, sticky="w")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.ui_queue: queue.Queue = queue.Queue()
        self.root.after(100, self._process_queue)

        greeting = "Hello. I am here with you. Take a slow breath — we are going to work through this together."
        self._append_chat("Assistant", greeting)
        if self.speak_var.get():
            threading.Thread(target=speak, args=(greeting,), daemon=True).start()

    # ------------------------------------------------------------------

    def _append_chat(self, speaker: str, text: str) -> None:
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{speaker}: {text}\n\n")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def _set_buttons(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.send_btn.config(state=state)
        self.mic_btn.config(state=state)

    # ------------------------------------------------------------------

    def on_send(self) -> None:
        user_text = self.entry.get().strip()
        if not user_text:
            return
        self.entry.delete(0, tk.END)
        self._append_chat("You", user_text)
        self._set_buttons(False)
        threading.Thread(target=self._generate_response, args=(user_text,), daemon=True).start()

    def on_mic(self) -> None:
        if sr is None:
            self._append_chat(
                "Assistant",
                "Microphone speech recognition is not installed.\n"
                "Run: pip install SpeechRecognition pyaudio\n"
                "On macOS: brew install portaudio && pip install pyaudio"
            )
            return
        self._set_buttons(False)
        threading.Thread(target=self._capture_mic_and_respond, daemon=True).start()

    # ------------------------------------------------------------------

    def _capture_mic_and_respond(self) -> None:
        try:
            text = listen()
        except Exception as e:
            self.ui_queue.put(("Assistant", f"Mic error: {e}"))
            self.ui_queue.put(("__enable__", ""))
            return

        if text:
            self.ui_queue.put(("You", text))
            self._generate_response(text)
        else:
            self.ui_queue.put(("__enable__", ""))

    def _generate_response(self, text: str) -> None:
        try:
            response = self.assistant.respond(text)
        except Exception as e:
            response = f"[Error generating response: {e}]"

        self.ui_queue.put(("Assistant", response))

        if self.speak_var.get():
            threading.Thread(target=speak, args=(response,), daemon=True).start()

        self.ui_queue.put(("__enable__", ""))

    # ------------------------------------------------------------------

    def _process_queue(self) -> None:
        try:
            while True:
                speaker, msg = self.ui_queue.get_nowait()
                if speaker == "__enable__":
                    self._set_buttons(True)
                else:
                    self._append_chat(speaker, msg)
        except queue.Empty:
            pass
        self.root.after(100, self._process_queue)

    def run(self) -> None:
        self.entry.focus_set()
        self.root.mainloop()


if __name__ == "__main__":
    assistant = EmergencyAssistant()
    gui = ChatGUI(assistant)
    gui.run()
