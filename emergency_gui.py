"""
emergency_gui.py - the emergency assistant in a small chat window.

Type a message or press "Speak" and talk; replies can be read out loud.
Everything runs on this computer (Gemma 3 + Vosk) unless ELEVENLABS_API_KEY
is set, in which case ElevenLabs is used for listening and talking.

The window opens straight away; the model loads in the background.
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext

from emergency_assistant import GREETING, EmergencyAssistant
from speech_module import SpeechAssistant


class ChatGUI:
    def __init__(self):
        self.assistant = None
        self.speech = None
        self.busy = True
        self.events: queue.Queue = queue.Queue()   # worker threads -> Tk thread

        self.root = tk.Tk()
        self.root.title("Emergency Assistant (Gemma 3)")
        self.root.minsize(520, 420)

        self.chat = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, height=20, width=70, state=tk.DISABLED,
            font=("Helvetica", 14), padx=8, pady=8,
        )
        self.chat.grid(row=0, column=0, columnspan=4, padx=8, pady=8, sticky="nsew")

        self.entry = tk.Entry(self.root, width=60, font=("Helvetica", 14))
        self.entry.grid(row=1, column=0, columnspan=3, padx=8, pady=4, sticky="ew")
        self.entry.bind("<Return>", lambda _event: self.on_send())

        self.send_btn = tk.Button(self.root, text="Send", command=self.on_send)
        self.send_btn.grid(row=1, column=3, padx=8, pady=4)

        self.mic_btn = tk.Button(self.root, text="Speak", command=self.on_mic)
        self.mic_btn.grid(row=2, column=0, padx=8, pady=4, sticky="w")

        self.speak_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Read replies aloud", variable=self.speak_var,
                       command=self._on_toggle_speech).grid(row=2, column=1, padx=8, pady=4, sticky="w")

        self.status = tk.Label(self.root, text="", anchor="w", fg="gray40")
        self.status.grid(row=3, column=0, columnspan=4, padx=8, pady=(0, 8), sticky="ew")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._append("Assistant", GREETING)
        self._set_busy(True, "Loading the model... (a few seconds)")
        threading.Thread(target=self._load, args=(self.speak_var.get(),), daemon=True).start()
        self.root.after(100, self._process_events)

    # ------------------------------------------------------------------
    # Background work (never touches Tk directly - it posts events)
    # ------------------------------------------------------------------

    def _load(self, greet_aloud: bool) -> None:
        try:
            self.speech = SpeechAssistant(api_key=os.environ.get("ELEVENLABS_API_KEY", ""))
        except Exception as exc:
            self.speech = None
            print(f"Voice is off: {exc}")
        if greet_aloud and self.speech:
            threading.Thread(target=self.speech.text_to_speech, args=(GREETING,), daemon=True).start()
        try:
            self.assistant = EmergencyAssistant()
        except Exception as exc:
            self.events.put(("fatal", f"Could not load the model: {exc}"))
            return
        self.events.put(("ready", None))

    def _listen_then_reply(self) -> None:
        try:
            text = self.speech.speech_to_text(max_seconds=12)
        except Exception as exc:
            self.events.put(("problem", f"Microphone problem: {exc}"))
            return
        if not text:
            self.events.put(("idle", "I didn't catch that - try again, or type your message."))
            return
        self.events.put(("you", text))
        self._reply(text)

    def _reply(self, text: str) -> None:
        self.events.put(("status", "Thinking..."))
        try:
            reply = self.assistant.respond(text)
        except Exception as exc:
            self.events.put(("problem", f"Sorry, something went wrong: {exc}"))
            return
        self.events.put(("reply", reply))

    def _say(self, text: str) -> None:
        if self.speech:
            self.speech.stop_speaking()
            self.speech.text_to_speech(text)

    # ------------------------------------------------------------------
    # Tk thread
    # ------------------------------------------------------------------

    def _process_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "ready":
                    can_listen = bool(self.speech and self.speech.can_listen)
                    self._set_busy(False, "Ready." if can_listen
                                   else "Ready. (Voice input is off - no Vosk model found.)")
                elif kind == "fatal":
                    self._append("Assistant", payload)
                    self.status.config(text=payload)
                elif kind == "status":
                    self.status.config(text=payload)
                elif kind == "you":
                    self._append("You", payload)
                elif kind == "reply":
                    self._append("Assistant", payload)
                    if self.speak_var.get():
                        threading.Thread(target=self._say, args=(payload,), daemon=True).start()
                    self._set_busy(False, "Ready.")
                elif kind == "problem":
                    self._append("Assistant", payload)
                    self._set_busy(False, "Ready.")
                elif kind == "idle":
                    self._set_busy(False, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._process_events)

    def _append(self, speaker: str, text: str) -> None:
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{speaker}: {text}\n\n")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def _set_busy(self, busy: bool, status: str = "") -> None:
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.send_btn.config(state=state)
        can_listen = bool(self.speech and self.speech.can_listen)
        self.mic_btn.config(state=tk.NORMAL if (not busy and can_listen) else tk.DISABLED)
        if status:
            self.status.config(text=status)
        if not busy:
            self.entry.focus_set()

    def on_send(self) -> None:
        text = self.entry.get().strip()
        if self.busy or not text:   # Enter still works while busy - ignore it
            return
        self.entry.delete(0, tk.END)
        self._append("You", text)
        self._set_busy(True, "Thinking...")
        threading.Thread(target=self._reply, args=(text,), daemon=True).start()

    def on_mic(self) -> None:
        if self.busy or not (self.speech and self.speech.can_listen):
            return
        self.speech.stop_speaking()   # don't record our own voice
        self._set_busy(True, "Listening... speak now (it stops when you stop talking)")
        threading.Thread(target=self._listen_then_reply, daemon=True).start()

    def _on_toggle_speech(self) -> None:
        if not self.speak_var.get() and self.speech:
            self.speech.stop_speaking()

    def on_close(self) -> None:
        if self.speech:
            self.speech.stop_speaking()
        self.root.destroy()

    def run(self) -> None:
        self.entry.focus_set()
        self.root.mainloop()


if __name__ == "__main__":
    ChatGUI().run()
