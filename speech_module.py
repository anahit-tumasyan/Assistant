"""
speech_module.py - listening and talking.

Speech to text
  * offline: Vosk with the small English model in Includes/model. Needs no
    internet and no account.
  * online: ElevenLabs Scribe, used when ELEVENLABS_API_KEY is set and the
    key works. If a request fails, the Vosk transcript is used instead.
  Listening stops on its own about a second after you stop talking (Vosk
  notices the pause, in both modes), with a hard time limit as a safety net.

Text to speech
  * online: ElevenLabs (voice from ELEVENLABS_VOICE_ID, "Rachel" by default)
  * offline: the macOS `say` command (pyttsx3 on other systems)

Try it:  python speech_module.py   (say something, it repeats it back)
"""

import io
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import wave

import requests
import sounddevice as sd
import vosk

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOSK_MODEL = os.path.join(PROJECT_DIR, "Includes", "model")

ELEVENLABS_API = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8mAMQ"      # the premade voice "Rachel"
DEFAULT_STT_MODEL = "scribe_v2"
DEFAULT_TTS_MODEL = "eleven_multilingual_v2"

# Words accepted when the assistant asks "yes or no?"
YES_WORDS = {"yes", "yeah", "yep", "sure", "okay", "run"}
NO_WORDS = {"no", "nope", "not", "cancel", "stop", "don't"}

CHUNK_SECONDS = 0.2


def pcm_to_wav(pcm: bytes, samplerate: int) -> bytes:
    """Wrap raw 16-bit mono samples in a WAV header."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(samplerate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def _clean_transcript(text: str) -> str:
    return " ".join(word for word in text.split() if word != "[unk]")


def interpret_yes_no(text: str) -> bool | None:
    """True for yes, False for no, None if it was neither (or both)."""
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    if words & NO_WORDS:
        return False
    if words & YES_WORDS:
        return True
    return None


class SpeechAssistant:
    def __init__(self, api_key: str | None = None, model_path: str = DEFAULT_VOSK_MODEL,
                 voice_id: str | None = None, samplerate: int = 16000):
        self.api_key = (api_key or "").strip()
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID
        self.samplerate = samplerate
        self._samplerate_checked = False
        self._audio_proc = None
        self._audio_lock = threading.Lock()
        self._pyttsx3_engine = None

        self.vosk_model = None
        if os.path.isdir(model_path):
            vosk.SetLogLevel(-1)  # Kaldi is very chatty otherwise
            self.vosk_model = vosk.Model(model_path)

        self.use_internet = bool(self.api_key) and self._elevenlabs_key_works()
        if self.use_internet:
            print("[speech] ElevenLabs key works - using ElevenLabs for listening and talking.")
        elif self.vosk_model is not None:
            print("[speech] Offline mode - Vosk for listening, the system voice for talking.")
        else:
            print(f"[speech] Voice input is off: no Vosk model in {model_path} "
                  "and no working ElevenLabs key.")

    @property
    def can_listen(self) -> bool:
        return self.use_internet or self.vosk_model is not None

    # ------------------------------------------------------------------
    # ElevenLabs
    # ------------------------------------------------------------------

    def _elevenlabs_key_works(self) -> bool:
        try:
            response = requests.get(f"{ELEVENLABS_API}/voices",
                                    headers={"xi-api-key": self.api_key}, timeout=5)
        except requests.RequestException as exc:
            print(f"[speech] Can't reach ElevenLabs ({exc.__class__.__name__}) - staying offline.")
            return False
        if response.ok:
            return True
        # A key limited to speech features can't list voices but still works.
        if response.status_code == 401 and "missing_permissions" in response.text:
            return True
        print(f"[speech] ElevenLabs rejected the API key (HTTP {response.status_code}) - staying offline.")
        return False

    def _elevenlabs_stt(self, pcm: bytes) -> str | None:
        try:
            response = requests.post(
                f"{ELEVENLABS_API}/speech-to-text",
                headers={"xi-api-key": self.api_key},
                data={"model_id": os.environ.get("ELEVENLABS_STT_MODEL", DEFAULT_STT_MODEL),
                      "tag_audio_events": "false"},
                files={"file": ("speech.wav", pcm_to_wav(pcm, self.samplerate), "audio/wav")},
                timeout=30,
            )
        except requests.RequestException as exc:
            print(f"[speech] ElevenLabs speech-to-text failed ({exc}) - using Vosk instead.")
            return None
        if not response.ok:
            print(f"[speech] ElevenLabs speech-to-text failed (HTTP {response.status_code}: "
                  f"{response.text[:150]}) - using Vosk instead.")
            return None
        return response.json().get("text", "").strip()

    def _elevenlabs_tts(self, text: str) -> bool:
        player = shutil.which("afplay") or shutil.which("mpg123") or shutil.which("ffplay")
        if not player:
            return False
        try:
            response = requests.post(
                f"{ELEVENLABS_API}/text-to-speech/{self.voice_id}",
                headers={"xi-api-key": self.api_key, "Accept": "audio/mpeg"},
                json={"text": text,
                      "model_id": os.environ.get("ELEVENLABS_TTS_MODEL", DEFAULT_TTS_MODEL)},
                timeout=30,
            )
        except requests.RequestException as exc:
            print(f"[speech] ElevenLabs text-to-speech failed ({exc}) - using the system voice.")
            return False
        if not response.ok:
            print(f"[speech] ElevenLabs text-to-speech failed (HTTP {response.status_code}: "
                  f"{response.text[:150]}) - using the system voice.")
            return False

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as file:
            file.write(response.content)
        try:
            args = [player, file.name]
            if player.endswith("ffplay"):
                args = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", file.name]
            self._play(args)
        finally:
            os.remove(file.name)
        return True

    # ------------------------------------------------------------------
    # Microphone
    # ------------------------------------------------------------------

    def _check_samplerate(self) -> None:
        """Use 16 kHz if the microphone allows it, else its own rate (Vosk
        and ElevenLabs both cope with any rate)."""
        if self._samplerate_checked:
            return
        try:
            sd.check_input_settings(samplerate=self.samplerate, channels=1, dtype="int16")
        except Exception:
            try:
                rate = int(sd.query_devices(kind="input")["default_samplerate"])
                sd.check_input_settings(samplerate=rate, channels=1, dtype="int16")
                self.samplerate = rate
            except Exception as exc:
                raise RuntimeError(f"No usable microphone found ({exc}).") from exc
        self._samplerate_checked = True

    def _microphone_chunks(self):
        """Yield ~0.2 s blocks of 16-bit mono audio until the caller stops."""
        blocks = queue.Queue()

        def callback(indata, frames, time_info, status):
            blocks.put(bytes(indata))

        with sd.RawInputStream(samplerate=self.samplerate, channels=1, dtype="int16",
                               blocksize=int(self.samplerate * CHUNK_SECONDS),
                               callback=callback):
            while True:
                try:
                    yield blocks.get(timeout=3)
                except queue.Empty:
                    raise RuntimeError("The microphone stopped sending audio.") from None

    def _record_fixed(self, seconds: float) -> bytes:
        """Plain recording, for when there is no Vosk model to spot pauses."""
        audio = bytearray()
        chunks = self._microphone_chunks()
        try:
            for chunk in chunks:
                audio += chunk
                if len(audio) >= seconds * self.samplerate * 2:
                    break
        finally:
            chunks.close()
        return bytes(audio)

    def transcribe_stream(self, chunks, max_seconds: float = 10.0,
                          silence_after_speech: float = 1.0, grammar: list[str] | None = None):
        """Feed audio chunks to Vosk until the speaker pauses.

        Stops once something was heard and then nothing new for
        `silence_after_speech` seconds, or after `max_seconds` of audio.
        Returns (text, raw_audio). Works with any iterable of 16-bit mono
        chunks, so it can be tested with a recording instead of a microphone.
        """
        if grammar:
            recognizer = vosk.KaldiRecognizer(self.vosk_model, self.samplerate, json.dumps(grammar))
        else:
            recognizer = vosk.KaldiRecognizer(self.vosk_model, self.samplerate)

        pieces, audio = [], bytearray()
        elapsed = last_change = 0.0
        heard_something = False
        last_partial = ""
        for chunk in chunks:
            audio += chunk
            elapsed += len(chunk) / (2 * self.samplerate)
            if recognizer.AcceptWaveform(chunk):
                text = _clean_transcript(json.loads(recognizer.Result()).get("text", ""))
                if text:
                    pieces.append(text)
                    heard_something, last_change = True, elapsed
                last_partial = ""
            else:
                partial = _clean_transcript(json.loads(recognizer.PartialResult()).get("partial", ""))
                if partial and partial != last_partial:
                    heard_something, last_change = True, elapsed
                last_partial = partial
            if heard_something and elapsed - last_change >= silence_after_speech:
                break
            if elapsed >= max_seconds:
                break

        # FinalResult (not Result) flushes words still waiting for a pause,
        # so speech right up to the end isn't lost.
        final = _clean_transcript(json.loads(recognizer.FinalResult()).get("text", ""))
        if final:
            pieces.append(final)
        return " ".join(pieces).strip(), bytes(audio)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speech_to_text(self, max_seconds: float = 10.0, silence_after_speech: float = 1.0) -> str:
        """Listen to the microphone and return what was said ("" if nothing)."""
        if not self.can_listen:
            raise RuntimeError("Voice input is off (no Vosk model and no ElevenLabs key).")
        self._check_samplerate()

        if self.vosk_model is None:  # online only: no pause detection available
            audio = self._record_fixed(min(max_seconds, 5.0))
            return self._elevenlabs_stt(audio) or ""

        chunks = self._microphone_chunks()
        try:
            text, audio = self.transcribe_stream(chunks, max_seconds, silence_after_speech)
        finally:
            chunks.close()

        if self.use_internet and text:  # only call the API when someone spoke
            online = self._elevenlabs_stt(audio)
            if online:
                return online
        return text

    def confirm(self, max_seconds: float = 4.0) -> bool | None:
        """Listen for a spoken yes/no. Returns True, False or None (unclear)."""
        if self.vosk_model is None:
            return interpret_yes_no(self.speech_to_text(max_seconds, 0.8))
        self._check_samplerate()
        grammar = sorted(YES_WORDS | NO_WORDS) + ["[unk]"]
        chunks = self._microphone_chunks()
        try:
            text, _ = self.transcribe_stream(chunks, max_seconds, 0.6, grammar=grammar)
        finally:
            chunks.close()
        return interpret_yes_no(text)

    def text_to_speech(self, text: str) -> None:
        """Say the text out loud (blocks until finished)."""
        text = (text or "").strip()
        if not text:
            return
        if self.use_internet and self._elevenlabs_tts(text):
            return
        self._speak_offline(text)

    def stop_speaking(self) -> None:
        """Interrupt whatever is being said right now."""
        with self._audio_lock:
            proc = self._audio_proc
        if proc and proc.poll() is None:
            proc.terminate()

    # ------------------------------------------------------------------

    def _play(self, args: list[str]) -> None:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with self._audio_lock:
            self._audio_proc = proc
        proc.wait()
        with self._audio_lock:
            if self._audio_proc is proc:
                self._audio_proc = None

    def _speak_offline(self, text: str) -> None:
        say = shutil.which("say")
        if say:  # macOS: reliable from any thread ("--" so text can't look like an option)
            self._play([say, "--", text])
            return
        try:
            import pyttsx3
        except ImportError:
            print(f"[speech] (no text-to-speech available) {text}")
            return
        if self._pyttsx3_engine is None:
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", 150)
        self._pyttsx3_engine.say(text)
        self._pyttsx3_engine.runAndWait()


if __name__ == "__main__":
    assistant = SpeechAssistant(api_key=os.environ.get("ELEVENLABS_API_KEY", ""))
    print("Say something...")
    heard = assistant.speech_to_text()
    print(f"Recognized: {heard or '(nothing)'}")
    if heard:
        assistant.text_to_speech(f"You said: {heard}")
