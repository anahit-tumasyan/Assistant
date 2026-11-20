import os
import json
import queue
import requests
import sounddevice as sd
import vosk
import pyttsx3


class SpeechAssistant:
    def __init__(self, api_key=None, voice_id="Rachel", model_path="model"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.samplerate = 16000
        self.q = queue.Queue()
        self.model_path = model_path

        if self._is_connected():
            print("Internet connected. Using ElevenLabs.")
            self.use_internet = True
        else:
            print("No internet connection. Falling back to offline mode.")
            self.use_internet = False

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Vosk model path '{model_path}' not found. "
                    "Download a model from https://alphacephei.com/vosk/models"
                )

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)
            self.vosk_model = vosk.Model(model_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_connected(self) -> bool:
        """
        Check ElevenLabs reachability using the correct v1/voices endpoint
        with proper API-key authentication.
        """
        if not self.api_key:
            return False
        try:
            response = requests.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": self.api_key},
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _record_audio(self, duration: int = 5) -> bytes:
        """Record raw PCM audio from the microphone for `duration` seconds."""

        def callback(indata, frames, time, status):
            if status:
                print(f"[Audio stream] {status}")
            self.q.put(bytes(indata))

        audio_data = b""
        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            print("Listening…")
            try:
                num_chunks = int(self.samplerate / 8000 * duration)
                for _ in range(num_chunks):
                    audio_data += self.q.get()
            except KeyboardInterrupt:
                pass

        return audio_data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def text_to_speech(self, text: str, output_file_path: str = "output.mp3") -> None:
        """Convert text to speech using ElevenLabs (online) or pyttsx3 (offline)."""
        if self.use_internet:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
            headers = {
                "Content-Type": "application/json",
                "xi-api-key": self.api_key,
            }
            response = requests.post(url, headers=headers, data=json.dumps({"text": text}))

            if response.status_code == 200:
                with open(output_file_path, "wb") as f:
                    f.write(response.content)
                print(f"Audio saved as {output_file_path}")
                # Play the file on macOS; adjust for other platforms as needed
                os.system(f"afplay {output_file_path}")
            else:
                print(f"ElevenLabs TTS error: {response.status_code} — {response.text}")
        else:
            self.engine.say(text)
            self.engine.runAndWait()

    def speech_to_text(self, duration: int = 5) -> str:
        """Record audio and convert to text using ElevenLabs (online) or Vosk (offline)."""
        audio_data = self._record_audio(duration)

        if self.use_internet:
            stt_url = "https://api.elevenlabs.io/v1/speech-to-text"
            headers = {"xi-api-key": self.api_key}
            files = {"audio": ("audio.wav", audio_data, "audio/wav")}

            try:
                response = requests.post(stt_url, headers=headers, files=files)
                if response.status_code == 200:
                    return response.json().get("text", "")
                else:
                    print(f"ElevenLabs STT error: {response.status_code} — {response.text}")
                    return ""
            except Exception as e:
                print(f"ElevenLabs STT exception: {e}")
                return ""
        else:
            try:
                recognizer = vosk.KaldiRecognizer(self.vosk_model, self.samplerate)
                recognizer.AcceptWaveform(audio_data)
                result = json.loads(recognizer.Result())
                return result.get("text", "")
            except Exception as e:
                print(f"Vosk STT error: {e}")
                return ""


if __name__ == "__main__":
    try:
        api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        assistant = SpeechAssistant(
            api_key=api_key,
            model_path=os.path.abspath("Includes/model"),
        )

        print("Say something…")
        text = assistant.speech_to_text(duration=5)
        print(f"Recognized: {text}")

        if text:
            assistant.text_to_speech(f"You said: {text}")
    except Exception as e:
        print(f"Error: {e}")
