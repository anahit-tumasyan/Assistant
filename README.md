# My Assistant

A small hands-free assistant for the Mac. Hold your hand up to the webcam and
it becomes your mouse: move, click, drag, scroll and zoom with simple hand
poses. Talk to it ("open my downloads folder", "how much disk space is
left?") and a language model running on your own machine turns what you said
into a terminal command and runs it.

There's also a second, separate assistant in here: an emergency-help chat
that keeps someone calm, makes sure they've called for help, and walks them
through basic first aid while they wait.

Everything works offline. Speech recognition is [Vosk](https://alphacephei.com/vosk/),
the language model is Google's [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-it),
and hand tracking is [MediaPipe](https://developers.google.com/mediapipe).
If you have an [ElevenLabs](https://elevenlabs.io) API key, it'll use that
for better speech recognition and a nicer voice, but you don't need one.

## What you can run

| Command | What it does |
| --- | --- |
| `python run_all.py` | The whole thing: gesture mouse control and voice commands at the same time |
| `python main.py` | Voice (or typed) requests become terminal commands; you confirm each one |
| `python hand_landmark.py` | Just the gesture mouse control |
| `python emergency_gui.py` | The emergency assistant in a chat window, with a Speak button |
| `python emergency_assistant.py` | The emergency assistant in the terminal |
| `python Predict_command.py "open safari"` | Shows which command a sentence would become, without running it |
| `python Run_command.py` | Optional WebSocket service that runs commands and streams their output |

## Setting it up

You'll need a Mac (Apple Silicon is fastest, Intel works on the CPU), Python
3.12, a webcam, a microphone and about 3 GB of disk space.

**1. Python packages**

```bash
python3.12 -m venv env312
source env312/bin/activate
pip install -r requirements.txt
```

Stick with Python 3.12. Newer versions like 3.14 don't have wheels for all of
these packages yet. If you already have the `env312` folder, just activate it.

**2. Models**

- **Gemma 3 1B** goes in `Includes/gemma-3-1b-it` (about 2 GB). Google asks
  you to accept its license first: open the
  [model page](https://huggingface.co/google/gemma-3-1b-it), accept, then
  ```bash
  hf auth login
  hf download google/gemma-3-1b-it --local-dir Includes/gemma-3-1b-it
  ```
- **Vosk speech model** goes in `Includes/model`. Download
  `vosk-model-small-en-us-0.15` from the [Vosk models page](https://alphacephei.com/vosk/models),
  unzip it, and rename the folder to `model`.
- **Hand tracking model** (`hand_landmarker.task`) is already here. If you
  delete it, it downloads itself the next time you run the gestures.

**3. macOS permissions**

This is the step everyone trips over. In *System Settings → Privacy &
Security*, allow the app you run Python from (Terminal, iTerm or VS Code)
to use:

- **Camera**, for hand tracking
- **Microphone**, for voice commands
- **Accessibility**, so it can move the mouse. Without this, your hand is
  tracked just fine but the cursor doesn't move, usually without any error.

Restart the terminal app after changing these.

## Hand gestures

Hold your hand about half a metre from the camera, palm facing it, with
reasonable light.

| Pose | Action |
| --- | --- |
| All five fingers open | Move the cursor |
| Four fingers up, thumb tucked in | Freeze the cursor, so you can reposition your hand (like lifting a mouse off the desk) |
| Fist | Hold the left button and drag. Open your hand to drop. |
| Only the thumb up | Right click |
| Only the index finger up | Left click |
| Index + middle (V sign) | Long press. Hold it and move to drag. |
| Index + middle + ring | Scroll. Move your hand up or down. |
| Thumb + index (an L) | Zoom. Spread them to zoom in, pinch to zoom out. |

The cursor moves relative to where it already is, like a trackpad, and slows
down for precise work when you move slowly. It glides rather than jumping
with every camera frame. A resting hand keeps it perfectly still, and even
very slow, careful movements come through smoothly. While your fingers are
forming a click it stays where you pointed, so it doesn't slide off the
button. Losing track of your hand for a moment (motion blur, the edge of
the picture) doesn't make it jump or drop what you're dragging.

A pose has to be held for a split second (three camera frames) before it
counts. That's deliberate: when you go from an open hand to a single finger,
your hand passes through a few other poses on the way, and those shouldn't
click anything. Press `q` or `Esc` in the camera window to quit.

## Voice commands

Just talk normally. It stops listening by itself about a second after you
finish. These are built in as examples, so when they're heard correctly they
always give the right command:

- "open the downloads folder", "open safari", "open youtube"
- "go to the desktop", "where am i", "list the files here"
- "how much disk space is left", "how big is my downloads folder"
- "show cpu and memory usage", "is python running", "ping google"
- "what is the git status", "show the last five commits"
- "say hello world", "what is on my clipboard"

Anything else goes through Gemma. It's a small model, so expect it to be
right most of the time but not always. On a test of 37 everyday requests it
had never seen, about four out of five came out right. It's also quick: a
few seconds to load, then around half a second per answer on an M-series Mac.

That's why you always get to check what it came up with. `main.py` shows the
command and runs it only if you type `y`. In `run_all.py` there's no
keyboard, so harmless commands run straight away. Anything that could change
your system or run code (python, pip, brew, curl, wget, osascript, sed, awk,
and git apart from status/log/diff/show) waits until you answer "yes" out
loud.

## What it's allowed to run

The model's output is never trusted. `Run_command.py` checks every command
before it runs:

- Every program on the line has to be in `WHITELISTED_COMMANDS`. That
  includes both sides of a pipe, `&&` or `;`, so `ls; rm -rf ~` is refused.
- `$(...)` and backticks are refused, and so is writing to files with `>`.
  `> /dev/null` is the one exception.
- A few options that would let an allowed program delete or overwrite
  things are blocked: `find -exec` / `-delete`, `sed -i`, `sort -o`.
- Commands get 30 seconds (5 minutes for installs, downloads, git and python
  scripts) and can't wait for keyboard input. `top` and `ping` automatically get a sample count
  so they don't run forever.
- `cd` is handled by the assistant itself, so "go to my desktop" sticks for
  the commands that follow.

Deleting files and shutting down or restarting the computer are refused on
purpose. To allow another program, add it to `WHITELISTED_COMMANDS` in
`Run_command.py`, and add an example or two to `EXAMPLES` in
`Predict_command.py` so the model learns to use it.

This is a guard rail, not a sandbox. Allowed programs like `python` or
`curl` can still do a lot, which is exactly why they ask first.

## The emergency assistant

`emergency_gui.py` opens a chat window. You can type, or press **Speak** and
talk, and replies are read aloud unless you untick that box. It remembers
the conversation, starts by checking that emergency services have been
called, asks about injuries and dangers, and gives simple instructions:
pressure on a bleeding wound, breathing slowly, saving phone battery,
knocking in sets of three so rescuers can find you.

Please treat it as what it is: a one-billion-parameter model running on a
laptop. It can keep someone company and remind them of the basics, but it
isn't a doctor and it can be wrong. Always call 112, 911 or your local
emergency number first.

## Settings

| Environment variable | What it does |
| --- | --- |
| `ELEVENLABS_API_KEY` | Use ElevenLabs for speech recognition and the voice. Optional. If a request fails, it falls back to the offline engines. |
| `ELEVENLABS_VOICE_ID` | Which ElevenLabs voice reads replies. The default is "Rachel". |
| `ELEVENLABS_STT_MODEL`, `ELEVENLABS_TTS_MODEL` | Override the ElevenLabs models (defaults: `scribe_v2`, `eleven_multilingual_v2`) |
| `ASSISTANT_DEVICE` | Force where Gemma runs: `mps`, `cuda` or `cpu` |

Gesture tuning lives at the top of `hand_landmark.py`: cursor sensitivity,
smoothing, click cooldown, scroll speed, and which camera to use
(`CAMERA_INDEX`). For the feel of the cursor, two settings matter most:

- `GLIDE_TIME`: lower it (say 0.02) if the cursor feels floaty, raise it
  (0.05) for an even silkier glide.
- `STILL_RADIUS`: raise it a little (0.004) if the cursor still twitches
  while your hand rests, which can happen in dim light. Lower it if it
  feels slow to start moving.

## Tests

```bash
python -m unittest discover -s tests
```

These cover the command safety checks, the gesture logic (with a fake
mouse), cleaning up the model's answers, and speech recognition (using audio
made with macOS's `say`). They never touch your camera, microphone or mouse,
and finish in about ten seconds. To also test the real Gemma model:

```bash
ASSISTANT_SLOW_TESTS=1 python -m unittest discover -s tests
```

## When something doesn't work

- **The cursor doesn't move.** The terminal app needs Accessibility
  permission (see setup). Restart the terminal afterwards.
- **"Cannot open the camera".** Check the Camera permission and make sure no
  other app is using the camera. If you have several cameras (an iPhone
  through Continuity Camera, say), try `CAMERA_INDEX = 1` in
  `hand_landmark.py`.
- **It never hears anything.** Check the Microphone permission, and which
  input is selected under System Settings → Sound.
- **"Gemma model folder not found".** The model download step was skipped.
- **The first answer is slow.** The first run warms up the GPU; after that
  it's quick.
- **`pip install` fails.** Make sure the virtual environment uses Python 3.12.

## What's in the folder

```
main.py                voice/typed requests -> confirmed terminal commands
run_all.py             gestures + voice together (hands-free)
hand_landmark.py       gesture recognition and mouse control
speech_module.py       microphone, Vosk / ElevenLabs speech-to-text, text-to-speech
Predict_command.py     Gemma prompt, examples and clean-up of its answers
Run_command.py         command safety checks, running commands, WebSocket service
Gemma3.py              loads Gemma once and runs it (shared by everything)
emergency_assistant.py emergency chat logic + terminal version
emergency_gui.py       emergency chat window
tests/                 unit tests
Includes/              the Gemma and Vosk models
hand_landmarker.task   MediaPipe hand model
```

## Credits

Gemma 3 is made by Google and released under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms), which you accept
when you download it. Vosk is by Alpha Cephei (Apache 2.0) and MediaPipe by
Google (Apache 2.0). Mouse control uses PyAutoGUI, and the camera and
drawing use OpenCV.
