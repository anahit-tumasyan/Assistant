"""
Predict_command.py - turns a spoken request into one terminal command.

The local Gemma 3 (1B) model does the translation. A model this small copies
examples much better than it follows instructions, so for every request we
pick the dozen most similar examples from EXAMPLES and show those. Every
example uses a program that Run_command.py allows, so the model learns to
answer with commands that can actually run. If you allow a new program
there, add an example or two for it here.

A few mistakes the model makes all the time are then fixed by hand - mostly
`open`: "open -a Downloads" becomes "open ~/Downloads", "open -a Chrome"
becomes "open -a 'Google Chrome'", "open github.com" becomes a web link, and
"go to my documents" ends up as `cd`, not `ls`. Requests to delete things or
shut the computer down get a polite "no" without asking the model at all.

Nothing is executed here. Try it without a microphone:
    python Predict_command.py "open my downloads folder"
"""

import functools
import os
import re
import shlex
import sys

from Gemma3 import DEFAULT_MODEL_PATH, Gemma3
from Run_command import current_directory

INSTRUCTIONS = """You translate requests into a single macOS terminal command.
Reply with the command only. No explanation, no markdown, no quotes around it.
To open a file, folder, app or website, use the open command.
The home folder is ~, so the desktop is ~/Desktop, downloads is ~/Downloads and documents is ~/Documents."""

# (request, command) pairs. Only allowed programs, macOS flavour.
EXAMPLES = [
    # open apps
    ("open safari", "open -a Safari"),
    ("open the calculator", "open -a Calculator"),
    ("launch notes", "open -a Notes"),
    ("start the terminal app", "open -a Terminal"),
    ("open microsoft word", 'open -a "Microsoft Word"'),
    ("open system settings", 'open -a "System Settings"'),
    # open folders and files
    ("open the downloads folder", "open ~/Downloads"),
    ("open my documents", "open ~/Documents"),
    ("show the desktop folder in finder", "open ~/Desktop"),
    ("open folder MIT on the desktop", "open ~/Desktop/MIT"),
    ("open the photos folder inside documents", "open ~/Documents/photos"),
    ("open this folder", "open ."),
    ("open the file report.pdf", "open report.pdf"),
    ("open notes.txt from the desktop", "open ~/Desktop/notes.txt"),
    # open websites
    ("open youtube", "open https://www.youtube.com"),
    ("go to google dot com", "open https://www.google.com"),
    ("open the website example.org", "open https://example.org"),
    # move between folders
    ("go to the desktop", "cd ~/Desktop"),
    ("change directory to downloads", "cd ~/Downloads"),
    ("go into the folder src", "cd src"),
    ("go back to my home folder", "cd ~"),
    ("go to the parent folder", "cd .."),
    ("go to the pictures folder", "cd ~/Pictures"),
    ("where am i", "pwd"),
    ("which folder am i in", "pwd"),
    # list
    ("list the files here", "ls"),
    ("show all files including hidden ones", "ls -la"),
    ("what is in my documents folder", "ls ~/Documents"),
    ("list the downloads folder with sizes", "ls -lh ~/Downloads"),
    ("list files sorted by date", "ls -lt"),
    ("list the biggest files in documents first", "ls -lS ~/Documents"),
    # create
    ("make a new folder called projects", "mkdir projects"),
    ("create a directory named backup on the desktop", "mkdir ~/Desktop/backup"),
    ("create an empty file called notes.txt", "touch notes.txt"),
    # read
    ("show me what is inside config.json", "cat config.json"),
    ("print the file hello.py", "cat hello.py"),
    ("show the first 10 lines of log.txt", "head -n 10 log.txt"),
    ("show the last 20 lines of server.log", "tail -n 20 server.log"),
    ("how many lines are in data.csv", "wc -l data.csv"),
    ("how many words are in essay.txt", "wc -w essay.txt"),
    # find and search
    ("find all python files", 'find . -name "*.py"'),
    ("find every jpg in my pictures folder", 'find ~/Pictures -name "*.jpg"'),
    ("find files bigger than 100 megabytes in downloads", "find ~/Downloads -size +100M"),
    ("find the file called budget.xlsx", 'find ~ -name "budget.xlsx"'),
    ("search for the word hello in the text files here", 'grep -rn "hello" --include="*.txt" .'),
    ("search for TODO in this folder", 'grep -rn "TODO" .'),
    ("which lines of app.py mention error", 'grep -n "error" app.py'),
    ("sort the lines of names.txt", "sort names.txt"),
    ("show the unique lines in list.txt", "sort list.txt | uniq"),
    # information
    ("what is the date today", "date"),
    ("what time is it now", "date +%H:%M"),
    ("who am i logged in as", "whoami"),
    ("show my account name", "whoami"),
    ("how long has the computer been on", "uptime"),
    ("show the system information", "uname -a"),
    ("how much disk space is left", "df -h"),
    ("how big is my downloads folder", "du -sh ~/Downloads"),
    ("how big is the music folder", "du -sh ~/Music"),
    ("show the running processes", "ps aux"),
    ("is python running", "ps aux | grep -i python"),
    ("is zoom running", "ps aux | grep -i zoom"),
    ("show cpu and memory usage", "top -l 1 -n 10"),
    ("how much memory is being used", "top -l 1 -n 0"),
    # network
    ("ping google", "ping -c 4 google.com"),
    ("am i connected to the internet", "ping -c 2 apple.com"),
    ("look up the ip address of github.com", "nslookup github.com"),
    ("download https://example.com/file.zip", "curl -O https://example.com/file.zip"),
    # git
    ("what is the git status", "git status"),
    ("show the last five commits", "git log --oneline -n 5"),
    ("show what changed in git", "git diff"),
    # speech, clipboard, power
    ("say hello world", "say hello world"),
    ("say good night everyone", "say good night everyone"),
    ("read my clipboard out loud", "pbpaste | say"),
    ("what is on my clipboard", "pbpaste"),
    ("copy the word hello to the clipboard", "echo hello | pbcopy"),
    ("keep the mac awake for an hour", "caffeinate -d -t 3600"),
    # developer tools
    ("run the python script hello.py", "python3 hello.py"),
    ("check the python version", "python3 --version"),
    ("what version of git is installed", "git --version"),
    ("install the python package requests", "pip3 install requests"),
    ("list the installed python packages", "pip3 list"),
    ("install wget with homebrew", "brew install wget"),
    ("update homebrew", "brew update"),
]

EXAMPLES_PER_REQUEST = 12

# Requests we answer with a polite "no" without asking the model (it would
# otherwise reach for rm or invent something odd).
REFUSALS = [
    (re.compile(r"\b(delete|remove|erase|wipe|trash|destroy)\b", re.I),
     "Sorry, I'm not allowed to delete anything."),
    (re.compile(r"\b(shut ?down|turn off|power off|restart|reboot|log ?out)\b", re.I),
     "Sorry, I can't shut down, restart or log out the computer."),
]

_STOP_WORDS = {
    "the", "a", "an", "my", "me", "to", "in", "on", "of", "is", "are", "please",
    "can", "could", "you", "i", "it", "this", "that", "for", "with", "and", "from",
    "here", "there", "some", "do", "does", "be", "now", "what", "whats", "all",
}


def _words(text: str) -> set[str]:
    words = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in _STOP_WORDS:
            continue
        if len(word) > 3 and word.endswith("s"):  # files -> file, folders -> folder
            word = word[:-1]
        words.add(word)
    return words


def _normalise(text: str) -> str:
    words = re.findall(r"[a-z0-9.'-]+", text.lower())
    return " ".join(word for word in words if word != "please").strip(".")


_KNOWN_REQUESTS = {_normalise(example): command for example, command in EXAMPLES}


def _first_word(text: str) -> str:
    found = re.findall(r"[a-z]+", text.lower())
    return found[0] if found else ""


def select_examples(request: str, k: int = EXAMPLES_PER_REQUEST) -> list[tuple[str, str]]:
    """The k examples that share the most words with the request, most
    similar last (closest to the real question)."""
    wanted, verb = _words(request), _first_word(request)
    scored = []
    for index, (example, command) in enumerate(EXAMPLES):
        score = len(wanted & _words(example)) + (0.5 if _first_word(example) == verb else 0)
        scored.append((score, -index, example, command))
    best = sorted(scored, reverse=True)[:k]
    return [(example, command) for _, _, example, command in reversed(best)]


def build_messages(request: str) -> list[dict]:
    examples = "\n".join(f"User: {ex}\n{cmd}" for ex, cmd in select_examples(request))
    system = f"{INSTRUCTIONS}\n\nExamples:\n{examples}\n\nNow translate the next request into one command."
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"User: {request}"}]


# ---------------------------------------------------------------------------
# Cleaning up the model's answer
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```[a-zA-Z]*[ \t]*\n?(.*?)```", re.S)
_PREFIX = re.compile(r"^(?:\$|>|command:|terminal:|shell:|bash:|zsh:)\s*", re.I)


def clean_command(text: str) -> str:
    """Pull the bare command out of whatever the model wrote.

    Small models sometimes wrap the answer in ```code fences```, backticks
    or quotes, or add a "$ " prompt or a line of explanation.
    """
    text = (text or "").strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("```", "#")):  # fences and comments
            continue
        if line.lower().startswith("user:"):  # the model started inventing examples
            break
        line = _PREFIX.sub("", line).strip().strip("`").strip()
        if len(line) > 1 and line[0] == line[-1] and line[0] in "\"'":
            line = line[1:-1].strip()
        return line
    return ""


APP_FOLDERS = ["/Applications", "/Applications/Utilities", "/System/Applications",
               "/System/Applications/Utilities", "~/Applications"]
HOME_FOLDERS = {"home": "~", "desktop": "~/Desktop", "documents": "~/Documents",
                "downloads": "~/Downloads", "pictures": "~/Pictures", "movies": "~/Movies",
                "music": "~/Music", "applications": "/Applications"}
SEARCH_FOLDERS = ["~", "~/Desktop", "~/Documents", "~/Downloads"]
_WEB_ENDINGS = {"com", "org", "net", "io", "edu", "gov", "dev", "ai", "co", "me", "am",
                "ru", "uk", "de", "fr", "es", "it", "eu", "info", "tv", "us", "ly"}


@functools.lru_cache(maxsize=1)
def _installed_apps() -> dict[str, str]:
    apps = {}
    for folder in APP_FOLDERS:
        try:
            names = os.listdir(os.path.expanduser(folder))
        except OSError:
            continue
        for name in names:
            if name.endswith(".app"):
                apps.setdefault(name[:-4].lower(), name[:-4])
    return apps


def find_app(name: str) -> str | None:
    """'chrome' -> 'Google Chrome', 'settings' -> 'System Settings'."""
    key = name.lower().removesuffix(".app").strip()
    if not key:
        return None
    apps = _installed_apps()
    if key in apps:
        return apps[key]
    whole_word = [real for low, real in apps.items() if key in low.split()]
    partial = [real for low, real in apps.items() if len(key) >= 3 and key in low]
    for matches in (whole_word, partial):
        if matches:
            return min(matches, key=len)
    return None


def _looks_like_website(target: str) -> bool:
    if target.startswith(("http://", "https://")):
        return True
    match = re.fullmatch(r"(?:www\.)?[\w-]+(?:\.[\w-]+)*\.([a-z]{2,4})(?:/\S*)?", target.lower())
    return bool(match) and match.group(1) in _WEB_ENDINGS


def _find_folder(name: str) -> str | None:
    if name.lower() in HOME_FOLDERS:
        return HOME_FOLDERS[name.lower()]
    for base in SEARCH_FOLDERS:
        candidate = f"{base}/{name}"
        if os.path.isdir(os.path.expanduser(candidate)):
            return candidate
    return None


def _quote(path: str) -> str:
    """Quote a path for the shell but keep a leading ~ working."""
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return path if path == "~" else shlex.quote(path)


_GO_TO = re.compile(r"^\s*(?:cd\b|(?:go|move|switch|navigate|change)\b.*\b(?:to|into|directory)\b)", re.I)


def fix_command(command: str, request: str = "") -> str:
    """Repair the model's most common slips: `open` misuse, and listing or
    opening a folder when the user asked to *go to* it."""
    try:
        words = shlex.split(command)
    except ValueError:
        return command

    if (request and _GO_TO.search(request) and len(words) == 2
            and words[0] in ("ls", "open") and not words[1].startswith("-")
            and not _looks_like_website(words[1])
            and os.path.isdir(os.path.join(current_directory(), os.path.expanduser(words[1])))):
        return f"cd {_quote(words[1])}"

    if len(words) < 2 or words[0] != "open":
        return command

    if words[1] == "-a" and len(words) >= 3:
        target = " ".join(words[2:])
        if target.startswith(("~", "/", ".")):                # a path, not an app
            return f"open {_quote(target)}"
        if _looks_like_website(target):
            return "open " + (target if "://" in target else f"https://{target}")
        app = find_app(target)
        if app:
            return f"open -a {shlex.quote(app)}"
        folder = _find_folder(target)
        return f"open {_quote(folder)}" if folder else command

    if len(words) == 2 and not words[1].startswith("-"):
        target = words[1]
        expanded = os.path.join(current_directory(), os.path.expanduser(target))
        if os.path.exists(expanded) or "/" in target:
            return command
        if _looks_like_website(target):
            return f"open https://{target}"
        folder = _find_folder(target)
        if folder:
            return f"open {_quote(folder)}"
        app = find_app(target)
        if app:
            return f"open -a {shlex.quote(app)}"
    return command


class TerminalAssistant:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str | None = None,
                 llm: Gemma3 | None = None):
        # Pass `llm` to share an already loaded model.
        self.llm = llm or Gemma3(model_path, device)

    def predict_command(self, input_text: str) -> str:
        """Return a single terminal command for the request ("" if none)."""
        request = (input_text or "").strip()
        if not request:
            return ""
        for pattern, message in REFUSALS:
            if pattern.search(request):
                return f'echo "{message}"'
        known = _KNOWN_REQUESTS.get(_normalise(request))
        if known:  # word for word one of the examples: no need to guess
            return known
        raw = self.llm.generate(build_messages(request), max_new_tokens=64, do_sample=False)
        return fix_command(clean_command(raw), request)


if __name__ == "__main__":
    from Run_command import check_command

    request = " ".join(sys.argv[1:]) or "open the desktop folder"
    print("Loading Gemma 3...")
    assistant = TerminalAssistant()
    command = assistant.predict_command(request)
    ok, reason = check_command(command)
    print(f"Request: {request}")
    print(f"Command: {command or '(none)'}")
    print("Allowed - main.py would offer to run it." if ok else f"Would not run: {reason}")
