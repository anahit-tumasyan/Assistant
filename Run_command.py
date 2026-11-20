"""
Run_command.py - decides whether a shell command may run, then runs it.

The commands come from a language model listening to a microphone, so
nothing it writes is trusted:

  * every program on the line (including each side of a pipe, && or ;)
    must be in WHITELISTED_COMMANDS,
  * command substitution - $(...) and backticks - is refused, and so is
    writing to files with > (only > /dev/null is fine),
  * a few dangerous options are refused: find -exec / -delete, sed -i,
    sort -o, rg --pre,
  * every command gets a time limit and no keyboard input, so nothing can
    hang the assistant (top and ping are given a sample count for you).

This is a guard rail, not a sandbox: allowed programs such as python or
curl can still do a lot, which is why needs_confirmation() flags them and
the hands-free mode asks before running them.

`cd` is handled here rather than by the shell, so "go to my desktop"
changes the folder for the commands that follow.

Run this file directly to start the optional WebSocket service that runs
commands and streams their output (ws://localhost:4444).
"""

import asyncio
import glob
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

# Programs the assistant may run. Add a name here to allow more.
WHITELISTED_COMMANDS = {
    # Files and folders
    "ls", "pwd", "cd", "find", "tree", "mkdir", "touch", "open",
    # Reading files
    "cat", "head", "tail", "less", "more", "wc",
    # Searching and text
    "grep", "rg", "awk", "sed", "echo", "printf", "sort", "uniq", "cut", "tr",
    # System info
    "date", "whoami", "uname", "uptime", "df", "du", "top", "ps",
    # Network
    "ping", "curl", "wget", "nslookup", "dig",
    # Developer tools
    "python", "python3", "pip", "pip3", "git", "brew",
    # macOS helpers
    "say", "pbcopy", "pbpaste", "caffeinate", "osascript",
}

# Allowed, but able to change your system or run arbitrary code, so the
# hands-free mode asks "yes or no?" first. (main.py always asks.)
ASK_FIRST = {
    "python", "python3", "pip", "pip3", "brew", "osascript",
    "curl", "wget", "awk", "sed",
}
# git is only asked about when it isn't one of these read-only subcommands.
GIT_READ_ONLY = {"status", "log", "diff", "show"}

# Options that would let an allowed program delete, overwrite or execute.
FIND_BLOCKED = {
    "-exec", "-execdir", "-ok", "-okdir", "-delete",
    "-fprint", "-fprint0", "-fprintf", "-fls",
}

# Programs that keep running until stopped: started in the background.
BACKGROUND_COMMANDS = {"caffeinate"}

# Time limits in seconds.
DEFAULT_TIMEOUT = 30
LONG_TIMEOUT = 300
LONG_RUNNING = {"brew", "pip", "pip3", "python", "python3", "git", "curl", "wget"}

_OPERATORS = {"|", "&&", "||", ";"}
_PUNCTUATION = set("();<>|&")

# Folder the commands run in. Starts where the program was launched;
# `cd` changes it.
_cwd = os.getcwd()
_background = []  # keep handles so finished background jobs get reaped


def current_directory() -> str:
    return _cwd


# ---------------------------------------------------------------------------
# Checking a command
# ---------------------------------------------------------------------------

def _split_segments(command: str):
    """Split a command line into simple commands.

    Returns (segments, None) where each segment is a list of words, or
    (None, reason) if the line uses shell features we don't allow.
    """
    if not command or not command.strip():
        return None, "the command is empty"
    if any(ord(ch) < 32 and ch != "\t" for ch in command):
        return None, "the command must be a single line"
    if "`" in command or "$(" in command:
        return None, "command substitution ($(...) or backticks) is not allowed"

    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""  # the shell only treats # as a comment at a word start
    try:
        tokens = list(lexer)
    except ValueError as exc:
        return None, f"could not read the command ({exc})"

    segments, current = [], []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if tok in _OPERATORS:
            if not current:
                return None, f"unexpected '{tok}'"
            segments.append(current)
            current = []
        elif tok in (">", ">>", "&>", "&>>"):
            if nxt != "/dev/null":
                return None, "writing to files with > is not allowed (only > /dev/null)"
            i += 1
        elif tok == ">&":
            if nxt not in ("1", "2"):
                return None, f"'>&{nxt or ''}' is not allowed"
            i += 1
        elif tok == "<":
            if nxt is None:
                return None, "a file name is missing after <"
            i += 1
        elif all(ch in _PUNCTUATION for ch in tok):
            return None, f"'{tok}' is not allowed (no background jobs, subshells or other redirections)"
        else:
            current.append(tok)
        i += 1

    if not current:
        return None, "the command ends with an operator"
    segments.append(current)
    return segments, None


def _program_name(word: str) -> str | None:
    """'ls' -> 'ls'. A path is only accepted if it is the real program
    (so /tmp/evil/ls can't pose as ls)."""
    if "/" not in word:
        return word
    name = os.path.basename(word)
    real = shutil.which(name)
    if real and os.path.realpath(real) == os.path.realpath(word):
        return name
    return None


def _short_flag_has(arg: str, letter: str) -> bool:
    return arg.startswith("-") and not arg.startswith("--") and letter in arg[1:]


def _blocked_option(program: str, args: list[str]) -> str | None:
    if program == "find":
        bad = next((a for a in args if a in FIND_BLOCKED), None)
        if bad:
            return f"find {bad} is not allowed (it can delete files or run other programs)"
    elif program == "sed":
        if any(a.startswith("--in-place") or _short_flag_has(a, "i") for a in args):
            return "sed -i (editing files in place) is not allowed"
    elif program == "sort":
        if any(a.startswith("--output") or a.startswith("--compress-program")
               or _short_flag_has(a, "o") for a in args):
            return "sort -o (writing to a file) is not allowed"
    elif program == "rg":
        if any(a.startswith("--pre") for a in args):
            return "rg --pre (running another program) is not allowed"
    return None


def check_command(command: str) -> tuple[bool, str]:
    """Return (True, "") if the command may run, else (False, reason)."""
    segments, error = _split_segments(command)
    if error:
        return False, error
    for words in segments:
        program = _program_name(words[0])
        if program is None:
            return False, f"'{words[0]}' is not a trusted program path"
        if program not in WHITELISTED_COMMANDS:
            return False, (f"'{program}' is not on the allowed list "
                           "(add it to WHITELISTED_COMMANDS in Run_command.py if you trust it)")
        reason = _blocked_option(program, words[1:])
        if reason:
            return False, reason
    return True, ""


def is_whitelisted(command: str) -> bool:
    return check_command(command)[0]


def needs_confirmation(command: str) -> bool:
    """True if the command could change the system or run arbitrary code."""
    segments, error = _split_segments(command)
    if error:
        return True
    for words in segments:
        program = os.path.basename(words[0])
        if program in ASK_FIRST:
            return True
        if program == "git" and len(words) > 1 and words[1] not in GIT_READ_ONLY:
            return True
    return False


# ---------------------------------------------------------------------------
# Running a command
# ---------------------------------------------------------------------------

def _programs(segments) -> list[str]:
    return [os.path.basename(words[0]) for words in segments]


def _make_non_interactive(command: str, segments) -> str:
    """`top` and `ping` run forever unless told otherwise; give them a count."""
    def lacks(program, flag):
        return any(os.path.basename(w[0]) == program and not any(a.startswith(flag) for a in w[1:])
                   for w in segments)

    if sys.platform == "darwin" and lacks("top", "-l"):
        command = re.sub(r"(^|[|;&]\s*)top(?=\s|$)", r"\1top -l 1", command)
    if lacks("ping", "-c"):
        command = re.sub(r"(^|[|;&]\s*)ping(?=\s|$)", r"\1ping -c 4", command)
    return command


def _timeout_for(segments) -> int:
    return LONG_TIMEOUT if set(_programs(segments)) & LONG_RUNNING else DEFAULT_TIMEOUT


def _change_directory(args: list[str]) -> str:
    global _cwd
    target = args[0] if args else "~"
    path = os.path.expandvars(os.path.expanduser(target))
    if not os.path.isabs(path):
        path = os.path.join(_cwd, path)
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        matches = [m for m in glob.glob(path) if os.path.isdir(m)]
        if len(matches) != 1:
            return f"Error: no such folder: {target}"
        path = matches[0]
    _cwd = path
    return f"Now in {path}"


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        pass


def _as_text(data) -> str:
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data or ""


def run_command(command: str, timeout: float | None = None) -> str:
    """Run an allowed command and return what it printed (or why it didn't run)."""
    command = command.strip()
    ok, reason = check_command(command)
    if not ok:
        return f"Blocked: {reason}"

    segments, _ = _split_segments(command)
    first = os.path.basename(segments[0][0])
    if len(segments) == 1 and first == "cd":
        return _change_directory(segments[0][1:])
    if len(segments) == 1 and first in BACKGROUND_COMMANDS:
        proc = subprocess.Popen(command, shell=True, cwd=_cwd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        _background[:] = [p for p in _background if p.poll() is None] + [proc]
        return f"Started in the background (process {proc.pid})."

    shell_command = _make_non_interactive(command, segments)
    limit = timeout or _timeout_for(segments)
    try:
        # Own process group, so a timeout can stop the whole pipeline.
        proc = subprocess.Popen(shell_command, shell=True, cwd=_cwd, text=True,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
    except OSError as exc:
        return f"Could not run the command: {exc}"

    try:
        out, err = proc.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid)
        out, _ = proc.communicate()
        partial = _as_text(out).rstrip()
        return (partial + "\n" if partial else "") + f"(stopped after {limit:g} seconds)"
    except BaseException:  # e.g. Ctrl-C while waiting: don't leave it running
        _kill_group(proc.pid)
        proc.wait()
        raise

    if proc.returncode == 0:
        return out or err or "(done - the command printed nothing)"
    details = err.strip() or out.strip() or "no details"
    return f"Error (exit code {proc.returncode}): {details}"


# ---------------------------------------------------------------------------
# Optional WebSocket service
#   send:    {"type": "execute", "command": "ls -la"}
#   receive: {"type": "output" | "error" | "info", "message": "..."}
# ---------------------------------------------------------------------------

# Only local, non-browser clients and pages served from this computer may
# connect, so a random website can't drive the service.
LOCAL_ORIGINS = [None, re.compile(r"https?://(localhost|127\.0\.0\.1)(:\d+)?\Z")]


async def _send(websocket, kind: str, message: str) -> None:
    await websocket.send(json.dumps({"type": kind, "message": message}))


async def stream_command(websocket, command: str) -> int:
    """Run an allowed command and stream its output line by line."""
    ok, reason = check_command(command)
    if not ok:
        await _send(websocket, "error", f"Blocked: {reason}")
        return 1

    segments, _ = _split_segments(command)
    if len(segments) == 1 and _programs(segments)[0] in {"cd"} | BACKGROUND_COMMANDS:
        await _send(websocket, "output", run_command(command))
        return 0

    limit = _timeout_for(segments)
    proc = await asyncio.create_subprocess_shell(
        _make_non_interactive(command, segments), cwd=_cwd,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True, limit=1024 * 1024,
    )

    async def pump(stream, kind):
        async for raw in stream:
            await _send(websocket, kind, raw.decode(errors="replace").rstrip("\n"))

    try:
        async with asyncio.timeout(limit):
            await asyncio.gather(pump(proc.stdout, "output"), pump(proc.stderr, "error"))
            code = await proc.wait()
    except TimeoutError:
        await _send(websocket, "error", f"Stopped after {limit:g} seconds.")
        code = -1
    finally:
        if proc.returncode is None:  # timed out or the client went away
            _kill_group(proc.pid)
            await proc.wait()

    await _send(websocket, "info", f"Command exited with return code: {code}")
    return code


async def socket_handler(websocket) -> None:
    import websockets

    try:
        await _send(websocket, "info", "Connected to command execution service")
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await _send(websocket, "error", "Invalid JSON message")
                continue
            if not isinstance(data, dict) or data.get("type") != "execute":
                await _send(websocket, "error", 'Expected {"type": "execute", "command": "..."}')
                continue
            command = str(data.get("command", "")).strip()
            if not command:
                await _send(websocket, "error", "Empty command received.")
                continue
            await _send(websocket, "info", f"Executing: {command}")
            await stream_command(websocket, command)
    except websockets.ConnectionClosed:
        pass
    print("Client disconnected")


async def start_server(host: str = "localhost", port: int = 4444) -> None:
    from websockets.asyncio.server import serve

    async with serve(socket_handler, host, port, origins=LOCAL_ORIGINS):
        print(f"Command execution service running on ws://{host}:{port}  (Ctrl-C to stop)")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\nStopped.")
