import json
import shlex
import asyncio
import subprocess
import websockets


# ---------------------------------------------------------------------------
# Whitelisted shell commands
# Only the base command name (no path) is checked.
# Add entries here to allow more commands.
# ---------------------------------------------------------------------------
WHITELISTED_COMMANDS = {
    # File navigation
    "ls", "pwd", "cd", "find", "tree",
    # File viewing
    "cat", "head", "tail", "less", "more", "wc",
    # Search
    "grep", "rg", "awk", "sed",
    # System info
    "date", "whoami", "uname", "uptime", "df", "du", "top", "ps",
    # Network (read-only)
    "ping", "curl", "wget", "nslookup", "dig",
    # Text / output
    "echo", "printf", "sort", "uniq", "cut", "tr",
    # Python / pip
    "python", "python3", "pip", "pip3",
    # Git (read-only subset)
    "git",
    # macOS helpers
    "open", "say", "pbcopy", "pbpaste", "caffeinate", "osascript",
    # Package manager
    "brew",
}


def _base_command(command: str) -> str:
    """Extract the bare command name from a shell command string."""
    try:
        parts = shlex.split(command)
        if not parts:
            return ""
        # Strip any absolute/relative path prefix
        return os.path.basename(parts[0]) if "/" in parts[0] else parts[0]
    except ValueError:
        return ""


import os


def is_whitelisted(command: str) -> bool:
    """Return True only if the command's base name is in WHITELISTED_COMMANDS."""
    base = _base_command(command)
    return base in WHITELISTED_COMMANDS


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def run_command_and_stream_output(websocket, command: str):
    """Execute a whitelisted shell command and stream its output over WebSocket."""
    if not is_whitelisted(command):
        await websocket.send(json.dumps({
            "type": "error",
            "message": f"Command not allowed: '{_base_command(command)}'. "
                       f"Add it to WHITELISTED_COMMANDS to enable it."
        }))
        return 1

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Stream stdout line by line
    while True:
        line = process.stdout.readline()
        if line:
            await websocket.send(json.dumps({
                "type": "output",
                "message": line.rstrip("\n"),
            }))
        else:
            if process.poll() is not None:
                break
            await asyncio.sleep(0.05)

    # Flush any remaining stderr
    error_output = process.stderr.read()
    if error_output:
        for err_line in error_output.strip().split("\n"):
            await websocket.send(json.dumps({
                "type": "error",
                "message": err_line,
            }))

    return_code = process.wait()
    await websocket.send(json.dumps({
        "type": "info",
        "message": f"Command exited with return code: {return_code}",
    }))
    return return_code


async def socket_handler(websocket):
    """Handle each WebSocket client — accept commands and execute them."""
    try:
        await websocket.send(json.dumps({
            "type": "info",
            "message": "Connected to command execution service",
        }))

        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "execute":
                    command = data.get("command", "").strip()
                    if not command:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Empty command received.",
                        }))
                        continue

                    if not is_whitelisted(command):
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": (
                                f"Blocked: '{_base_command(command)}' is not in "
                                f"the whitelist."
                            ),
                        }))
                        continue

                    await websocket.send(json.dumps({
                        "type": "info",
                        "message": f"Executing: {command}",
                    }))
                    await run_command_and_stream_output(websocket, command)

            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON message",
                }))

    except websockets.ConnectionClosed:
        print("Client disconnected")


async def start_server(host="localhost", port=4444):
    """Start the WebSocket command-execution server."""
    async with websockets.serve(socket_handler, host, port):
        print(f"Command execution service running on ws://{host}:{port}")
        await asyncio.Future()  # run forever


# ---------------------------------------------------------------------------
# Synchronous helper (used by other modules)
# ---------------------------------------------------------------------------

def run_command(command: str) -> str:
    """
    Run a whitelisted shell command synchronously.

    Returns stdout on success or an error/block message.
    """
    if not is_whitelisted(command):
        return (
            f"Blocked: '{_base_command(command)}' is not in the whitelist. "
            f"Add it to WHITELISTED_COMMANDS to enable it."
        )
    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout or "(command completed with no output)"
        return f"Error (code {result.returncode}): {result.stderr.strip()}"
    except Exception as e:
        return f"Exception occurred: {e}"


if __name__ == "__main__":
    asyncio.run(start_server())
