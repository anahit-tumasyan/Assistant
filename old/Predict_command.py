# Predict_command.py

import os
import asyncio
import subprocess
import websockets
from transformers import AutoTokenizer, pipeline
from Run_command import run_command


class TerminalAssistant:
    def __init__(self,
                 model_path: str = "./Includes/gemma-3-1b-it",
                 device: str = "mps"):
        """
        Initialize the assistant with model details and device.
        """
        self.model_path = model_path
        self.device = device

        # Load tokenizer & text-generation pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.pipe = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.tokenizer,
            device=self.device,
            torch_dtype="auto"
        )

        # System prompt for terminal command translation
        self.prompt_text = """You are a helpful assistant that translates natural language commands into terminal commands.
Your task is to extract the precise command to execute in a Unix-like terminal.
You must return ONLY the command itself, with no explanations, no markdown, no code blocks — just the raw command.
If the user says open anything, use the 'open' command.

Examples:
User: list files in the current directory
ls

User: create a new directory named 'projects'
mkdir projects

User: change directory to downloads
cd ~/Downloads

User: open the file 'report.txt' with the default text editor
open report.txt

User: copy the file 'data.csv' to the directory 'backup'
cp data.csv backup

User: remove the file 'temp.log'
rm temp.log

User: open folder MIT in desktop
open ~/Desktop/MIT/

User: search for files containing 'keyword' in the home directory
grep -r 'keyword' ~

User: show the current working directory
pwd

User: display the contents of 'config.json'
cat config.json

User: list all in the Documents
ls ~/Documents/

User: run the python script 'script.py'
python3 script.py

User: install the package 'requests' using pip
pip install requests

User: compress the directory 'images' into 'images.zip'
zip -r images.zip images

User: extract the archive 'archive.tar.gz'
tar -xzf archive.tar.gz

User: shutdown the system
sudo shutdown -h now

User: restart the system
sudo reboot

User: show the network interfaces
ifconfig

User: list all running processes
ps aux

User: kill the process with PID 1234
kill 1234

User: find all files named 'example.txt'
find . -name example.txt

User: count the number of lines in 'largefile.txt'
wc -l largefile.txt

User: display the first 10 lines of 'logfile.log'
head -n 10 logfile.log

User: display the last 10 lines of 'logfile.log'
tail -n 10 logfile.log

User: download a file from 'http://example.com/file.zip'
curl -O http://example.com/file.zip

User: update the system packages
sudo apt update && sudo apt upgrade -y

User: install git
sudo apt install git -y

User: remove a directory called temp recursively
rm -rf temp

User: create a new empty file called newfile.txt
touch newfile.txt

User: move a file called oldfile.txt to newfolder
mv oldfile.txt newfolder

User: give execute permissions to script.sh
chmod +x script.sh

User: display the disk usage in human readable format
df -h

User: show the memory usage
free -h

User: find all files larger than 10MB
find . -size +10M

User: sort the contents of data.txt
sort data.txt

User: replace all occurrences of 'old' with 'new' in text.txt
sed 's/old/new/g' text.txt

Now translate the following user command into a terminal command:"""

    def predict_command(self, input_text: str) -> str:
        """
        Uses the text-generation pipeline to translate natural language
        into a terminal command.
        """
        messages = [
            {"role": "system", "content": self.prompt_text},
            {"role": "user", "content": input_text},
        ]

        # Apply chat template to get the properly formatted prompt string
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.pipe(prompt, max_new_tokens=64, do_sample=False)

        # Strip off the prompt prefix — return only the newly generated text
        generated = outputs[0]["generated_text"][len(prompt):].strip()

        # Keep only the first line (the command itself)
        command = generated.splitlines()[0].strip()
        return command

    async def _stream_command(self, websocket, command: str):
        """
        Internal helper: run a subprocess and stream stdout/stderr to a websocket.
        """
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Stream STDOUT line by line
        while True:
            line = proc.stdout.readline()
            if line:
                await websocket.send(f"[STDOUT] {line.rstrip()}")
            elif proc.poll() is not None:
                break
            else:
                await asyncio.sleep(0.05)

        # Stream any remaining STDERR
        for err in proc.stderr.read().splitlines():
            if err:
                await websocket.send(f"[STDERR] {err}")

        code = proc.wait()
        await websocket.send(f"[INFO] Command exited with return code: {code}")

    def stream_command_over_ws(self,
                               command: str,
                               host: str = "localhost",
                               port: int = 4444):
        """
        Spin up a one-shot WebSocket server on ws://{host}:{port}.
        Connecting clients will receive live command output, then the server closes.
        """
        async def handler(ws, path):
            await ws.send(f"[INFO] Running: {command}")
            await self._stream_command(ws, command)
            await ws.close()

        async def server_main():
            async with websockets.serve(handler, host, port):
                print(f"WS server listening on ws://{host}:{port}")
                await asyncio.Future()

        try:
            asyncio.run(server_main())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    assistant = TerminalAssistant()

    nl = "Open the Desktop"
    cmd = assistant.predict_command(nl)
    print(f"Predicted command: {cmd}")
    print(run_command(cmd))
