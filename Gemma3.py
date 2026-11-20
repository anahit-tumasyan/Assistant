"""
Gemma3.py - the one place where the local Gemma 3 (1B, instruction-tuned)
model is loaded and run.

Both the voice-to-command translator (Predict_command.py) and the emergency
chat (emergency_assistant.py) go through this class, so the model is loaded
the same way everywhere and only once per program.
"""

import os
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_DIR, "Includes", "gemma-3-1b-it")


def pick_device(preferred: str | None = None) -> str:
    """Pick where the model runs: NVIDIA GPU, Apple GPU (mps) or CPU.

    An explicit choice wins - either the argument or the ASSISTANT_DEVICE
    environment variable (for example ASSISTANT_DEVICE=cpu).
    """
    choice = preferred or os.environ.get("ASSISTANT_DEVICE")
    if choice:
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str) -> torch.dtype:
    # Gemma 3 was trained in bfloat16. float16 overflows with this model, so
    # GPUs without bfloat16 support, and plain CPUs, get float32 instead.
    if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        return torch.float32
    return torch.float32 if device == "cpu" else torch.bfloat16


class Gemma3:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str | None = None):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Gemma model folder not found: {model_path}\n"
                "Download it with:\n"
                "  hf download google/gemma-3-1b-it --local-dir Includes/gemma-3-1b-it"
            )
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=_pick_dtype(self.device)
        ).to(self.device)
        self.model.eval()
        # One generation at a time, even if several threads share the model.
        self._lock = threading.Lock()

    def generate(self, messages: list[dict], max_new_tokens: int = 256,
                 do_sample: bool = True) -> str:
        """Answer a chat.

        messages: [{"role": "system" | "user" | "assistant", "content": "..."}]
        do_sample=False gives the same answer every time (good for commands);
        do_sample=True gives more natural, varied replies (good for chat).
        """
        # The chat template already adds the <bos> token. Tokenising the
        # messages here (instead of passing a pre-rendered string to a
        # pipeline) avoids a second <bos>, which noticeably hurts Gemma.
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        options = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if not do_sample:
            # The model ships with sampling settings; switch them off for
            # greedy decoding so transformers doesn't warn about them.
            options.update(temperature=None, top_p=None, top_k=None)

        with self._lock, torch.inference_mode():
            output = self.model.generate(**inputs, **options)

        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
