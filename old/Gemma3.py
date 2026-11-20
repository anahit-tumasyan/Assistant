import torch
from transformers import AutoTokenizer, pipeline


class Gemma3:
    def __init__(self, model_path: str = "./Includes/gemma-3-1b-it", device: str = "mps"):
        """
        Args:
            model_path: Local path or HuggingFace model ID.
            device: 'cpu', 'cuda:0', 'mps', etc.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.gen_pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=self.tokenizer,
            device=device,
            torch_dtype="auto",
        )

    def generate(self, messages: list, max_new_tokens: int = 256) -> str:
        """
        Generate a response given a list of chat-style messages.

        Args:
            messages: List of dicts with 'role' and 'content' keys, e.g.
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": "Hello!"},
                ]
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The model's response as a plain string.
        """
        # Apply the model's chat template to produce the raw prompt string
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.gen_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True)

        # Return only the newly generated portion (strip the prompt prefix)
        generated = outputs[0]["generated_text"][len(prompt):].strip()
        return generated
