# gemma3_server.py
import time
from transformers import AutoTokenizer, pipeline


def gemma3_process_func(child_conn, model_path: str = "./Includes/gemma-3-1b-it", device: str = "cpu"):
    """
    Runs in its own process:
      - Loads the Gemma3 model once.
      - Waits for messages via a multiprocessing Pipe.
      - Expected message format: {"prompt": "...", "system": "..."}
        (the 'system' key is optional)
      - Sends back {"generated_text": "..."} or {"error": "..."}.
      - Exits cleanly when it receives the string "TERMINATE".
    """
    print("[Gemma3 Process] Loading model…")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    gen_pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        device=device,
        torch_dtype="auto",
    )
    print("[Gemma3 Process] Model loaded. Waiting for prompts.")

    while True:
        if child_conn.poll():
            msg = child_conn.recv()

            if msg == "TERMINATE":
                print("[Gemma3 Process] Terminate signal received. Shutting down.")
                break

            if not isinstance(msg, dict) or "prompt" not in msg:
                child_conn.send({"error": "Invalid message format. Expected {'prompt': '...'}."})
                continue

            user_prompt = msg["prompt"].strip()
            system_prompt = msg.get("system", "You are a helpful assistant.")

            if not user_prompt:
                child_conn.send({"error": "Empty prompt."})
                continue

            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
                prompt_str = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                outputs = gen_pipe(prompt_str, max_new_tokens=256, do_sample=True)
                generated = outputs[0]["generated_text"][len(prompt_str):].strip()
                child_conn.send({"generated_text": generated})
            except Exception as e:
                child_conn.send({"error": str(e)})

        else:
            time.sleep(0.02)  # avoid busy-waiting
