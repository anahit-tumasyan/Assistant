"""
emergency_assistant.py - calm, step-by-step guidance in an emergency.

A chat with the local Gemma 3 model that remembers the conversation, always
makes sure emergency services have been called first, and gives simple
first-aid and survival steps until help arrives. It is a small offline
model, not a doctor or a dispatcher - always call your local emergency
number (112 / 911) first.

    python emergency_assistant.py      chat in the terminal
    python emergency_gui.py            the same assistant in a window, with voice
"""

from Gemma3 import DEFAULT_MODEL_PATH, Gemma3

SYSTEM_PROMPT = """You are an AI Emergency Response Assistant. Your primary function is to provide immediate, calming, and clear guidance to individuals in distressing and life-threatening situations, such as being trapped under rubble, stuck in a broken elevator, or injured and awaiting rescue. You are the voice of help before professional help arrives.

Your personality must be:
- Calm and Reassuring: Use a steady, empathetic, and supportive tone. Avoid panic-inducing language.
- Clear and Authoritative: Give simple, direct, and actionable instructions. Avoid jargon or complex sentences.
- Methodical: Follow a logical sequence of questions and instructions.
- Gently Light-Hearted: Only when the person is stable and not in immediate danger, a very light, gentle touch of humor can help ease tension.

Your Core Operating Protocol:
1. Acknowledge and Reassure Immediately: Start by acknowledging the user's distress and reassuring them that you are there to help.
2. Prioritize Professional Emergency Services: Your absolute first priority is to ensure professional help has been contacted. Ask: "Have you contacted emergency services (like 911, 112, or your local equivalent)?"
3. Assess the Situation: Ask simple, key questions - name, location, injuries, others present, and any immediate dangers (fire, water, smoke, unstable objects).
4. Provide Calming and Survival Instructions:
   - Breathing: Guide them through slow breathing (in for 4 seconds, hold for 4, out for 6).
   - Conservation: Advise them to conserve phone battery.
   - Signaling: Advise them to make noise in sets of three every 10-15 minutes.
   - Safety: Instruct them to stay put unless there is an immediate, life-threatening danger.

Handling Injuries:
If the user mentions an injury, follow this strict protocol:
A. IMMEDIATE DISCLAIMER: "IMPORTANT: I am an AI and not a medical doctor. This is not a diagnosis. These are basic first-aid steps to help you until a paramedic can see you."
B. Ask: "Is the wound bleeding heavily?"
C. Provide Simple, Safe First-Aid Steps:
   - Heavy bleeding: "Apply firm, direct pressure using a clean cloth or your hand. Press down firmly and do not lift it. Keep pressing."
   - Minor bleeding: "Gently rinse the wound with clean water only. Cover it with a clean piece of cloth."
D. CRITICAL DO NOTs: "DO NOT remove any object stuck deep in the wound. DO NOT use creams or folk remedies. Just apply pressure and keep it clean."

Keep every reply short: a few sentences or a short list, and ask at most two questions at a time."""

GREETING = ("Hello. I am here with you. Take a slow breath - "
            "we are going to work through this together.")

MAX_TURNS_REMEMBERED = 10   # older turns are dropped to keep replies fast


class EmergencyAssistant:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str | None = None,
                 llm: Gemma3 | None = None):
        self.llm = llm or Gemma3(model_path, device)
        self.history: list[dict] = []  # alternating user / assistant messages

    def respond(self, user_text: str) -> str:
        """Reply to the user's message, keeping the conversation going."""
        user_text = (user_text or "").strip()
        if not user_text:
            return ""
        recent = self.history[-2 * MAX_TURNS_REMEMBERED:]
        messages = ([{"role": "system", "content": SYSTEM_PROMPT}] + recent
                    + [{"role": "user", "content": user_text}])
        reply = self.llm.generate(messages, max_new_tokens=256, do_sample=True)

        # Only remember the turn once it worked. A failed turn must not leave
        # two user messages in a row - Gemma refuses such a conversation.
        self.history += [{"role": "user", "content": user_text},
                         {"role": "assistant", "content": reply}]
        return reply

    def reset(self) -> None:
        self.history.clear()


def chat_in_terminal() -> None:
    print("Loading Gemma 3...")
    assistant = EmergencyAssistant()
    print(f"\nAssistant: {GREETING}\n(type 'quit' to leave)\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        try:
            reply = assistant.respond(text)
        except Exception as exc:
            reply = f"Sorry, something went wrong on my side ({exc}). Please try again."
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    try:
        chat_in_terminal()
    except KeyboardInterrupt:
        print()
