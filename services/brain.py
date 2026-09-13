import ollama
import re


class SofiaBrain:

    def __init__(self, model="qwen3:1.7b"): #gemma3:1b 815MG # qwen3:1.7b 1GB # qwen3:0.6b 500MB

        self.model = model

        self.history = [
            {
                "role": "system",
                "content": (
                    "Your name is SOFIA. "
                    "You are Miguel's personal assistant. "
                    "Speak naturally, clearly and concisely. "
                    "Do not think out loud. "
                    "Do not explain your reasoning. "
                    "Give only the final answer."
                )
            }
        ]

    def clean_response(self, text):

        # Remove complete thinking blocks
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL
        )

        # Remove incomplete thinking block
        if "<think>" in text:
            text = text.split("<think>", 1)[0]

        # Remove remaining tags
        text = text.replace("<think>", "")
        text = text.replace("</think>", "")

        # Remove line breaks and duplicate spaces
        text = " ".join(text.split())

        return text.strip()

    def think(self, text):

        self.history.append({
            "role": "user",
            "content": text
        })

        response = ollama.chat(
            model=self.model,
            messages=self.history,
            think=False,
            options={
                "temperature": 0.7,
                "num_predict": 100
            }
        )

        answer = response["message"]["content"]

        print(f"RAW SOFIA: {repr(answer)}")

        answer = self.clean_response(answer)

        if not answer:
            answer = "I'm sorry, I couldn't formulate a response."

        self.history.append({
            "role": "assistant",
            "content": answer
        })

        return answer