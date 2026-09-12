import ollama
import re


class SofiaBrain:

    def __init__(self, model="qwen3:1.7b"):

        self.model = model

        self.history = [
            {
                "role": "system",
                "content": (
                    "Your name is SOFIA. "
                    "You are Miguel's personal assistant. "
                    "Speak naturally, clearly and concisely. "
                    "Do not explain your reasoning. "
                    "Give only the final answer."
                )
            }
        ]

    def clean_response(self, text):

        # delete everything before the last </think> tag, if it exists
        if "</think>" in text:

            text = text.split("</think>", 1)[1]
        text = text.replace("<think>", "")
        text = text.replace("</think>", "")
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
            options={
                "temperature": 0.7,
                "num_predict": 100
            }
        )

        answer = response["message"]["content"]

        answer = self.clean_response(answer)

        self.history.append({
            "role": "assistant",
            "content": answer
        })

        return answer