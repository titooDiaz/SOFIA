#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shlex
import platform
import subprocess
import ollama


class SofiaBrain:
    """
    Brain conversacional (como el tuyo) + modo comando.
    """
    def __init__(self, model="qwen3:1.7b"):
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
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "<think>" in text:
            text = text.split("<think>", 1)[0]
        text = text.replace("<think>", "").replace("</think>", "")
        text = " ".join(text.split())
        return text.strip()

    def think(self, text):
        self.history.append({"role": "user", "content": text})

        response = ollama.chat(
            model=self.model,
            messages=self.history,
            think=False,
            options={"temperature": 0.2, "num_predict": 120}
        )

        answer = response["message"]["content"]
        answer = self.clean_response(answer)

        if not answer:
            answer = "I'm sorry, I couldn't formulate a response."

        self.history.append({"role": "assistant", "content": answer})
        return answer

    def command_from_user(self, user_text, os_name, cwd):
        """
        Pide al mismo modelo que responda SOLO con un comando de consola.
        """
        prompt = (
            f"Convert the user request into exactly ONE shell command for {os_name}. "
            f"Current working directory: {cwd}. "
            "Rules: return only the command, no quotes around it, no markdown, no explanation, no extra text. "
            "If the request is ambiguous, return: echo Ambiguous request. "
            f"User request: {user_text}"
        )
        cmd = self.think(prompt)
        cmd = self.clean_response(cmd)
        return cmd.strip()


class ConsoleRunner:
    def __init__(self, safe_mode=False, timeout_seconds=25):
        self.safe_mode = safe_mode
        self.timeout_seconds = timeout_seconds
        self.current_dir = os.getcwd()
        self.is_windows = platform.system().lower().startswith("win")

        self.allowed_base = {
            "pwd", "cd", "ls", "dir", "echo", "whoami", "date", "time",
            "python", "python3", "pip", "pip3",
            "start", "open", "google-chrome", "chrome",
            "code", "notepad", "cat", "type", "mkdir"
        }

        self.blocked_patterns = {
            " rm ", "rmdir", " del ", "format", "mkfs", "shutdown", "reboot",
            "poweroff", "halt", "sudo", " su ", "dd ", "diskpart"
        }

    def _tokenize(self, command):
        try:
            return shlex.split(command, posix=not self.is_windows)
        except Exception:
            return command.split()

    def _check_safety(self, command):
        c = f" {command.strip().lower()} "
        if not c.strip():
            return False, "Empty command."

        for p in self.blocked_patterns:
            if p in c:
                return False, f"Blocked pattern: {p.strip()}"

        tokens = self._tokenize(command.strip())
        if not tokens:
            return False, "Could not parse command."

        base = tokens[0].lower()

        if base == "cd":
            return True, "OK"

        if self.safe_mode and base not in self.allowed_base:
            return False, f"Command not allowed in safe mode: {base}"

        return True, "OK"

    def run(self, command):
        cmd = command.strip()

        if cmd.lower().startswith("cd "):
            path = cmd[3:].strip().strip('"').strip("'")
            new_path = os.path.abspath(os.path.join(self.current_dir, path))
            if os.path.isdir(new_path):
                self.current_dir = new_path
                return cmd, f"Directory changed to: {self.current_dir}", "", 0
            return cmd, "", f"Directory does not exist: {new_path}", 1

        ok, reason = self._check_safety(cmd)
        if not ok:
            return cmd, "", f"[DENIED] {reason}", 126

        try:
            done = subprocess.run(
                cmd,
                shell=True,
                cwd=self.current_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            return cmd, done.stdout.strip(), done.stderr.strip(), done.returncode
        except subprocess.TimeoutExpired:
            return cmd, "", f"Timeout > {self.timeout_seconds}s", 124
        except Exception as e:
            return cmd, "", f"Execution error: {e}", 1


def main():
    brain = SofiaBrain(model="qwen3:1.7b")
    runner = ConsoleRunner(safe_mode=False, timeout_seconds=25)

    os_name = "windows" if runner.is_windows else ("macos" if platform.system().lower().startswith("darwin") else "linux")

    print("SOFIA Console Test (Brain-style)")
    print(f"Model: {brain.model}")
    print(f"OS: {os_name}")
    print(f"Safe mode: {runner.safe_mode}")
    print('Type "salir" to exit.\n')

    while True:
        try:
            user_text = input("Tú -> SOFIA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break

        command = brain.command_from_user(user_text, os_name=os_name, cwd=runner.current_dir)

        cmd, out, err, code = runner.run(command)

        print("\n" + "=" * 60)
        print("SOFIA -> COMMAND SENT:")
        print(cmd)
        print("-" * 60)
        print("STDOUT:")
        print(out if out else "(empty)")
        print("-" * 60)
        print("STDERR:")
        print(err if err else "(empty)")
        print("-" * 60)
        print(f"RETURN CODE: {code}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()