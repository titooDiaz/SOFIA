#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shlex
import platform
import subprocess
import ollama


class SofiaBrain:
    """Selects a command template and fills variables from user intent."""

    def __init__(self, model="qwen2.5:0.5b"):
        self.model = model
        self.history = [
            {
                "role": "system",
                "content": (
                    "Your name is SOFIA. "
                    "You are a fast command-line assistant. "
                    "Output ONLY valid JSON."
                )
            }
        ]

    def think_json(self, prompt):
        response = ollama.chat(
            model=self.model,
            messages=self.history + [{"role": "user", "content": prompt}],
            format="json",
            options={
                "temperature": 0.0,
                "num_predict": 100
            }
        )
        return response["message"]["content"].strip()

    def choose_command_json(self, user_text, os_name, cwd, command_items):
        catalog = []
        for item in command_items:
            catalog.append({
                "id": item["id"],
                "template": item["template"],
                "args": item.get("args", [])
            })

        prompt = (
            "Select ONE command from the catalog and fill args.\n"
            f"OS: {os_name}\n"
            f"CWD: {cwd}\n\n"
            "JSON schema: {\"id\":\"...\",\"args\":{...},\"confidence\":0.0}\n"
            f"User request: {user_text}\n"
            f"Catalog: {json.dumps(catalog, ensure_ascii=False)}"
        )

        raw = self.think_json(prompt)

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return {"id": "NONE", "args": {}, "confidence": 0}
            parsed.setdefault("id", "NONE")
            parsed.setdefault("args", {})
            return parsed
        except Exception:
            return {"id": "NONE", "args": {}, "confidence": 0}


class CommandCatalog:
    """Loads command definitions from JSON."""

    def __init__(self, path):
        self.path = path
        self.items = self.load()

    def load(self):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Catalog file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("commands.json must be a JSON array")

        required = {"id", "description", "template"}
        cleaned = []
        for i, item in enumerate(data):
            if not isinstance(item, dict) or not required.issubset(item.keys()):
                raise ValueError(f"Invalid command at index {i}")
            cleaned.append(item)

        return cleaned

    def by_id(self, cmd_id):
        for item in self.items:
            if item["id"] == cmd_id:
                return item
        return None


class TemplateEngine:
    """Replaces {variables} in templates with validated user values."""

    PLACEHOLDER_RE = re.compile(r"{([a-zA-Z_][a-zA-Z0-9_]*)}")

    @staticmethod
    def extract_placeholders(template):
        return list(dict.fromkeys(TemplateEngine.PLACEHOLDER_RE.findall(template)))

    @staticmethod
    def sanitize_value(val):
        # Allow letters, numbers, spaces, underscore, dash, dot
        val = val.strip()
        if not val:
            return ""
        if not re.fullmatch(r"[A-Za-z0-9_\-\. ]{1,80}", val):
            return ""
        return val

    @staticmethod
    def render(template, args):
        result = template
        for key in TemplateEngine.extract_placeholders(template):
            if key not in args:
                raise ValueError(f"Missing argument: {key}")
            safe_val = TemplateEngine.sanitize_value(str(args[key]))
            if not safe_val:
                raise ValueError(f"Invalid value for: {key}")
            result = result.replace("{" + key + "}", safe_val)
        return result


class ConsoleRunner:
    def __init__(self, timeout_seconds=25):
        self.timeout_seconds = timeout_seconds
        self.current_dir = os.getcwd()
        self.is_windows = platform.system().lower().startswith("win")
        self.blocked_patterns = {
            " rm ", "rmdir", " del ", "format", "mkfs", "shutdown", "reboot",
            "poweroff", "halt", "sudo", " su ", "dd ", "diskpart", "mkfs."
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


def detect_os():
    if platform.system().lower().startswith("win"):
        return "windows"
    if platform.system().lower().startswith("darwin"):
        return "macos"
    return "linux"


def main():
    catalog_path = "documents/commands.json"
    brain = SofiaBrain(model="qwen3:0.6b")
    catalog = CommandCatalog(catalog_path)
    runner = ConsoleRunner(timeout_seconds=25)
    os_name = detect_os()

    print("SOFIA Console (JSON command catalog)")
    print(f"Model: {brain.model}")
    print(f"OS: {os_name}")
    print(f"Catalog: {catalog_path} ({len(catalog.items)} commands loaded)")
    print('Type "salir" to exit.\n')

    while True:
        try:
            user_text = input("You -> SOFIA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"salir", "exit", "quit"}:
            print("See you.")
            break

        selection = brain.choose_command_json(
            user_text=user_text,
            os_name=os_name,
            cwd=runner.current_dir,
            command_items=catalog.items
        )

        selected_id = selection.get("id", "NONE")
        args = selection.get("args", {})

        cmd_def = catalog.by_id(selected_id)
        if not cmd_def:
            print("\nNo suitable command found.\n")
            continue

        if os_name not in cmd_def.get("platforms", ["linux", "macos", "windows"]):
            print("\nSelected command is not supported on this OS.\n")
            continue

        try:
            final_command = TemplateEngine.render(cmd_def["template"], args)
        except Exception as e:
            print("\nCould not build command:", str(e), "\n")
            continue

        cmd, out, err, code = runner.run(final_command)

        print("\n" + "=" * 60)
        print("SELECTED COMMAND ID:")
        print(selected_id)
        print("-" * 60)
        print("SELECTED COMMAND:")
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