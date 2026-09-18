#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shlex
import platform
import subprocess
import hashlib
import unicodedata
import ollama


class SofiaBrain:
    """Selects a command template and fills variables from user intent."""

    ALL_PLATFORMS = ("linux", "macos", "windows")
    FAST_PATTERNS = {
        "list_files": (
            "listar archivos", "lista archivos", "mostrar archivos", "ver archivos",
            "list files", "show files", "ls"
        ),
        "pwd": (
            "directorio actual", "donde estoy", "current directory",
            "working directory", "where am i", "pwd"
        ),
    }

    def __init__(self, model="qwen3:1.7b"):
        self.model = model
        self.decision_cache = {}
        self.history = [
            {
                "role": "system",
                "content": (
                    "Your name is SOFIA. "
                    "You are Miguel's personal assistant. "
                    "Be concise. Return only final answers."
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

    def think(self, prompt, options=None):
        if options is None:
            options = {"temperature": 0, "num_predict": 80}
        response = ollama.chat(
            model=self.model,
            messages=self.history + [{"role": "user", "content": prompt}],
            think=False,
            options=options
        )
        answer = self.clean_response(response["message"]["content"])
        return answer

    @staticmethod
    def _normalize_text(text):
        normalized = unicodedata.normalize("NFKD", text.lower())
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return " ".join(normalized.split())

    @classmethod
    def _platforms_for_item(cls, item):
        return item.get("platforms", list(cls.ALL_PLATFORMS))

    @classmethod
    def _filter_for_platform(cls, command_items, os_name):
        return [
            item for item in command_items
            if os_name in cls._platforms_for_item(item)
        ]

    def _catalog_signature(self, command_items):
        canonical = []
        for item in command_items:
            canonical.append({
                "id": item.get("id"),
                "description": item.get("description"),
                "template": item.get("template"),
                "args": item.get("args", []),
                "platforms": sorted(self._platforms_for_item(item)),
            })
        raw = json.dumps(sorted(canonical, key=lambda x: str(x["id"])), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _command_hint_text(item):
        return " ".join([
            str(item.get("id", "")),
            str(item.get("description", "")),
            str(item.get("template", "")),
        ]).lower()

    def _resolve_fast_command(self, intent, command_items):
        if intent == "list_files":
            for item in command_items:
                hint = self._command_hint_text(item)
                if any(k in hint for k in (" ls", "dir", "list", "listar", "archivo", "file")):
                    return item["id"], {}
        if intent == "pwd":
            for item in command_items:
                hint = self._command_hint_text(item)
                if any(k in hint for k in ("pwd", "directorio actual", "current directory", "working directory")):
                    return item["id"], {}
        return None

    def fast_decision(self, user_text, os_name, command_items):
        normalized_text = self._normalize_text(user_text)
        compatible_items = self._filter_for_platform(command_items, os_name)
        for intent, patterns in self.FAST_PATTERNS.items():
            if any(pattern in normalized_text for pattern in patterns):
                resolved = self._resolve_fast_command(intent, compatible_items)
                if resolved:
                    cmd_id, args = resolved
                    return {"id": cmd_id, "args": args, "confidence": 1.0, "fast_path": True}
        return None

    def choose_command_json(self, user_text, os_name, cwd, command_items):
        """
        Returns JSON:
        {
          "id": "mkdir_named",
          "args": {"nombre":"proyecto_x"},
          "confidence": 0.0-1.0
        }
        or {"id":"NONE","args":{},"confidence":0}
        """
        fast = self.fast_decision(user_text, os_name, command_items)
        if fast is not None:
            return fast

        compatible_items = self._filter_for_platform(command_items, os_name)
        if not compatible_items:
            return {"id": "NONE", "args": {}, "confidence": 0}

        cache_key = (
            self._normalize_text(user_text),
            os_name,
            os.path.abspath(cwd),
            self._catalog_signature(command_items),
        )
        cached = self.decision_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        catalog = []
        for item in compatible_items:
            row = {
                "id": item["id"],
                "description": item["description"],
                "args": item.get("args", []),
            }
            if "platforms" in item:
                row["platforms"] = self._platforms_for_item(item)
            catalog.append(row)

        prompt = (
            "You must select ONE command definition from the catalog and fill args.\n"
            f"OS: {os_name}\n"
            f"CWD: {cwd}\n\n"
            "Rules:\n"
            "1) Return ONLY valid JSON.\n"
            "2) JSON schema: {\"id\":\"...\",\"args\":{...},\"confidence\":0.0}\n"
            "3) id must exist in catalog, or id='NONE' if no good match.\n"
            "4) Fill args from user request when possible.\n"
            "5) Do not include markdown.\n\n"
            f"User request: {user_text}\n\n"
            f"Catalog: {json.dumps(catalog, ensure_ascii=False)}"
        )

        raw = self.think(
            prompt,
            options={"temperature": 0, "num_predict": 80}
        )

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return {"id": "NONE", "args": {}, "confidence": 0}
            parsed.setdefault("id", "NONE")
            parsed.setdefault("args", {})
            parsed.setdefault("confidence", 0)
            if not isinstance(parsed["args"], dict):
                parsed["args"] = {}
            if not isinstance(parsed.get("id"), str):
                parsed["id"] = "NONE"
            valid_ids = {item["id"] for item in compatible_items}
            if parsed["id"] != "NONE" and parsed["id"] not in valid_ids:
                parsed["id"] = "NONE"
                parsed["args"] = {}
                parsed["confidence"] = 0
            if not isinstance(parsed["confidence"], (int, float)):
                parsed["confidence"] = 0
            parsed["confidence"] = float(parsed["confidence"])
            self.decision_cache[cache_key] = dict(parsed)
            return parsed
        except Exception:
            return {"id": "NONE", "args": {}, "confidence": 0}


class CommandCatalog:
    """Loads command definitions from JSON."""

    def __init__(self, path):
        self.path = path
        self.items = self.load()
        self.items_by_id = {item["id"]: item for item in self.items}

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
        return self.items_by_id.get(cmd_id)


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
    brain = SofiaBrain(model="qwen3:1.7b")
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