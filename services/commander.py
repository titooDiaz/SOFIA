import os
import re
import json
import shlex
import platform
import subprocess
import ollama


# ============================================================
# COMMANDER BRAIN
# ============================================================

class CommanderBrain:

    def __init__(self, model="qwen3:0.6b"):
        self.model = model

        self.history = [
            {
                "role": "system",
                "content": (
                    "Your name is SOFIA. "
                    "You are a command selection system. "
                    "Your job is to select exactly ONE command "
                    "from the provided catalog and fill its arguments. "
                    "You MUST output valid JSON only. "
                    "Do not use markdown. "
                    "Do not explain your answer."
                )
            }
        ]

    def think_json(self, prompt):

        response = ollama.chat(
            model=self.model,
            messages=self.history + [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json",
            options={
                "temperature": 0.0,
                "num_predict": 150
            }
        )

        raw = response["message"]["content"].strip()

        print("\n----- LLM RAW RESPONSE -----")
        print(raw)
        print("----------------------------\n")

        return raw

    def choose_command_json(
        self,
        user_text,
        os_name,
        cwd,
        command_items,
        last_error="None"
    ):

        catalog = []

        for item in command_items:

            catalog.append({
                "id": item["id"],
                "description": item.get("description", ""),
                "template": item["template"],
                "args": item.get("args", []),
                "platforms": item.get(
                    "platforms",
                    ["linux", "macos", "windows"]
                )
            })

        prompt = f"""
You are selecting a command for SOFIA.

USER REQUEST:
{user_text}

OPERATING SYSTEM:
{os_name}

CURRENT DIRECTORY:
{cwd}

PREVIOUS ERROR:
{last_error}

COMMAND CATALOG:
{json.dumps(catalog, ensure_ascii=False)}

RULES:

1. Select exactly ONE command from the catalog.
2. The "id" MUST be copied exactly from an existing catalog entry.
3. Never invent a command id.
4. "args" MUST be a JSON object.
5. The keys inside "args" MUST correspond exactly to the argument names
   listed in the selected command's "args" array.
6. If the selected command has no arguments, use an empty object.
7. Do not put the complete command inside "args".
8. Do not use markdown.
9. Do not explain anything.
10. Return ONLY this JSON structure:

{{
    "id": "command_id",
    "args": {{}},
    "confidence": 1.0
}}

Select the most appropriate command for the user's request.
"""

        raw = self.think_json(prompt)

        try:

            parsed = json.loads(raw)

        except json.JSONDecodeError as e:

            print("JSON PARSE ERROR:")
            print(e)

            return {
                "id": "NONE",
                "args": {},
                "confidence": 0
            }

        if not isinstance(parsed, dict):

            print("LLM returned something other than an object.")

            return {
                "id": "NONE",
                "args": {},
                "confidence": 0
            }

        selected_id = parsed.get("id", "NONE")
        args = parsed.get("args", {})
        confidence = parsed.get("confidence", 0)

        if not isinstance(selected_id, str):
            selected_id = "NONE"

        if not isinstance(args, dict):
            args = {}

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0

        return {
            "id": selected_id,
            "args": args,
            "confidence": confidence
        }


# ============================================================
# COMMAND CATALOG
# ============================================================

class CommandCatalog:

    def __init__(self, path):

        self.path = path
        self.items = self.load()

    def load(self):

        if not os.path.isfile(self.path):

            raise FileNotFoundError(
                f"Catalog file not found: {self.path}"
            )

        with open(
            self.path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if not isinstance(data, list):

            raise ValueError(
                "commands.json must be a JSON array"
            )

        required = {
            "id",
            "description",
            "template"
        }

        cleaned = []

        for i, item in enumerate(data):

            if (
                not isinstance(item, dict)
                or not required.issubset(item.keys())
            ):

                raise ValueError(
                    f"Invalid command at index {i}"
                )

            if "args" not in item:
                item["args"] = []

            if "platforms" not in item:

                item["platforms"] = [
                    "linux",
                    "macos",
                    "windows"
                ]

            cleaned.append(item)

        print(
            f"Loaded {len(cleaned)} commands "
            f"from {self.path}"
        )

        return cleaned

    def by_id(self, cmd_id):

        for item in self.items:

            if item["id"] == cmd_id:
                return item

        return None


# ============================================================
# TEMPLATE ENGINE
# ============================================================

class TemplateEngine:

    PLACEHOLDER_RE = re.compile(
        r"{([a-zA-Z_][a-zA-Z0-9_]*)}"
    )

    @staticmethod
    def extract_placeholders(template):

        return list(
            dict.fromkeys(
                TemplateEngine.PLACEHOLDER_RE.findall(
                    template
                )
            )
        )

    @staticmethod
    def sanitize_value(val):

        val = str(val).strip()

        if not val:
            return ""

        # Allow normal command argument characters.
        #
        # This intentionally excludes shell metacharacters such as:
        # ; & | > < ` $ ( )
        #
        # because arguments are eventually passed to a shell.

        if not re.fullmatch(
            r"[A-Za-z0-9_\-./: ]{1,200}",
            val
        ):

            return ""

        return val

    @staticmethod
    def render(template, args):

        result = template

        placeholders = (
            TemplateEngine.extract_placeholders(
                template
            )
        )

        for key in placeholders:

            if key not in args:

                raise ValueError(
                    f"Missing argument: {key}"
                )

            safe_val = (
                TemplateEngine.sanitize_value(
                    args[key]
                )
            )

            if not safe_val:

                raise ValueError(
                    f"Invalid value for: {key}"
                )

            result = result.replace(
                "{" + key + "}",
                safe_val
            )

        return result


# ============================================================
# CONSOLE RUNNER
# ============================================================

class ConsoleRunner:

    def __init__(self, timeout_seconds=25):

        self.timeout_seconds = timeout_seconds

        self.current_dir = os.getcwd()

        self.is_windows = (
            platform.system()
            .lower()
            .startswith("win")
        )

        self.blocked_patterns = {

            " rm ",
            "rmdir",
            " del ",
            "format",
            "mkfs",
            "shutdown",
            "reboot",
            "poweroff",
            "halt",
            "sudo",
            " su ",
            "dd ",
            "diskpart",
            "mkfs."
        }

    def _tokenize(self, command):

        try:

            return shlex.split(
                command,
                posix=not self.is_windows
            )

        except Exception:

            return command.split()

    def _check_safety(self, command):

        c = (
            f" {command.strip().lower()} "
        )

        if not c.strip():

            return False, "Empty command."

        for pattern in self.blocked_patterns:

            if pattern in c:

                return (
                    False,
                    f"Blocked pattern: {pattern.strip()}"
                )

        tokens = self._tokenize(
            command.strip()
        )

        if not tokens:

            return (
                False,
                "Could not parse command."
            )

        return True, "OK"

    def run(self, command):

        cmd = command.strip()

        # ----------------------------------------------------
        # Handle cd separately
        # ----------------------------------------------------

        if cmd.lower().startswith("cd "):

            path = (
                cmd[3:]
                .strip()
                .strip('"')
                .strip("'")
            )

            new_path = os.path.abspath(
                os.path.join(
                    self.current_dir,
                    path
                )
            )

            if os.path.isdir(new_path):

                self.current_dir = new_path

                return (
                    cmd,
                    f"Directory changed to: "
                    f"{self.current_dir}",
                    "",
                    0
                )

            return (
                cmd,
                "",
                f"Directory does not exist: "
                f"{new_path}",
                1
            )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        ok, reason = self._check_safety(cmd)

        if not ok:

            return (
                cmd,
                "",
                f"[DENIED] {reason}",
                126
            )

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        try:

            done = subprocess.run(

                cmd,

                shell=True,

                cwd=self.current_dir,

                capture_output=True,

                text=True,

                timeout=self.timeout_seconds
            )

            return (
                cmd,
                done.stdout.strip(),
                done.stderr.strip(),
                done.returncode
            )

        except subprocess.TimeoutExpired:

            return (
                cmd,
                "",
                f"Timeout > "
                f"{self.timeout_seconds}s",
                124
            )

        except Exception as e:

            return (
                cmd,
                "",
                f"Execution error: {e}",
                1
            )


# ============================================================
# OPERATING SYSTEM
# ============================================================

def detect_os():

    system = platform.system().lower()

    if system.startswith("win"):

        return "windows"

    if system.startswith("darwin"):

        return "macos"

    return "linux"


# ============================================================
# DEVICE COMMANDER
# ============================================================

class DeviceCommander:

    def __init__(
        self,
        catalog_path="documents/commands.json"
    ):

        self.brain = CommanderBrain(
            model="qwen3:0.6b"
        )

        self.catalog = CommandCatalog(
            catalog_path
        )

        self.runner = ConsoleRunner(
            timeout_seconds=25
        )

        self.os_name = detect_os()

        print(
            f"Commander initialized | "
            f"OS: {self.os_name} | "
            f"CWD: {self.runner.current_dir}"
        )

    def execute_intent(self, user_intent):

        max_attempts = 5

        last_error = "None"

        for attempt in range(max_attempts):

            print("\n")
            print("=" * 70)
            print(
                f"ATTEMPT {attempt + 1}/{max_attempts}"
            )
            print("=" * 70)

            # ------------------------------------------------
            # 1. Ask LLM for command
            # ------------------------------------------------

            selection = (
                self.brain.choose_command_json(

                    user_text=user_intent,

                    os_name=self.os_name,

                    cwd=self.runner.current_dir,

                    command_items=self.catalog.items,

                    last_error=last_error
                )
            )

            print(
                f"LLM SELECTION: {selection}"
            )

            selected_id = selection.get(
                "id",
                "NONE"
            )

            args = selection.get(
                "args",
                {}
            )

            confidence = selection.get(
                "confidence",
                0
            )

            print(
                f"Selected command: "
                f"{selected_id}"
            )

            print(
                f"Arguments: {args}"
            )

            print(
                f"Confidence: "
                f"{confidence}"
            )

            # ------------------------------------------------
            # 2. Validate selected command
            # ------------------------------------------------

            cmd_def = self.catalog.by_id(
                selected_id
            )

            if not cmd_def:

                last_error = (
                    f"No command with id "
                    f"'{selected_id}' exists."
                )

                print(
                    f"ERROR: {last_error}"
                )

                continue

            print(
                f"Description: "
                f"{cmd_def.get('description', '')}"
            )

            print(
                f"Template: "
                f"{cmd_def['template']}"
            )

            print(
                f"Expected args: "
                f"{cmd_def.get('args', [])}"
            )

            # ------------------------------------------------
            # 3. Validate platform
            # ------------------------------------------------

            platforms = cmd_def.get(
                "platforms",
                [
                    "linux",
                    "macos",
                    "windows"
                ]
            )

            if self.os_name not in platforms:

                last_error = (
                    f"Command '{selected_id}' "
                    f"is not supported on "
                    f"{self.os_name}."
                )

                print(
                    f"ERROR: {last_error}"
                )

                continue

            # ------------------------------------------------
            # 4. Validate arguments
            # ------------------------------------------------

            expected_args = cmd_def.get(
                "args",
                []
            )

            missing_args = [
                arg
                for arg in expected_args
                if arg not in args
            ]

            if missing_args:

                last_error = (
                    "Missing arguments: "
                    + ", ".join(missing_args)
                )

                print(
                    f"ERROR: {last_error}"
                )

                continue

            # ------------------------------------------------
            # 5. Render command
            # ------------------------------------------------

            try:

                final_command = (
                    TemplateEngine.render(

                        cmd_def["template"],

                        args
                    )
                )

            except Exception as e:

                last_error = (
                    f"Template building error: {e}"
                )

                print(
                    f"ERROR: {last_error}"
                )

                continue

            print(
                f"FINAL COMMAND: "
                f"{final_command}"
            )

            # ------------------------------------------------
            # 6. Execute command
            # ------------------------------------------------

            cmd, out, err, code = (
                self.runner.run(
                    final_command
                )
            )

            print(
                f"RETURN CODE: {code}"
            )

            print(
                f"STDOUT: {out}"
            )

            print(
                f"STDERR: {err}"
            )

            # ------------------------------------------------
            # 7. Success
            # ------------------------------------------------

            if code == 0:

                print(
                    "=" * 70
                )

                print(
                    "COMMAND SUCCESS"
                )

                print(
                    "=" * 70
                )

                message = (
                    out
                    if out
                    else (
                        f"Command executed "
                        f"successfully: {cmd}"
                    )
                )

                return True, message

            # ------------------------------------------------
            # 8. Failure
            # ------------------------------------------------

            last_error = (
                f"Command: {cmd}\n"
                f"Return code: {code}\n"
                f"STDOUT: {out}\n"
                f"STDERR: {err}"
            )

            print(
                f"COMMAND FAILED:\n"
                f"{last_error}"
            )

        # ----------------------------------------------------
        # All attempts failed
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("ALL ATTEMPTS FAILED")
        print("=" * 70)

        return (
            False,
            "Failed to satisfy intent after "
            f"{max_attempts} attempts."
        )
