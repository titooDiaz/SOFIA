import base64
import io
import json
import threading
import time
from typing import Any

import ollama


class DesktopAgent:
    """Runs a short Ollama vision/action loop without shell access."""

    MAX_STEPS = 20
    MAX_TEXT = 500
    MODEL = "qwen3-vl:2b"

    def __init__(self, model: str | None = None):
        self.model = model or self.MODEL
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._running = False

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def stop(self) -> None:
        self._stop.set()

    def run(self, goal: str, max_steps: int = MAX_STEPS) -> dict[str, Any]:
        goal = goal.strip()
        if not goal or len(goal) > 2000:
            raise ValueError("La tarea debe tener entre 1 y 2000 caracteres.")
        if self.running:
            raise RuntimeError("Ya hay una tarea en ejecución.")

        self._stop.clear()
        with self._lock:
            self._running = True
        actions: list[dict[str, Any]] = []
        try:
            for step in range(min(max_steps, self.MAX_STEPS)):
                if self._stop.is_set():
                    return {"status": "stopped", "actions": actions}
                image = self._screenshot()
                decision = self._decide(goal, image, actions)
                action = decision.get("action", "finish")
                if action == "finish":
                    return {"status": "completed", "message": decision.get("message", ""), "actions": actions}
                result = self._execute(action, decision)
                actions.append({"step": step + 1, "action": action, "result": result})
                if result.startswith("DENIED"):
                    return {"status": "denied", "message": result, "actions": actions}
                time.sleep(0.25)
            return {"status": "limit_reached", "message": "Se alcanzó el límite de acciones.", "actions": actions}
        finally:
            with self._lock:
                self._running = False

    def _screenshot(self) -> bytes:
        try:
            import pyautogui
            image = pyautogui.screenshot()
        except ImportError as exc:
            raise RuntimeError("Instala pyautogui y Pillow para usar visión de pantalla.") from exc
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _decide(self, goal: str, image: bytes, actions: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = (
            "You control a local desktop to complete the user's goal. "
            "Return JSON only. Choose exactly one action. "
            "Allowed actions: move(x,y), click(x,y,button), type(text), "
            "hotkey(keys), wait(seconds), finish(message). "
            "Never use shell, terminal, passwords, payments, deletion, or system settings. "
            "Coordinates must be visible and plausible. Keep text short. "
            f"Goal: {goal}\nPrevious actions: {json.dumps(actions[-5:], ensure_ascii=False)}\n"
            'Schema: {"action":"finish","message":"..."}'
        )
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [base64.b64encode(image).decode("ascii")]}],
            format="json",
            options={"temperature": 0, "num_predict": 180},
        )
        parsed = json.loads(response["message"]["content"])
        if not isinstance(parsed, dict):
            raise ValueError("Ollama devolvió una acción inválida.")
        return parsed

    def _execute(self, action: str, data: dict[str, Any]) -> str:
        if action in {"type", "hotkey"}:
            text = str(data.get("text", data.get("keys", "")))
            if len(text) > self.MAX_TEXT:
                return "DENIED: texto demasiado largo."
        if action in {"click", "move"}:
            try:
                x, y = int(data["x"]), int(data["y"])
            except (KeyError, TypeError, ValueError):
                return "DENIED: coordenadas inválidas."
            if x < 0 or y < 0:
                return "DENIED: coordenadas fuera de pantalla."
        if action == "hotkey":
            keys = {key.lower() for key in str(data.get("keys", "")).split("+")}
            if keys & {"ctrl", "alt", "shift"} and {"delete", "esc"} & keys:
                return "DENIED: combinación potencialmente destructiva."

        try:
            import pyautogui
        except ImportError:
            return "DENIED: pyautogui no está instalado."
        if action == "move":
            pyautogui.moveTo(int(data["x"]), int(data["y"]), duration=0.15)
        elif action == "click":
            pyautogui.click(int(data["x"]), int(data["y"]), button=str(data.get("button", "left")))
        elif action == "type":
            pyautogui.write(str(data["text"])[: self.MAX_TEXT], interval=0.01)
        elif action == "hotkey":
            pyautogui.hotkey(*[key.strip() for key in str(data["keys"]).split("+")])
        elif action == "wait":
            time.sleep(min(max(float(data.get("seconds", 1)), 0), 5))
        else:
            return "DENIED: acción no permitida."
        return "ok"
