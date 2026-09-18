import json
import sys
import types
import tempfile
import unittest
import importlib.util
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("computer_control.py")


def load_module():
    if "ollama" not in sys.modules:
        sys.modules["ollama"] = types.SimpleNamespace(chat=lambda **kwargs: {"message": {"content": "{}"}})
    spec = importlib.util.spec_from_file_location("computer_control_under_test", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ComputerControlTests(unittest.TestCase):
    def test_fast_path_uses_existing_command_id(self):
        module = load_module()
        brain = module.SofiaBrain()
        command_items = [
            {"id": "cmd_listar_archivos", "description": "Listar archivos", "template": "ls", "args": [], "platforms": ["linux"]},
            {"id": "cmd_pwd", "description": "Mostrar directorio actual", "template": "pwd", "args": [], "platforms": ["linux"]},
        ]

        with mock.patch.object(module.ollama, "chat", side_effect=AssertionError("Fast path should skip Ollama")):
            result = brain.choose_command_json("por favor listar archivos", "linux", "/tmp", command_items)

        self.assertEqual(result["id"], "cmd_listar_archivos")
        self.assertEqual(result["args"], {})
        self.assertTrue(result["fast_path"])

    def test_choose_command_filters_platform_and_omits_template(self):
        module = load_module()
        brain = module.SofiaBrain()
        command_items = [
            {"id": "linux_only", "description": "List Linux files", "template": "ls", "args": [], "platforms": ["linux"]},
            {"id": "windows_only", "description": "List Windows files", "template": "dir", "args": [], "platforms": ["windows"]},
        ]
        captured = {}

        def fake_chat(**kwargs):
            captured["prompt"] = kwargs["messages"][-1]["content"]
            return {"message": {"content": json.dumps({"id": "linux_only", "args": {}, "confidence": 0.9})}}

        with mock.patch.object(module.ollama, "chat", side_effect=fake_chat):
            result = brain.choose_command_json("open folders", "linux", "/tmp", command_items)

        self.assertEqual(result["id"], "linux_only")
        self.assertNotIn("windows_only", captured["prompt"])
        self.assertNotIn('"template"', captured["prompt"])

    def test_choose_command_uses_cache_and_invalidates_on_catalog_change(self):
        module = load_module()
        brain = module.SofiaBrain()
        command_items = [
            {"id": "linux_only", "description": "List Linux files", "template": "ls", "args": [], "platforms": ["linux"]},
        ]
        calls = {"count": 0}

        def fake_chat(**kwargs):
            calls["count"] += 1
            return {"message": {"content": json.dumps({"id": "linux_only", "args": {}, "confidence": 0.9})}}

        with mock.patch.object(module.ollama, "chat", side_effect=fake_chat):
            brain.choose_command_json("open folders", "linux", "/tmp", command_items)
            brain.choose_command_json("open folders", "linux", "/tmp", command_items)
            self.assertEqual(calls["count"], 1)

            changed_catalog = [
                {"id": "linux_only", "description": "List Linux files now", "template": "ls", "args": [], "platforms": ["linux"]},
            ]
            brain.choose_command_json("open folders", "linux", "/tmp", changed_catalog)
            self.assertEqual(calls["count"], 2)

    def test_command_catalog_by_id_uses_prebuilt_index(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_file = Path(temp_dir) / "commands.json"
            catalog_file.write_text(
                json.dumps([
                    {"id": "a", "description": "desc a", "template": "echo a"},
                    {"id": "b", "description": "desc b", "template": "echo b"},
                ]),
                encoding="utf-8",
            )

            catalog = module.CommandCatalog(str(catalog_file))
            catalog.items = None
            self.assertEqual(catalog.by_id("b")["template"], "echo b")

    def test_console_runner_security_and_cd_behavior(self):
        module = load_module()
        runner = module.ConsoleRunner(timeout_seconds=1)
        _, _, denied_err, denied_code = runner.run("rm -rf /")
        self.assertEqual(denied_code, 126)
        self.assertIn("[DENIED]", denied_err)

        original_dir = runner.current_dir
        _, _, cd_err, cd_code = runner.run("cd .")
        self.assertEqual(cd_code, 0)
        self.assertEqual(cd_err, "")
        self.assertEqual(runner.current_dir, original_dir)


if __name__ == "__main__":
    unittest.main()
