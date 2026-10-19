"""Static checks for rollout registries; no training dependencies are imported.

These checks validate repository layout, not runtime service/worker behavior.
"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "light_scale/async_rollout_v2/registries.py"


class TestRolloutRegistryLayout(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(REGISTRY_PATH.read_text(encoding="utf-8"))

    def _registry(self, name):
        for node in self.tree.body:
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                self.assertIsInstance(node.value, ast.Dict)
                return node.value
        self.fail(f"Registry declaration not found: {name}")

    def test_project_imports_resolve_to_existing_classes(self):
        for node in self.tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith("light_scale."):
                continue
            with self.subTest(module=node.module):
                module_path = ROOT.joinpath(*node.module.split(".")).with_suffix(".py")
                self.assertTrue(module_path.is_file(), f"Missing module: {node.module}")
                module_tree = ast.parse(module_path.read_text(encoding="utf-8"))
                classes = {
                    item.name for item in module_tree.body if isinstance(item, ast.ClassDef)
                }
                for alias in node.names:
                    self.assertIn(alias.name, classes)

    def test_registry_entries_are_unique_and_imported(self):
        imported_names = {
            alias.asname or alias.name
            for node in self.tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        for name in ("SERVICE_CLASS_REGISTRY", "WORKER_CLASS_REGISTRY"):
            with self.subTest(registry=name):
                registry = self._registry(name)
                keys = [ast.literal_eval(key) for key in registry.keys]
                self.assertTrue(all(isinstance(key, str) and key for key in keys))
                self.assertEqual(len(keys), len(set(keys)))
                for value in registry.values:
                    self.assertIsInstance(value, ast.Name)
                    self.assertIn(value.id, imported_names)

    def test_common_services_remain_registered(self):
        keys = {ast.literal_eval(key) for key in self._registry("SERVICE_CLASS_REGISTRY").keys}
        self.assertTrue({"rock", "sglang", "sglang_native"}.issubset(keys))

    def test_common_workers_remain_registered(self):
        keys = {ast.literal_eval(key) for key in self._registry("WORKER_CLASS_REGISTRY").keys}
        self.assertTrue(
            {
                "single_turn", "code", "if_rubric", "math", "math_tool",
                "llm_judge", "rule_llm_mixed", "rock",
            }.issubset(keys)
        )


if __name__ == "__main__":
    unittest.main()
