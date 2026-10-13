# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fnmatch
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import python315_cpu_rehearsal as rehearsal
import yaml


class RehearsalGuards(unittest.TestCase):
    def test_wheel_selection_rejects_wrong_tag_duplicates_and_staleness(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(sys, "platform", "linux"):
            root = Path(temporary)
            wrong_tag = f"cp{sys.version_info.major}{sys.version_info.minor + 1}"
            wrong = root / f"onnxruntime-1.0-{wrong_tag}-{wrong_tag}-linux_x86_64.whl"
            wrong.touch()
            with self.assertRaisesRegex(ValueError, "Wrong interpreter"):
                rehearsal.select_wheel(root, "x64", 0)
            wrong.unlink()
            tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            correct = root / f"onnxruntime-1.0-{tag}-{tag}-linux_x86_64.whl"
            correct.touch()
            self.assertEqual(rehearsal.select_wheel(root, "x64", 0), correct)
            with self.assertRaisesRegex(ValueError, "Stale wheel"):
                rehearsal.select_wheel(root, "x64", time.time() + 10)
            wrong.touch()
            with self.assertRaisesRegex(ValueError, "exactly one wheel"):
                rehearsal.select_wheel(root, "x64", 0)

    def test_wheel_selection_rejects_unrelated_distribution(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(sys, "platform", "linux"):
            root = Path(temporary)
            tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            (root / f"other_package-1.0-{tag}-{tag}-linux_x86_64.whl").touch()
            with self.assertRaisesRegex(ValueError, "Wrong interpreter"):
                rehearsal.select_wheel(root, "x64", 0)

    def test_wheel_selection_rejects_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(sys, "platform", "linux"):
            root = Path(temporary)
            tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            (root / f"onnxruntime-1.0-{tag}-{tag}-linux_aarch64.whl").touch()
            with self.assertRaisesRegex(ValueError, "Wrong platform"):
                rehearsal.select_wheel(root, "x64", 0)

    def test_import_provenance_rejects_checkout_and_build_shadow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "fresh-venv"
            installed = prefix / "lib/site-packages/onnxruntime/__init__.py"
            native = prefix / "lib/site-packages/onnxruntime/capi/native.so"
            rehearsal.validate_provenance([installed, native], prefix)
            for shadow in (root / "checkout/onnxruntime/__init__.py", root / "build/onnxruntime/capi/native.so"):
                with self.assertRaisesRegex(ValueError, "outside fresh"):
                    rehearsal.validate_provenance([installed, shadow], prefix)

    def test_macos_wheel_selection_requires_arm64(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(sys, "platform", "darwin"):
            root = Path(temporary)
            tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            universal = root / f"onnxruntime-1.0-{tag}-{tag}-macosx_15_0_universal2.whl"
            universal.touch()
            with self.assertRaisesRegex(ValueError, "Wrong platform"):
                rehearsal.select_wheel(root, "arm64", 0)
            universal.unlink()
            for minimum_os in ("14_0", "15_0"):
                with self.subTest(minimum_os=minimum_os):
                    arm64 = root / f"onnxruntime-1.0-{tag}-{tag}-macosx_{minimum_os}_arm64.whl"
                    arm64.touch()
                    self.assertEqual(rehearsal.select_wheel(root, "arm64", 0), arm64)
                    arm64.unlink()

    def test_workflow_environment_files_have_real_newlines(self):
        source = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load((source / ".github/workflows/python315-cpu-rehearsal.yml").read_text())
        block = workflow["jobs"]["cpu-wheel"]["steps"][2]["run"]
        for platform_name in ("darwin", "linux", "win32"):
            with self.subTest(platform=platform_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                environment = {
                    "GITHUB_WORKSPACE": str(root),
                    "GITHUB_ENV": str(root / "env"),
                    "GITHUB_PATH": str(root / "path"),
                    "GITHUB_SHA": "fixture-commit",
                    "LEG": "fixture",
                    "REQUIREMENTS": "requirements.txt",
                    "EXPECTED_PYTHON": f"{sys.version_info.major}.{sys.version_info.minor}",
                }
                with (
                    patch.dict(os.environ, environment),
                    patch.object(sys, "platform", platform_name),
                    patch("venv.EnvBuilder"),
                    patch("subprocess.run"),
                    patch("platform.platform", return_value="fixture-platform"),
                ):
                    exec(compile(block, "workflow-setup", "exec"), {})
                env_text = (root / "env").read_text()
                lines = env_text.splitlines()
                self.assertEqual(len(lines), 4 if platform_name == "darwin" else 2)
                self.assertTrue(lines[0].startswith("BUILD_PYTHON="))
                self.assertTrue(lines[1].startswith("WHEEL_NOT_BEFORE="))
                if platform_name == "darwin":
                    self.assertEqual(
                        lines[2:],
                        ["MACOSX_DEPLOYMENT_TARGET=14.0", "_PYTHON_HOST_PLATFORM=macosx-14.0-arm64"],
                    )
                    self.assertTrue(
                        env_text.endswith("MACOSX_DEPLOYMENT_TARGET=14.0\n_PYTHON_HOST_PLATFORM=macosx-14.0-arm64\n")
                    )
                else:
                    self.assertNotIn("MACOSX_DEPLOYMENT_TARGET", env_text)
                    self.assertNotIn("_PYTHON_HOST_PLATFORM", env_text)
                self.assertEqual(len((root / "path").read_text().splitlines()), 1)
                environment["EXPECTED_PYTHON"] = f"{sys.version_info.major}.{sys.version_info.minor + 1}"
                with (
                    patch.dict(os.environ, environment),
                    patch("venv.EnvBuilder"),
                    patch("subprocess.run"),
                    self.assertRaisesRegex(RuntimeError, "Expected Python"),
                ):
                    exec(compile(block, "workflow-setup", "exec"), {})

    def test_windows_build_discovers_multiconfig_artifacts_and_packages_once(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(sys, "platform", "win32"):
            root = Path(temporary)
            args = type("Args", (), {"build_dir": root / "build", "source": root, "evidence": root})()
            calls = []

            def fake_run(command, cwd, log):
                calls.append(command)
                cache = args.build_dir / "Release/CMakeCache.txt"
                cache.parent.mkdir(exist_ok=True)
                cache.touch()

            with patch.object(rehearsal, "run", side_effect=fake_run):
                rehearsal.build(args)
            self.assertEqual(args.wheel_dir, args.build_dir / "Release/Release/dist")
            self.assertEqual(args.custom_op, args.build_dir / "Release/Release/custom_op_library.dll")
            self.assertIn("--build_wheel", calls[1])
            self.assertNotIn("--build_wheel", calls[2])
            self.assertIn("--enable_pybind", calls[2])
            self.assertIn("custom_op_library", calls[2])

    def test_workflow_repository_and_push_branch_guards(self):
        source = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load((source / ".github/workflows/python315-cpu-rehearsal.yml").read_text())
        trigger = workflow.get("on", workflow.get(True))
        patterns = trigger["push"]["branches"]
        guard = workflow["jobs"]["cpu-wheel"]["if"]
        self.assertEqual(guard, "github.repository == 'dannyota/onnxruntime'")
        self.assertEqual(patterns, ["rehearsal/python315-*"])
        for repository, branch, expected in (
            ("dannyota/onnxruntime", "rehearsal/python315-cpu", True),
            ("microsoft/onnxruntime", "rehearsal/python315-cpu", False),
            ("someone/onnxruntime", "rehearsal/python315-cpu", False),
            ("dannyota/onnxruntime", "main", False),
            ("dannyota/onnxruntime", "rehearsal/other", False),
        ):
            allowed = repository == "dannyota/onnxruntime" and any(fnmatch.fnmatchcase(branch, p) for p in patterns)
            self.assertEqual(allowed, expected, (repository, branch))
        self.assertEqual(workflow["permissions"], {"contents": "read"})


if __name__ == "__main__":
    unittest.main()
