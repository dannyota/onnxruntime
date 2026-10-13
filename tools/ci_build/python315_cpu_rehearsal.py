# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Fork CPU rehearsal: build targeted artifacts or test an explicitly selected wheel."""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
import venv
import zipfile
from pathlib import Path

SUITES = tuple(
    f"onnxruntime_test_python{suffix}.py"
    for suffix in ("", "_mlops", "_sparse_matmul", "_global_threadpool", "_backend", "_symlink_data")
)
REQUIRED = (
    "test_run_async",
    "test_global_threadpool",
    "test_raise_error_if_use_per_session_threads",
    "test_session_scoped_cpu_ortvalue",
    "test_device_ortvalue_provenance_rules",
    "test_run_session_owned_output_is_zero_copy",
    "test_run_output_does_not_alias_input_passthrough",
    "test_cpu_io_binding_numerical",
)


def validate_provenance(paths, prefix):
    prefix = Path(prefix).resolve()
    for path in paths:
        if not Path(path).resolve().is_relative_to(prefix):
            raise ValueError(f"Import outside fresh test environment: {path}")


def select_wheel(directory, arch, not_before):
    candidates = list(Path(directory).glob("*.whl"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one wheel, found {len(candidates)} in {directory}")
    wheel = candidates[0]
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    parts = wheel.stem.rsplit("-", 3)
    if len(parts) != 4 or parts[1:3] != [tag, tag] or parts[0].split("-")[0] != "onnxruntime":
        raise ValueError(f"Wrong interpreter/ABI tag: {wheel.name}; expected {tag}")
    platforms = parts[3].split(".")
    valid = {
        ("linux", "x64"): lambda p: p == "linux_x86_64",
        ("win32", "x64"): lambda p: p == "win_amd64",
        ("darwin", "arm64"): lambda p: p.startswith("macosx_") and p.endswith("_arm64"),
    }
    if not all(valid[(sys.platform, arch)](p) for p in platforms):
        raise ValueError(f"Wrong platform/architecture tag: {wheel.name}")
    if wheel.stat().st_mtime < not_before:
        raise ValueError(f"Stale wheel: {wheel}")
    return wheel.resolve()


def run(command, cwd, log):
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    with Path(log).open("w", encoding="utf-8") as output:
        subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)


def child(suite_name, result_path):
    # These imports must occur only in the fresh installed-wheel child process.
    np = importlib.import_module("numpy")
    ort = importlib.import_module("onnxruntime")
    native = importlib.import_module("onnxruntime.capi.onnxruntime_pybind11_state")

    paths = [ort.__file__, native.__file__]
    validate_provenance(paths, sys.prefix)

    class CpuBinding(unittest.TestCase):
        def test_cpu_io_binding_numerical(self):
            session = ort.InferenceSession("testdata/mul_1.onnx", providers=["CPUExecutionProvider"])
            data = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
            expected = session.run(None, {session.get_inputs()[0].name: data})
            binding = session.io_binding()
            binding.bind_cpu_input(session.get_inputs()[0].name, data)
            for output in session.get_outputs():
                binding.bind_output(output.name, "cpu")
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            actual = binding.copy_outputs_to_cpu()
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected, strict=True):
                np.testing.assert_allclose(left, right)

    class EvidenceResult(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.passed = []

        def addSuccess(self, test):  # noqa: N802
            self.passed.append(test.id())
            super().addSuccess(test)

    loader = unittest.TestLoader()
    suite = (
        loader.loadTestsFromTestCase(CpuBinding)
        if suite_name == "cpu_binding"
        else loader.loadTestsFromModule(importlib.import_module(Path(suite_name).stem))
    )
    discovered = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
    evidence = {
        "suite": suite_name,
        "pid": os.getpid(),
        "python": sys.version,
        "prefix": sys.prefix,
        "imports": paths,
        "discovered": discovered,
        "executed": result.testsRun - len(result.skipped),
        "skipped": [(test.id(), reason) for test, reason in result.skipped],
        "passed": result.passed,
        "successful": result.wasSuccessful(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "onnx", "onnxruntime")},
    }
    Path(result_path).write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if not result.wasSuccessful() or evidence["executed"] == 0:
        raise SystemExit(1)


def build(args):
    args.build_dir.mkdir(parents=True, exist_ok=False)
    common = [
        sys.executable,
        str(args.source / "tools/ci_build/build.py"),
        "--build_dir",
        str(args.build_dir),
        "--config",
        "Release",
        "--skip_tests",
        "--skip_pip_install",
        "--parallel",
        "2",
        "--cmake_extra_defines",
        "onnxruntime_MATERIALIZE_ONNX_NODE_TESTS=ON",
    ]
    if sys.platform == "win32":
        common += ["--cmake_generator", "Visual Studio 17 2022"]
    run([*common, "--update", "--build_wheel"], args.source, args.evidence / "configure.log")
    run(
        [*common, "--build", "--build_wheel", "--target", "onnxruntime_pybind11_state"],
        args.source,
        args.evidence / "build-wheel.log",
    )
    run(
        [*common, "--build", "--enable_pybind", "--target", "custom_op_library"],
        args.source,
        args.evidence / "build-custom-op.log",
    )
    shutil.copy2(args.build_dir / "Release/CMakeCache.txt", args.evidence / "CMakeCache.txt")
    binary_dir = args.build_dir / ("Release/Release" if sys.platform == "win32" else "Release")
    args.wheel_dir = binary_dir / "dist"
    for compiler_file in (args.build_dir / "Release/CMakeFiles").glob("*/CMakeCXXCompiler.cmake"):
        shutil.copy2(compiler_file, args.evidence / compiler_file.name)
    name = {"win32": "custom_op_library.dll", "darwin": "libcustom_op_library.dylib"}.get(
        sys.platform, "libcustom_op_library.so"
    )
    args.custom_op = binary_dir / name


def test_wheel(args):
    wheel = select_wheel(args.wheel_dir, args.arch, args.not_before)
    shutil.copy2(wheel, args.evidence / wheel.name)
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Wheel must contain exactly one METADATA file")
        classifiers = [
            line for line in archive.read(metadata[0]).decode().splitlines() if line.startswith("Classifier:")
        ]
    (args.evidence / "classifiers.json").write_text(json.dumps(classifiers, indent=2), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="ort-cpu-wheel-") as temporary:
        root = Path(temporary).resolve()
        if any(root.is_relative_to(path) for path in (args.source, args.build_dir)):
            raise ValueError("Temporary test directory must be outside source and build trees")
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                str(wheel),
                "-r",
                str(args.source / "requirements-materialize-onnx-node-tests.txt"),
            ],
            root,
            args.evidence / "install.log",
        )
        run([str(python), "-m", "pip", "check"], root, args.evidence / "pip-check.log")
        run([str(python), "-m", "pip", "freeze"], root, args.evidence / "test-dependencies.txt")
        for name in (*SUITES, "helper.py"):
            shutil.copy2(args.source / "onnxruntime/test/python" / name, root / name)
        shutil.copytree(args.source / "onnxruntime/test/testdata", root / "testdata")
        shutil.copy2(args.custom_op, root / args.custom_op.name)
        shutil.copy2(Path(__file__), root / "rehearsal.py")
        results = []
        for suite in (*SUITES, "cpu_binding"):
            result_path = args.evidence / f"{suite}.json"
            run(
                [str(python), "rehearsal.py", "--suite", suite, "--result", str(result_path)],
                root,
                args.evidence / f"{suite}.log",
            )
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        passed = {test.rsplit(".", 1)[-1] for result in results for test in result["passed"]}
        missing = set(REQUIRED) - passed
        if missing or sum(result["executed"] for result in results) < 50:
            raise ValueError(f"Missing required coverage or insufficient executed cases: {sorted(missing)}")
        summary = {
            "wheel": str(wheel),
            "source_commit": args.source_commit,
            "discovered": sum(result["discovered"] for result in results),
            "executed": sum(result["executed"] for result in results),
            "skipped": sum(len(result["skipped"]) for result in results),
            "required_passed": list(REQUIRED),
            "suites": results,
        }
        (args.evidence / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--custom-op", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--arch", choices=("x64", "arm64"))
    parser.add_argument("--not-before", type=float)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--suite")
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.suite:
        child(args.suite, args.result)
        return
    for name in ("source", "build_dir", "evidence", "source_commit", "arch", "not_before"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if platform.python_implementation() != "CPython" or sysconfig_free_threaded():
        parser.error("Ordinary CPython is required")
    machines = {"x64": {"x86_64", "amd64"}, "arm64": {"arm64", "aarch64"}}
    if platform.machine().lower() not in machines[args.arch]:
        parser.error(f"Unexpected interpreter architecture: {platform.machine()}")
    for name in ("source", "build_dir", "evidence"):
        setattr(args, name, getattr(args, name).resolve())
    args.evidence.mkdir(parents=True, exist_ok=True)
    if args.build:
        build(args)
    if not args.wheel_dir or not args.custom_op:
        parser.error("--wheel-dir and --custom-op are required without --build")
    test_wheel(args)


def sysconfig_free_threaded():
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


if __name__ == "__main__":
    main()
