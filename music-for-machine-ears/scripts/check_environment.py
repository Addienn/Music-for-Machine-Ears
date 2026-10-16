#!/usr/bin/env python3
"""Check MME prerequisites without installing packages or downloading models.

This entrypoint and its configuration helpers use only the standard library.
Package imports are tested in child interpreters, never in this process.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
DEFAULT_TIMBRE_CONFIG = {
    "enabled": False,
    "python": "",
    "model": "htdemucs",
    "device": "cpu",
    "jobs": 1,
    "slice_seconds": 20.0,
    "cleanup_temporary_audio": True,
    "allow_model_download": False,
}
BASE_MODULES = ("numpy", "scipy", "matplotlib")
# Demucs' own import checks version-specific dependencies (some releases need
# torchaudio; others do not). Do not impose extra packages independently.
TIMBRE_MODULES = BASE_MODULES + ("soundfile", "librosa", "torch", "demucs.separate")


def load_mme_config(path=None):
    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH
    config = {"timbre_layers": dict(DEFAULT_TIMBRE_CONFIG)}
    if not config_path.is_file() and path is None:
        return config
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not read MME config {config_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("timbre_layers", {}), dict):
        raise RuntimeError("MME config must be an object with a timbre_layers object.")
    config["timbre_layers"].update(payload.get("timbre_layers", {}))
    configured = config["timbre_layers"].get("python")
    if isinstance(configured, str) and configured.strip():
        expanded = os.path.expandvars(os.path.expanduser(configured.strip()))
        if not Path(expanded).is_absolute() and ("/" in expanded or "\\" in expanded):
            expanded = str((config_path.parent / expanded).resolve())
        config["timbre_layers"]["python"] = expanded
    return config


def resolve_timbre_config(config, enabled_override=None):
    merged = dict(DEFAULT_TIMBRE_CONFIG, **(config or {}))
    if enabled_override is None:
        if not isinstance(merged.get("enabled"), bool):
            raise RuntimeError("MME config timbre_layers.enabled must be true or false.")
    else:
        merged["enabled"] = bool(enabled_override)
    if not merged["enabled"]:
        return merged
    for key in ("cleanup_temporary_audio", "allow_model_download"):
        if not isinstance(merged.get(key), bool):
            raise RuntimeError(f"MME config timbre_layers.{key} must be true or false.")
    jobs, seconds = merged.get("jobs"), merged.get("slice_seconds")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise RuntimeError("MME timbre config jobs must be an integer greater than or equal to 1.")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("MME timbre config slice_seconds must be a finite positive number.")
    for key in ("model", "device"):
        if not isinstance(merged.get(key), str) or not merged[key].strip():
            raise RuntimeError(f"MME timbre config {key} must be a non-empty string.")
    if not isinstance(merged.get("python"), str):
        raise RuntimeError("MME timbre config python must be a string.")
    return merged


def candidate_timbre_python(configured):
    """Never silently substitute the base interpreter for the optional runtime."""
    explicit = str(configured or "").strip()
    environment = os.environ.get("MME_TIMBRE_PYTHON", "").strip()
    if explicit or environment:
        return os.path.expandvars(os.path.expanduser(explicit or environment))
    return str(DEFAULT_CONFIG_PATH.parent / ".venv-timbre" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    ))


# Real imports detect broken binary wheels as well as missing distributions.
# No Demucs model-loading function is invoked here.
IMPORT_PROBE = '''import importlib, json, sys
result = {"python": sys.version.split()[0], "supported": sys.version_info >= (3, 11), "modules": {}}
for name in sys.argv[1:]:
    try:
        module = importlib.import_module(name)
        result["modules"][name] = {"ok": True, "version": str(getattr(module, "__version__", "installed"))}
    except Exception as exc:
        result["modules"][name] = {"ok": False, "error": type(exc).__name__ + ": " + str(exc).splitlines()[-1][:300]}
print(json.dumps(result))
'''


def probe_runtime(python, modules):
    try:
        result = subprocess.run(
            [python, "-c", IMPORT_PROBE, *modules], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot run Python {python!r}: {exc}") from exc
    try:
        if result.returncode:
            raise ValueError("interpreter/import probe failed")
        return json.loads(result.stdout.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        detail = (result.stderr or result.stdout or str(exc)).strip().splitlines()[-1:]
        raise RuntimeError(f"Could not check Python {python!r}: {' '.join(detail)[:300]}") from exc


def runtime_problems(report):
    problems = []
    if not report["supported"]:
        problems.append("Python 3.11+ is required")
    for name, status in report["modules"].items():
        if not status["ok"]:
            problems.append(f"{name}: {status['error']}")
    return problems


def preflight_timbre(config):
    """Fail before audio analysis/graphs, without loading or fetching a model."""
    python = candidate_timbre_python(config.get("python"))
    problems = []
    if not shutil.which("ffmpeg"):
        problems.append("FFmpeg not found; install FFmpeg and add its executable to PATH")
    try:
        problems.extend(runtime_problems(probe_runtime(python, TIMBRE_MODULES)))
    except RuntimeError as exc:
        problems.append(str(exc))
    if problems:
        raise RuntimeError(
            "Timbre preflight failed:\n- " + "\n- ".join(problems)
            + "\nCreate/fix an isolated timbre environment and set timbre_layers.python or "
            "MME_TIMBRE_PYTHON (or use <skill>/.venv-timbre). "
            "Run scripts/check_environment.py --timbre for details. Nothing was installed."
        )
    return python


def print_runtime(label, python, modules, optional=()):
    print(f"{label} interpreter: {python}")
    try:
        report = probe_runtime(python, modules)
    except RuntimeError as exc:
        print(f"MISSING: {exc}")
        return False
    print(f"Python {report['python']}: {'OK' if report['supported'] else 'requires 3.11+'}")
    ok = report["supported"]
    for name, status in report["modules"].items():
        if status["ok"]:
            print(f"  OK {name}: {status['version']}")
        else:
            required = name not in optional
            print(f"  {'MISSING' if required else 'OPTIONAL'} {name}: {status['error']}")
            ok = ok and not required
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timbre", action="store_true", help="Also require the optional timbre runtime and FFmpeg")
    parser.add_argument("--config", type=Path, help="Configuration JSON (default: <skill>/config.json)")
    args = parser.parse_args(argv)
    ok = print_runtime("Base", sys.executable, BASE_MODULES + ("soundfile",), optional=("soundfile",))
    ffmpeg = shutil.which("ffmpeg")
    ffmpeg_note = "not found (required for timbre layers)" if args.timbre else "not found (optional for ordinary compatible WAV input)"
    print(f"FFmpeg: {ffmpeg or ffmpeg_note}")
    if not ok:
        print("Remedy: use Python 3.11+ and install numpy scipy matplotlib in that environment; soundfile is recommended.")
    if args.timbre:
        try:
            config = resolve_timbre_config(load_mme_config(args.config)["timbre_layers"], True)
            python = candidate_timbre_python(config.get("python"))
            timbre_ok = print_runtime("Timbre", python, TIMBRE_MODULES)
            ok = ok and timbre_ok and bool(ffmpeg)
            print(f"Model: {config['model']}; cache status not inspected. No model loaded or downloaded.")
            print("Demucs manages its model cache; generation may download weights when permitted.")
            if not timbre_ok or not ffmpeg:
                print("Remedy: create <skill>/.venv-timbre or set timbre_layers.python / MME_TIMBRE_PYTHON. "
                      "Install compatible demucs, torch, librosa, numpy, scipy, matplotlib, soundfile there "
                      "(and dependencies required by your Demucs version). "
                      "Install FFmpeg separately and add it to PATH. See the optional setup documentation.")
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            ok = False
    elif args.config:
        try:
            load_mme_config(args.config)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            ok = False
    print("No packages were installed and no model downloads were requested.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
