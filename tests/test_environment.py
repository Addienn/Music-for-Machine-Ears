"""Portable environment/consent tests; no Torch, Demucs or models needed."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'music-for-machine-ears'
sys.path.insert(0, str(ROOT / "scripts"))
import check_environment as CHECK


def generator():
    for name in CHECK.BASE_MODULES:
        pytest.importorskip(name)
    spec = importlib.util.spec_from_file_location("generate_mme", ROOT / "scripts/generate-mme.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checker_help_works_without_site_packages():
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/check_environment.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--timbre" in result.stdout
    assert "--config" in result.stdout


def test_checker_missing_runtime_fails_cleanly(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"timbre_layers": {"python": str(tmp_path / "missing-python")}}))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/check_environment.py"),
                             "--timbre", "--config", str(config)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Cannot run Python" in result.stdout
    assert "Remedy:" in result.stdout
    assert "Traceback" not in result.stderr + result.stdout
    assert "No packages were installed" in result.stdout


def test_missing_explicit_config_is_error(tmp_path):
    with pytest.raises(RuntimeError, match="Could not read"):
        CHECK.load_mme_config(tmp_path / "absent.json")


def test_relative_config_runtime_and_environment_precedence(tmp_path, monkeypatch):
    config = tmp_path / "custom.json"
    config.write_text(json.dumps({"timbre_layers": {"python": "./runtime/python"}}))
    monkeypatch.setenv("MME_TIMBRE_PYTHON", "environment-python")
    settings = CHECK.load_mme_config(config)["timbre_layers"]
    assert CHECK.candidate_timbre_python(settings["python"]) == str(tmp_path / "runtime/python")
    assert CHECK.candidate_timbre_python("") == "environment-python"


def test_default_runtime_has_no_agent_discovery_or_base_fallback(monkeypatch):
    monkeypatch.delenv("MME_TIMBRE_PYTHON", raising=False)
    monkeypatch.setenv("AGENT_ID", "ignored")
    expected = ROOT / ".venv-timbre" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert CHECK.candidate_timbre_python("") == str(expected)
    assert CHECK.candidate_timbre_python("") != sys.executable


def test_disabled_override_ignores_invalid_optional_settings():
    assert not CHECK.resolve_timbre_config({"enabled": "invalid", "jobs": 0}, False)["enabled"]
    with pytest.raises(RuntimeError, match="jobs"):
        CHECK.resolve_timbre_config({"enabled": False, "jobs": 0}, True)


@pytest.mark.parametrize("field,value", [("jobs", 0), ("slice_seconds", float("nan")),
                                          ("allow_model_download", "yes"), ("python", 3)])
def test_enabled_config_validation(field, value):
    with pytest.raises(RuntimeError):
        CHECK.resolve_timbre_config({field: value}, True)


def test_preflight_reports_missing_ffmpeg_and_packages(monkeypatch):
    monkeypatch.setattr(CHECK.shutil, "which", lambda _: None)
    monkeypatch.setattr(CHECK, "probe_runtime", lambda *_: {
        "supported": True, "modules": {"librosa": {"ok": False, "error": "missing"}}
    })
    with pytest.raises(RuntimeError, match="FFmpeg") as exc:
        CHECK.preflight_timbre({})
    assert "librosa" in str(exc.value)


def test_no_timbre_probe_in_ordinary_checker(monkeypatch, capsys):
    calls = []
    def fake_probe(python, modules):
        calls.append(tuple(modules))
        return {"python": "3.11", "supported": True,
                "modules": {name: {"ok": name != "soundfile", "version": "test", "error": "missing"}
                            for name in modules}}
    monkeypatch.setattr(CHECK, "probe_runtime", fake_probe)
    monkeypatch.setattr(CHECK.shutil, "which", lambda _: None)
    assert CHECK.main([]) == 0
    assert calls == [CHECK.BASE_MODULES + ("soundfile",)]
    assert "OPTIONAL soundfile" in capsys.readouterr().out


def test_missing_base_package_fails_checker(monkeypatch):
    monkeypatch.setattr(CHECK, "probe_runtime", lambda *_: {
        "python": "3.11", "supported": True,
        "modules": {"numpy": {"ok": False, "error": "missing"}}
    })
    assert CHECK.main([]) == 1


def test_probe_invokes_imports_only(monkeypatch):
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({"supported": True, "modules": {}}), "")
    monkeypatch.setattr(CHECK.subprocess, "run", fake_run)
    CHECK.probe_runtime("isolated-python", CHECK.TIMBRE_MODULES)
    assert commands[0][:2] == ["isolated-python", "-c"]
    assert "get_model" not in commands[0][2]
    assert "pip" not in commands[0]


def test_layer_preflight_precedes_base_analysis(tmp_path, monkeypatch):
    module = generator()
    analyzed = []
    monkeypatch.setattr(module, "_generate_mme_unlocked", lambda *args, **kwargs: analyzed.append(True))
    def unavailable(_):
        raise RuntimeError("missing optional runtime")
    monkeypatch.setattr(module, "preflight_timbre", unavailable)
    with pytest.raises(RuntimeError, match="missing optional runtime"):
        module.generate_mme("missing.wav", tmp_path / "out", timbre_enabled=True,
                            timbre_config={"allow_model_download": True})
    assert analyzed == []
    assert not (tmp_path / "out").exists()


def test_ordinary_generation_never_preflights_optional_runtime(tmp_path, monkeypatch):
    module = generator()
    def unexpected(_):
        raise AssertionError("ordinary mode probed timbre")
    monkeypatch.setattr(module, "preflight_timbre", unexpected)
    monkeypatch.setattr(module, "_generate_mme_unlocked", lambda *args, **kwargs: "ordinary")
    assert module.generate_mme("input.wav", tmp_path / "out") == "ordinary"


def test_download_consent_before_generation(tmp_path, monkeypatch):
    module = generator()
    monkeypatch.setattr(module, "preflight_timbre", lambda _: "python")
    def unexpected(*args, **kwargs):
        raise AssertionError("analysis started without model retrieval consent")
    monkeypatch.setattr(module, "_generate_mme_unlocked", unexpected)
    with pytest.raises(RuntimeError, match="allow.model.download"):
        module.generate_mme("missing.wav", tmp_path / "out", timbre_enabled=True)


def test_explicit_ordinary_cli_bypasses_broken_config(tmp_path):
    module = generator()
    config = tmp_path / "config.json"
    config.write_text("not JSON")
    # main() is exercised with a mocked generator, avoiding expensive graphs.
    from unittest.mock import patch
    with patch.object(sys, "argv", ["generate-mme.py", "--audio", "input.wav", "--out_dir", str(tmp_path),
                                     "--config", str(config), "--no-timbre-layers"]), \
         patch.object(module, "generate_mme", return_value={}) as generate:
        module.main()
    assert generate.call_args.kwargs["timbre_enabled"] is False


@pytest.mark.parametrize("enabled,flag,expected", [(False, "--timbre-layers", True),
                                                   (True, "--no-timbre-layers", False),
                                                   (False, "--allow-model-download", False)])
def test_cli_overrides_config_without_implicitly_enabling(tmp_path, monkeypatch, enabled, flag, expected):
    module = generator()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"timbre_layers": {"enabled": enabled}}))
    captured = {}
    def fake_generate(*args, **kwargs):
        captured.update(kwargs)
        return {}
    monkeypatch.setattr(module, "generate_mme", fake_generate)
    monkeypatch.setattr(sys, "argv", ["generate-mme.py", "--audio", "input.wav", "--config", str(config), flag])
    module.main()
    assert captured["timbre_enabled"] is expected
    assert captured["timbre_config"]["allow_model_download"] is (flag == "--allow-model-download")


def test_generator_cli_missing_optional_runtime_has_no_traceback_or_outputs(tmp_path):
    generator()  # Skip only when base test packages are unavailable.
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"timbre_layers": {"python": str(tmp_path / "missing-python")}}))
    output = tmp_path / "not-created"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/generate-mme.py"),
                             "--audio", str(tmp_path / "also-missing.wav"), "--out_dir", str(output),
                             "--config", str(config), "--timbre-layers", "--allow-model-download"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Timbre preflight failed" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()
