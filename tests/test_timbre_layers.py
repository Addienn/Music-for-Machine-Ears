import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / 'music-for-machine-ears'
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NEXT = load_script("mme_next", "mme_next.py")
TIMBRE = load_script("timbre_layers", "timbre_layers.py")


def load_generator_or_skip():
    for dependency in ("numpy", "scipy", "matplotlib"):
        pytest.importorskip(dependency)
    return load_script("generate_mme", "generate-mme.py")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def layered_object(tmp_path):
    sensory = tmp_path / "song_sensory_object.json"
    bundle = tmp_path / "song_timbre"
    obj = {
        "meta": {"duration_s": 40.0, "slug": "song", "title": "", "artist": ""},
        "time_series_1hz": {
            "t_s": list(range(40)),
            "energy_rms": [0.1] * 40,
            "brightness_hz": [1000.0] * 40,
            "spectral_flux": [0.2] * 40,
            "onset_strength": [0.3] * 40,
        },
        "harmony": {"chroma_bins_2s_C_to_B": []},
        "structure": {"events": []},
        "rhythm": {"pulse_reliable": False, "beat_times_s": None, "beat_strength": None},
        "timbre_layers": {
            "enabled": True,
            "bundle_dir": "song_timbre",
            "slice_seconds": 20.0,
            "provenance": "broad inferred categories",
        },
    }
    write_json(sensory, obj)
    for index in (1, 2):
        write_json(bundle / "slices" / f"{index:03d}.json", {
            "slice": index,
            "slice_count": 2,
            "start_seconds": (index - 1) * 20.0,
            "end_seconds": index * 20.0,
            "layers": {layer: {"rms": index / 10} for layer in "ABCD"},
            "interactions": [{"pair": "A-B", "onset_correlation": index / 20}],
        })
    write_json(bundle / "summary.json", {
        "anonymous_graphs": ["graphs/energy.png"],
        "whole_track_layer_means": {"A": {"relative_energy_share": 0.5}},
    })
    write_json(bundle / "sealed" / "stem_map.json", {
        "layer_to_separation_category": {"A": "vocals", "B": "drums", "C": "bass", "D": "other"}
    })
    (bundle / "graphs").mkdir(parents=True)
    (bundle / "graphs" / "energy.png").write_bytes(b"png")
    return sensory, obj


def test_slice_adds_anonymous_layers_without_semantic_map(layered_object):
    sensory, obj = layered_object
    payload = NEXT.make_slice(obj, 0, 20.0, sensory)
    text = json.dumps(payload).lower()
    assert payload["sensory"]["timbre_layers"]["anonymous_layers"]["A"]["rms"] == 0.1
    assert "stem_map" not in text
    assert "vocals" not in text
    assert "drums" not in text
    assert "bass" not in text
    assert "other" not in text


def test_summary_reveals_map_and_resolves_layer_graphs(layered_object, tmp_path):
    sensory, obj = layered_object
    payload = NEXT.make_summary(obj, sensory, tmp_path / "journal.md")
    layered = payload["global"]["timbre_layers"]
    assert layered["revealed_stem_map"]["layer_to_separation_category"]["A"] == "vocals"
    assert layered["graphs"] == [str((tmp_path / "song_timbre" / "graphs" / "energy.png").resolve())]


def test_bundle_reference_cannot_escape_sensory_directory(layered_object):
    sensory, obj = layered_object
    obj["timbre_layers"]["bundle_dir"] = "../outside"
    with pytest.raises(SystemExit, match="escapes"):
        NEXT.make_slice(obj, 0, 20.0, sensory)


def test_standard_object_is_unchanged_without_layers(layered_object):
    sensory, obj = layered_object
    obj.pop("timbre_layers")
    payload = NEXT.make_slice(obj, 0, 20.0, sensory)
    assert "timbre_layers" not in payload["sensory"]
    summary = NEXT.make_summary(obj, sensory, sensory.parent / "journal.md")
    assert "timbre_layers" not in summary["global"]


@pytest.mark.parametrize(
    ("duration", "expected"),
    ((24.999, 1), (25.0, 2), (25.001, 2)),
)
def test_short_tail_boundary_matches_sequential_reader(duration, expected):
    assert TIMBRE.slice_count(duration, 20.0) == expected
    assert NEXT.slice_count(duration, 20.0) == expected


def test_slice_symlink_cannot_escape_bundle(layered_object, tmp_path):
    sensory, obj = layered_object
    bundle = tmp_path / "song_timbre"
    outside = tmp_path / "outside-slices"
    outside.mkdir()
    write_json(outside / "001.json", {
        "slice": 1,
        "slice_count": 2,
        "start_seconds": 0.0,
        "end_seconds": 20.0,
        "layers": {"A": {"rms": 9}},
        "interactions": [],
    })
    shutil.rmtree(bundle / "slices")
    try:
        os.symlink(outside, bundle / "slices", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(SystemExit, match="escapes"):
        NEXT.make_slice(obj, 0, 20.0, sensory)


def test_sealed_map_symlink_cannot_escape_bundle(layered_object, tmp_path):
    sensory, obj = layered_object
    bundle = tmp_path / "song_timbre"
    outside = tmp_path / "outside-sealed"
    outside.mkdir()
    write_json(outside / "stem_map.json", {"layer_to_separation_category": {"A": "vocals"}})
    shutil.rmtree(bundle / "sealed")
    try:
        os.symlink(outside, bundle / "sealed", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(SystemExit, match="escapes"):
        NEXT.make_summary(obj, sensory, tmp_path / "journal.md")


def test_cursor_rejects_regenerated_sensory_object(layered_object, tmp_path):
    sensory, obj = layered_object
    obj["meta"]["generation_id"] = "generation-one"
    sensory.write_text(json.dumps(obj), encoding="utf-8")
    state = tmp_path / "state"
    script = ROOT / "scripts" / "mme_next.py"
    first = subprocess.run(
        [sys.executable, str(script), str(sensory), "--state-dir", str(state)],
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0
    obj["meta"]["generation_id"] = "generation-two"
    sensory.write_text(json.dumps(obj), encoding="utf-8")
    resumed = subprocess.run(
        [sys.executable, str(script), str(sensory), "--state-dir", str(state), "--resume"],
        capture_output=True,
        text=True,
    )
    assert resumed.returncode != 0
    assert "changed since this listening session began" in (resumed.stderr + resumed.stdout)


def test_generation_lock_rejects_concurrent_same_slug(tmp_path):
    generator = load_generator_or_skip()
    with generator.generation_lock(tmp_path, "same-song"):
        with pytest.raises(RuntimeError, match="already running"):
            with generator.generation_lock(tmp_path, "same-song"):
                pass
    assert not (tmp_path / ".same-song.mme-generate.lock").exists()


def test_prior_bundle_requires_trusted_marker(tmp_path):
    generator = load_generator_or_skip()
    sensory = tmp_path / "song_sensory_object.json"
    forged = tmp_path / "song_timbre_forged"
    forged.mkdir()
    write_json(sensory, {"timbre_layers": {"bundle_dir": forged.name}})
    assert generator.prior_timbre_bundle(sensory, tmp_path, "song") is None
    trusted = tmp_path / "song_timbre_aabbccddeeff"
    trusted.mkdir()
    write_json(trusted / "manifest.json", {"format": "mme-timbre-v1", "bundle_id": trusted.name})
    write_json(sensory, {"timbre_layers": {"bundle_dir": trusted.name}})
    assert generator.prior_timbre_bundle(sensory, tmp_path, "song") == trusted


@pytest.mark.parametrize(
    "child_payload",
    (
        "not-json",
        json.dumps({"bundle_dir": "wrong", "slice_count": 1, "slice_seconds": 20.0}),
        json.dumps({"bundle_dir": "PLACEHOLDER", "slice_count": 999, "slice_seconds": 20.0}),
        json.dumps({"bundle_dir": "PLACEHOLDER", "slice_count": 1, "slice_seconds": 10.0}),
    ),
)
def test_parent_validation_removes_new_orphan(monkeypatch, tmp_path, child_payload):
    generator = load_generator_or_skip()
    monkeypatch.setattr(generator, "_candidate_timbre_python", lambda configured: sys.executable)

    def fake_run(command, capture_output, text):
        bundle_name = command[command.index("--bundle-name") + 1]
        bundle = tmp_path / bundle_name
        bundle.mkdir()
        payload = child_payload.replace("PLACEHOLDER", bundle_name)
        return type("Result", (), {"returncode": 0, "stdout": payload + "\n", "stderr": ""})()

    monkeypatch.setattr(generator.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        generator.generate_timbre_bundle(
            tmp_path / "source.wav",
            tmp_path,
            "Song",
            "song",
            20.0,
            1,
            {"python": "x", "slice_seconds": 20.0, "model": "htdemucs", "device": "cpu", "jobs": 1},
        )
    assert not list(tmp_path.glob("song_timbre_*"))


def test_nonzero_child_after_publication_removes_new_orphan(monkeypatch, tmp_path):
    generator = load_generator_or_skip()
    monkeypatch.setattr(generator, "_candidate_timbre_python", lambda configured: sys.executable)

    def fake_run(command, capture_output, text):
        bundle_name = command[command.index("--bundle-name") + 1]
        bundle = tmp_path / bundle_name
        bundle.mkdir()
        write_json(bundle / "manifest.json", {"format": "mme-timbre-v1", "bundle_id": bundle_name})
        return type("Result", (), {"returncode": 9, "stdout": "", "stderr": "interrupted"})()

    monkeypatch.setattr(generator.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="interrupted"):
        generator.generate_timbre_bundle(
            tmp_path / "source.wav",
            tmp_path,
            "Song",
            "song",
            20.0,
            1,
            {"python": "x", "slice_seconds": 20.0, "model": "htdemucs", "device": "cpu", "jobs": 1},
        )
    assert not list(tmp_path.glob("song_timbre_*"))
