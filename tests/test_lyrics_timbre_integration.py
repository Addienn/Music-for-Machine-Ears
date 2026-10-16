"""Synthetic, download-free coverage of lyrics and layers in one publication."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from test_lyrics_delivery import LYRICS, cli
from test_timbre_layers import layered_object, load_generator_or_skip, write_json


def test_generator_publishes_lyrics_and_timbre_together(tmp_path, monkeypatch):
    generator = load_generator_or_skip()
    audio = tmp_path / "tone.wav"
    sr = 22050
    samples = np.arange(sr * 8) / sr
    wavfile.write(audio, sr, (0.15 * np.sin(2 * np.pi * 440 * samples)).astype(np.float32))
    lrc = tmp_path / "supplied.lrc"
    lrc.write_text("[00:02]synthetic first\n[00:06]synthetic last\n", encoding="utf-8")
    output = tmp_path / "out"
    events = []
    original_parse = generator.parse_lrc
    original_publish = generator.write_json_atomic

    def parse(path):
        events.append("parse")
        assert not output.exists()
        return original_parse(path)

    def preflight(config):
        assert events == ["parse"]
        assert not output.exists()
        events.append("preflight")

    def bundle(audio_path, out_dir, title, slug, duration, count, config):
        events.append("bundle")
        bundle_path = Path(out_dir) / f"{slug}_timbre_mock"
        bundle_path.mkdir()
        return {"enabled": True, "bundle_dir": bundle_path.name,
                "slice_seconds": 20.0, "provenance": "synthetic mocked bundle"}

    def publish(path, obj):
        events.append("publish")
        assert obj["lyrics"]["lines"][0]["text"] == "synthetic first"
        assert obj["timbre_layers"]["enabled"] is True
        assert (output / obj["timbre_layers"]["bundle_dir"]).is_dir()
        original_publish(path, obj)

    monkeypatch.setattr(generator, "parse_lrc", parse)
    monkeypatch.setattr(generator, "preflight_timbre", preflight)
    monkeypatch.setattr(generator, "generate_timbre_bundle", bundle)
    monkeypatch.setattr(generator, "write_json_atomic", publish)
    # Preserve the existing positional timbre arguments and keyword-only LRC.
    result = generator.generate_mme(audio, output, "", "", None, True,
                                    {"allow_model_download": True}, lyrics_path=lrc)
    obj = json.loads(Path(result["json"]).read_text())
    assert obj["lyrics"] == original_parse(lrc)
    assert obj["timbre_layers"]["enabled"] is True
    assert events == ["parse", "preflight", "bundle", "publish"]


def test_missing_lrc_precedes_timbre_preflight_audio_and_output(tmp_path, monkeypatch):
    generator = load_generator_or_skip()

    def unexpected(*args, **kwargs):
        pytest.fail("Missing LRC must fail before optional setup, audio, or output")

    monkeypatch.setattr(generator, "preflight_timbre", unexpected)
    monkeypatch.setattr(generator, "load_audio_with_fallback", unexpected)
    monkeypatch.setattr(generator, "generation_lock", unexpected)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Cannot load --lyrics"):
        generator.generate_mme(tmp_path / "missing.wav", output,
                               timbre_enabled=True, lyrics_path=tmp_path / "missing.lrc")
    assert not output.exists()


@pytest.fixture
def combined_object(layered_object, tmp_path):
    sensory, obj = layered_object
    lrc = tmp_path / "private-source.lrc"
    lrc.write_text("[ti:Synthetic title]\n[00:02]early words\n"
                   "[00:20]later words\n[00:40]endpoint words\n", encoding="utf-8")
    obj["lyrics"] = LYRICS.parse_lrc(lrc)
    obj["meta"]["generation_id"] = "generation-one"
    write_json(sensory, obj)
    return sensory, obj


@pytest.mark.parametrize("reveal", [False, True])
def test_combined_sequence_resume_and_final_disclosure(combined_object, tmp_path, reveal):
    sensory, obj = combined_object
    state = tmp_path / "state"

    def call(*args):
        return json.loads(cli("mme_next.py", sensory, "--state-dir", state, *args).stdout)

    first = call(*(["--reveal-lyrics"] if reveal else []))
    assert first["sensory"]["timbre_layers"]["anonymous_layers"]["A"]["rms"] == 0.1
    for secret in ["vocals", "stem_map", "private-source", "lyric_onset_s", "later words"]:
        assert secret not in json.dumps(first)
    if reveal:
        assert first["lyrics"]["lines"] == [{"time_s": 2.0, "text": "early words"}]
    else:
        assert "lyrics" not in first
    replay = call("--resume")
    replay.pop("resumed")
    replay.pop("resume_message")
    assert replay == first
    second = call("--note", "Synthetic test note, not a listening claim.")
    assert second["sensory"]["timbre_layers"]["anonymous_layers"]["A"]["rms"] == 0.2
    assert "vocals" not in json.dumps(second)
    if reveal:
        assert [line["text"] for line in second["lyrics"]["lines"]] == ["later words", "endpoint words"]
        assert "early words" not in json.dumps(second)
    else:
        assert "lyrics" not in second
    replay = call("--resume")
    replay.pop("resumed")
    replay.pop("resume_message")
    assert replay == second
    summary = call("--note", "Synthetic final note.")
    assert summary["global"]["lyrics"] == obj["lyrics"]
    assert summary["global"]["timbre_layers"]["revealed_stem_map"]["layer_to_separation_category"]["A"] == "vocals"
    assert call("--resume") == summary


@pytest.mark.parametrize("change", ["generation", "digest", "bundle"])
def test_combined_session_rejects_changes_until_reset(combined_object, tmp_path, change):
    sensory, obj = combined_object
    state = tmp_path / "state"
    cli("mme_next.py", sensory, "--state-dir", state, "--reveal-lyrics")
    cursor = (state / "cursor.json").read_bytes()
    journal = (state / "journal.md").read_bytes()
    if change == "generation":
        obj["meta"]["generation_id"] = "generation-two"
    elif change == "digest":
        obj["lyrics"]["lines"][0]["text"] = "changed words"
    else:
        old = sensory.parent / obj["timbre_layers"]["bundle_dir"]
        new = old.with_name("replacement_timbre")
        old.rename(new)
        obj["timbre_layers"]["bundle_dir"] = new.name
    write_json(sensory, obj)
    failed = cli("mme_next.py", sensory, "--state-dir", state, "--resume", success=False)
    assert failed.returncode != 0
    assert "changed" in failed.stderr and "--reset" in failed.stderr
    assert (state / "cursor.json").read_bytes() == cursor
    assert (state / "journal.md").read_bytes() == journal
    reset = json.loads(cli("mme_next.py", sensory, "--state-dir", state,
                           "--reset", "--reveal-lyrics").stdout)
    assert reset["slice_index"] == 1
    assert reset["lyrics"]["lines"] == [obj["lyrics"]["lines"][0]]
    assert "anonymous_layers" in reset["sensory"]["timbre_layers"]
