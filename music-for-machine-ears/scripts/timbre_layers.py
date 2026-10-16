#!/usr/bin/env python3
"""Optional anonymous timbre-layer preparation for Music for Machine Ears.

This script intentionally runs as a separate process so Demucs and librosa can
live in an isolated optional runtime. Ordinary MME generation never imports
those dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LAYERS = ("A", "B", "C", "D")
SEMANTIC_NAMES = ("vocals", "drums", "bass", "other")
EPS = 1e-10


def optional_dependencies():
    try:
        import librosa
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Timbre layers require an isolated Python runtime containing "
            "demucs, librosa, matplotlib, numpy, and soundfile. Configure "
            "timbre_layers.python or MME_TIMBRE_PYTHON."
        ) from exc
    return librosa, plt, np, sf


def compact(np, value, digits=5):
    return round(float(np.nan_to_num(value)), digits)


def slice_count(duration, seconds):
    total = max(1, math.ceil(duration / seconds))
    tail = duration % seconds
    if total > 1 and 0 < tail < seconds * 0.25:
        total -= 1
    return total


def band_ratios(np, power, frequencies):
    bands = ((20, 250), (250, 2000), (2000, float(frequencies[-1]) + 1))
    total = float(power.sum()) + EPS
    values = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        values.append(compact(np, power[mask].sum() / total))
    return values


def analyze_segment(librosa, np, signal, sample_rate):
    if len(signal) < 2048:
        signal = np.pad(signal, (0, 2048 - len(signal)))
    hop = 512
    magnitude = np.abs(librosa.stft(signal, n_fft=2048, hop_length=hop))
    power = magnitude ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    rms = librosa.feature.rms(S=magnitude, frame_length=2048, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    zero_crossing = librosa.feature.zero_crossing_rate(signal, frame_length=2048, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=signal, sr=sample_rate, hop_length=hop)
    flux = np.sqrt(np.sum(np.maximum(np.diff(magnitude, axis=1), 0.0) ** 2, axis=0))
    harmonic, _ = librosa.effects.hpss(signal)
    harmonicity = np.sum(harmonic ** 2) / (np.sum(signal ** 2) + EPS)
    metrics = {
        "rms": compact(np, np.sqrt(np.mean(signal ** 2))),
        "spectral_centroid_hz": compact(np, np.mean(centroid), 2),
        "spectral_bandwidth_hz": compact(np, np.mean(bandwidth), 2),
        "rolloff85_hz": compact(np, np.mean(rolloff), 2),
        "flatness": compact(np, np.mean(flatness)),
        "zero_crossing_rate": compact(np, np.mean(zero_crossing)),
        "band_energy_ratio_low_mid_high": band_ratios(np, power, frequencies),
        "spectral_flux": compact(np, np.mean(flux) / (np.mean(magnitude) + EPS)),
        "onset_strength": compact(np, np.mean(onset)),
        "change_variability": compact(np, np.std(rms) / (np.mean(rms) + EPS)),
        "harmonicity_proxy": compact(np, harmonicity),
    }
    return metrics, {"rms": rms, "onset": onset, "centroid": centroid}


def correlation(np, first, second):
    length = min(len(first), len(second))
    if length < 3 or np.std(first[:length]) < EPS or np.std(second[:length]) < EPS:
        return 0.0
    return compact(np, np.corrcoef(first[:length], second[:length])[0, 1])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_bundle(
    semantic_dir,
    bundle,
    title,
    source_label,
    slice_seconds,
    canonical_duration,
    expected_slices,
    bundle_id,
):
    librosa, plt, np, sf = optional_dependencies()
    found = {name: semantic_dir / f"{name}.wav" for name in SEMANTIC_NAMES}
    missing = [str(path) for path in found.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Demucs did not produce the expected four stems: {missing}")

    labels = list(LAYERS)
    secrets.SystemRandom().shuffle(labels)
    semantic_to_layer = dict(zip(SEMANTIC_NAMES, labels))
    layer_to_semantic = {layer: semantic for semantic, layer in semantic_to_layer.items()}
    signals = {}
    sample_rate = None
    for semantic, source in found.items():
        layer = semantic_to_layer[semantic]
        signal, sample_rate = librosa.load(source, sr=22050, mono=True)
        signals[layer] = signal.astype(np.float32)

    target_samples = int(round(canonical_duration * sample_rate))
    if target_samples <= 0:
        raise RuntimeError("Canonical MME duration produced no timbre samples.")
    for layer, signal in signals.items():
        if len(signal) < target_samples:
            signals[layer] = np.pad(signal, (0, target_samples - len(signal)))
        else:
            signals[layer] = signal[:target_samples]

    duration = canonical_duration
    total_slices = slice_count(duration, slice_seconds)
    if total_slices != expected_slices:
        raise RuntimeError(
            f"Canonical slice mismatch: expected {expected_slices}, computed {total_slices}."
        )
    series = {layer: [] for layer in LAYERS}
    slice_dir = bundle / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    for index in range(total_slices):
        start = index * slice_seconds
        end = duration if index == total_slices - 1 else min(duration, (index + 1) * slice_seconds)
        low, high = int(start * sample_rate), int(end * sample_rate)
        layer_metrics, frames, energies = {}, {}, {}
        for layer in LAYERS:
            metrics, frame = analyze_segment(librosa, np, signals[layer][low:high], sample_rate)
            layer_metrics[layer] = metrics
            frames[layer] = frame
            energies[layer] = metrics["rms"] ** 2
        energy_sum = sum(energies.values()) + EPS
        for layer in LAYERS:
            layer_metrics[layer]["relative_energy_share"] = compact(np, energies[layer] / energy_sum)
            others = (energy_sum - energies[layer]) / 3
            layer_metrics[layer]["relative_dominance_db"] = compact(
                np, 10 * np.log10((energies[layer] + EPS) / (others + EPS)), 2
            )
            series[layer].append(layer_metrics[layer])
        interactions = []
        for position, first in enumerate(LAYERS):
            for second in LAYERS[position + 1:]:
                interactions.append({
                    "pair": f"{first}-{second}",
                    "level_balance_db": compact(
                        np, 10 * np.log10((energies[first] + EPS) / (energies[second] + EPS)), 2
                    ),
                    "rms_envelope_correlation": correlation(np, frames[first]["rms"], frames[second]["rms"]),
                    "onset_correlation": correlation(np, frames[first]["onset"], frames[second]["onset"]),
                    "centroid_separation_hz": compact(
                        np, abs(np.mean(frames[first]["centroid"]) - np.mean(frames[second]["centroid"])), 2
                    ),
                })
        write_json(slice_dir / f"{index + 1:03d}.json", {
            "slice": index + 1,
            "slice_count": total_slices,
            "start_seconds": compact(np, start, 3),
            "end_seconds": compact(np, end, 3),
            "layers": layer_metrics,
            "interactions": interactions,
        })

    graph_dir = bundle / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    positions = np.arange(1, total_slices + 1)
    plots = (
        ("relative_energy_share", "Relative energy share", "energy_share.png"),
        ("spectral_centroid_hz", "Spectral centroid (Hz)", "centroid.png"),
        ("spectral_bandwidth_hz", "Spectral bandwidth (Hz)", "bandwidth.png"),
        ("flatness", "Spectral flatness", "flatness.png"),
        ("onset_strength", "Onset strength", "onset_strength.png"),
        ("harmonicity_proxy", "Harmonicity proxy", "harmonicity.png"),
    )
    graph_paths = []
    for key, ylabel, filename in plots:
        figure, axes = plt.subplots(figsize=(10, 4))
        for layer in LAYERS:
            axes.plot(positions, [metrics[key] for metrics in series[layer]], label=f"Layer {layer}")
        axes.set(xlabel=f"{slice_seconds:g}-second slice", ylabel=ylabel, title=f"Anonymous layers — {ylabel}")
        axes.grid(alpha=0.25)
        axes.legend(ncol=4)
        figure.tight_layout()
        output = graph_dir / filename
        figure.savefig(output, dpi=140)
        plt.close(figure)
        graph_paths.append(str(Path("graphs") / filename))

    manifest = {
        "format": "mme-timbre-v1",
        "bundle_id": bundle_id,
        "title": title,
        "source_label": source_label,
        "sample_rate": sample_rate,
        "duration_seconds": compact(np, duration, 3),
        "slice_seconds": slice_seconds,
        "slice_count": total_slices,
        "anonymous_layers": list(LAYERS),
        "descriptor_note": "Offline machine descriptors; values are summaries, not human-like hearing.",
    }
    write_json(bundle / "manifest.json", manifest)
    write_json(bundle / "sealed" / "stem_map.json", {
        "sealed_until_summary": True,
        "layer_to_separation_category": layer_to_semantic,
        "warning": "Demucs categories are inferred separation categories, not ground truth or specific instruments.",
    })
    write_json(bundle / "summary.json", {
        "title": title,
        "duration_seconds": manifest["duration_seconds"],
        "slice_count": total_slices,
        "anonymous_graphs": graph_paths,
        "whole_track_layer_means": {
            layer: {
                key: compact(np, np.mean([metrics[key] for metrics in series[layer]]), 3)
                for key in (
                    "relative_energy_share", "spectral_centroid_hz", "spectral_bandwidth_hz",
                    "flatness", "onset_strength", "harmonicity_proxy",
                )
            }
            for layer in LAYERS
        },
    })
    return manifest


def normalize_audio(audio, destination):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Optional timbre-layer mode requires ffmpeg to prepare Demucs input.")
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(audio), "-ar", "44100", "-ac", "2", str(destination)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not prepare Demucs input: {result.stderr[-1000:]}")


def main():
    parser = argparse.ArgumentParser(description="Prepare optional anonymous timbre layers for MME.")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--bundle-name", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--source-label", default="")
    parser.add_argument("--slice-seconds", type=float, default=20.0)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--slice-count", type=int, required=True)
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--keep-temporary-audio", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_download:
        raise RuntimeError(
            "Timbre layers require model-download permission: use --allow-model-download "
            "or config timbre_layers.allow_model_download=true when running the generator."
        )
    print(
        f"Model {args.model}: cache status unknown; Demucs may retrieve/cache model weights.",
        file=sys.stderr,
    )
    if not math.isfinite(args.slice_seconds) or args.slice_seconds <= 0:
        raise SystemExit("--slice-seconds must be a finite number greater than zero.")
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        raise SystemExit("--duration-seconds must be a finite number greater than zero.")
    if args.slice_count < 1:
        raise SystemExit("--slice-count must be at least one.")
    if Path(args.bundle_name).name != args.bundle_name:
        raise SystemExit("--bundle-name must be a single filesystem-safe name.")

    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_bundle = output_dir / args.bundle_name
    if final_bundle.exists():
        raise SystemExit(f"Refusing to replace existing versioned timbre bundle: {final_bundle}")
    bundle = Path(tempfile.mkdtemp(prefix=f".{args.bundle_name}.pending-", dir=output_dir))
    work = Path(tempfile.mkdtemp(prefix=f".{args.slug}-timbre-", dir=output_dir))
    success = False
    try:
        source = work / "source.wav"
        normalize_audio(args.audio.resolve(), source)
        command = [
            sys.executable, "-m", "demucs", "-n", args.model,
            "--device", args.device, "--jobs", str(args.jobs),
            "--out", str(work / "separated"), str(source),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            if args.keep_temporary_audio:
                (work / "preparation.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
            detail = (result.stderr or result.stdout or "unknown error")[-1000:]
            raise RuntimeError(f"Demucs failed with exit code {result.returncode}: {detail}")
        semantic_dir = work / "separated" / args.model / "source"
        manifest = prepare_bundle(
            semantic_dir,
            bundle,
            args.title,
            args.source_label or args.audio.name,
            args.slice_seconds,
            args.duration_seconds,
            args.slice_count,
            final_bundle.name,
        )
        os.replace(bundle, final_bundle)
        success = True
        print(json.dumps({
            "enabled": True,
            "format": manifest["format"],
            "bundle_dir": final_bundle.name,
            "slice_seconds": manifest["slice_seconds"],
            "slice_count": manifest["slice_count"],
            "anonymous_layers": manifest["anonymous_layers"],
            "provenance": (
                "Anonymous descriptors from Demucs-inferred broad separation categories; "
                "not human-like hearing, ground truth, or specific-instrument identification."
            ),
        }, ensure_ascii=False))
    except Exception as exc:
        if args.keep_temporary_audio:
            (work / "failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        raise
    finally:
        if not success:
            shutil.rmtree(bundle, ignore_errors=True)
        if not args.keep_temporary_audio:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as exc:
        raise SystemExit(f"MME timbre layers: {exc}") from None
