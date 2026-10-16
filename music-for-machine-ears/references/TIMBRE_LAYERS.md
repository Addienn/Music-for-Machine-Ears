# Optional timbre layers

Add anonymous A-D texture descriptors and pair interactions to the existing
MME listening sequence. Ordinary MME stays available without Demucs, model
weights, or a GPU. Layers are **off by default**.

## Contents

- [Before you start](#before-you-start)
- [Install once](#install-once)
- [Check and enable](#check-and-enable)
- [Listen and manage files](#listen-and-manage-files)
- [Troubleshooting](#troubleshooting)
- [What the descriptors mean](#what-the-descriptors-mean)

## Before you start

Importing the `.skill` file does not install executable dependencies. You need
a host that can run Python, access the source audio, and persist local files.

Layered preparation additionally needs:

- A **separate Python environment**, containing Demucs, PyTorch and Demucs's
  dependencies, librosa, NumPy, SciPy, Matplotlib, and SoundFile.
- **FFmpeg**, available on PATH. It is a separate executable, not a Python
  package. Check with `ffmpeg -version`. If missing, choose an installation
  appropriate for your OS from [FFmpeg's download page](https://ffmpeg.org/download.html).
- **Model weights**. Installing Python packages and retrieving the model are
  separate steps. Demucs manages its model cache and may access remote services
  during retrieval. Do not assume a warm cache means no network access.
- **Disk space and memory** for the environment, weights, temporary audio, and
  model inference. CPU works; a GPU is not required. Full songs can take minutes
  or longer depending on hardware, length, and model. No universal minimum RAM
  or processing-time guarantee has been established.

One measured Windows x64 / Python 3.12 CPU installation used **962.7 MB** of
logical runtime files, plus **84.0 MB** for cached HTDemucs weights (about
**1.05 GB combined**). This is an observed example, **not a minimum or download
size**. It excludes the base Python installation, FFmpeg, package-manager
caches, temporary audio, and generated results. Different builds, particularly
GPU builds, can be much larger. The measured packages were Demucs 4.1.0,
PyTorch 2.14.1+cpu, and librosa 1.0.0; these are not required version pins.

Audio processing runs locally, without a paid processing API. Dependency and
model downloads require network access. Keep private source audio and stems
local; only share outputs you intend to share.

**For agents:** explain these costs and obtain permission before package
installation or model retrieval. Permission to enable layers is not blanket
permission for downloads. Never install into a user's global Python or project
environment without agreement.

## Install once

Run these commands **from the extracted `music-for-machine-ears` folder** after
agreeing to the downloads. Python 3.11 or 3.12 is a conservative starting point
for optional ML dependency compatibility; do not assume every newer Python has
matching wheels.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv-timbre
.\.venv-timbre\Scripts\python.exe -m pip install demucs librosa numpy scipy matplotlib soundfile
```

Linux/macOS:

```bash
python3.12 -m venv .venv-timbre
.venv-timbre/bin/python -m pip install demucs librosa numpy scipy matplotlib soundfile
```

Substitute an available compatible Python if necessary. No activation script
is required. On distributions without `venv` support, install the matching OS
Python-venv package with permission first. For a CPU-specific or GPU-specific
PyTorch build, follow [PyTorch's installer selector](https://pytorch.org/get-started/locally/)
inside this environment before installing Demucs. Do not guess CUDA versions.

The `.venv-timbre` directory is machine-local and must not be included in a
redistributed skill archive. Alternatively create the environment elsewhere
and set `timbre_layers.python` to its interpreter in `config.json`, or set
`MME_TIMBRE_PYTHON`. Paths containing separators in the config are relative to
that config file; an absolute interpreter path is least ambiguous.

## Check and enable

Use the ordinary MME Python for these commands. The checker is standard-library
only; it does not install packages, load model weights, or download them:

```bash
python scripts/check_environment.py --timbre
```

Runtime selection is: config `timbre_layers.python`, then
`MME_TIMBRE_PYTHON`, then the skill's `.venv-timbre` interpreter. There is no
host-specific path and no fallback to the ordinary interpreter.

Once the environment is ready and model retrieval has been approved:

```bash
python scripts/generate-mme.py --audio "song.wav" --out_dir "out" --timbre-layers --allow-model-download
```

`--allow-model-download` grants permission for Demucs-managed retrieval, which
reuses cached files where supported. It is deliberately conservative: permission
is required even if weights might already be cached. Cache locations and
retrieval behavior vary by Demucs version. The checker does not call
`get_model()` as a supposedly read-only cache test.

For persistent opt-in, edit these values under `timbre_layers` in `config.json`:

```json
{
  "enabled": true,
  "allow_model_download": true
}
```

These are two fields within the existing configuration, not a replacement for
the whole file. `enabled` controls layers; `allow_model_download` records ongoing
permission for retrieval. Leave both false to retain the default. You can also
keep retrieval permission false and supply the CLI flag only on approved runs.

`--config path/to/config.json` selects another configuration.
`--no-timbre-layers` explicitly selects ordinary generation even if layers are
enabled in the config. Missing prerequisites in requested layered mode fail
clearly rather than silently dropping the layers.

## Listen and manage files

Use `scripts/mme_next.py` normally. It adds anonymous A-D data to each slice;
only the final summary reveals the model's category map and six extra graphs.
All slices still require a private note before advancing. Do not open the
bundle's `sealed/` directory or full-track charts ahead of that reveal.

- Keep the sensory JSON and its versioned timbre directory together when moving
  a prepared song. Preparation defaults to 20-second slices; the reader must
  use the same slice size.
- Temporary converted audio and named stems are removed on normal success and
  handled failure with `cleanup_temporary_audio: true`. The original supplied
  source file is **never deleted**. A force-killed process can leave temporary
  directories; inspect these only after confirming no generation is running.
- Runtime packages and model caches remain for reuse. JSON, graphs, sealed
  category maps, and private journals remain until you delete them.
- Generation uses a per-slug lock and publishes versioned bundles. A failed
  replacement leaves the old sensory JSON and layer bundle available. Ordinary
  graph files are not versioned; use a fresh output directory when preserving
  all artifacts from an earlier run matters.
- Successful regeneration removes the previously referenced, validated layer
  bundle. A listening cursor detects a changed generation and requires an
  explicit reset rather than mixing old impressions with a new A-D mapping.

No lyric extraction is added here. A surrounding lyric-reveal workflow can
continue independently after listening.

## Troubleshooting

| Symptom | Next step |
|---|---|
| Missing optional interpreter | Create `.venv-timbre` or configure its actual Python executable, not the environment directory. |
| Missing/broken imports | Run the checker; install or repair packages in that interpreter only. Check compatible Python/PyTorch builds. |
| FFmpeg not found | Install FFmpeg with permission, reopen the shell if PATH changed, and run `ffmpeg -version`. |
| Model retrieval not approved | Explain downloads/cache costs; use the explicit permission flag only after agreement. |
| Model retrieval fails | Check network access and Demucs's reported error. Do not silently switch models or disable layers. |
| Processing is slow | CPU separation can be slow. Keep one job, monitor the existing process, and avoid duplicate runs. |
| Out of memory | Close competing jobs or use a machine with more memory; ordinary MME remains available by explicit choice. |
| Generation lock exists | Check whether the original job is alive. Remove only a confirmed stale lock; never bypass an active job. |
| Reader reports changed generation | Use the original prepared files or explicitly reset to listen to the replacement. |

## What the descriptors mean

Per-layer descriptors include RMS, spectral centroid/bandwidth/rolloff,
flatness, zero crossings, frequency-band energy ratios, spectral change,
onset strength, variability, harmonicity proxy, and relative energy share.
Pair data describe envelope/onset correlation and spectral/level separation.

The A-D mapping is randomized for each preparation. At completion, Demucs's
classes (`vocals`, `drums`, `bass`, `other`) are revealed. They are **inferred
separation categories**, with possible leakage/artifacts, not verified source
identity. In particular, `other` does not identify individual instruments.

Energy shares compare separated stems, not perceived loudness or exact shares
of the original mix. Whole-track means average slices, including the shorter
tail, rather than measuring duration-weighted total energy. Harmonicity is an
HPSS-based proxy, not a singer or instrument detector. Near-silent layers can
produce misleading spectral statistics; interpret them alongside RMS. Slice
summaries cannot locate intra-slice transitions to exact timestamps.
