# Detect — multi-camera people and activity analytics

**Մանրամասն հայերեն նկարագրություն և տեղադրման ուղեցույց՝
[PROJECT_AND_INSTALLATION_HY.md](PROJECT_AND_INSTALLATION_HY.md)**։

Detect is a local Windows application for Hikvision cameras. It combines live video,
pose and activity recognition, face recognition, trained handgun detection, daily
timelines, clothing search, and a calibrated floor map. Its FastAPI web UI runs at
**http://localhost:8000**.

Repository: https://github.com/gevorgsc/detect-deploy  
Models and SDK: https://github.com/gevorgsc/detect-deploy/releases/tag/migration-v1

## Included

- Complete application source in `camera/`, `recognition/`, and `web/`.
- Installation scripts, pinned dependencies, public downloads and validation tools.
- All migration models and the complete Hikvision SDK in release assets.
- Floor-map geometry, calibration, model toggles and generic example connections.
- Optional model exports and source-GPU engines, kept outside active model paths.

**Enrolled people, face crops, identity embeddings, SQLite databases and history are
excluded. Camera IPs are examples; usernames and passwords are blank.** Enter actual
camera connections locally after installation. The generic AntelopeV2 neural network
is included; none of the source installation's enrolled identity vectors accompany it.

## Install on another PC

Use **64-bit Windows, Python 3.11, an NVIDIA GPU with a compatible driver, and camera
network access**. The original configuration ran on an RTX 5080 with 16 GB VRAM.
Memory fit and throughput on another GPU must be checked there.

Download **Code → Download ZIP** and extract it, or use Git:

```powershell
git clone https://github.com/gevorgsc/detect-deploy.git
cd detect-deploy
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\.venv\Scripts\python.exe scripts\configure_cameras.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

GitHub login and GitHub CLI are not required. The installer downloads public release
assets directly, verifies SHA-256, creates `.venv`, installs CUDA packages and tests
all active models without connecting to cameras. The configuration tool hides
password entry and saves settings locally. `start.ps1` checks the configuration
before making camera login attempts.

Open **http://localhost:8000** after model initialization and staggered camera startup.
Stop with Ctrl+C or:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\stop.ps1
```

## Release groups

| Group | Contents | Default install |
|---|---|---|
| `core` | Three active `.pt` models, AntelopeV2, Hikvision SDK, InsightFace wheel | Yes |
| `config` | Example connections, toggles, branch, floor plan, calibration and zones | Yes; existing local config is preserved |
| `engines` | Source GPU's TensorRT exports, installed under `reference_engines/` | Optional |
| `optional` | YOLOE/YOLO-World, MobileCLIP, CLIP, PAR and alternate ONNX exports | Optional |

`assets-manifest.json` records every archive and member SHA-256, size and destination.
Default installation runs portable `.pt` weights on CUDA. To build TensorRT engines
on the destination GPU, stop the app and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -BuildEngines
```

Source-machine engines are hardware references, never automatically placed in the
active model directory. To retrieve optional archives:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_assets.py --groups engines optional
```

## Local data and checks

`camera/config/cameras.json` and `data/` stay local and are ignored by Git. Empty
databases are created at first startup. The original computer's files were untouched.

```powershell
.\.venv\Scripts\python.exe scripts\configure_cameras.py --check
.\.venv\Scripts\python.exe scripts\doctor.py --models
```

InsightFace installs from its bundled wheel with `--no-deps`; functional dependencies
are pinned explicitly to preserve `onnxruntime-gpu`. `pip check` may still report its
distribution-level CPU `onnxruntime` requirement; actual CUDA sessions are tested.

See [INSTALL_HANDOFF.md](INSTALL_HANDOFF.md) for destination setup with Codex,
[source-environment.json](source-environment.json) for the original environment, and
[VALIDATION.json](VALIDATION.json) for preparation checks. Actual destination GPU and
camera verification remain part of installing on that machine.

