# Detect — installation on another Windows PC

Hikvision multi-camera analytics with pose and activity recognition, face recognition,
handgun detection, timelines, and a live floor map. This repository contains the
working application's source and deployment tools. Its private release contains the
models, Hikvision SDK, and original camera/floor-map configuration.

**No enrolled people, face crops, identity embeddings, event history, or SQLite
databases are included.** The application creates empty databases on its first run.
The generic AntelopeV2 face-recognition network is included; it contains no enrollment
from this installation.

## Install

Use 64-bit Windows, Python 3.11, a working NVIDIA driver, and network access to the
configured cameras. The source machine runs an RTX 5080 with 16 GB VRAM, PyTorch
2.11.0/CUDA 12.8. Exact GPU capacity requirements on another card are not measured.
The image sizes and inference batch sizes currently target that source configuration.

Install Python 3.11 and GitHub CLI, authenticate to the account with access to this
private repository, and download the source (Git is optional):

```powershell
gh auth login
gh repo clone gevorgsc/detect-deploy
cd detect-deploy
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

`gh repo clone` requires Git. Alternatively, use the repository's **Code → Download
ZIP**, extract it, and run the same `install.ps1` inside the extracted folder. If
GitHub CLI is not installed, download the `core` and `config` ZIP assets from the
private release into `.downloads/` before running the installer.

The installer creates an isolated `.venv`, downloads and checks the release assets,
installs the pinned CUDA packages, and tests the SDK and all active models without
logging into cameras. If Hikvision DLL loading reports a missing runtime dependency,
install the Microsoft Visual C++ 2015–2022 Redistributable (x64).

Open **http://localhost:8000** after starting. Camera login is staggered, so the web
server becomes available after the model initialization and camera startup sequence.

Stop cleanly with Ctrl+C in the server terminal or:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\stop.ps1
```

Avoid force-killing the process: each camera SDK session needs to close normally.
When switching PCs, stop the old instance before starting the new one.

## Models and release groups

`assets-manifest.json` lists each asset, SHA-256, member, destination, and size.

| Group | Contents | Installed by default |
|---|---|---|
| `core` | `yolo11s-pose.pt`, `yolo26x.pt`, `weapon.pt`; AntelopeV2 ONNX networks; complete Hikvision SDK; InsightFace wheel | Yes |
| `config` | Original camera connections, model toggles, branch, floor plan, calibration and zones | Yes, existing local config is preserved |
| `engines` | Source GPU's active FP16/INT8 engines | No; saved to `reference_engines/` only |
| `optional` | YOLOE/YOLO-World, MobileCLIP, CLIP, PAR and alternate ONNX exports | No; archived alternatives, unused in the default runtime |

The source code selects `.engine` files automatically when present beside `.pt`.
**Source-machine engines are deliberately installed in `reference_engines/`, never
as active engines on the new PC.** The default install runs portable `.pt` weights
on CUDA. Performance may differ from the source machine's optimized TensorRT runtime.

To build and test fresh TensorRT engines on the destination GPU:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -BuildEngines
```

This is optional and can take time and extra GPU memory. Stop Detect first.
`scripts/build_engines.py` exports the actual rectangular input sizes and batch limits
used by `web/web_server.py`. It validates each engine before placing it in
`recognition/`. If a destination card cannot run these sizes, adjust the runtime
sizes/batches and export settings together. For the optional YOLOE backend, the
source TensorRT engine also needs rebuilding for a different GPU; the default backend
uses the included trained `weapon.pt`.

Fetch every archived alternative, without enabling them:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_assets.py --groups engines optional
```

## Configuration and storage

The original runtime configuration is transferred through the private release,
outside Git history. `camera/config/cameras.example.json` has blank passwords.
`camera/config/cameras.json` is restored only if it does not already exist. The
application's camera and floor-map UI can update it normally afterward.

`data/analytics.sqlite3`, `data/faces.sqlite3`, crops, and unknown-face records are
created locally. Existing data on the original computer is untouched. The original
`data/`, camera captures, experiments with captured people, logs, and unrelated
publication files were excluded from this transfer.

To check an installation without starting the server or connecting to cameras:

```powershell
.\.venv\Scripts\python.exe scripts\doctor.py --models
```

InsightFace is installed from the bundled wheel with `--no-deps`; its actual runtime
dependencies are explicitly listed in `requirements.txt`. This avoids installing
the CPU `onnxruntime` distribution over `onnxruntime-gpu`. Consequently, `pip check`
may report that InsightFace's distribution-level `onnxruntime` requirement is
missing; the installer verifies the actual CUDA sessions instead. The source
machine also had two OpenCV distributions; the new install uses one matching the
source's actual imported version, OpenCV 4.11.

## Continue setup with Codex

On the destination computer, open the downloaded repository folder and tell Codex:
«Տեղադրիր այս repo-ի ծրագիրը՝ README.md-ի և INSTALL_HANDOFF.md-ի համաձայն»։
The repo includes the source environment, model manifests, installation validation,
and a handoff explaining which steps remain specific to the destination PC.

