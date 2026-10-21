# Destination-PC handoff

The owner requested a complete transferable Detect installation, excluding enrolled
people, then explicitly requested a public repository. The public release contains
sanitized example connections. Enter actual camera credentials locally; the source
computer's configuration was not changed.

1. Read PROJECT_AND_INSTALLATION_HY.md, README.md, source-environment.json and assets-manifest.json.
2. Inspect destination Windows/Python, NVIDIA GPU/VRAM, driver, disk space and camera
   network. Source sizes and batches ran on an RTX 5080/16 GB; check the new GPU's fit.
3. Use Python 3.11 x64 and install.ps1. Public assets download through Python's standard
   library. No GitHub login or GitHub CLI is needed. Manually downloaded core/config
   ZIPs can also be placed in .downloads/.
4. Default models use CUDA PyTorch .pt. AntelopeV2 goes into the destination user's
   .insightface/models/antelopev2. Source-GPU engines stay outside active paths.
   InsightFace installs without dependencies to preserve ONNX Runtime GPU; do not
   replace it with CPU onnxruntime to satisfy pip check's metadata warning.
5. Run .venv\Scripts\python.exe scripts\configure_cameras.py. It prompts for IPv4,
   SDK port, channel, username and hidden password, and saves them in ignored local
   config. Keep camera IDs stable for floor-map calibration. Public defaults are
   documentation IPs and blank credentials; start.ps1 refuses those before login.
6. Run doctor.py --models. It loads DLLs and runs all active networks without camera
   login. This does not prove simultaneous live batch memory fit or video throughput.
7. Stop the old instance before using the same cameras on the new PC. Use Ctrl+C or
   POST /api/shutdown. The source instance was left running during package preparation.
8. Start with start.ps1. Verify HTTP 200, real frames, statuses, detectors, new face
   records and map calibration. Enrollments/history were not transferred. New records
   may appear from live frames. Recalibrate if camera placement changed.
9. If necessary, stop the app and run install.ps1 -BuildEngines on the target GPU.
   Change both runtime and exporter settings together if reducing sizes/batches.

No source people DB, identity vector, crop, unknown-face image, history, calibration
capture or diagnostic log is included. Local credentials and runtime data are ignored
by Git. The original private config release asset was removed and replaced with a
sanitized archive before the repository became public.

Preparation checks cover code parsing, dependency resolution, archive/member hashes,
isolated app import and extraction, blank people stores, config preservation, public
downloads, and rejection of unconfigured cameras. Test destination hardware, driver,
live video and camera access on that PC.

