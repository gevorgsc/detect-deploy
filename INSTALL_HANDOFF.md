# Destination-PC handoff

The owner requested a complete transferable installation on 2026-10-21, with all
necessary code/models/settings, explicitly excluding enrolled people. This is an
installation package, not a copy of the source machine's live databases.

1. Read README.md, `source-environment.json`, and `assets-manifest.json`.
2. Inspect destination OS, Python, NVIDIA GPU/VRAM, driver, free disk and camera LAN.
   The preserved settings were used with an RTX 5080/16 GB. Do not promise the same
   throughput or memory fit on a different GPU before checking.
3. Use Python 3.11 x64 and the isolated `.venv` from `install.ps1`. GitHub CLI must
   authenticate to an account with access to this private repo, unless the release
   ZIPs have already been downloaded manually into `.downloads/`.
4. Run install.ps1. Default models are CUDA PyTorch `.pt`, avoiding incompatible
   source-GPU TensorRT engines. AntelopeV2 is restored into the destination user's
   `.insightface/models/antelopev2`; it is a pretrained generic model, not the user's
   identity database. The bundled InsightFace wheel is installed without dependencies
   to preserve GPU ONNX Runtime. Do not install CPU onnxruntime over it.
5. `scripts/doctor.py --models` verifies model inference and SDK loading without
   logging into cameras or creating face/event databases.
6. Ensure the old PC's Detect instance is stopped before running a duplicate against
   the same cameras. Use POST /api/shutdown or Ctrl+C, not force-kill. The original
   process was left running while the transfer package was prepared.
7. Start with start.ps1, wait for staggered startup and HTTP 200 on localhost:8000,
   then check actual camera frames and face/pose/object/weapon processing. Fresh
   people records may be created from new live camera frames; old enrollments were
   not transferred. Verify the UI and map/calibration persisted from the config.
8. If performance needs TensorRT, stop the app and use install.ps1 -BuildEngines;
   build on the actual destination GPU. Exact source image dimensions and batch
   limits are in build_engines.py. Source engines are references only.

No source-machine people DB, identity vector, cropped face, unknown-face image,
analytics history, capture/calibration image, or diagnostic log was uploaded. The
only `data/` assets permitted are floor-plan images if referenced by configuration.
Full original camera configuration, including credentials, is inside the **private**
config release asset. Do not print those credentials in chat, logs, or screenshots.
The sanitized example config in Git keeps connections/calibration but clears passwords.

Local preparation validation checks Python and PowerShell parsing, dependencies,
release/member SHA-256, a clean extraction, preservation of existing config, exclusion
of people/history files, and import of the app against empty test storage. This does
not substitute for destination GPU/driver and real-camera verification.

