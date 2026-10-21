"""Check installation without connecting to cameras or creating people records."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', action='store_true', help='Also run all active models on a blank frame.')
    parser.add_argument('--files-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'assets-manifest.json').read_text(encoding='utf-8'))
    missing = []
    for asset in manifest['assets']:
        if asset['group'] not in ('core', 'config'):
            continue
        for entry in asset['files']:
            base = ROOT if entry['root'] == 'project' else Path.home() / '.insightface'
            if not (base / entry['target']).is_file():
                missing.append(entry['target'])
    if missing:
        raise SystemExit('Missing files: ' + ', '.join(missing))
    config = json.loads((ROOT / 'camera/config/cameras.json').read_text(encoding='utf-8'))
    for camera in config['cameras']:
        if not all(key in camera for key in ('id', 'ip', 'port', 'username', 'password', 'channel')):
            raise SystemExit('Incomplete camera configuration')
    print(f'Files OK. Cameras: {len(config["cameras"])}. No camera login was attempted.')
    if args.files_only:
        return
    if platform.system() != 'Windows' or struct.calcsize('P') != 8 or sys.version_info[:2] != (3, 11):
        raise SystemExit('Use 64-bit Windows and Python 3.11.')
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. Install the NVIDIA driver and the CUDA PyTorch build using install.ps1.')
    print(f'GPU: {torch.cuda.get_device_name(0)}; torch {torch.__version__}; CUDA {torch.version.cuda}')
    if torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3:
        print('This configuration was used on a 16 GB RTX 5080. Smaller GPUs may require reduced image sizes/batches.')
    sys.path.insert(0, str(ROOT / 'camera'))
    import camera_sdk
    camera_sdk._add_dll_directory(camera_sdk._SDK_DIR)
    camera_sdk._add_dll_directory(camera_sdk._SDK_DIR / 'HCNetSDKCom')
    camera_sdk._load_dll('hpr.dll')
    camera_sdk._load_dll('HCCore.dll')
    camera_sdk._load_dll('PlayCtrl.dll')
    camera_sdk._load_dll('HCNetSDK.dll')
    print('Hikvision DLLs loaded.')
    # Keep these directory handles alive until all ONNX Runtime sessions close.
    import os
    dll_handles = [os.add_dll_directory(str(Path(torch.__file__).parent / 'lib'))]
    import onnxruntime as ort
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise SystemExit('ONNX Runtime has no CUDA provider.')
    if args.models:
        from ultralytics import YOLO
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for name, size in [('yolo11s-pose', (736, 1280)), ('yolo26x', (1088, 1920)), ('weapon', (1088, 1920))]:
            pt = ROOT / 'recognition' / (name + '.pt')
            engine = pt.with_suffix('.engine')
            model = YOLO(str(engine if engine.exists() else pt))
            model.predict(frame, imgsz=size, device=0, half=True, verbose=False)
            print(f'Model OK: {name}')
            del model
            gc.collect()
            torch.cuda.empty_cache()
        from insightface.app import FaceAnalysis
        face = FaceAnalysis(name='antelopev2', allowed_modules=['detection', 'recognition'],
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        face.prepare(ctx_id=0, det_size=(160, 160))
        for name, model in face.models.items():
            if 'CUDAExecutionProvider' not in model.session.get_providers():
                raise SystemExit(f'Face model {name} fell back to CPU.')
        face.get(frame)
        print('Face models OK (CUDA); no face samples or database were created.')
    print('Installation checks passed.')


if __name__ == '__main__':
    main()

