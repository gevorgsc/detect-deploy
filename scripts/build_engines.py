"""Build TensorRT engines on the destination GPU; never reuse another GPU's cache."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import torch
    import numpy as np
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise SystemExit('A working NVIDIA CUDA GPU is required.')
    config = json.loads((ROOT / 'camera/config/cameras.json').read_text(encoding='utf-8'))
    jobs = [('yolo11s-pose', (736, 1280), 4, True),
            ('yolo26x', (1088, 1920), 1, False),
            ('weapon', (1088, 1920), max(2, len(config['cameras'])), True)]
    (ROOT / '.cache').mkdir(exist_ok=True)
    for name, size, batch, dynamic in jobs:
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as folder:
            source = Path(folder) / (name + '.pt')
            shutil.copy2(ROOT / 'recognition' / source.name, source)
            model = YOLO(str(source))
            exported = Path(model.export(format='engine', device=0, half=True,
                                        imgsz=size, batch=batch, dynamic=dynamic, workspace=4))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            test = YOLO(str(exported))
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            test.predict([frame] * min(batch, 2), imgsz=size, device=0, half=True, verbose=False)
            del test
            gc.collect()
            torch.cuda.empty_cache()
            shutil.copy2(exported, ROOT / 'recognition' / exported.name)
            print(f'Built and verified: {exported.name}', flush=True)
    import tensorrt
    (ROOT / 'recognition/engine-build.json').write_text(json.dumps({
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'tensorrt': tensorrt.__version__,
        'jobs': jobs,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

