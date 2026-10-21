# Included third-party components

This migration preserves the models and SDK already used by the owner.
Their respective upstream licenses and usage terms continue to apply; this package
does not grant a new license to those third-party artifacts.

- Hikvision HCNetSDK/PlayCtrl x64: copied from the working `camera/SDK/lib` folder.
- Ultralytics YOLO/YOLOE/YOLO-World: `.pt` weights, exports and runtime dependencies.
- `weapon.pt`: the source installation's trained handgun detector.
- InsightFace 1.0.1: bundled unchanged PyPI wheel, including its package metadata.
- InsightFace AntelopeV2: the source user's pretrained model pack. These are generic
  network weights; none of the installation's enrolled identity embeddings accompany them.
- Optional MobileCLIP, OpenAI CLIP and PAR models: archived source alternatives.
- PyTorch, CUDA runtime packages, ONNX Runtime and TensorRT: obtained separately by
  install.ps1 from their upstream package sources.

