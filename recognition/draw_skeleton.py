"""
draw_skeleton.py — локальный тест отрисовки скелета на вебкамере/видеофайле.

Для просмотра скелета с реальных IP-камер используйте сайт (web/web_server.py) —
там распознавание уже подключено через Hikvision SDK (camera/camera_worker.py).

Модель по умолчанию — yolo11s-pose.pt, лёгкая и быстрая pose-модель семейства YOLO11.

Запуск (из корня проекта):
    python recognition/draw_skeleton.py
    python recognition/draw_skeleton.py --source office.mp4

Выход — клавиша q.
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from skeleton import draw_white_skeleton


def _prefer_engine(pt_path: Path) -> str:
    engine_path = pt_path.with_suffix(".engine")
    return str(engine_path if engine_path.exists() else pt_path)


DEFAULT_MODEL = _prefer_engine(Path(__file__).resolve().parent / "yolo11s-pose.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0=веб-камера или путь к видеофайлу")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="веса pose-модели (yolo11s-pose по умолчанию)")
    ap.add_argument("--device", default="0", help="GPU id, напр. 0; или cpu")
    ap.add_argument("--conf", type=float, default=0.35, help="порог уверенности детекции")
    args = ap.parse_args()

    model = YOLO(args.model)

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Не удалось открыть источник: {source}")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, device=args.device, conf=args.conf, verbose=False)[0]
        if result.keypoints is not None and len(result.keypoints):
            kpts = result.keypoints.data.cpu().numpy()  # [n,17,3]
            draw_white_skeleton(frame, kpts)

        cv2.imshow("skeleton", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
