"""
skeleton.py — отрисовка скелета человека белым цветом поверх кадра (BGR, OpenCV).
Общая логика для сайта (camera_worker.py) и локального теста (draw_skeleton.py).
"""
import cv2
import numpy as np

WHITE = (255, 255, 255)
KP_CONF = 0.30  # ниже этого — точку не рисуем (закрыта/не уверена)
FINGER_EXTEND = 0.63  # насколько удлинять линию руки за запястье (доля длины локоть-запястье);
# синхронизировано с activity_classifier.py — тот же отрезок используется для классификации
FINGER_WIDTH_RATIO = 0.18  # ширина полосы пальцев на КАЖДУЮ сторону, в долях длины отрезка;
# синхронизировано с activity_classifier.py::_finger_rect — рисуем именно ту зону,
# по которой реально проверяется пересечение с клавиатурой/мышью/телефоном

# 17 связей скелета COCO (0-индексация)
FINGER_WIDTH_RATIO = 0.198

SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]

# (локоть, запястье) для каждой руки — чтобы удлинить линию до пальцев
ARM_ENDS = [(7, 9), (8, 10)]


def draw_white_skeleton(frame, keypoints):
    """keypoints: [n_people, 17, 3] (x, y, conf). Рисует поверх frame на месте."""
    for kps in keypoints:
        for i, j in SKELETON:
            if kps[i, 2] < KP_CONF or kps[j, 2] < KP_CONF:
                continue
            p1 = (int(kps[i, 0]), int(kps[i, 1]))
            p2 = (int(kps[j, 0]), int(kps[j, 1]))
            cv2.line(frame, p1, p2, WHITE, 2)
        for x, y, c in kps:
            if c < KP_CONF:
                continue
            cv2.circle(frame, (int(x), int(y)), 4, WHITE, -1)

        # удлиняем руку от запястья в сторону пальцев (направление локоть->запястье),
        # без отдельной модели — просто продолжаем ту же линию на FINGER_EXTEND длины,
        # и рисуем не тонкую линию, а расширенный прямоугольник ("полоса пальцев") —
        # именно эта зона (не линия) используется в activity_classifier.py для
        # проверки пересечения с телефоном, поэтому она должна быть видна как есть
        for elbow_idx, wrist_idx in ARM_ENDS:
            if kps[elbow_idx, 2] < KP_CONF or kps[wrist_idx, 2] < KP_CONF:
                continue
            elbow = kps[elbow_idx, :2]
            wrist = kps[wrist_idx, :2]
            fingers = wrist + (wrist - elbow) * FINGER_EXTEND
            rect = _finger_rect(wrist, fingers)
            if rect is None:
                continue
            pts = np.array([[int(p[0]), int(p[1])] for p in rect], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=WHITE, thickness=2)


def _finger_rect(wrist, fingertip, width_ratio=FINGER_WIDTH_RATIO):
    """4 угла прямоугольника вокруг отрезка запястье->кончики пальцев, расширенного
    перпендикулярно направлению на width_ratio длины с каждой стороны. Та же геометрия,
    что и в activity_classifier.py::_finger_rect — тут только для отрисовки."""
    direction = fingertip - wrist
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return None
    unit = direction / length
    perp = np.array([-unit[1], unit[0]]) * (length * width_ratio)
    return [wrist - perp, wrist + perp, fingertip + perp, fingertip - perp]
