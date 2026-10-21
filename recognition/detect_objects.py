"""
detect_objects.py — распознавание объектов на видео: стол, стул, телефон,
компьютерная мышь, клавиатура. Модель — YOLO26 (детектор, не pose).

Локальный тест на вебкамере/видеофайле (по аналогии с draw_skeleton.py), либо на
реальной камере через Hikvision SDK (--camera-id) — SDK работает в отдельном
процессе (camera_worker.py), YOLO — в этом; так надёжнее (см. camera_worker.py).

Запуск (из корня проекта):
    python recognition/detect_objects.py
    python recognition/detect_objects.py --source office.mp4
    python recognition/detect_objects.py --camera-id cam_01

Выход — клавиша q.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _prefer_engine(pt_path: Path) -> str:
    engine_path = pt_path.with_suffix(".engine")
    return str(engine_path if engine_path.exists() else pt_path)


DEFAULT_MODEL = _prefer_engine(Path(__file__).resolve().parent / "yolo26x.pt")
CAMERA_DIR = Path(__file__).resolve().parent.parent / "camera"

# ultralytics по умолчанию гоняет инференс на 640px — на кадре 3200x1800 с камеры
# это уменьшает мышь/телефон/дальнюю клавиатуру до нескольких пикселей и они пропадают.
# Поднимаем разрешение инференса, чтобы мелкие объекты оставались различимы.
DEFAULT_IMGSZ = 1920

WHITE = (255, 255, 255)

# классы COCO, которые нужно распознавать
OBJECT_CLASSES = {
    56: "chair",
    60: "table",
    62: "monitor",   # COCO "tv" — сюда же модель относит компьютерные мониторы
    63: "laptop",    # ноутбук — тоже экран на столе, и класс у модели крепкий
    64: "mouse",
    66: "keyboard",
    67: "phone",
    73: "notebook",  # COCO "book" — блокноты, тетради и стопки бумаг
}

# Порог уверенности СВОЙ на каждый класс. Общий порог 0.35 годился, пока в наборе были
# стул, мышь и клавиатура — их модель находит уверенно. Мониторы на столах она даёт
# слабее большого экрана на стене (0.61..0.75 против 0.85), а бумаги и тетради COCO
# знает плохо: отдельного класса для них нет, и стопка документов проходит как "book"
# на 0.34..0.52 — ровно по краю прежнего порога, то есть мигая.
OBJECT_CONF = {
    56: 0.35,   # chair
    60: 0.35,   # table
    62: 0.30,   # monitor
    63: 0.40,   # laptop
    64: 0.35,   # mouse
    66: 0.35,   # keyboard
    67: 0.35,   # phone
    73: 0.25,   # notebook / бумаги
}
OBJECT_CONF_MIN = min(OBJECT_CONF.values())


def filter_by_class_conf(boxes, cls_ids, confs):
    """Оставить то, что прошло порог СВОЕГО класса."""
    if not len(cls_ids):
        return boxes, cls_ids, confs
    keep = np.array([c >= OBJECT_CONF.get(int(k), 0.35)
                     for k, c in zip(cls_ids, confs)], dtype=bool)
    return boxes[keep], cls_ids[keep], confs[keep]

PHONE_CLASS_ID = 67


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _center_distance(a, b):
    ac = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    return float(np.linalg.norm(ac - bc))


def _box_corners(box):
    x1, y1, x2, y2 = box
    return [
        np.array([x1, y1], dtype=np.float32),
        np.array([x2, y1], dtype=np.float32),
        np.array([x2, y2], dtype=np.float32),
        np.array([x1, y2], dtype=np.float32),
    ]


def _polygon_axes(poly):
    axes = []
    for i in range(len(poly)):
        edge = poly[(i + 1) % len(poly)] - poly[i]
        normal = np.array([-edge[1], edge[0]], dtype=np.float32)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            axes.append(normal / norm)
    return axes


def _polygons_overlap(poly1, poly2):
    for axis in _polygon_axes(poly1) + _polygon_axes(poly2):
        proj1 = [np.dot(p, axis) for p in poly1]
        proj2 = [np.dot(p, axis) for p in poly2]
        if max(proj1) < min(proj2) or max(proj2) < min(proj1):
            return False
    return True


def _move_box_to_intersect_hand(box, hand_rect):
    box = np.asarray(box, dtype=np.float32)
    hand_rect = np.asarray(hand_rect, dtype=np.float32)
    if _polygons_overlap(_box_corners(box), hand_rect):
        return box.copy()

    w, h = box[2] - box[0], box[3] - box[1]
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
    hand_x1, hand_y1 = np.min(hand_rect, axis=0)
    hand_x2, hand_y2 = np.max(hand_rect, axis=0)
    center[0] = np.clip(center[0], hand_x1, hand_x2)
    center[1] = np.clip(center[1], hand_y1, hand_y2)
    return np.array(
        [center[0] - w / 2, center[1] - h / 2, center[0] + w / 2, center[1] + h / 2],
        dtype=np.float32,
    )


class PhoneBoxTracker:
    """Keeps phone boxes alive and lets them follow a hand after pickup-like contact.

    Кадр сюда приходит ЯРКОСТНОЙ плоскостью (gray, [h,w]), а не BGR: всё, что этому
    трекеру нужно от кадра, — это размер и серые патчи для matchTemplate. У YV12
    плоскость яркости уже готова, разворачивать кадр в цвет и убирать из него цвет
    обратно незачем. Трёхканальный кадр тоже принимается — тогда конвертация
    делается как раньше, чтобы сторонние вызывающие не сломались.
    """

    MAX_MISSED_FRAMES = 15
    MATCH_IOU = 0.05
    MATCH_CENTER_DIST = 90.0
    TRACKED_CONF = 0.20
    ATTACHED_MAX_FRAMES = 50
    BOX_EXPAND_RATIO = 0.10
    PIXEL_CHANGE_LIMIT = 0.60
    TEMPLATE_MIN_SCORE = 0.45
    OCCLUSION_GRACE_SEC = 0.50
    SEARCH_SCALE = 3.5
    MIN_SEARCH_PAD = 55.0
    PIXEL_DIFF_THRESHOLD = 35
    HAND_VISUAL_DELAY_FRAMES = 15

    def __init__(self):
        self.tracks = []
        self.on_phone_now = False
        self.on_phone_boxes = []

    def update(self, boxes, cls_ids, confs, hand_rects=None, gray=None):
        now = time.monotonic()
        self.on_phone_now = False
        self.on_phone_boxes = []
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        cls_ids = np.asarray(cls_ids, dtype=np.float32).reshape(-1)
        confs = np.asarray(confs, dtype=np.float32).reshape(-1)
        hand_rects = [] if hand_rects is None else [np.asarray(rect, dtype=np.float32) for rect in hand_rects]
        for track in self.tracks:
            track["visible"] = False

        phone_indices = [i for i, cls_id in enumerate(cls_ids) if int(cls_id) == PHONE_CLASS_ID]
        matched_tracks = set()

        for idx in phone_indices:
            box = self._expand_box(boxes[idx], gray)
            boxes[idx] = box
            hand_i = self._intersecting_hand(box, hand_rects)
            if hand_i is not None:
                self.on_phone_now = True
                self.on_phone_boxes.append(box.copy())
            best_i, best_score = None, 0.0
            for i, track in enumerate(self.tracks):
                if i in matched_tracks:
                    continue
                iou = _iou(box, track["box"])
                dist = _center_distance(box, track["box"])
                score = iou if iou >= self.MATCH_IOU else (1.0 - dist / self.MATCH_CENTER_DIST)
                if (iou >= self.MATCH_IOU or dist <= self.MATCH_CENTER_DIST) and score > best_score:
                    best_i, best_score = i, score

            if best_i is None:
                self.tracks.append(self._new_track(box, float(confs[idx]), hand_i, hand_rects, gray, now))
                matched_tracks.add(len(self.tracks) - 1)
            else:
                self._set_box(self.tracks[best_i], box, gray, now, reset_bad=True)
                self.tracks[best_i]["conf"] = float(confs[idx])
                self.tracks[best_i]["missed"] = 0
                self.tracks[best_i]["visible"] = True
                self.tracks[best_i]["visual_delay_frames"] = (
                    self.HAND_VISUAL_DELAY_FRAMES if hand_i is not None else 0
                )
                if hand_i is None:
                    self._detach(self.tracks[best_i])
                else:
                    self._remember_hand_contact(self.tracks[best_i], hand_i, hand_rects)
                matched_tracks.add(best_i)

        drop_attached = set()
        if phone_indices:
            has_free_phone_detection = any(
                self._intersecting_hand(boxes[idx], hand_rects) is None
                for idx in phone_indices
            )
            for i, track in enumerate(self.tracks):
                if i in matched_tracks or not track.get("attached", False):
                    continue
                if has_free_phone_detection:
                    drop_attached.add(i)

        for i, track in enumerate(self.tracks):
            if i in drop_attached:
                continue
            if i not in matched_tracks:
                track["missed"] += 1
                self._update_missing_track(track, hand_rects, gray, now)
        self.tracks = [
            track for i, track in enumerate(self.tracks)
            if i not in drop_attached
            and not track.get("lost", False)
            and (not track.get("attached", False) or track.get("attached_frames", 0) <= self.ATTACHED_MAX_FRAMES)
            and (track.get("attached", False)
                 or track["missed"] <= self.MAX_MISSED_FRAMES
                 or track.get("visual_tracking", False)
                 or not self._bad_timed_out(track, now))
        ]

        remembered = [
            track for track in self.tracks
            if track["missed"] > 0
            or ((track.get("attached") or track.get("visual_tracking")) and not track.get("visible", False))
        ]
        if any(track.get("attached", False) for track in self.tracks):
            self.on_phone_now = True
            self.on_phone_boxes.extend(
                track["box"].copy() for track in self.tracks if track.get("attached", False)
            )
        if not remembered:
            return boxes, cls_ids, confs

        remembered_boxes = np.asarray([track["box"] for track in remembered], dtype=np.float32)
        remembered_cls = np.full((len(remembered),), PHONE_CLASS_ID, dtype=np.float32)
        remembered_confs = np.asarray(
            [self.TRACKED_CONF if track.get("attached") or track.get("visual_tracking") else
             max(0.05, self.TRACKED_CONF * (1.0 - min(track["missed"], self.MAX_MISSED_FRAMES) / (self.MAX_MISSED_FRAMES + 1)))
             for track in remembered],
            dtype=np.float32,
        )
        return (
            np.concatenate([boxes, remembered_boxes], axis=0),
            np.concatenate([cls_ids, remembered_cls], axis=0),
            np.concatenate([confs, remembered_confs], axis=0),
        )

    def _new_track(self, box, conf, hand_i, hand_rects, gray=None, now=None):
        now = time.monotonic() if now is None else now
        track = {
            "box": box.copy(),
            "conf": conf,
            "missed": 0,
            "attached": False,
            "attached_frames": 0,
            "pending_hand_i": None,
            "hand_i": None,
            "visible": True,
            "patch": None,
            "bad_since": None,
            "visual_tracking": False,
            "lost": False,
            "last_update_t": now,
            "velocity": np.zeros(2, dtype=np.float32),
            "last_hand_center": None,
            "visual_delay_frames": self.HAND_VISUAL_DELAY_FRAMES if hand_i is not None else 0,
        }
        self._remember_patch(track, gray)
        if hand_i is not None:
            self._remember_hand_contact(track, hand_i, hand_rects)
        return track

    def _expand_box(self, box, gray=None):
        box = np.asarray(box, dtype=np.float32)
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = (box[2] - box[0]) * (1.0 + self.BOX_EXPAND_RATIO)
        h = (box[3] - box[1]) * (1.0 + self.BOX_EXPAND_RATIO)
        expanded = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
        return self._clip_box(expanded, gray)

    def _clip_box(self, box, gray):
        if gray is None:
            return np.asarray(box, dtype=np.float32)
        h, w = gray.shape[:2]
        clipped = np.asarray(box, dtype=np.float32).copy()
        clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, max(w - 1, 0))
        clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, max(h - 1, 0))
        return clipped

    def _crop_patch(self, gray, box):
        if gray is None:
            return None
        box = self._clip_box(box, gray)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        if x2 <= x1 or y2 <= y1:
            return None
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        if patch.ndim != 2:
            return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Срез — вид с шагом по строке. Раньше cvtColor всегда возвращал плотный
        # массив, и дальше по коду (matchTemplate/absdiff) это молча подразумевается.
        return np.ascontiguousarray(patch)

    def _remember_patch(self, track, gray):
        patch = self._crop_patch(gray, track["box"])
        if patch is not None and patch.shape[0] >= 4 and patch.shape[1] >= 4:
            track["patch"] = patch.copy()

    def _center(self, box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)

    def _set_box(self, track, box, gray, now, reset_bad=False, remember_patch=True):
        old_center = self._center(track["box"])
        new_box = self._clip_box(box, gray)
        new_center = self._center(new_box)
        dt = max(now - track.get("last_update_t", now), 1e-3)
        track["velocity"] = (new_center - old_center) / dt
        track["box"] = new_box.copy()
        track["last_update_t"] = now
        if reset_bad:
            track["bad_since"] = None
            track["visual_tracking"] = True
        if remember_patch:
            self._remember_patch(track, gray)

    def _mark_bad(self, track, now):
        if track.get("bad_since") is None:
            track["bad_since"] = now

    def _bad_timed_out(self, track, now):
        bad_since = track.get("bad_since")
        return bad_since is not None and now - bad_since > self.OCCLUSION_GRACE_SEC

    def _pixel_change_ratio(self, old_patch, new_patch):
        if old_patch is None or new_patch is None:
            return None
        if old_patch.shape != new_patch.shape:
            new_patch = cv2.resize(new_patch, (old_patch.shape[1], old_patch.shape[0]))
        diff = cv2.absdiff(old_patch, new_patch)
        return float(np.mean(diff > self.PIXEL_DIFF_THRESHOLD))

    def _smooth_enough(self, track, new_box):
        old_center = self._center(track["box"])
        new_center = self._center(new_box)
        bw = max(track["box"][2] - track["box"][0], 1.0)
        bh = max(track["box"][3] - track["box"][1], 1.0)
        max_jump = max(130.0, np.hypot(bw, bh) * 4.0)
        return float(np.linalg.norm(new_center - old_center)) <= max_jump

    def _visual_track(self, track, gray):
        patch = track.get("patch")
        if gray is None or patch is None:
            return None
        h, w = gray.shape[:2]
        box = track["box"]
        bw = max(box[2] - box[0], 1.0)
        bh = max(box[3] - box[1], 1.0)
        pad_x = max(self.MIN_SEARCH_PAD, bw * self.SEARCH_SCALE)
        pad_y = max(self.MIN_SEARCH_PAD, bh * self.SEARCH_SCALE)
        sx1 = int(max(0, np.floor(box[0] - pad_x)))
        sy1 = int(max(0, np.floor(box[1] - pad_y)))
        sx2 = int(min(w, np.ceil(box[2] + pad_x)))
        sy2 = int(min(h, np.ceil(box[3] + pad_y)))
        search = gray[sy1:sy2, sx1:sx2]
        if search.size == 0 or search.shape[0] < patch.shape[0] or search.shape[1] < patch.shape[1]:
            return None
        search_gray = (np.ascontiguousarray(search) if search.ndim == 2
                       else cv2.cvtColor(search, cv2.COLOR_BGR2GRAY))
        result = cv2.matchTemplate(search_gray, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < self.TEMPLATE_MIN_SCORE:
            return None
        x1 = sx1 + max_loc[0]
        y1 = sy1 + max_loc[1]
        candidate = np.array([x1, y1, x1 + patch.shape[1], y1 + patch.shape[0]], dtype=np.float32)
        new_patch = self._crop_patch(gray, candidate)
        change_ratio = self._pixel_change_ratio(patch, new_patch)
        if change_ratio is None or change_ratio > self.PIXEL_CHANGE_LIMIT:
            return None
        if not self._smooth_enough(track, candidate):
            return None
        return candidate

    def _intersecting_hand(self, box, hand_rects):
        box_poly = _box_corners(box)
        for i, hand_rect in enumerate(hand_rects):
            if _polygons_overlap(hand_rect, box_poly):
                return i
        return None

    def _hand_center(self, hand_rect):
        return np.mean(np.asarray(hand_rect, dtype=np.float32), axis=0)

    def _hand_motion_smooth(self, track, hand_rect):
        center = self._hand_center(hand_rect)
        prev = track.get("last_hand_center")
        if prev is None:
            return True
        return float(np.linalg.norm(center - prev)) <= max(140.0, np.linalg.norm(np.ptp(hand_rect, axis=0)) * 4.0)

    def _move_box_with_hand_delta(self, track, hand_rect):
        center = self._hand_center(hand_rect)
        prev = track.get("last_hand_center")
        if prev is None:
            return _move_box_to_intersect_hand(track["box"], hand_rect)
        delta = center - prev
        moved = track["box"] + np.array([delta[0], delta[1], delta[0], delta[1]], dtype=np.float32)
        if not _polygons_overlap(_box_corners(moved), hand_rect):
            moved = _move_box_to_intersect_hand(moved, hand_rect)
        return moved

    def _is_same_phone_place(self, a, b):
        return _iou(a, b) >= self.MATCH_IOU or _center_distance(a, b) <= self.MATCH_CENTER_DIST

    def _remember_hand_contact(self, track, hand_i, hand_rects):
        track["pending_hand_i"] = hand_i
        track["hand_i"] = hand_i
        track["attached_frames"] = 0
        track["visual_delay_frames"] = self.HAND_VISUAL_DELAY_FRAMES
        if hand_i is not None and hand_i < len(hand_rects):
            track["last_hand_center"] = self._hand_center(hand_rects[hand_i])

    def _attach_to_hand(self, track, hand_i, hand_rects, gray=None, now=None):
        now = time.monotonic() if now is None else now
        track["attached"] = True
        track["attached_frames"] = 1
        track["hand_i"] = hand_i
        track["pending_hand_i"] = None
        track["missed"] = 0
        new_box = self._move_box_with_hand_delta(track, hand_rects[hand_i])
        self._set_box(track, new_box, gray, now, reset_bad=True, remember_patch=False)
        track["last_hand_center"] = self._hand_center(hand_rects[hand_i])

    def _detach(self, track):
        track["attached"] = False
        track["attached_frames"] = 0
        track["pending_hand_i"] = None
        track["hand_i"] = None
        track["last_hand_center"] = None

    def _update_missing_track(self, track, hand_rects, gray=None, now=None):
        now = time.monotonic() if now is None else now
        if track.get("attached"):
            track["attached_frames"] = track.get("attached_frames", 0) + 1
            hand_i = track.get("hand_i")
            if hand_i is not None and hand_i < len(hand_rects):
                new_box = self._move_box_with_hand_delta(track, hand_rects[hand_i])
                if self._hand_motion_smooth(track, hand_rects[hand_i]) and self._smooth_enough(track, new_box):
                    self._set_box(track, new_box, gray, now, reset_bad=True, remember_patch=False)
                    track["last_hand_center"] = self._hand_center(hand_rects[hand_i])
                else:
                    self._mark_bad(track, now)
                    if self._bad_timed_out(track, now):
                        track["lost"] = True
                        return
            else:
                self._mark_bad(track, now)
                if self._bad_timed_out(track, now):
                    track["lost"] = True
                    return
            track["missed"] = 0
            return

        hand_i = track.get("pending_hand_i")
        if hand_i is not None and hand_i < len(hand_rects):
            self._attach_to_hand(track, hand_i, hand_rects, gray, now)
            return

        hand_i = self._intersecting_hand(track["box"], hand_rects)
        if hand_i is not None:
            self._remember_hand_contact(track, hand_i, hand_rects)
            self._attach_to_hand(track, hand_i, hand_rects, gray, now)
            return

        if track["missed"] <= track.get("visual_delay_frames", 0):
            return

        candidate = self._visual_track(track, gray)
        if candidate is not None:
            self._set_box(track, candidate, gray, now, reset_bad=True, remember_patch=True)
            track["missed"] = 0
            return

        self._mark_bad(track, now)
        if self._bad_timed_out(track, now):
            track["visual_tracking"] = False
            track["lost"] = True


class ObjectBoxTracker:
    """Tracks stable non-phone objects so object YOLO can run less often.

    Кадр — яркостная плоскость (gray), по той же причине, что и в PhoneBoxTracker.
    """

    # Что этот трекер вообще ведёт. Класса нет в наборе — детекция молча выбрасывается
    # и до кадра не доходит, сколько её ни находи: именно поэтому добавленные в
    # OBJECT_CLASSES мониторы и бумаги не появлялись в видео.
    TRACK_CLASS_IDS = {56, 60, 62, 63, 64, 66, 73}
    MATCH_IOU = 0.20
    MATCH_CENTER_DIST = 70.0
    TRACKED_CONF = 0.35
    TEMPLATE_MIN_SCORE = 0.42
    PIXEL_CHANGE_LIMIT = 0.65
    PIXEL_DIFF_THRESHOLD = 35
    SEARCH_SCALE = 2.0
    MIN_SEARCH_PAD = 45.0
    TTL_BY_CLASS = {
        56: 180,  # chair
        60: 180,  # table
        62: 180,  # monitor — стоит на месте так же прочно, как стол
        63: 120,  # laptop — переносят чаще монитора, но реже мыши
        64: 45,   # mouse
        66: 120,  # keyboard
        73: 90,   # notebook / бумаги — их двигают, но не каждую минуту
    }
    MIN_CONF_BY_CLASS = {
        56: 0.30,
        60: 0.30,
        62: 0.28,
        63: 0.35,
        64: 0.30,
        66: 0.30,
        73: 0.22,   # COCO знает бумагу плохо, ниже порога её просто не будет
    }
    MIN_AREA_BY_CLASS = {
        56: 1200.0,
        60: 1500.0,
        62: 2000.0,   # монитор мелким не бывает; так отсекаются случайные пятна
        63: 1500.0,
        64: 250.0,
        66: 700.0,
        73: 600.0,
    }
    VISUAL_CLASS_IDS = set()

    def __init__(self):
        self.tracks = []

    def update(self, boxes, cls_ids, confs, gray=None, detector_ran=True):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        cls_ids = np.asarray(cls_ids, dtype=np.float32).reshape(-1)
        confs = np.asarray(confs, dtype=np.float32).reshape(-1)
        for track in self.tracks:
            track["visible"] = False

        matched_tracks = set()
        for idx, cls_id_f in enumerate(cls_ids):
            cls_id = int(cls_id_f)
            if cls_id not in self.TRACK_CLASS_IDS:
                continue
            box = boxes[idx]
            if not self._valid_detection(box, cls_id, float(confs[idx])):
                continue
            best_i, best_score = None, 0.0
            for i, track in enumerate(self.tracks):
                if i in matched_tracks or track["cls_id"] != cls_id:
                    continue
                iou = _iou(box, track["box"])
                dist = _center_distance(box, track["box"])
                score = iou if iou >= self.MATCH_IOU else (1.0 - dist / self.MATCH_CENTER_DIST)
                if (iou >= self.MATCH_IOU or dist <= self.MATCH_CENTER_DIST) and score > best_score:
                    best_i, best_score = i, score

            if best_i is None:
                self.tracks.append(self._new_track(box, cls_id, float(confs[idx]), gray))
                matched_tracks.add(len(self.tracks) - 1)
            else:
                self._set_box(self.tracks[best_i], box, gray, float(confs[idx]))
                matched_tracks.add(best_i)

        for i, track in enumerate(self.tracks):
            if i in matched_tracks:
                continue
            track["missed"] += 1
            candidate = self._visual_track(track, gray)
            if candidate is not None:
                self._set_box(track, candidate, gray, self.TRACKED_CONF)
            elif detector_ran:
                track["detector_missed"] = track.get("detector_missed", 0) + 1

        self.tracks = [
            track for track in self.tracks
            if track["missed"] <= self.TTL_BY_CLASS.get(track["cls_id"], 60)
            and track.get("detector_missed", 0) <= 6
        ]
        if not self.tracks:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        return (
            np.asarray([track["box"] for track in self.tracks], dtype=np.float32),
            np.asarray([track["cls_id"] for track in self.tracks], dtype=np.float32),
            np.asarray([track["conf"] for track in self.tracks], dtype=np.float32),
        )

    def _valid_detection(self, box, cls_id, conf):
        if conf < self.MIN_CONF_BY_CLASS.get(cls_id, 0.45):
            return False
        w = max(float(box[2] - box[0]), 0.0)
        h = max(float(box[3] - box[1]), 0.0)
        return w * h >= self.MIN_AREA_BY_CLASS.get(cls_id, 500.0)

    def _new_track(self, box, cls_id, conf, gray):
        track = {
            "box": np.asarray(box, dtype=np.float32).copy(),
            "cls_id": int(cls_id),
            "conf": float(conf),
            "missed": 0,
            "detector_missed": 0,
            "visible": True,
            "patch": None,
        }
        self._remember_patch(track, gray)
        return track

    def _set_box(self, track, box, gray, conf):
        track["box"] = self._clip_box(box, gray).copy()
        track["conf"] = float(conf)
        track["missed"] = 0
        track["detector_missed"] = 0
        track["visible"] = True
        self._remember_patch(track, gray)

    def _clip_box(self, box, gray):
        if gray is None:
            return np.asarray(box, dtype=np.float32)
        h, w = gray.shape[:2]
        clipped = np.asarray(box, dtype=np.float32).copy()
        clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, max(w - 1, 0))
        clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, max(h - 1, 0))
        return clipped

    def _crop_patch(self, gray, box):
        if gray is None:
            return None
        box = self._clip_box(box, gray)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        if x2 <= x1 or y2 <= y1:
            return None
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        if patch.ndim != 2:
            return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(patch)

    def _remember_patch(self, track, gray):
        patch = self._crop_patch(gray, track["box"])
        if patch is not None and patch.shape[0] >= 5 and patch.shape[1] >= 5:
            track["patch"] = patch.copy()

    def _pixel_change_ratio(self, old_patch, new_patch):
        if old_patch is None or new_patch is None:
            return None
        if old_patch.shape != new_patch.shape:
            new_patch = cv2.resize(new_patch, (old_patch.shape[1], old_patch.shape[0]))
        diff = cv2.absdiff(old_patch, new_patch)
        return float(np.mean(diff > self.PIXEL_DIFF_THRESHOLD))

    def _visual_track(self, track, gray):
        if track["cls_id"] not in self.VISUAL_CLASS_IDS:
            return None
        patch = track.get("patch")
        if gray is None or patch is None:
            return None
        h, w = gray.shape[:2]
        box = track["box"]
        bw = max(box[2] - box[0], 1.0)
        bh = max(box[3] - box[1], 1.0)
        pad_x = max(self.MIN_SEARCH_PAD, bw * self.SEARCH_SCALE)
        pad_y = max(self.MIN_SEARCH_PAD, bh * self.SEARCH_SCALE)
        sx1 = int(max(0, np.floor(box[0] - pad_x)))
        sy1 = int(max(0, np.floor(box[1] - pad_y)))
        sx2 = int(min(w, np.ceil(box[2] + pad_x)))
        sy2 = int(min(h, np.ceil(box[3] + pad_y)))
        search = gray[sy1:sy2, sx1:sx2]
        if search.size == 0 or search.shape[0] < patch.shape[0] or search.shape[1] < patch.shape[1]:
            return None
        search_gray = (np.ascontiguousarray(search) if search.ndim == 2
                       else cv2.cvtColor(search, cv2.COLOR_BGR2GRAY))
        result = cv2.matchTemplate(search_gray, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < self.TEMPLATE_MIN_SCORE:
            return None
        x1 = sx1 + max_loc[0]
        y1 = sy1 + max_loc[1]
        candidate = np.array([x1, y1, x1 + patch.shape[1], y1 + patch.shape[0]], dtype=np.float32)
        new_patch = self._crop_patch(gray, candidate)
        change_ratio = self._pixel_change_ratio(patch, new_patch)
        if change_ratio is None or change_ratio > self.PIXEL_CHANGE_LIMIT:
            return None
        return candidate


def draw_objects(frame, boxes, cls_ids, confs):
    for box, cls_id, conf in zip(boxes, cls_ids, confs):
        label_name = OBJECT_CLASSES.get(int(cls_id))
        if label_name is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 2)
        text = f"{label_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), WHITE, -1)
        cv2.putText(frame, text, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def _run_on_video_source(args, model, class_ids):
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

        result = model(frame, classes=class_ids, device=args.device,
                        conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            draw_objects(frame, boxes, cls_ids, confs)

        cv2.imshow("objects", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def _run_on_sdk_camera(camera_id, args, model, class_ids):
    """SDK-подключение (отдельный процесс, camera_worker.py) -> YOLO здесь же."""
    sys.path.insert(0, str(CAMERA_DIR))
    from camera_worker import run_camera_worker

    with open(CAMERA_DIR / "config" / "cameras.json", encoding="utf-8") as f:
        cameras = json.load(f)["cameras"]
    cam = next((c for c in cameras if c["id"] == camera_id), None)
    if cam is None:
        print(f"Камера '{camera_id}' не найдена в {CAMERA_DIR / 'config' / 'cameras.json'}")
        return

    frame_queue: mp.Queue = mp.Queue(maxsize=4)
    status_queue: mp.Queue = mp.Queue(maxsize=8)
    proc = mp.Process(target=run_camera_worker, args=(cam, frame_queue, status_queue), daemon=True)
    proc.start()

    try:
        while True:
            try:
                _, jpeg = frame_queue.get(timeout=5)
            except Exception:
                continue
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            result = model(frame, classes=class_ids, device=args.device,
                            conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                draw_objects(frame, boxes, cls_ids, confs)

            cv2.imshow("objects", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        proc.terminate()
        cv2.destroyAllWindows()


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0=веб-камера или путь к видеофайлу")
    ap.add_argument("--camera-id", default=None,
                    help="id камеры из camera/config/cameras.json — подключение через Hikvision SDK")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="веса детектора (по умолчанию YOLO26)")
    ap.add_argument("--device", default="0", help="GPU id, напр. 0; или cpu")
    ap.add_argument("--conf", type=float, default=0.35, help="порог уверенности детекции")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                    help="разрешение инференса — выше = мелкие объекты (мышь/телефон) заметнее, но медленнее")
    args = ap.parse_args()

    model = YOLO(args.model)
    class_ids = list(OBJECT_CLASSES.keys())

    if args.camera_id:
        _run_on_sdk_camera(args.camera_id, args, model, class_ids)
    else:
        _run_on_video_source(args, model, class_ids)


if __name__ == "__main__":
    main()
