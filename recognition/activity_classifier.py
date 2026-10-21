"""
activity_classifier.py — математика классификации: поза (sitting/standing) +
активность (working/idle). Без отдельной ML-модели — геометрия поверх
уже посчитанных pose-keypoints и рамок объектов (detect_objects.py).
"""
import cv2
import numpy as np

WHITE = (255, 255, 255)

KP_CONF = 0.30  # ниже этого — точку не используем (закрыта/не уверена)

NOSE = 0
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

FINGER_EXTEND = 0.63       # укорочено на 10% от прежних 0.7 (как в skeleton.py)
FINGER_WIDTH_RATIO = 0.18  # ширина "полосы пальцев" на КАЖДУЮ сторону, в долях длины отрезка;

KNEE_STANDING_ANGLE = 120  # > этого угла в колене — считаем "стоит", иначе "сидит"
FURNITURE_IOU = 0.04        # перекрытие рамки человека с креслом — сигнал "сидит"
# Насколько близко к краю кадра должна подходить рамка, чтобы считать человека
# обрезанным. В долях высоты кадра.
FRAME_EDGE_MARGIN = 0.02
# Запас вокруг рамки кресла, в долях её размера: таз сидящего приходится на сиденье,
# но рамка кресла и рамка человека размечаются каждая по-своему, и точное попадание
# требовать нельзя.
CHAIR_HIP_PAD = 0.15
# «Рамка человека совпала с рамкой кресла» — сидит. Сидящий занимает ровно кресло: и
# по месту, и по размеру. У стоящего рядом рамка вытянута вверх и вниз, у идущего мимо
# она пересекает кресло, но выше и уже. Поэтому требуем ОБА условия сразу: сильное
# перекрытие И близкие размеры. По отдельности каждое ошибается — дальний стоящий
# человек размером с ближнее кресло, а стоящий вплотную к креслу его перекрывает.
CHAIR_SAME_IOU = 0.45
CHAIR_SIZE_LO, CHAIR_SIZE_HI = 0.65, 1.55

FINGER_WIDTH_RATIO = 0.198


def _angle(a, b, c):
    """Угол в точке b (градусы), между отрезками b-a и b-c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _fills_chair(person_box, chair_boxes):
    """Совпадает ли рамка человека с рамкой кресла — по месту И по размеру."""
    px1, py1, px2, py2 = person_box
    pw, ph = max(px2 - px1, 1.0), max(py2 - py1, 1.0)
    for cb in chair_boxes:
        cw = max(cb[2] - cb[0], 1.0)
        ch = max(cb[3] - cb[1], 1.0)
        if not (CHAIR_SIZE_LO <= pw / cw <= CHAIR_SIZE_HI
                and CHAIR_SIZE_LO <= ph / ch <= CHAIR_SIZE_HI):
            continue
        if _iou(person_box, cb) >= CHAIR_SAME_IOU:
            return True
    return False


def _hip_in_chair(kpts, chair_boxes):
    """Лежит ли таз человека внутри рамки кресла.

    Куда более точный признак, чем пересечение рамок: у сидящего таз физически на
    сиденье, а идущий между столами пересекается с креслами постоянно — рамки плоские,
    и человек за креслом накрывает его целиком. Таз к тому же и виден чаще коленей
    (замерено: бёдра проходят порог в 81% кадров, колени — в 31%).
    """
    hips = [i for i in (L_HIP, R_HIP) if kpts[i, 2] > KP_CONF]
    if not hips:
        return False
    hx = float(np.mean([kpts[i, 0] for i in hips]))
    hy = float(np.mean([kpts[i, 1] for i in hips]))
    for cx1, cy1, cx2, cy2 in chair_boxes:
        px = (cx2 - cx1) * CHAIR_HIP_PAD
        py = (cy2 - cy1) * CHAIR_HIP_PAD
        if cx1 - px <= hx <= cx2 + px and cy1 - py <= hy <= cy2 + py:
            return True
    return False


def classify_posture(kpts, person_box, chair_boxes, frame_shape=None):
    """kpts: [17,3]; person_box/chair_boxes: [x1,y1,x2,y2]. Голосование сигналов.

    frame_shape — (высота, ширина) кадра. Нужен, чтобы отличить «ног не видно, потому
    что их закрывает стол» от «ног не видно, потому что человек не влез в кадр».
    Первое — сильный признак того, что человек сидит; второе не говорит ни о чём.
    """
    votes = {"sitting": 0.0, "standing": 0.0}
    knee_angle = None

    knee_conf = min(kpts[L_KNEE, 2], kpts[R_KNEE, 2], kpts[L_ANK, 2], kpts[R_ANK, 2])
    if knee_conf > 0.4:
        hip = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2
        angles = []
        if kpts[L_KNEE, 2] > KP_CONF and kpts[L_ANK, 2] > KP_CONF:
            angles.append(_angle(kpts[L_HIP, :2], kpts[L_KNEE, :2], kpts[L_ANK, :2]))
        if kpts[R_KNEE, 2] > KP_CONF and kpts[R_ANK, 2] > KP_CONF:
            angles.append(_angle(kpts[R_HIP, :2], kpts[R_KNEE, :2], kpts[R_ANK, :2]))
        if angles:
            knee_angle = float(np.mean(angles))
            if knee_angle > KNEE_STANDING_ANGLE:
                votes["standing"] += 0.9
            else:
                votes["sitting"] += 0.9

        sho = (kpts[L_SHO, :2] + kpts[R_SHO, :2]) / 2
        scale = np.linalg.norm(sho - hip)
        if scale > 1e-3:
            hipknee_drop = ((kpts[L_KNEE, 1] - hip[1]) + (kpts[R_KNEE, 1] - hip[1])) / 2 / scale
            if hipknee_drop > 0.7:
                votes["standing"] += 0.8
            else:
                votes["sitting"] += 0.8

    x1, y1, x2, y2 = person_box
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)

    # Обрезан ли человек краем кадра. Если да, форма рамки — это форма видимого куска,
    # а не человека, и судить по ней нельзя.
    cropped = False
    if frame_shape is not None:
        fh, fw = frame_shape[0], frame_shape[1]
        m = fh * FRAME_EDGE_MARGIN
        cropped = (y2 >= fh - m or y1 <= m or x1 <= m or x2 >= fw - m)

    # Пропорция рамки — только для целиком видимого человека. Раньше считалась всегда,
    # и у стоящего с закрытыми ногами рамка выходила почти квадратной, то есть этот
    # сигнал сам голосовал за "сидит" — ровно в том случае, где и так ошибались.
    if not cropped:
        aspect = h / w
        if aspect > 2.2:
            votes["standing"] += 0.5
        elif aspect < 1.6:
            votes["sitting"] += 0.5

    # Таз на сиденье — признак сильнее простого пересечения рамок, поэтому и вес выше.
    # Пересечение оставляем как слабый запасной сигнал: бывает, что таз не разметился.
    if knee_angle is None or knee_angle <= KNEE_STANDING_ANGLE:
        if _hip_in_chair(kpts, chair_boxes):
            votes["sitting"] += 0.9
        elif any(_iou(person_box, cb) > FURNITURE_IOU for cb in chair_boxes):
            votes["sitting"] += 0.3

    # Рамка человека совпала с креслом — сидит. Голос сильный и не зависит от коленей:
    # именно тогда, когда ног не видно, этот признак и нужен больше всего.
    if _fills_chair(person_box, chair_boxes):
        votes["sitting"] += 1.0

    # Ног не видно, хотя человек целиком в кадре — значит их что-то закрывает, и почти
    # всегда это стол. Если же рамка упирается в край кадра, ноги просто не поместились,
    # и вывода отсюда никакого.
    legs_seen = max(kpts[L_KNEE, 2], kpts[R_KNEE, 2],
                    kpts[L_ANK, 2], kpts[R_ANK, 2]) > KP_CONF
    if not legs_seen and not cropped:
        votes["sitting"] += 0.6

    # По умолчанию человек стоит. "Сидит" — это утверждение, и оно должно быть на чём-то
    # основано: перевесить признаками сидения. Раньше по умолчанию было "сидит", и
    # стоящий с закрытыми ногами молча становился сидящим; потом было "unknown", но
    # незнание в кадре бесполезно, а человек на ногах — куда более частый случай.
    return "sitting" if votes["sitting"] > votes["standing"] else "standing"


_ARM_SIDES = ((L_ELB, L_WRI), (R_ELB, R_WRI))


def _finger_rect(wrist, fingertip, width_ratio=FINGER_WIDTH_RATIO):
    """4 угла прямоугольника ("полоса пальцев") вокруг отрезка запястье->кончики,
    расширенного перпендикулярно направлению на width_ratio длины с каждой стороны."""
    direction = fingertip - wrist
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return None
    unit = direction / length
    perp = np.array([-unit[1], unit[0]]) * (length * width_ratio)
    return [wrist - perp, wrist + perp, fingertip + perp, fingertip - perp]


def _box_corners(box):
    x1, y1, x2, y2 = box
    return [np.array([x1, y1]), np.array([x2, y1]), np.array([x2, y2]), np.array([x1, y2])]


def _polygon_axes(poly):
    axes = []
    n = len(poly)
    for i in range(n):
        edge = poly[(i + 1) % n] - poly[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            axes.append(normal / norm)
    return axes


def _polygons_overlap(poly1, poly2):
    """SAT (separating axis theorem) — пересекаются ли два выпуклых многоугольника."""
    for axis in _polygon_axes(poly1) + _polygon_axes(poly2):
        proj1 = [np.dot(p, axis) for p in poly1]
        proj2 = [np.dot(p, axis) for p in poly2]
        if max(proj1) < min(proj2) or max(proj2) < min(proj1):
            return False
    return True


def person_pos_scale(kpts):
    """Примерный центр человека (середина торса) и масштаб (длина торса) —
    для сопоставления 'того же' человека между кадрами без ID/трекинга."""
    sho = (kpts[L_SHO, :2] + kpts[R_SHO, :2]) / 2
    hip = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2
    scale = np.linalg.norm(sho - hip)
    return (sho + hip) / 2, max(float(scale), 30.0)


class ObjectSlotMemory:
    """Запоминает последние известные позиции СТАТИЧНЫХ объектов (мышь/клавиатура —
    лежат на столе и не двигаются сами). Без протухания по таймеру для самой логики
    (рука над местом = working, сколько угодно долго) — только очень долгая гигиеническая
    очистка на случай, если предмет реально унесли навсегда."""
    HYGIENE_TTL = 300.0  # сек без единого повторного обнаружения — забыть совсем
    MATCH_IOU = 0.05

    def __init__(self):
        self.slots = []  # [{"box": np.ndarray, "last_seen": float}]

    def update(self, boxes, t):
        matched = set()
        for box in boxes:
            best_i, best_score = None, 0.0
            for i, slot in enumerate(self.slots):
                if i in matched:
                    continue
                score = _iou(box, slot["box"])
                if score > best_score:
                    best_score, best_i = score, i
            if best_i is not None and best_score > self.MATCH_IOU:
                self.slots[best_i]["box"] = box
                self.slots[best_i]["last_seen"] = t
                matched.add(best_i)
            else:
                self.slots.append({"box": np.asarray(box), "last_seen": t})
        self.slots = [s for s in self.slots if t - s["last_seen"] < self.HYGIENE_TTL]

    def all_boxes(self):
        return [s["box"] for s in self.slots]


class WorkingContactMemory:
    REQUIRED_SECONDS = 0.5
    TRACK_TTL = 2.0
    MATCH_DIST_RATIO = 0.75

    def __init__(self):
        self.tracks = []

    def update(self, person_pos, person_scale, raw_working_wrists, t):
        self._cleanup(t)
        track = self._match_track(person_pos, person_scale, t)
        raw_working_wrists = set(raw_working_wrists)

        for wrist_idx in list(track["starts"]):
            if wrist_idx not in raw_working_wrists:
                del track["starts"][wrist_idx]

        for wrist_idx in raw_working_wrists:
            track["starts"].setdefault(wrist_idx, t)

        track["pos"] = np.asarray(person_pos, dtype=np.float32)
        track["scale"] = float(person_scale)
        track["last_seen"] = t
        return [
            wrist_idx for wrist_idx, start_t in track["starts"].items()
            if t - start_t >= self.REQUIRED_SECONDS
        ]

    def _match_track(self, person_pos, person_scale, t):
        person_pos = np.asarray(person_pos, dtype=np.float32)
        person_scale = float(person_scale)
        best_i, best_dist = None, None
        for i, track in enumerate(self.tracks):
            dist = float(np.linalg.norm(person_pos - track["pos"]))
            max_dist = max(35.0, min(person_scale, track["scale"]) * self.MATCH_DIST_RATIO)
            if dist <= max_dist and (best_dist is None or dist < best_dist):
                best_i, best_dist = i, dist
        if best_i is not None:
            return self.tracks[best_i]

        track = {
            "pos": person_pos,
            "scale": person_scale,
            "last_seen": t,
            "starts": {},
        }
        self.tracks.append(track)
        return track

    def _cleanup(self, t):
        self.tracks = [
            track for track in self.tracks
            if t - track["last_seen"] <= self.TRACK_TTL
        ]


def classify_activity(kpts, person_pos, person_scale, keyboard_mouse_boxes,
                      km_memory: "ObjectSlotMemory | None" = None, t: float = 0.0,
                      phone_wrists=None, working_memory: "WorkingContactMemory | None" = None):
    """Возвращает МНОЖЕСТВО меток активности: working/on_phone/idle."""
    activities = set()
    phone_wrists = set(phone_wrists or [])

    km_all = list(keyboard_mouse_boxes)
    if km_memory is not None:
        km_all += km_memory.all_boxes()

    raw_working_wrists = []
    for elbow_idx, wrist_idx in _ARM_SIDES:
        if kpts[elbow_idx, 2] < KP_CONF or kpts[wrist_idx, 2] < KP_CONF:
            continue
        elbow, wrist = kpts[elbow_idx, :2], kpts[wrist_idx, :2]
        fingers = wrist + (wrist - elbow) * FINGER_EXTEND

        rect = _finger_rect(wrist, fingers)
        if rect is None:
            continue
        if any(_polygons_overlap(rect, _box_corners(box)) for box in km_all):
            raw_working_wrists.append(wrist_idx)

    if working_memory is None:
        working_wrists = raw_working_wrists
    else:
        working_wrists = working_memory.update(person_pos, person_scale, raw_working_wrists, t)

    if working_wrists:
        activities.add("working")
    if any(w not in raw_working_wrists for w in phone_wrists):
        activities.add("on_phone")

    if not activities:
        activities.add("idle")
    return activities


def render_label(posture, activities):
    parts = [posture.upper()]
    if "working" in activities:
        parts.append("working")
    if "on_phone" in activities:
        parts.append("ON PHONE")
    return " | ".join(parts)


def draw_activity_label(frame, person_box, posture, activities):
    """Подпись под рамкой человека, тем же стилем, что и у объектов (белый фон, чёрный текст)."""
    text = render_label(posture, activities)
    x1, y1, x2, _ = [int(v) for v in person_box]
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    frame_h, frame_w = frame.shape[:2]
    lx = int(np.clip((x1 + x2 - tw) / 2, 0, max(frame_w - tw - 8, 0)))
    top = y1 - th - base - 10
    if top < 0:
        top = max(y1 + 4, 0)
    bottom = min(top + th + base + 8, frame_h - 1)
    ly = min(top + th + 4, frame_h - 1)
    cv2.rectangle(frame, (lx, top), (min(lx + tw + 8, frame_w - 1), bottom), WHITE, -1)
    cv2.putText(frame, text, (lx + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
