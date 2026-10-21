"""План этажа: где человек НА САМОМ ДЕЛЕ, а не какая камера его видит.

Маршрут по камерам врал там, где две камеры смотрят на одно место: имя живёт ровно
в одном месте, оно перескакивает к той камере, что лучше видит лицо, и в записи
получалось «cam_03 -> cam_04 -> cam_03» по нулю минут, хотя человек не сдвинулся.
Камера — это точка зрения, а не место.

Здесь каждая камера привязана к плану гомографией: оператор указывает 4+ пары точек
«на кадре — на плане» (пол, углы столов, дверные проёмы), и точка ног на кадре
переводится в точку на плане. На плане нарисованы зоны (стол, кухня, коридор), и
маршрут дня — это смена зон человека, а не смена камер. Две камеры, видящие человека
у одного стола, дают одну и ту же зону и один отрезок.

Координаты везде нормированы 0..1: на кадре — к размеру кадра, на плане — к размеру
картинки плана. Так калибровка переживает смену разрешения потока, а картинку плана
можно заменить на такую же в другом размере.

План человек рисует здесь же, а не приносит картинкой: нарисованное — это вектор со
смыслом («это стол», «это стена»), и по нему видно, что искать в кадре, тогда как
картинку пришлось бы разбирать обратно в объекты. Рисует приблизительно и без метров:
масштаб съедает гомография, а кривизну чертежа исправляет человек — потом, в кадре,
перетаскивая те же самые точки на их настоящее место.
"""

import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from activity_classifier import KP_CONF

PROJECT_DIR = Path(__file__).resolve().parent.parent
FLOORPLAN_DIR = PROJECT_DIR / "data" / "floorplans"
FLOORPLAN_DIR.mkdir(parents=True, exist_ok=True)

KP_L_ANK, KP_R_ANK = 15, 16
MIN_CALIB_POINTS = 4
# Дольше этого точка на плане считается устаревшей и на живой карте не показывается:
# трек уже мог уйти, а последняя запись о нём — нет.
LIVE_TTL = 3.0

# --- нарисованный план ----------------------------------------------------
# Что рисуем и сколько точек в этом имеет смысл. Стена и дверь — незамкнутые,
# комната и мебель — замкнутые: комната даёт зону, мебель нужна только для того,
# чтобы было что совмещать с кадром.
PLAN_KINDS = (("walls", 2, False), ("rooms", 3, True),
              ("furniture", 3, True), ("doors", 2, False))
PLAN_ASPECT = 1.6
# Высоту камеры и наклон мы не спрашиваем — вместо них берём, сколько пола камера
# видит перед собой. Дальний край — там, где взгляд упирается в край листа: обзорная
# камера в помещении видит его до стены, а не на фиксированную долю плана.
GUESS_FOV_DEG = 90.0
GUESS_NEAR_FRAC = 0.15          # ближний край — доля от дальнего
GUESS_FAR_MIN, GUESS_FAR_MAX = 0.2, 1.5

# --- след человека --------------------------------------------------------
# Точку в базу кладём не чаще раза в секунду и только если человек заметно
# сдвинулся; стоящий на месте отмечается редким биением, чтобы «стоял» отличалось
# от «пропал». Одного порога по движению мало: точка ног дрожит на каждом кадре
# заметно больше TRAIL_MIN_MOVE, и без паузы выходило по десять строк в секунду
# на человека. Секунды хватает — на плане идущий человек за секунду сдвигается
# на шаг, а отмотка дня по шагам и нужна.
TRAIL_MIN_GAP = 1.0
TRAIL_MIN_MOVE = 0.004
TRAIL_HEARTBEAT = 15.0
TRAIL_FLUSH_SEC = 3.0
TRAIL_MAX_BUF = 200


def foot_point(kpts, box, frame_shape) -> tuple[float, float]:
    """Где человек стоит — нормированная точка на кадре.

    Середина между лодыжками, когда обе видны: это и есть точка касания пола. Иначе
    низ рамки по центру — за столом ноги закрыты, и низ рамки лежит на краю стола или
    на коленях; для зон такой точности хватает, зона — это метры, а не сантиметры.
    """
    h, w = frame_shape[:2]
    la, ra = kpts[KP_L_ANK], kpts[KP_R_ANK]
    if la[2] >= KP_CONF and ra[2] >= KP_CONF:
        x = (float(la[0]) + float(ra[0])) / 2.0
        y = (float(la[1]) + float(ra[1])) / 2.0
    else:
        x = (float(box[0]) + float(box[2])) / 2.0
        y = float(box[3])
    return x / float(w), y / float(h)


def _point_in_polygon(u: float, v: float, poly: list) -> bool:
    """Обычный луч: чётное число пересечений — снаружи."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > v) != (yj > v):
            x_cross = (xj - xi) * (v - yi) / (yj - yi + 1e-12) + xi
            if u < x_cross:
                inside = not inside
        j = i
    return inside


def solve_homography(points: list) -> tuple[np.ndarray | None, list[float]]:
    """Гомография кадр -> план по парам точек плюс ошибка на каждой паре.

    Ошибка возвращается оператору: одна неверно поставленная точка портит всё
    преобразование, и по списку сразу видно, какую переставить. Ошибка в долях плана;
    0.02 на плане в 1000 px — это 20 px.
    """
    if len(points) < MIN_CALIB_POINTS:
        return None, []
    src = np.array([p["img"] for p in points], dtype=np.float64)
    dst = np.array([p["map"] for p in points], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    # Точки на одной прямой: OpenCV возвращает не None, а вырожденную матрицу,
    # которая всё переводит в одну точку.
    if H is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        return None, []
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errs = [float(np.linalg.norm(proj[i] - dst[i])) for i in range(len(points))]
    return H, errs


def _clamp01(v) -> float:
    return min(1.0, max(0.0, float(v)))


def _poly_area(poly) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def poly_centre(poly) -> tuple[float, float]:
    """Середина фигуры — тем же способом, что подпись зоны на плане."""
    n = max(1, len(poly))
    return (sum(float(p[0]) for p in poly) / n, sum(float(p[1]) for p in poly) / n)


def sanitize_plan(d: dict) -> dict:
    """Чертёж из браузера: только то, что мы понимаем, и всё в 0..1."""
    aspect = min(4.0, max(0.25, float(d.get("aspect") or PLAN_ASPECT)))
    out: dict = {"aspect": aspect, "cams": {}}
    for kind, min_pts, _closed in PLAN_KINDS:
        items = []
        for it in (d.get(kind) or []):
            pts = [[_clamp01(p[0]), _clamp01(p[1])] for p in (it.get("points") or [])]
            if len(pts) < min_pts:
                continue
            row = {"id": str(it.get("id") or f"{kind[0]}{len(items) + 1}")[:40],
                   "points": pts}
            name = str(it.get("name") or "").strip()[:60]
            if name:
                row["name"] = name
            items.append(row)
        out[kind] = items
    for cid, c in (d.get("cams") or {}).items():
        try:
            row = {"pos": [_clamp01(c["pos"][0]), _clamp01(c["pos"][1])],
                   "dir": float(c.get("dir") or 0.0) % 360.0}
            # Угол и дальность человек правит на самом секторе. Не задал — считаем
            # обзорной камерой, которая видит до края листа.
            if c.get("fov") is not None:
                row["fov"] = min(170.0, max(10.0, float(c["fov"])))
            if c.get("far") is not None:
                row["far"] = min(GUESS_FAR_MAX, max(GUESS_FAR_MIN, float(c["far"])))
            out["cams"][str(cid)[:40]] = row
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


def plan_is_empty(plan: dict) -> bool:
    return not any((plan or {}).get(k) for k, _m, _c in PLAN_KINDS)


def zones_from_plan(plan: dict) -> list:
    """Комнаты чертежа — это и есть зоны: человек назвал их, когда рисовал.

    Меньшую кладём раньше: zone_at берёт первое совпадение, и стол, обведённый
    внутри зала, должен выигрывать у зала, а не наоборот.
    """
    rooms = [r for r in ((plan or {}).get("rooms") or [])
             if r.get("name") and len(r["points"]) >= 3]
    rooms.sort(key=lambda r: _poly_area(r["points"]))
    return [{"id": r["id"], "name": r["name"],
             "points": [[p[0], p[1]] for p in r["points"]]} for r in rooms]


def plan_shapes(plan: dict) -> tuple[list, list]:
    """Каркас для наложения на кадр: фигуры и их точки с постоянными id.

    Совпавшие углы — это одна точка, а не две: контур стены замыкается обратно на
    первую вершину, а стол человек притягивает углом к стене. Две ручки в одном
    месте нечего двигать порознь, да и гомография от повторной пары вырождается.
    """
    shapes, handles = [], []
    seen: dict = {}
    for kind, _min_pts, closed in PLAN_KINDS:
        for it in (plan or {}).get(kind) or []:
            ids = []
            for i, p in enumerate(it["points"]):
                key = (round(float(p[0]), 5), round(float(p[1]), 5))
                hid = seen.get(key)
                if hid is None:
                    hid = seen[key] = f"{it['id']}:{i}"
                    handles.append({"id": hid, "kind": kind,
                                    "map": [float(p[0]), float(p[1])]})
                ids.append(hid)
            shapes.append({"id": it["id"], "kind": kind, "closed": closed,
                           "name": it.get("name", ""), "pts": ids})
    return shapes, handles


def guess_far(plan: dict, pos, direction: float) -> float:
    """Докуда камера смотрит: до края листа вдоль стрелки, в изотропных единицах.

    То же самое считает браузер, когда рисует сектор обзора на чертеже, — иначе
    нарисованный сектор обещал бы одно, а каркас в кадре показывал другое.
    """
    aspect = float((plan or {}).get("aspect") or PLAN_ASPECT)
    cx, cy = float(pos[0]), float(pos[1]) / aspect
    th = math.radians(float(direction or 0.0))
    dx, dy = math.cos(th), math.sin(th)
    ts = []
    if abs(dx) > 1e-6:
        ts.append(((1.0 if dx > 0 else 0.0) - cx) / dx)
    if abs(dy) > 1e-6:
        ts.append(((1.0 / aspect if dy > 0 else 0.0) - cy) / dy)
    ts = [t for t in ts if t > 0]
    return min(GUESS_FAR_MAX, max(GUESS_FAR_MIN, min(ts) if ts else GUESS_FAR_MAX))


def guess_homography(plan: dict, cam_id: str) -> np.ndarray | None:
    """Грубая гомография кадр -> план по точке и стрелке, нарисованным человеком.

    Камера видит перед собой трапецию пола: дальний край там, где взгляд упирается
    в край листа, ближний — рядом с собой, ширина по углу обзора. Четыре её угла
    это четыре угла кадра, а четырёх пар достаточно для гомографии. Она будет
    неточной, и это нормально: её задача — положить каркас примерно туда, где
    человек его узнает и подвинет.
    """
    cam = ((plan or {}).get("cams") or {}).get(cam_id)
    if not cam:
        return None
    aspect = float(plan.get("aspect") or PLAN_ASPECT)
    # Считаем в изотропных единицах ширины плана: по вертикали 0..1 — это другое
    # расстояние, чем по горизонтали, и без поправки трапеция вышла бы косой.
    cx, cy = float(cam["pos"][0]), float(cam["pos"][1]) / aspect
    th = math.radians(float(cam.get("dir") or 0.0))
    dx, dy = math.cos(th), math.sin(th)
    rx, ry = -dy, dx                    # «направо» от смотрящего: на плане y вниз
    half = math.tan(math.radians(float(cam.get("fov") or GUESS_FOV_DEG) / 2.0))
    far = (float(cam["far"]) if cam.get("far")
           else guess_far(plan, cam["pos"], cam.get("dir") or 0.0))
    near = far * GUESS_NEAR_FRAC
    quad = []
    for dist, side in ((far, -1), (far, 1), (near, 1), (near, -1)):
        w = dist * half * side
        quad.append([cx + dx * dist + rx * w, (cy + dy * dist + ry * w) * aspect])
    # Углы кадра в том же порядке: дальний край — верх кадра.
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    H, _ = cv2.findHomography(src, np.array(quad, dtype=np.float64), 0)
    if H is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        return None
    return H


def _nudge(pred: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Поправка в кадре по 1-3 подвинутым точкам: сдвиг, поворот, перекос.

    Пока пар мало, гомографию по ним не решить, но и бросать их жалко: одна пара
    уже говорит, куда съехал весь каркас. Двигаем его целиком — человек видит
    отклик с первой же точки, а не после четвёртой.
    """
    n = len(pred)
    if n == 1:
        d = tgt[0] - pred[0]
        return np.array([[1.0, 0.0, d[0]], [0.0, 1.0, d[1]], [0.0, 0.0, 1.0]]), "shift"
    if n == 2:
        p = pred[:, 0] + 1j * pred[:, 1]
        q = tgt[:, 0] + 1j * tgt[:, 1]
        if abs(p[1] - p[0]) < 1e-9:
            return None, "guess"
        a = (q[1] - q[0]) / (p[1] - p[0])
        b = q[0] - a * p[0]
        return np.array([[a.real, -a.imag, b.real],
                         [a.imag, a.real, b.imag],
                         [0.0, 0.0, 1.0]]), "turn"
    M = cv2.getAffineTransform(pred[:3].astype(np.float32), tgt[:3].astype(np.float32))
    if M is None or not np.isfinite(M).all():
        return None, "guess"
    return np.vstack([M, [0.0, 0.0, 1.0]]), "skew"


def _project_handles(P: np.ndarray, handles: list) -> tuple[np.ndarray, np.ndarray]:
    """Точки чертежа в кадре плюс признак «перед камерой».

    Знак w за горизонтом меняется, и точка уезжает в бесконечность. Опорный знак
    берём у точки плана, которую видно в центре кадра: она перед камерой по самому
    построению. Позицию самой камеры для этого брать нельзя — она лежит под ней,
    то есть ближе ближнего края, и у неё знак бывает уже другой.
    """
    M = np.array([[h["map"][0], h["map"][1], 1.0] for h in handles], dtype=np.float64).T
    proj = P @ M
    try:
        centre = np.linalg.inv(P) @ np.array([0.5, 0.5, 1.0])
        w_ref = 1.0 / centre[2] if abs(centre[2]) > 1e-12 else 1.0
    except np.linalg.LinAlgError:
        w_ref = 1.0
    ahead = (np.isfinite(proj).all(axis=0) & (np.abs(proj[2]) > 1e-9)
             & (proj[2] * w_ref > 0.0))
    return proj, ahead


def _fit_to_frame(handles: list) -> np.ndarray:
    """Разложить чертёж по кадру, когда прикидка увела его за края.

    Каркас, которого не видно, — это тупик: ухватить нечего, значит и точек не
    поставить, а без точек гомографии не будет. Условное положение, зато целиком
    в кадре и под курсором, честнее пустого экрана.
    """
    pts = np.array([h["map"] for h in handles], dtype=np.float64)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    k = float(min(0.8 / span[0], 0.8 / span[1]))
    off = 0.5 - k * (lo + hi) / 2.0
    return np.array([[k, 0.0, off[0]], [0.0, k, off[1]], [0.0, 0.0, 1.0]])


def align_fit(plan: dict, cam_id: str, adjust: dict) -> dict:
    """Где точки чертежа лежат на кадре при нынешних поправках человека.

    Пар набралось четыре — решаем гомографию прямо по ним и забываем прикидку:
    дальше каркас садится сам, и оставшиеся точки человеку трогать уже не нужно.
    """
    shapes, handles = plan_shapes(plan)
    by_id = {h["id"]: h for h in handles}
    keys = [k for k in adjust if k in by_id]
    errors: dict = {}
    fit, warn = "guess", None
    P = None                                    # план -> кадр
    if len(keys) >= MIN_CALIB_POINTS:
        src = np.array([adjust[k] for k in keys], dtype=np.float64)          # кадр
        dst = np.array([by_id[k]["map"] for k in keys], dtype=np.float64)    # план
        Hi, _ = cv2.findHomography(src, dst, 0)
        if Hi is not None and np.isfinite(Hi).all() and np.linalg.matrix_rank(Hi) == 3:
            proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hi).reshape(-1, 2)
            for i, k in enumerate(keys):
                errors[k] = float(np.linalg.norm(proj[i] - dst[i]))
            P = np.linalg.inv(Hi)
            fit = "homography"
        else:
            # Точек хватает, а решения нет: значит, они лежат на одной прямой.
            # Молча откатываться на прикидку нельзя — человек решит, что готово.
            warn = "these points sit on one line — move a corner off it"
    if P is None:
        H = guess_homography(plan, cam_id)
        if H is None:
            return {"ok": False, "error": "put this camera on the plan first"}
        P = np.linalg.inv(H)
        if keys:
            src = np.array([by_id[k]["map"] for k in keys], dtype=np.float64)
            pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), P).reshape(-1, 2)
            T, fit = _nudge(pred, np.array([adjust[k] for k in keys], dtype=np.float64))
            if T is not None:
                P = T @ P
    if handles:
        proj, ahead = _project_handles(P, handles)
        # Прикидка могла увести весь чертёж за край кадра — тогда человеку нечего
        # тащить. Пока он не поставил ни одной точки, показать условное положение
        # лучше, чем пустой кадр; после четырёх решает уже гомография.
        if len(keys) < MIN_CALIB_POINTS:
            with np.errstate(divide="ignore", invalid="ignore"):
                xs, ys = proj[0] / proj[2], proj[1] / proj[2]
            seen = ahead & (xs > 0.0) & (xs < 1.0) & (ys > 0.0) & (ys < 1.0)
            if not seen.any():
                P = _fit_to_frame(handles)
                proj, ahead = _project_handles(P, handles)
                fit = "laid out"
        for i, h in enumerate(handles):
            w = float(proj[2, i])
            x, y = (proj[0, i] / w, proj[1, i] / w) if ahead[i] else (0.0, 0.0)
            h["img"] = [round(float(x), 4), round(float(y), 4)]
            # behind — нечего рисовать вообще; off — линию рисуем (браузер обрежет
            # её по рамке), а ручки нет: за край кадра её всё равно не утащить.
            h["behind"] = not bool(ahead[i])
            h["off"] = bool(ahead[i]) and not (-0.02 <= x <= 1.02 and -0.02 <= y <= 1.02)
            h["out"] = h["behind"] or h["off"]
            h["set"] = h["id"] in adjust
            if h["set"]:
                h["img"] = [float(adjust[h["id"]][0]), float(adjust[h["id"]][1])]
                h["behind"] = h["off"] = h["out"] = False
            h["err"] = errors.get(h["id"])
    return {"ok": True, "shapes": shapes, "handles": handles, "fit": fit,
            "warn": warn, "adjusted": len(keys), "min_points": MIN_CALIB_POINTS}


class FloorMap:
    def __init__(self):
        self.lock = threading.Lock()
        self._cameras: list = []
        self._branches: list = []
        # cam_id -> (гомография, филиал). Пересобирается при каждой правке конфига,
        # потоки камер только читают: словарь подменяется целиком.
        self._H: dict[str, tuple[np.ndarray, str]] = {}
        self._zones: dict[str, list] = {}         # филиал -> [(id, name, polygon)]
        # Живые точки: (камера, трек) -> {u, v, name, person_id, t}.
        self._live: dict[tuple, dict] = {}
        # Зона человека с выдержкой, как и остальные отрезки: на границе двух зон
        # точка дрожит, и без выдержки каждый кадр открывал бы новый отрезок.
        self._zone_now: dict[int, str | None] = {}
        self._zone_pending: dict[int, tuple] = {}
        self._zone_seen: dict[int, float] = {}
        self._sweep_at = 0.0
        # След: буфер строк для базы и последняя записанная точка человека.
        self._trail: list = []
        self._trail_at: dict[int, tuple] = {}
        self._trail_flush = 0.0

    # --- конфигурация -----------------------------------------------------
    def configure(self, cameras: list, branches: list) -> None:
        self._cameras = cameras
        self._branches = branches
        self.rebuild()

    def rebuild(self) -> None:
        H = {}
        for cam in self._cameras:
            calib = cam.get("floor_calib") or {}
            pts = calib.get("points") or []
            if len(pts) < MIN_CALIB_POINTS:
                continue
            m, _ = solve_homography(pts)
            if m is not None:
                H[cam["id"]] = (m, cam.get("branch") or "")
        zones = {}
        for b in self._branches:
            zones[b["id"]] = [(z["id"], z["name"], [(float(p[0]), float(p[1]))
                                                    for p in z["points"]])
                              for z in (b.get("zones") or []) if len(z.get("points", [])) >= 3]
        with self.lock:
            self._H = H
            self._zones = zones

    def branch(self, branch_id: str) -> dict | None:
        return next((b for b in self._branches if b["id"] == branch_id), None)

    def floorplan_path(self, branch_id: str) -> Path | None:
        b = self.branch(branch_id)
        fp = (b or {}).get("floorplan")
        if not fp:
            return None
        p = FLOORPLAN_DIR / fp["file"]
        return p if p.exists() else None

    def calibrated(self, cam_id: str) -> bool:
        return cam_id in self._H

    def plan(self, branch_id: str) -> dict:
        return (self.branch(branch_id) or {}).get("plan") or {}

    def has_plan(self, branch_id: str) -> bool:
        return (not plan_is_empty(self.plan(branch_id))
                or self.floorplan_path(branch_id) is not None)

    def uncalibrated(self, branch_id: str) -> list[str]:
        return [c["id"] for c in self._cameras
                if (c.get("branch") or "") == branch_id and c["id"] not in self._H]

    # --- геометрия ---------------------------------------------------------
    def project(self, cam_id: str, x: float, y: float) -> tuple[float, float] | None:
        entry = self._H.get(cam_id)
        if entry is None:
            return None
        H = entry[0]
        p = H @ np.array([x, y, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        u, v = float(p[0] / p[2]), float(p[1] / p[2])
        # Верх кадра лежит за горизонтом, и точка оттуда улетает на километры.
        # Такой человек не в комнате — это ошибка ног или калибровки, а не позиция.
        if not (-0.5 <= u <= 1.5 and -0.5 <= v <= 1.5):
            return None
        return u, v

    def zone_at(self, branch_id: str, u: float, v: float) -> tuple[str, str] | None:
        for zid, name, poly in self._zones.get(branch_id, []):
            if _point_in_polygon(u, v, poly):
                return zid, name
        return None

    # --- живые точки -------------------------------------------------------
    def update_live(self, cam_id: str, track_id: str, uv: tuple, name: str | None,
                    person_id: int | None, t: float) -> None:
        branch = self._H.get(cam_id, (None, ""))[1]
        with self.lock:
            self._live[(cam_id, track_id)] = {
                "camera_id": cam_id, "track_id": track_id, "branch": branch,
                "u": round(uv[0], 4), "v": round(uv[1], 4),
                "name": name, "person_id": person_id, "t": t,
            }

    def live(self, branch_id: str) -> list[dict]:
        now = time.time()
        with self.lock:
            stale = [k for k, r in self._live.items() if now - r["t"] > LIVE_TTL]
            for k in stale:
                self._live.pop(k, None)
            return [dict(r) for r in self._live.values()
                    if not branch_id or r["branch"] == branch_id]

    # --- зоны как отрезки --------------------------------------------------
    def observe_zone(self, store, branch_id: str, person_id: int, uv: tuple, t: float,
                     debounce_sec: float) -> None:
        """Человек сейчас в этой точке плана — записать смену зоны, если она устоялась.

        Ключ — человек, а не (камера, трек): именно поэтому передача имени между
        камерами отрезок не рвёт. Писать сюда имеет право только держатель имени, и
        он один в каждый момент, так что гонки по одному person_id нет.
        """
        found = self.zone_at(branch_id, uv[0], uv[1])
        label = f"{branch_id}:{found[1]}" if found else None
        with self.lock:
            self._zone_seen[person_id] = t
            if self._zone_now.get(person_id, None) == label and person_id in self._zone_now:
                self._zone_pending.pop(person_id, None)
                return
            pending = self._zone_pending.get(person_id)
            if pending is None or pending[0] != label:
                self._zone_pending[person_id] = (label, t)
                return
            if t - pending[1] < debounce_sec:
                return
            self._zone_pending.pop(person_id, None)
            self._zone_now[person_id] = label
        store.record_zone(branch_id, person_id, found[1] if found else None, ts=t)

    def record_point(self, store, branch_id: str, person_id: int, uv: tuple,
                     t: float) -> None:
        """Точку в след человека — если он сдвинулся или давно не отмечался.

        Пишем пачками: точки идут из четырёх потоков камер, и коммит на каждую
        обошёлся бы дороже самой аналитики. В базу уходим вне блокировки, чтобы
        поток камеры не ждал диск.
        """
        with self.lock:
            last = self._trail_at.get(person_id)
            if last is not None:
                gap = t - last[0]
                if gap < TRAIL_MIN_GAP:
                    return
                moved = math.hypot(uv[0] - last[1], uv[1] - last[2])
                if moved < TRAIL_MIN_MOVE and gap < TRAIL_HEARTBEAT:
                    return
            self._trail_at[person_id] = (t, uv[0], uv[1])
            self._trail.append((branch_id, int(person_id), float(t),
                                round(float(uv[0]), 4), round(float(uv[1]), 4)))
            if len(self._trail) < TRAIL_MAX_BUF and t - self._trail_flush < TRAIL_FLUSH_SEC:
                return
            self._trail_flush = t
            batch, self._trail = self._trail, []
        store.record_positions(batch)

    def flush_trail(self, store) -> None:
        """Дописать остаток буфера: при остановке и перед чтением следа за сегодня."""
        with self.lock:
            batch, self._trail = self._trail, []
        store.record_positions(batch)

    def sweep(self, store, t: float, grace_sec: float, every_sec: float) -> None:
        """Закрыть зоны тех, кого давно не видели — раз в несколько секунд, из любого потока."""
        with self.lock:
            if t - self._sweep_at < every_sec:
                return
            self._sweep_at = t
            gone = [(pid, ts) for pid, ts in self._zone_seen.items() if t - ts > grace_sec]
            for pid, _ in gone:
                self._zone_seen.pop(pid, None)
                self._zone_now.pop(pid, None)
                self._zone_pending.pop(pid, None)
        # Закрываем моментом, когда человека видели в последний раз, а не моментом
        # подметания: полминуты ожидания — это не полминуты в зоне.
        for pid, last in gone:
            store.close_zone(pid, ts=last)

    # --- для страницы -----------------------------------------------------
    def describe(self, branch_id: str) -> dict:
        b = self.branch(branch_id) or {}
        fp = b.get("floorplan")
        plan = b.get("plan") or {}
        placed = (plan.get("cams") or {})
        cams = []
        for c in self._cameras:
            if (c.get("branch") or "") != branch_id:
                continue
            pts = (c.get("floor_calib") or {}).get("points") or []
            _, errs = solve_homography(pts) if len(pts) >= MIN_CALIB_POINTS else (None, [])
            cams.append({"id": c["id"], "name": c.get("name", c["id"]),
                         "calibrated": c["id"] in self._H,
                         "placed": c["id"] in placed,
                         "points": pts, "errors": errs})
        return {
            "branch": branch_id,
            "plan": plan,
            "drawn": not plan_is_empty(plan),
            "floorplan": ({"url": f"/floorplans/{fp['file']}?v={fp.get('stamp', 0)}",
                           "width": fp["width"], "height": fp["height"]} if fp else None),
            "zones": b.get("zones") or [],
            "cameras": cams,
            "min_points": MIN_CALIB_POINTS,
            # Сектор обзора на чертеже рисует браузер, а гомографию по нему считает
            # сервер: числа должны быть одни и те же, иначе стрелка врёт.
            "guess": {"fov": GUESS_FOV_DEG, "near_frac": GUESS_NEAR_FRAC,
                      "far_min": GUESS_FAR_MIN, "far_max": GUESS_FAR_MAX},
        }


floor_map = FloorMap()
