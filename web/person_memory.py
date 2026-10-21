import itertools
import time
from dataclasses import dataclass, field

import numpy as np


def _center(box) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def _box_size(box) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(40.0, max(x2 - x1, y2 - y1))


def _contains_point(box, point) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    px, py = [float(v) for v in point]
    return x1 <= px <= x2 and y1 <= py <= y2


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class PersonTrack:
    track_id: str
    camera_id: str
    box: np.ndarray
    last_seen: float
    person_id: int | None = None
    external_id: str | None = None
    name: str | None = None
    status: dict = field(default_factory=dict)


class CameraTrackMemory:
    TRACK_TTL = 8.0
    MATCH_DIST_RATIO = 0.65

    def __init__(self):
        self._counter = itertools.count(1)
        self.tracks: list[PersonTrack] = []

    def observe(self, camera_id: str, box, ts: float | None = None) -> PersonTrack:
        ts = ts or time.time()
        self.cleanup(ts)
        box_arr = np.asarray(box, dtype=np.float32)
        pos = _center(box_arr)
        size = _box_size(box_arr)

        best = None
        best_dist = None
        for track in self.tracks:
            if track.camera_id != camera_id:
                continue
            dist = float(np.linalg.norm(pos - _center(track.box)))
            max_dist = max(70.0, min(size, _box_size(track.box)) * self.MATCH_DIST_RATIO)
            if dist <= max_dist and (best_dist is None or dist < best_dist):
                best = track
                best_dist = dist

        if best is None:
            best = PersonTrack(
                track_id=f"{camera_id}_track_{next(self._counter)}",
                camera_id=camera_id,
                box=box_arr,
                last_seen=ts,
            )
            self.tracks.append(best)
        else:
            best.box = box_arr
            best.last_seen = ts
        return best

    def bind_identity(
        self,
        camera_id: str,
        person_id: int,
        external_id: str,
        name: str | None = None,
        box=None,
        ts: float | None = None,
    ) -> PersonTrack | None:
        ts = ts or time.time()
        self.cleanup(ts)
        target = None
        if box is not None:
            box_arr = np.asarray(box, dtype=np.float32)
            box_center = _center(box_arr)
            candidates = [t for t in self.tracks if t.camera_id == camera_id]
            containing = [
                t for t in candidates
                if _contains_point(t.box, box_center)
            ]
            if containing:
                target = min(containing, key=lambda t: float(np.linalg.norm(box_center - _center(t.box))))
            else:
                scored = []
                box_size = _box_size(box_arr)
                for track in candidates:
                    dist = float(np.linalg.norm(box_center - _center(track.box)))
                    max_dist = max(90.0, min(box_size, _box_size(track.box)) * 0.9)
                    overlap = _iou(box_arr, track.box)
                    if dist <= max_dist or overlap >= 0.15:
                        scored.append((dist, -overlap, track))
                if scored:
                    target = min(scored, key=lambda item: (item[0], item[1]))[2]
            if target is not None:
                target.last_seen = ts

        if target is not None:
            target.person_id = person_id
            target.external_id = external_id
            target.name = name or external_id
            target.last_seen = ts
        return target

    def bind_track_id(
        self,
        camera_id: str,
        track_id: str,
        person_id: int,
        external_id: str,
        name: str | None = None,
        ts: float | None = None,
    ) -> PersonTrack | None:
        ts = ts or time.time()
        self.cleanup(ts)
        for track in self.tracks:
            if track.camera_id == camera_id and track.track_id == track_id:
                track.person_id = person_id
                track.external_id = external_id
                track.name = name or external_id
                track.last_seen = ts
                return track
        return None

    def cleanup(self, ts: float | None = None):
        ts = ts or time.time()
        self.tracks = [t for t in self.tracks if ts - t.last_seen <= self.TRACK_TTL]

    def active_track_ids(self, camera_id: str, ts: float | None = None) -> set[str]:
        self.cleanup(ts)
        return {t.track_id for t in self.tracks if t.camera_id == camera_id}

    def snapshot(self):
        now = time.time()
        self.cleanup(now)
        return [
            {
                "track_id": t.track_id,
                "camera_id": t.camera_id,
                "person_id": t.person_id,
                "external_id": t.external_id,
                "name": t.name,
                "box": [round(float(v), 1) for v in t.box.tolist()],
                "last_seen_ago": round(now - t.last_seen, 2),
                "status": t.status,
            }
            for t in self.tracks
        ]


track_memory = CameraTrackMemory()
