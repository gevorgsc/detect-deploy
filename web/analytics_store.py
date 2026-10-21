import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "analytics.sqlite3"


class AnalyticsStore:
    def __init__(self, db_path: Path = DB_PATH):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        # WAL + NORMAL: commit перестаёт ждать fsync диска на каждой записи — 2.1 мс
        # превращаются в 0.01 (замерено). Отрезки пишутся при каждой смене состояния
        # на каждом треке каждой камеры, и эти миллисекунды складывались в заметную
        # долю «tracks+db». Цена: при отключении питания теряются последние доли
        # секунды записей; целостность базы при этом не страдает.
        # Побочное: рядом с базой появляются файлы -wal и -shm. Копировать базу
        # простым copy на ходу нельзя — часть данных ещё в -wal; брать conn.backup().
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self):
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT UNIQUE,
                    name TEXT,
                    created_at REAL NOT NULL,
                    last_seen REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS identity_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    camera_id TEXT NOT NULL,
                    track_id TEXT,
                    confidence REAL,
                    raw_payload TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(id)
                );

                CREATE TABLE IF NOT EXISTS face_identities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_key TEXT UNIQUE NOT NULL,
                    person_id INTEGER NOT NULL,
                    source TEXT,
                    created_at REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(id)
                );

                CREATE TABLE IF NOT EXISTS unknown_face_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    track_id TEXT,
                    identity_key TEXT,
                    box TEXT,
                    snapshot_path TEXT,
                    raw_payload TEXT,
                    created_at REAL NOT NULL,
                    resolved_person_id INTEGER,
                    resolved_at REAL,
                    FOREIGN KEY(resolved_person_id) REFERENCES persons(id)
                );

                CREATE TABLE IF NOT EXISTS activity_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    person_id INTEGER,
                    label_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    start_ts REAL NOT NULL,
                    end_ts REAL,
                    duration_sec REAL DEFAULT 0,
                    FOREIGN KEY(person_id) REFERENCES persons(id)
                );

                -- Настройки, которые оператор меняет с сайта. Сейчас это порог
                -- бездействия в минутах: сколько человек может ничего не делать,
                -- прежде чем это станет записью.
                CREATE TABLE IF NOT EXISTS settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );

                -- След человека на плане: чтобы день можно было отмотать назад, а
                -- не только прочитать списком. Отрезки зон отвечают на вопрос «где
                -- был», но не «каким путём шёл», и по ним не видно, что два человека
                -- встретились. Пишем не каждый кадр: точка ложится, когда человек
                -- заметно сдвинулся, плюс редкое биение на месте — иначе сидящий за
                -- столом за день даёт десятки тысяч одинаковых строк.
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    branch_id TEXT NOT NULL,
                    person_id INTEGER NOT NULL,
                    ts REAL NOT NULL,
                    u REAL NOT NULL,
                    v REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_positions_branch_ts
                    ON positions(branch_id, ts);

                CREATE INDEX IF NOT EXISTS ix_segments_person_start
                    ON activity_segments(person_id, start_ts);
                CREATE INDEX IF NOT EXISTS ix_segments_track_open
                    ON activity_segments(camera_id, track_id, end_ts);
                CREATE INDEX IF NOT EXISTS ix_face_identities_person
                    ON face_identities(person_id);
                CREATE INDEX IF NOT EXISTS ix_unknown_faces_open
                    ON unknown_face_events(resolved_person_id, created_at);
                """
            )
            self.conn.commit()

    # --- настройки ---------------------------------------------------------
    def get_setting(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM settings WHERE key=?",
                                (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO settings (key, value) VALUES (?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)))
            self.conn.commit()

    # --- привязка отрезков к человеку -------------------------------------
    def reattribute_track(self, camera_id: str, track_id: str, person_id: int) -> int:
        """Переписать все отрезки трека на этого человека.

        Имя на треке может смениться: сначала лицо совпало слабо и не с тем, через
        секунду — уверенно и с настоящим. Всё, что трек успел наработать, должно
        числиться за тем, кем он оказался на самом деле, а не за первой догадкой.

        Только ОТКРЫТЫЕ отрезки. Закрытые уже разрезаны по сменам хозяина (см.
        switch_track_person), и переписать их значит стереть эту границу: половина,
        честно отданная другой камере, снова досталась бы этому треку — и время опять
        пошло бы дважды.
        """
        with self.lock:
            cur = self.conn.execute(
                "UPDATE activity_segments SET person_id=?"
                " WHERE camera_id=? AND track_id=? AND end_ts IS NULL"
                "   AND (person_id IS NULL OR person_id<>?)",
                (person_id, camera_id, track_id, person_id))
            self.conn.commit()
            return cur.rowcount

    def switch_track_person(self, camera_id: str, track_id: str,
                            person_id: int | None, ts: float | None = None) -> int:
        """Хозяин трека сменился: РАЗРЕЗАТЬ открытые отрезки ровно по этому моменту.

        Отрезок — промежуток, а хозяин меняется мгновенно. Пока отрезок один на всё
        время трека, любая смена хозяина либо переписывает прошлое, либо оставляет
        минуты за тем, кому они больше не принадлежат. Поэтому старую половину
        закрываем здесь, вместе с прежним хозяином, и тут же открываем такую же —
        с новым. Никаких дыр и никаких наложений.
        """
        ts = ts or time.time()
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, label_type, label, start_ts, person_id FROM activity_segments"
                " WHERE camera_id=? AND track_id=? AND end_ts IS NULL",
                (camera_id, track_id)).fetchall()
            changed = 0
            for r in rows:
                if r["person_id"] == person_id:
                    continue                       # хозяин тот же — резать нечего
                changed += 1
                self.conn.execute(
                    "UPDATE activity_segments SET end_ts=?, duration_sec=? WHERE id=?",
                    (ts, max(0.0, ts - float(r["start_ts"])), int(r["id"])))
                self.conn.execute(
                    "INSERT INTO activity_segments"
                    " (camera_id, track_id, person_id, label_type, label, start_ts)"
                    " VALUES (?,?,?,?,?,?)",
                    (camera_id, track_id, person_id, r["label_type"], r["label"], ts))
            self.conn.commit()
            return changed

    def release_track(self, camera_id: str, track_id: str,
                      ts: float | None = None) -> int:
        """Трек потерял имя — дальше он безымянный."""
        return self.switch_track_person(camera_id, track_id, None, ts)

    # --- бездействие -------------------------------------------------------
    def _idle_started_at(self, camera_id: str, track_id: str, open_start: float,
                         work_reset_sec: float, day_start: float) -> float:
        """С какого момента человек на самом деле бездействует.

        Короткая вспышка «работает» не должна обнулять счётчик: рука на пару секунд
        легла на мышь — и три накопленных минуты пропали, а такого человека список
        не покажет уже никогда. Поэтому идём назад по отрезкам трека и перешагиваем
        через рабочие куски короче work_reset_sec, сшивая простой в один.

        Само правило живёт здесь, при чтении, а не при записи. Отрезки в базе остаются
        точной летописью того, что было; поменяется представление о том, какая вспышка
        считается работой, — хватит поменять число, ничего не потеряв.
        """
        start = open_start
        rows = self.conn.execute(
            "SELECT label, start_ts, end_ts, duration_sec FROM activity_segments"
            " WHERE camera_id=? AND track_id=? AND label_type='activity'"
            "   AND start_ts >= ? AND start_ts < ?"
            " ORDER BY start_ts DESC", (camera_id, track_id, day_start, open_start)
        ).fetchall()
        for r in rows:
            if r["end_ts"] is None:
                continue
            # Между отрезками бывает зазор в доли секунды — это не работа.
            if start - float(r["end_ts"]) > 1.5:
                break
            if r["label"] == "idle":
                start = float(r["start_ts"])
            elif float(r["duration_sec"] or 0) < work_reset_sec:
                start = float(r["start_ts"])      # вспышка слишком коротка, шагаем через неё
            else:
                break                              # настоящая работа — простой начался после неё
        return start

    def idle_now(self, threshold_sec: float, day_start: float,
                 work_reset_sec: float = 10.0):
        """Кто прямо сейчас бездействует дольше порога.

        Берём ОТКРЫТЫЕ отрезки idle: закрытый отрезок — это уже прошлое, а вопрос
        оператора всегда про «сейчас».
        """
        now = time.time()
        rows = self.conn.execute(
            "SELECT person_id, camera_id, track_id, start_ts FROM activity_segments"
            " WHERE label_type='activity' AND label='idle' AND end_ts IS NULL"
            "   AND person_id IS NOT NULL AND start_ts >= ?"
            " ORDER BY start_ts", (day_start,)).fetchall()
        out = []
        for r in rows:
            since = self._idle_started_at(r["camera_id"], r["track_id"],
                                          float(r["start_ts"]), work_reset_sec,
                                          day_start)
            idle_for = now - since
            if idle_for >= threshold_sec:
                out.append({"person_id": int(r["person_id"]),
                            "camera_id": r["camera_id"], "track_id": r["track_id"],
                            "since": since, "idle_sec": round(idle_for, 1)})
        return out

    def idle_today(self, threshold_sec: float, day_start: float, limit: int = 200,
                   day_end: float | None = None):
        """Завершённые эпизоды бездействия за день — история, а не текущее состояние."""
        return [dict(r) for r in self.conn.execute(
            "SELECT person_id, camera_id, start_ts, end_ts, duration_sec"
            "  FROM activity_segments"
            " WHERE label_type='activity' AND label='idle' AND end_ts IS NOT NULL"
            "   AND person_id IS NOT NULL AND duration_sec >= ?"
            "   AND start_ts >= ? AND start_ts < ?"
            " ORDER BY start_ts DESC LIMIT ?",
            (threshold_sec, day_start,
             day_end if day_end is not None else time.time() + 86400, limit))]

    @staticmethod
    def _merge(intervals, join_gap: float = 0.0):
        """Склеить пересекающиеся отрезки (и разделённые зазором меньше join_gap).

        Складывать длительности напрямую нельзя: одного человека одновременно видят
        две камеры, и сумма даёт двойной счёт. Правильная величина — объединение.
        """
        if not intervals:
            return []
        out = []
        for s, e in sorted(intervals):
            if out and s - out[-1][1] <= join_gap:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out

    def recorded_days(self, limit: int = 60):
        """Дни, за которые вообще что-то записано — для выбора даты на странице."""
        return [r["day"] for r in self.conn.execute(
            "SELECT DISTINCT date(start_ts, 'unixepoch', 'localtime') day"
            "  FROM activity_segments ORDER BY day DESC LIMIT ?", (limit,))]

    def daily_summary(self, day_start: float, idle_threshold_sec: float,
                      work_reset_sec: float = 10.0, day_end: float | None = None):
        """Сколько каждый человек за день работал, говорил по телефону и бездельничал.

        Плюс сколько РАЗ бездействие затянулось дольше порога — оператору важно не
        только «сколько всего», но и «сколько раз подряд ничего не делал».
        """
        now = time.time()
        end_cap = day_end if day_end is not None else now
        rows = self.conn.execute(
            "SELECT person_id, label_type, label, start_ts, end_ts FROM activity_segments"
            " WHERE person_id IS NOT NULL AND start_ts >= ? AND start_ts < ?",
            (day_start, end_cap)).fetchall()

        by_person: dict[int, dict[str, list]] = {}
        seen: dict[int, list] = {}
        for r in rows:
            end = float(r["end_ts"]) if r["end_ts"] is not None else min(now, end_cap)
            if end <= float(r["start_ts"]):
                continue
            pid = int(r["person_id"])
            span = (float(r["start_ts"]), end)
            if r["label_type"] == "activity":
                by_person.setdefault(pid, {}).setdefault(r["label"], []).append(span)
            elif r["label_type"] == "presence":
                seen.setdefault(pid, []).append(span)

        out = []
        for pid in set(by_person) | set(seen):
            labels = by_person.get(pid, {})
            totals = {}
            for label, spans in labels.items():
                totals[label] = sum(e - s for s, e in self._merge(spans))
            # Короткая вспышка работы простой не прерывает — те же правила, что и в
            # idle_now, иначе число эпизодов и текущий статус противоречили бы друг другу.
            idle_runs = self._merge(labels.get("idle", []), join_gap=work_reset_sec)
            long_runs = [r for r in idle_runs if r[1] - r[0] >= idle_threshold_sec]
            # Присутствие: когда пришёл, когда ушёл и сколько всего пробыл. Объединение,
            # а не сумма: две камеры, видящие человека одновременно, дают одну минуту.
            spans = self._merge(seen.get(pid, []))
            out.append({
                "person_id": pid,
                "working_sec": round(totals.get("working", 0.0), 1),
                "phone_sec": round(totals.get("on_phone", 0.0), 1),
                "idle_sec": round(totals.get("idle", 0.0), 1),
                "idle_episodes": len(long_runs),
                "longest_idle_sec": round(max((r[1] - r[0] for r in long_runs),
                                              default=0.0), 1),
                "present_sec": round(sum(e - s for s, e in spans), 1),
                "first_seen": spans[0][0] if spans else None,
                "last_seen": spans[-1][1] if spans else None,
            })
        out.sort(key=lambda p: -p["present_sec"])
        return out

    def period_summary(self, day_starts, idle_threshold_sec: float,
                       work_reset_sec: float = 10.0):
        """То же самое, но за несколько дней сразу: 3, 7, 30.

        Считаем ПО ДНЯМ и складываем, а не одним куском от первого дня до последнего.
        Разница не косметическая. Эпизод бездействия, пойманный сплошным отрезком, мог
        бы «склеиться» через ночь — с вечера одного дня до утра следующего, — и вместо
        двух эпизодов по десять минут получился бы один на четырнадцать часов. По той
        же причине по дням считаются и «дни на работе»: за них отвечает факт, что в
        этот день человека вообще видели, а не длина общего отрезка.
        """
        totals: dict[int, dict] = {}
        for start in day_starts:
            for r in self.daily_summary(start, idle_threshold_sec, work_reset_sec,
                                        day_end=start + 86400):
                pid = r["person_id"]
                acc = totals.setdefault(pid, {
                    "person_id": pid, "working_sec": 0.0, "phone_sec": 0.0,
                    "idle_sec": 0.0, "present_sec": 0.0, "idle_episodes": 0,
                    "longest_idle_sec": 0.0, "days": 0,
                    "first_seen": None, "last_seen": None,
                })
                for k in ("working_sec", "phone_sec", "idle_sec", "present_sec"):
                    acc[k] += r[k]
                acc["idle_episodes"] += r["idle_episodes"]
                acc["longest_idle_sec"] = max(acc["longest_idle_sec"],
                                              r["longest_idle_sec"])
                if r["present_sec"] > 0:
                    acc["days"] += 1
                if r["first_seen"] and (acc["first_seen"] is None
                                        or r["first_seen"] < acc["first_seen"]):
                    acc["first_seen"] = r["first_seen"]
                if r["last_seen"] and (acc["last_seen"] is None
                                       or r["last_seen"] > acc["last_seen"]):
                    acc["last_seen"] = r["last_seen"]

        out = []
        for acc in totals.values():
            for k in ("working_sec", "phone_sec", "idle_sec", "present_sec",
                      "longest_idle_sec"):
                acc[k] = round(acc[k], 1)
            # Среднее за ДЕНЬ НА РАБОТЕ, а не за календарный день: человек в отпуске не
            # должен выглядеть лентяем оттого, что делить пришлось на тридцать.
            acc["avg_working_sec"] = round(
                acc["working_sec"] / acc["days"], 1) if acc["days"] else 0.0
            out.append(acc)
        out.sort(key=lambda p: -p["working_sec"])
        return out

    def clear_activity(self, day_start: float | None = None) -> dict:
        """Стереть записанные отрезки и события. Без аргумента — всё, что есть."""
        with self.lock:
            if day_start is None:
                seg = self.conn.execute("SELECT COUNT(*) c FROM activity_segments").fetchone()["c"]
                ev = self.conn.execute("SELECT COUNT(*) c FROM identity_events").fetchone()["c"]
                self.conn.execute("DELETE FROM activity_segments")
                self.conn.execute("DELETE FROM identity_events")
            else:
                seg = self.conn.execute(
                    "SELECT COUNT(*) c FROM activity_segments WHERE start_ts >= ?",
                    (day_start,)).fetchone()["c"]
                ev = self.conn.execute(
                    "SELECT COUNT(*) c FROM identity_events WHERE created_at >= ?",
                    (day_start,)).fetchone()["c"]
                self.conn.execute("DELETE FROM activity_segments WHERE start_ts >= ?",
                                  (day_start,))
                self.conn.execute("DELETE FROM identity_events WHERE created_at >= ?",
                                  (day_start,))
            self.conn.commit()
        return {"segments": seg, "events": ev}

    def forget_persons(self, person_ids) -> int:
        """Убрать записи людей, которых больше нет в галерее.

        Иначе на странице остаются маршруты и простои тех, кого удалили: отрезки
        ссылаются на person_id, а карточки с таким номером уже нет.
        """
        ids = [int(i) for i in person_ids]
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        with self.lock:
            n = self.conn.execute(
                f"SELECT COUNT(*) c FROM activity_segments WHERE person_id IN ({marks})",
                ids).fetchone()["c"]
            self.conn.execute(
                f"DELETE FROM activity_segments WHERE person_id IN ({marks})", ids)
            self.conn.execute(
                f"DELETE FROM identity_events WHERE person_id IN ({marks})", ids)
            self.conn.commit()
        return n

    # --- перемещения -------------------------------------------------------
    def routes(self, day_start: float, person_id: int | None = None, limit: int = 500,
               day_end: float | None = None):
        """Отрезки присутствия по камерам — из них и складывается маршрут."""
        sql = ("SELECT person_id, camera_id, track_id, start_ts, end_ts, duration_sec"
               "  FROM activity_segments"
               " WHERE label_type='presence' AND person_id IS NOT NULL"
               "   AND start_ts >= ? AND start_ts < ?")
        args = [day_start, day_end if day_end is not None else time.time() + 86400]
        if person_id is not None:
            sql += " AND person_id = ?"
            args.append(person_id)
        sql += " ORDER BY person_id, start_ts LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def upsert_person(self, external_id: str, name: str | None = None) -> int:
        now = time.time()
        external_id = str(external_id).strip()
        if not external_id:
            raise ValueError("external_id is required")
        display_name = name or external_id
        with self.lock:
            row = self.conn.execute(
                "SELECT id FROM persons WHERE external_id = ?",
                (external_id,),
            ).fetchone()
            if row:
                person_id = int(row["id"])
                self.conn.execute(
                    "UPDATE persons SET name = COALESCE(?, name), last_seen = ? WHERE id = ?",
                    (name, now, person_id),
                )
            else:
                cur = self.conn.execute(
                    "INSERT INTO persons (external_id, name, created_at, last_seen) VALUES (?, ?, ?, ?)",
                    (external_id, display_name, now, now),
                )
                person_id = int(cur.lastrowid)
            self.conn.commit()
            return person_id

    def find_person_by_face_identity(self, identity_key: str):
        identity_key = str(identity_key or "").strip()
        if not identity_key:
            return None
        with self.lock:
            row = self.conn.execute(
                """
                SELECT p.id, p.external_id, p.name
                  FROM face_identities f
                  JOIN persons p ON p.id = f.person_id
                 WHERE f.identity_key = ?
                """,
                (identity_key,),
            ).fetchone()
            return dict(row) if row else None

    def find_person_by_name(self, name: str):
        name = str(name or "").strip()
        if not name:
            return None
        with self.lock:
            row = self.conn.execute(
                "SELECT id, external_id, name FROM persons WHERE lower(name) = lower(?) ORDER BY id LIMIT 1",
                (name,),
            ).fetchone()
            return dict(row) if row else None

    def link_face_identity(self, identity_key: str, person_id: int, source: str | None = None):
        now = time.time()
        identity_key = str(identity_key or "").strip()
        if not identity_key:
            return
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO face_identities (identity_key, person_id, source, created_at, last_seen)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(identity_key) DO UPDATE SET
                    person_id = excluded.person_id,
                    source = COALESCE(excluded.source, face_identities.source),
                    last_seen = excluded.last_seen
                """,
                (identity_key, person_id, source, now, now),
            )
            self.conn.commit()

    def create_unknown_face_event(
        self,
        camera_id: str,
        track_id: str | None = None,
        identity_key: str | None = None,
        box: str | None = None,
        snapshot_path: str | None = None,
        raw_payload: str | None = None,
    ) -> int:
        now = time.time()
        with self.lock:
            cur = self.conn.execute(
                """
                INSERT INTO unknown_face_events
                    (camera_id, track_id, identity_key, box, snapshot_path, raw_payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (camera_id, track_id, identity_key, box, snapshot_path, raw_payload, now),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def list_unknown_face_events(self, limit: int = 20):
        with self.lock:
            return [
                dict(row)
                for row in self.conn.execute(
                    """
                    SELECT id, camera_id, track_id, identity_key, box, snapshot_path, created_at
                      FROM unknown_face_events
                     WHERE resolved_person_id IS NULL
                     ORDER BY created_at DESC
                     LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
            ]

    def get_unknown_face_event(self, event_id: int):
        with self.lock:
            row = self.conn.execute(
                """
                SELECT id, camera_id, track_id, identity_key, box, snapshot_path, raw_payload, created_at,
                       resolved_person_id, resolved_at
                  FROM unknown_face_events
                 WHERE id = ?
                """,
                (int(event_id),),
            ).fetchone()
            return dict(row) if row else None

    def resolve_unknown_face_event(self, event_id: int, person_id: int):
        now = time.time()
        with self.lock:
            self.conn.execute(
                """
                UPDATE unknown_face_events
                   SET resolved_person_id = ?, resolved_at = ?
                 WHERE id = ?
                """,
                (person_id, now, int(event_id)),
            )
            self.conn.commit()

    def record_identity_event(
        self,
        camera_id: str,
        person_id: int,
        track_id: str | None = None,
        confidence: float | None = None,
        raw_payload: str | None = None,
    ):
        now = time.time()
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO identity_events
                    (person_id, camera_id, track_id, confidence, raw_payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (person_id, camera_id, track_id, confidence, raw_payload, now),
            )
            self.conn.execute("UPDATE persons SET last_seen = ? WHERE id = ?", (now, person_id))
            if track_id:
                self.conn.execute(
                    """
                    UPDATE activity_segments
                       SET person_id = ?
                     WHERE camera_id = ?
                       AND track_id = ?
                       AND person_id IS NULL
                    """,
                    (person_id, camera_id, track_id),
                )
            self.conn.commit()

    def record_observation(
        self,
        camera_id: str,
        track_id: str,
        person_id: int | None,
        posture: str,
        activities,
        ts: float | None = None,
    ):
        ts = ts or time.time()
        active = {("posture", posture)}
        for activity in activities:
            active.add(("activity", str(activity)))

        with self.lock:
            # Только свои типы. Раньше выбирались ВСЕ открытые отрезки трека, и отрезок
            # присутствия, которого нет в наборе поз и действий, закрывался при каждой
            # смене действия — а затем открывался заново. В базе это выглядело как
            # семнадцать кусков присутствия подряд с нулевыми промежутками между ними.
            rows = self.conn.execute(
                """
                SELECT id, label_type, label, start_ts
                  FROM activity_segments
                 WHERE camera_id = ? AND track_id = ? AND end_ts IS NULL
                   AND label_type IN ('posture', 'activity')
                """,
                (camera_id, track_id),
            ).fetchall()
            open_labels = {(row["label_type"], row["label"]): row for row in rows}

            for key, row in open_labels.items():
                if key not in active:
                    duration = max(0.0, ts - float(row["start_ts"]))
                    self.conn.execute(
                        """
                        UPDATE activity_segments
                           SET end_ts = ?, duration_sec = ?
                         WHERE id = ?
                        """,
                        (ts, duration, int(row["id"])),
                    )

            for label_type, label in active:
                if (label_type, label) not in open_labels:
                    self.conn.execute(
                        """
                        INSERT INTO activity_segments
                            (camera_id, track_id, person_id, label_type, label, start_ts)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (camera_id, track_id, person_id, label_type, label, ts),
                    )
                elif person_id is not None and open_labels[(label_type, label)]["id"]:
                    self.conn.execute(
                        "UPDATE activity_segments SET person_id = ? WHERE id = ? AND person_id IS NULL",
                        (person_id, int(open_labels[(label_type, label)]["id"])),
                    )

            if person_id is not None:
                self.conn.execute("UPDATE persons SET last_seen = ? WHERE id = ?", (ts, person_id))
            self.conn.commit()

    def record_presence(self, camera_id: str, track_id: str, person_id: int | None,
                        ts: float | None = None):
        """Открыть отрезок присутствия трека на этой камере, если он ещё не открыт.

        Отдельно от позы и действий: маршрут и рабочее время строятся именно на нём, и
        он не должен рваться от того, что человек встал или сел.
        """
        ts = ts or time.time()
        with self.lock:
            row = self.conn.execute(
                "SELECT id, person_id FROM activity_segments"
                " WHERE camera_id=? AND track_id=? AND label_type='presence'"
                "   AND end_ts IS NULL LIMIT 1", (camera_id, track_id)).fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO activity_segments"
                    " (camera_id, track_id, person_id, label_type, label, start_ts)"
                    " VALUES (?,?,?,'presence',?,?)",
                    (camera_id, track_id, person_id, camera_id, ts))
            elif person_id is not None and row["person_id"] is None:
                self.conn.execute(
                    "UPDATE activity_segments SET person_id=? WHERE id=?",
                    (person_id, int(row["id"])))
            self.conn.commit()

    # --- зоны плана этажа ---------------------------------------------------
    # Отрезок зоны привязан к ЧЕЛОВЕКУ, а не к треку: camera_id хранит филиал,
    # track_id — «p<person_id>». Так передача имени между камерами его не рвёт,
    # а маршрут по зонам не зависит от того, какая камера сейчас держит имя.
    @staticmethod
    def _zone_track(person_id: int) -> str:
        return f"p{int(person_id)}"

    def record_zone(self, branch_id: str, person_id: int, zone: str | None,
                    ts: float | None = None) -> None:
        """Человек перешёл в зону (или вышел из всех, zone=None)."""
        ts = ts or time.time()
        tid = self._zone_track(person_id)
        with self.lock:
            row = self.conn.execute(
                "SELECT id, label FROM activity_segments"
                " WHERE track_id=? AND label_type='zone' AND end_ts IS NULL LIMIT 1",
                (tid,)).fetchone()
            if row is not None:
                if zone is not None and row["label"] == zone:
                    return
                self.conn.execute(
                    "UPDATE activity_segments SET end_ts=?, duration_sec=MAX(0, ?-start_ts)"
                    " WHERE id=?", (ts, ts, int(row["id"])))
            if zone is not None:
                self.conn.execute(
                    "INSERT INTO activity_segments"
                    " (camera_id, track_id, person_id, label_type, label, start_ts)"
                    " VALUES (?,?,?,'zone',?,?)",
                    (branch_id, tid, int(person_id), zone, ts))
            self.conn.commit()

    def close_zone(self, person_id: int, ts: float | None = None) -> None:
        self.record_zone("", person_id, None, ts)

    # --- след на плане ----------------------------------------------------
    def record_positions(self, rows: list) -> None:
        """Пачка точек (branch, person_id, ts, u, v) одной транзакцией.

        По одной вставке на точку не пишем: точки идут из четырёх потоков камер
        сразу, и коммит на каждую — это та же нагрузка, из-за которой аналитику
        когда-то и отключали.
        """
        if not rows:
            return
        with self.lock:
            self.conn.executemany(
                "INSERT INTO positions (branch_id, person_id, ts, u, v)"
                " VALUES (?,?,?,?,?)", rows)
            self.conn.commit()

    def positions(self, branch_id: str, t_from: float, t_to: float,
                  person_id: int | None = None, limit: int = 120000) -> list:
        sql = ("SELECT person_id, ts, u, v FROM positions"
               " WHERE branch_id=? AND ts >= ? AND ts < ?")
        args: list = [branch_id, t_from, t_to]
        if person_id is not None:
            sql += " AND person_id = ?"
            args.append(person_id)
        sql += " ORDER BY ts LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def prune_positions(self, days: float = 7.0) -> int:
        """След старше недели никто не отматывает, а строк он даёт больше всего."""
        cutoff = time.time() - days * 86400
        with self.lock:
            cur = self.conn.execute("DELETE FROM positions WHERE ts < ?", (cutoff,))
            self.conn.commit()
            return cur.rowcount

    def zone_routes(self, day_start: float, branch_id: str, person_id: int | None = None,
                    limit: int = 5000, day_end: float | None = None):
        sql = ("SELECT person_id, label, start_ts, end_ts, duration_sec"
               "  FROM activity_segments"
               " WHERE label_type='zone' AND camera_id=? AND person_id IS NOT NULL"
               "   AND start_ts >= ? AND start_ts < ?")
        args = [branch_id, day_start,
                day_end if day_end is not None else time.time() + 86400]
        if person_id is not None:
            sql += " AND person_id = ?"
            args.append(person_id)
        sql += " ORDER BY person_id, start_ts LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def close_stale_tracks(self, camera_id: str, active_track_ids: set[str], ts: float | None = None):
        ts = ts or time.time()
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, track_id, start_ts
                  FROM activity_segments
                 WHERE camera_id = ? AND end_ts IS NULL
                """,
                (camera_id,),
            ).fetchall()
            for row in rows:
                if row["track_id"] not in active_track_ids:
                    duration = max(0.0, ts - float(row["start_ts"]))
                    self.conn.execute(
                        "UPDATE activity_segments SET end_ts = ?, duration_sec = ? WHERE id = ?",
                        (ts, duration, int(row["id"])),
                    )
            self.conn.commit()

    def close_open_segments(self, ts: float | None = None) -> int:
        """Закрыть все отрезки, оставшиеся открытыми. Вызывается при остановке.

        Отрезок закрывает развёртка, когда трек пропал из кадра. Но при остановке
        программы треки не «пропадают» — исчезает тот, кто за ними следил, и отрезки
        остаются открытыми навсегда. А открытый отрезок в отчёте считается до конца
        суток: человек ушёл в шесть вечера, сервер выключили, и назавтра в графе
        «работал» стояли часы, которых не было.
        """
        ts = ts or time.time()
        with self.lock:
            cur = self.conn.execute(
                "UPDATE activity_segments"
                "   SET end_ts=?, duration_sec=MAX(0, ? - start_ts)"
                " WHERE end_ts IS NULL", (ts, ts))
            self.conn.commit()
            return cur.rowcount

    DOWNTIME_GAP = 30 * 60

    def repair_downtime(self) -> dict:
        """Убрать из отрезков время, когда программа не работала.

        Отрезок закрывается, когда трек пропал из кадра. При остановке программы треки
        не пропадают — пропадает тот, кто за ними следил. Отрезок остаётся открытым, а
        на следующем запуске развёртка закрывает его ТЕКУЩИМ временем: получается кусок
        с вечера до утра. Один такой отрезок дал человеку 15 часов 55 минут «работы»
        за ночь.

        Простой видно по самой базе: пока система работает, отрезки открываются
        непрерывно — за день их тысячи. Промежуток, в котором не начался ни один
        отрезок дольше получаса, — это остановка, а не затишье. Всё, что «шло» через
        такой провал, на самом деле шло только до его начала.

        Делается на каждом запуске: аварийная остановка обходит штатный путь, и
        чинить это руками пришлось бы после каждого падения.
        """
        with self.lock:
            starts = [r[0] for r in self.conn.execute(
                "SELECT start_ts FROM activity_segments ORDER BY start_ts")]
            gaps = [(a, b) for a, b in zip(starts, starts[1:])
                    if b - a > self.DOWNTIME_GAP]

            clipped = 0
            for g0, g1 in gaps:
                cur = self.conn.execute(
                    "UPDATE activity_segments"
                    "   SET end_ts=?, duration_sec=MAX(0, ? - start_ts)"
                    " WHERE start_ts <= ? AND (end_ts IS NULL OR end_ts > ?)",
                    (g0, g0, g0, g0))
                clipped += cur.rowcount

            # Всё, что осталось открытым и НЕ пересекает простой, — хвосты последнего
            # запуска: закрываем последним моментом, когда система точно работала.
            row = self.conn.execute(
                "SELECT MAX(end_ts) alive FROM activity_segments").fetchone()
            alive = row["alive"]
            cur = self.conn.execute(
                "UPDATE activity_segments"
                "   SET end_ts=MAX(start_ts, ?),"
                "       duration_sec=MAX(0, MAX(start_ts, ?) - start_ts)"
                " WHERE end_ts IS NULL", (alive, alive)) if alive is not None else \
                self.conn.execute(
                    "UPDATE activity_segments SET end_ts=start_ts, duration_sec=0"
                    " WHERE end_ts IS NULL")
            self.conn.commit()
            return {"gaps": len(gaps), "clipped": clipped, "closed": cur.rowcount}

    def list_persons(self):
        with self.lock:
            return [
                dict(row)
                for row in self.conn.execute(
                    "SELECT id, external_id, name, created_at, last_seen FROM persons ORDER BY name"
                ).fetchall()
            ]

    def recent_identity_events(self, limit: int = 50):
        with self.lock:
            return [
                dict(row)
                for row in self.conn.execute(
                    """
                    SELECT
                        e.id, e.camera_id, e.track_id, e.confidence, e.created_at,
                        p.id AS person_id, p.external_id, p.name
                    FROM identity_events e
                    JOIN persons p ON p.id = e.person_id
                    ORDER BY e.created_at DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
            ]

    def recent_segments(self, limit: int = 100):
        now_ts = time.time()
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT
                    s.id, s.camera_id, s.track_id, s.person_id,
                    COALESCE(p.external_id, NULL) AS external_id,
                    COALESCE(p.name, NULL) AS person_name,
                    s.label_type, s.label, s.start_ts, s.end_ts,
                    CASE
                        WHEN s.end_ts IS NULL THEN MAX(0, ? - s.start_ts)
                        ELSE s.duration_sec
                    END AS duration_sec
                FROM activity_segments s
                LEFT JOIN persons p ON p.id = s.person_id
                ORDER BY s.start_ts DESC
                LIMIT ?
                """,
                (now_ts, int(limit)),
            ).fetchall()
            return [dict(row) for row in rows]

    def daily_analytics(self, date_text: str | None = None):
        if date_text:
            day = datetime.strptime(date_text, "%Y-%m-%d")
        else:
            now = datetime.now()
            day = datetime(now.year, now.month, now.day)
        start = day.timestamp()
        end = (day + timedelta(days=1)).timestamp()
        now_ts = time.time()

        with self.lock:
            rows = self.conn.execute(
                """
                SELECT
                    s.person_id,
                    COALESCE(p.name, 'Unknown ' || s.track_id) AS person_name,
                    COALESCE(p.external_id, s.track_id) AS external_id,
                    s.label_type,
                    s.label,
                    SUM(
                        MAX(0, MIN(COALESCE(s.end_ts, ?), ?) - MAX(s.start_ts, ?))
                    ) AS seconds
                FROM activity_segments s
                LEFT JOIN persons p ON p.id = s.person_id
                WHERE s.start_ts < ? AND COALESCE(s.end_ts, ?) > ?
                GROUP BY s.person_id, external_id, person_name, s.label_type, s.label
                ORDER BY person_name, s.label_type, s.label
                """,
                (now_ts, end, start, end, now_ts, start),
            ).fetchall()

        people: dict[str, dict] = {}
        for row in rows:
            person_key = str(row["person_id"] or row["external_id"])
            item = people.setdefault(
                person_key,
                {
                    "person_id": row["person_id"],
                    "external_id": row["external_id"],
                    "name": row["person_name"],
                    "labels": {},
                },
            )
            seconds = float(row["seconds"] or 0.0)
            item["labels"][f"{row['label_type']}:{row['label']}"] = round(seconds, 2)

        for item in people.values():
            posture_total = sum(
                seconds for key, seconds in item["labels"].items()
                if key.startswith("posture:")
            )
            activity_total = sum(
                seconds for key, seconds in item["labels"].items()
                if key.startswith("activity:")
            )
            base = max(posture_total, activity_total, 1.0)
            item["percentages"] = {
                key: round(seconds * 100.0 / base, 2)
                for key, seconds in item["labels"].items()
            }
        return {"date": day.strftime("%Y-%m-%d"), "people": list(people.values())}


store = AnalyticsStore()
