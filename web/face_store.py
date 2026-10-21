"""Gallery of faces seen by the cameras, with automatic grouping by identity.

Flow this serves:
    camera captures a face -> recognition model gives a 512-number vector ->
    compare against everyone already known ->
        close enough  -> same person, attach the sighting to them
        not close     -> a new person, auto-numbered
    an admin can later replace the number with a real name

Why samples and not one vector per person
-----------------------------------------
The same face from a different angle or under different light produces a noticeably
different vector. Keeping only the first one means the second sighting of the same
person often fails to match and creates a duplicate. So every accepted sighting is
stored, and a new face is compared against ALL of them - the best match wins.
SAMPLES_PER_PERSON caps how many are kept so the comparison stays cheap and one
long visit cannot flood the gallery.

About the threshold
-------------------
MATCH_THRESHOLD decides whether two captures are the same person, and it is the one
number that has to be tuned on real data:
  too high -> the same person collects several numbers
  too low  -> two different people end up merged, and that is hard to undo
It cannot be copied from another system: it depends on the recognition model
(antelopev2/glintr100 here) and on how the cameras see faces. Every decision is logged
with its similarity so the distribution can be looked at and the number corrected.
"""

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "faces.sqlite3"
CROP_DIR = Path(__file__).resolve().parent.parent / "data" / "face_crops"

# 0.40, а не 0.45. На 0.45 узнавание проходило впритык: davo брался с 0.471, то есть
# чуть другой ракурс — и человек уже получал новый номер. Замеренные попадания лежали
# в 0.47..0.79, так что запас вниз есть.
# Меняется переменной окружения, без правки кода: число подбирается на живых данных, и
# перекомпилировать ради него нечего.
MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.40"))
SAMPLES_PER_PERSON = int(os.environ.get("FACE_SAMPLES_PER_PERSON", "20"))
# На сколько названному человеку разрешено проиграть безымянной карточке и всё равно
# выиграть: разброс похожести у ОДНОГО лица от кадра к кадру примерно такой и есть.
NAMED_MARGIN = float(os.environ.get("FACE_NAMED_MARGIN", "0.10"))


class FaceStore:
    def __init__(self, db_path: Path = DB_PATH, threshold: float = MATCH_THRESHOLD):
        self.db_path = db_path
        self.threshold = threshold
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        CROP_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL + NORMAL, как и у аналитики: засечки одежды и филиалы пишутся из горячего
        # цикла, и fsync на каждом commit там ни к чему. Копировать базу на ходу —
        # только через conn.backup(), простой copy не увидит хвост в -wal.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        # Whole gallery in memory: a few thousand vectors is under 10 MB and turns
        # matching into one matrix multiply. SQLite stays the durable copy.
        self._emb = np.zeros((0, 512), dtype=np.float32)
        self._owner = np.zeros(0, dtype=np.int64)
        self._n = 0                      # сколько строк матрицы реально заняты
        self._named_cache = None
        self._named_mask = None
        self._load()

    def _init_schema(self):
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS face_persons (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                label       TEXT,
                created_at  REAL NOT NULL,
                last_seen   REAL,
                seen_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS face_samples (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   INTEGER NOT NULL,
                embedding   BLOB NOT NULL,
                face_score  REAL,
                det_score   REAL,
                camera_id   TEXT,
                crop_file   TEXT,
                similarity  REAL,
                created_at  REAL NOT NULL,
                FOREIGN KEY (person_id) REFERENCES face_persons(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_samples_person ON face_samples(person_id);
            """
        )
        # Во что человек одет СЕГОДНЯ — ОДНА запись на человека в день. Пишется один
        # раз, с самого удачного кадра: когда все точки скелета уверенны разом и лицо
        # узнано уверенно, то есть человек стоит прямо и смотрит в камеру. Завтра
        # запись делается заново.
        old = {r["name"] for r in self._conn.execute("PRAGMA table_info(person_clothing)")}
        if old and "slot" in old:
            # Прежняя схема копила счётчики по каждому цвету; данные суточные и
            # одноразовые, переносить нечего.
            self._conn.execute("DROP TABLE person_clothing")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS person_clothing (
                person_id   INTEGER NOT NULL,
                day         TEXT NOT NULL,
                upper       TEXT,
                lower       TEXT,
                face_sim    REAL,
                kp_conf     REAL,
                camera_id   TEXT,
                created_at  REAL,
                PRIMARY KEY (person_id, day)
            );
            CREATE INDEX IF NOT EXISTS ix_clothing_day ON person_clothing(day);
            -- Отдельные удачные засечки до того, как цвет признан за человеком.
            -- Одной мало: она может прийтись на неудачный свет или на чужого, кого
            -- лицо приняло за своего. Цвет попадает в person_clothing только когда
            -- несколько засечек подряд говорят одно и то же.
            CREATE TABLE IF NOT EXISTS person_clothing_seen (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   INTEGER NOT NULL,
                day         TEXT NOT NULL,
                upper       TEXT,
                lower       TEXT,
                face_sim    REAL,
                kp_conf     REAL,
                camera_id   TEXT,
                created_at  REAL
            );
            CREATE INDEX IF NOT EXISTS ix_seen_person
                ON person_clothing_seen(person_id, day);
            """
        )
        # Полный кадр события: по вырезке не видно, что вокруг человека происходило,
        # а именно это и хотят увидеть, открыв карточку.
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(face_samples)")}
        if "frame_file" not in cols:
            self._conn.execute("ALTER TABLE face_samples ADD COLUMN frame_file TEXT")
        if "bbox" not in cols:
            self._conn.execute("ALTER TABLE face_samples ADD COLUMN bbox TEXT")

        # Карточка человека — те же поля, что ведёт работающая система на .179:
        # тип (сотрудник/посетитель/неизвестный), отдел, должность, телефон, заметка.
        # Заполняет их оператор; всё остальное считается само.
        pcols = {r["name"] for r in self._conn.execute("PRAGMA table_info(face_persons)")}
        for col in ("person_type", "department", "position", "phone", "notes",
                    "last_camera"):
            if col not in pcols:
                self._conn.execute(f"ALTER TABLE face_persons ADD COLUMN {col} TEXT")

        # Филиалы. Человек один на всю базу — карточка, снимки и векторы общие, — а вот
        # ГДЕ его видели, у каждого филиала своё. Поэтому не колонка «филиал» в карточке
        # (она хранила бы только последний), а отдельная строка на каждый филиал, где
        # человек появлялся: по ней и страница филиала, и общая база с пометками.
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS person_branches (
                person_id   INTEGER NOT NULL,
                branch_id   TEXT NOT NULL,
                first_seen  REAL,
                last_seen   REAL,
                seen_count  INTEGER NOT NULL DEFAULT 0,
                last_camera TEXT,
                PRIMARY KEY (person_id, branch_id)
            );
            CREATE INDEX IF NOT EXISTS ix_pbranch_branch ON person_branches(branch_id);
            -- Один и тот же человек в двух филиалах. Тревога хранится, а не только
            -- показывается: оператор мог отойти от экрана, а вопрос «он был и там,
            -- и там» задают потом, а не в ту же секунду.
            CREATE TABLE IF NOT EXISTS branch_alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id    INTEGER NOT NULL,
                branch_id    TEXT NOT NULL,
                other_branch TEXT NOT NULL,
                other_seen   REAL,
                camera_id    TEXT,
                created_at   REAL NOT NULL,
                acked        INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS ix_alerts_open ON branch_alerts(acked, id);
            """
        )
        self._conn.commit()

    def _load(self):
        rows = self._conn.execute(
            "SELECT person_id, embedding FROM face_samples ORDER BY id").fetchall()
        if rows:
            self._set_gallery(
                np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]),
                np.fromiter((int(r["person_id"]) for r in rows), dtype=np.int64,
                            count=len(rows)))
        logger.info(f"face store: {len(np.unique(self._owner_view()))} persons, "
                    f"{self._n} samples, threshold {self.threshold}")

    # --- галерея в памяти -------------------------------------------------
    # Матрица держится с ЗАПАСОМ строк, а не пересобирается на каждое добавление.
    # Раньше здесь был np.vstack на всю галерею: при ста тысячах человек это
    # копирование четырёх гигабайт на каждый新 снимок. Теперь дописываем в свободную
    # строку, а когда запас кончился — удваиваем ёмкость.
    def _set_gallery(self, emb: np.ndarray, owner: np.ndarray) -> None:
        self._emb = np.ascontiguousarray(emb, dtype=np.float32)
        self._owner = np.ascontiguousarray(owner, dtype=np.int64)
        self._n = len(owner)
        self._named_mask = None          # строки переехали — маска больше не про них

    def _emb_view(self) -> np.ndarray:
        return self._emb[:self._n]

    def _owner_view(self) -> np.ndarray:
        return self._owner[:self._n]

    def _append(self, vec: np.ndarray, person_id: int) -> None:
        if self._n >= len(self._owner):
            cap = max(1024, len(self._owner) * 2)
            grown = np.zeros((cap, 512), dtype=np.float32)
            grown[:self._n] = self._emb[:self._n]
            self._emb = grown
            owners = np.zeros(cap, dtype=np.int64)
            owners[:self._n] = self._owner[:self._n]
            self._owner = owners
            mask = np.zeros(cap, dtype=bool)
            if self._named_mask is not None:
                mask[:self._n] = self._named_mask[:self._n]
                self._named_mask = mask
        self._emb[self._n] = vec
        self._owner[self._n] = person_id
        if self._named_mask is not None:
            # Знаем владельца — знаем и ответ, пересобирать всю маску незачем.
            self._named_mask[self._n] = person_id in self._named_ids()
        self._n += 1

    def _named_row_mask(self) -> np.ndarray:
        """Для каждой строки галереи: принадлежит ли она названному человеку.

        Считается один раз на изменение имён. Раньше np.isin по двум миллионам строк
        выполнялся на КАЖДОМ распознавании и стоил около 18 мс — больше, чем всё
        остальное сравнение вместе взятое.
        """
        if self._named_mask is None:
            named = self._named_ids()
            full = np.zeros(len(self._owner), dtype=bool)
            if len(named):
                full[:self._n] = np.isin(self._owner_view(), named)
            self._named_mask = full
        return self._named_mask[:self._n]

    # ------------------------------------------------------------------
    def _named_ids(self) -> np.ndarray:
        """Кто из людей уже назван. Спрашивается на КАЖДОМ распознавании, а меняется
        только когда оператор переименовал, слил или удалил человека, — поэтому ответ
        держим до следующего такого события."""
        if self._named_cache is None:
            self._named_cache = np.fromiter(
                (int(r["id"]) for r in self._conn.execute(
                    "SELECT id FROM face_persons WHERE label IS NOT NULL AND label<>''")),
                dtype=np.int64)
        return self._named_cache

    def _forget_named(self) -> None:
        self._named_cache = None
        self._named_mask = None

    def match(self, embedding: np.ndarray):
        """Best match in the gallery. Returns (person_id, similarity) or (None, best).

        When the nearest card is unnamed but a NAMED card is nearly as close, the named
        one wins. The same person routinely ends up with both: he was captured and
        enrolled as #35 before anyone typed his name on another card. Then the plain
        nearest-neighbour answer alternates between "vazgen" (0.78) and "#35" (0.75)
        from one second to the next, and the name over his head blinks on and off.
        Preferring the named card settles that, and the operator can still fold #35 in
        by typing the same name on it.
        """
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if self._n == 0:
            return None, 0.0
        # Embeddings come out of the model already L2-normalised, so a dot product
        # is the cosine similarity.
        owners = self._owner_view()
        sims = self._emb_view() @ vec
        i = int(np.argmax(sims))
        best = float(sims[i])
        if best < self.threshold:
            return None, best

        owner = int(owners[i])
        named = self._named_ids()
        if len(named) and owner not in named:
            # Раньше здесь был проход по галерее питоновским циклом: на двух миллионах
            # векторов он занимал 127 мс из 210 — больше, чем само перемножение матриц.
            # Теперь готовая маска, посчитанная один раз на изменение имён.
            mask = self._named_row_mask()
            if mask.any():
                near = np.where(mask, sims, -1.0)
                j = int(np.argmax(near))
                s = float(near[j])
                if s >= self.threshold and s >= best - NAMED_MARGIN:
                    return int(owners[j]), s
        return owner, best

    @staticmethod
    def _remove_files(names) -> int:
        gone = 0
        for name in names:
            if not name:
                continue
            try:
                (CROP_DIR / name).unlink()
                gone += 1
            except OSError:
                pass
        return gone

    def prune_images(self) -> dict:
        """Убрать с диска всё, на что база не ссылается, и лишние снимки людей.

        Две разные кучи мусора. Первая — файлы-сироты: снимок писался до проверки
        лимита образцов, и когда лимит был исчерпан, файл оставался на диске без
        строки. Вторая — вторые и последующие снимки одного человека: теперь на
        карточку нужен один, самый чёткий.
        """
        with self._lock:
            # У каждого человека оставляем строку с лучшим det_score, у остальных
            # ссылки на файлы снимаем.
            self._conn.execute(
                "UPDATE face_samples SET crop_file=NULL, frame_file=NULL"
                " WHERE crop_file IS NOT NULL AND id NOT IN ("
                "   SELECT id FROM face_samples s WHERE crop_file IS NOT NULL"
                "     AND s.det_score = (SELECT MAX(det_score) FROM face_samples"
                "                        WHERE person_id = s.person_id"
                "                          AND crop_file IS NOT NULL)"
                "     GROUP BY s.person_id)")
            self._conn.commit()
            referenced = {r[0] for r in self._conn.execute(
                "SELECT crop_file FROM face_samples WHERE crop_file IS NOT NULL")}
            referenced |= {r[0] for r in self._conn.execute(
                "SELECT frame_file FROM face_samples WHERE frame_file IS NOT NULL")}

        removed = size = 0
        for f in CROP_DIR.glob("*.jpg"):
            if f.name in referenced:
                continue
            try:
                size += f.stat().st_size
                f.unlink()
                removed += 1
            except OSError:
                pass
        logger.info(f"face store: pruned {removed} unused images "
                    f"({size / 1e6:.0f} MB), {len(referenced)} kept")
        return {"removed": removed, "freed_mb": round(size / 1e6, 1),
                "kept": len(referenced)}

    def add_sighting(self, embedding, face_score=None, det_score=None,
                     camera_id=None, crop: bytes | None = None,
                     frame: bytes | None = None, bbox=None):
        """Attach a capture to whoever it matches, or open a new person for it."""
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        now = time.time()
        with self._lock:
            person_id, similarity = self.match(vec)
            is_new = person_id is None
            cur = self._conn.cursor()
            if is_new:
                cur.execute(
                    "INSERT INTO face_persons (label, created_at, last_seen, seen_count) "
                    "VALUES (NULL, ?, ?, 0)", (now, now))
                person_id = int(cur.lastrowid)

            stamp = int(now * 1000)

            def _save(data, prefix):
                if not data:
                    return None
                name = f"{prefix}{person_id}_{stamp}.jpg"
                try:
                    (CROP_DIR / name).write_bytes(data)
                    return name
                except OSError as e:
                    logger.error(f"face store: cannot save {prefix}: {e}")
                    return None

            # Картинку храним ОДНУ на человека — ту, где лицо видно лучше всего.
            # Смысла в сотне почти одинаковых снимков нет: оператору нужно узнать
            # человека в лицо, а для узнавания машиной служат векторы, не файлы.
            # Раньше файл писался ДО проверки лимита, и когда у человека уже было
            # 20 образцов, снимок ложился на диск, а строка про него — нет. Так
            # накопилось 28 тысяч файлов, на которые никто не ссылается.
            best = self._conn.execute(
                "SELECT id, crop_file, det_score FROM face_samples"
                " WHERE person_id=? AND crop_file IS NOT NULL"
                " ORDER BY det_score DESC LIMIT 1", (person_id,)).fetchone()
            better = best is None or (det_score or 0) > (best["det_score"] or 0)

            crop_file = _save(crop, "p") if (crop and better) else None
            frame_file = _save(frame, "f") if (frame and better) else None

            if crop_file and best is not None:
                # Новый снимок лучше — старый больше не нужен ни в базе, ни на диске.
                self._conn.execute(
                    "UPDATE face_samples SET crop_file=NULL, frame_file=NULL WHERE id=?",
                    (int(best["id"]),))
                self._remove_files([best["crop_file"]])

            keep = self._conn.execute(
                "SELECT COUNT(*) c FROM face_samples WHERE person_id=?",
                (person_id,)).fetchone()["c"]
            if keep < SAMPLES_PER_PERSON:
                cur.execute(
                    "INSERT INTO face_samples (person_id, embedding, face_score, det_score,"
                    " camera_id, crop_file, frame_file, bbox, similarity, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (person_id, vec.tobytes(), face_score, det_score, camera_id,
                     crop_file, frame_file,
                     ",".join(str(int(v)) for v in bbox) if bbox else None,
                     None if is_new else similarity, now))
                self._append(vec, person_id)
            elif crop_file:
                # Лимит образцов достигнут, но снимок лучше прежнего: вектор не берём,
                # а картинку обновляем — иначе лицо в карточке навсегда останется тем,
                # каким его увидели в первый раз.
                self._conn.execute(
                    "UPDATE face_samples SET crop_file=?, frame_file=?, det_score=?"
                    " WHERE id=(SELECT id FROM face_samples WHERE person_id=?"
                    "           ORDER BY det_score DESC LIMIT 1)",
                    (crop_file, frame_file, det_score, person_id))

            cur.execute(
                "UPDATE face_persons SET last_seen=?, seen_count=seen_count+1,"
                " last_camera=? WHERE id=?", (now, camera_id, person_id))
            self._conn.commit()

        return {"person_id": person_id, "is_new": is_new,
                "similarity": round(similarity, 4),
                "label": self.label_of(person_id)}

    # ------------------------------------------------------------------
    def label_of(self, person_id: int) -> str:
        row = self._conn.execute("SELECT label FROM face_persons WHERE id=?",
                                 (person_id,)).fetchone()
        if row and row["label"]:
            return row["label"]
        return f"#{person_id}"

    def rename(self, person_id: int, label: str):
        """Give a person a name — and if that name is already taken, fold the two together.

        This is how a split gets repaired. The recogniser keeps a person apart when the
        two sightings do not look similar enough (a face straight on and the same face in
        profile easily land below the threshold), so the operator types the same name on
        both cards and they become one entry.

        The samples of BOTH are kept: the whole point is that the person is now known by
        two different-looking vectors, and either of them will match next time. Throwing
        one away would leave the same gap that caused the split.
        """
        label = (label or "").strip()
        if not label:
            with self._lock:
                self._forget_named()
                self._conn.execute("UPDATE face_persons SET label=NULL WHERE id=?",
                                   (person_id,))
                self._conn.commit()
            return {"ok": True, "person_id": person_id, "label": self.label_of(person_id)}

        with self._lock:
            twin = self._conn.execute(
                "SELECT id FROM face_persons WHERE label IS NOT NULL"
                " AND lower(trim(label))=lower(?) AND id<>? ORDER BY id LIMIT 1",
                (label, person_id)).fetchone()

        if twin:
            keep = int(twin["id"])
            merged = self.merge(person_id, keep)          # takes the lock itself
            logger.info(f"face store: '{label}' already exists as #{keep}; "
                        f"folded #{person_id} into it, samples kept from both")
            return {"ok": True, "person_id": keep, "label": label,
                    "merged_from": person_id, "samples": merged["samples"]}

        with self._lock:
            self._forget_named()
            self._conn.execute("UPDATE face_persons SET label=? WHERE id=?",
                               (label, person_id))
            self._conn.commit()
        return {"ok": True, "person_id": person_id, "label": label}

    def list_persons(self, limit: int = 500, named: bool | None = None,
                     only_ids=None):
        """Люди, самые свежие первыми.

        Отбор идёт в запросе, а не после него, и это не украшательство: страница берёт
        двести человек, а безымянных в базе под четыреста. Отфильтруй их в Python —
        и во вкладке «названные» окажется горстка тех, кто случайно попал в эти
        двести, а остальные просто исчезнут из списка.
        """
        sql = ("SELECT p.id, p.label, p.created_at, p.last_seen, p.seen_count,"
               " p.person_type, p.department, p.position, p.phone, p.notes, p.last_camera,"
               " (SELECT COUNT(*) FROM face_samples s WHERE s.person_id=p.id) samples,"
               " (SELECT crop_file FROM face_samples s WHERE s.person_id=p.id"
               "   AND crop_file IS NOT NULL ORDER BY det_score DESC LIMIT 1) crop"
               " FROM face_persons p WHERE 1=1")
        args: list = []
        if named is True:
            sql += " AND p.label IS NOT NULL AND p.label<>''"
        elif named is False:
            sql += " AND (p.label IS NULL OR p.label='')"
        if only_ids is not None:
            ids = [int(i) for i in only_ids]
            if not ids:
                return []
            sql += f" AND p.id IN ({','.join('?' * len(ids))})"
            args += ids
        sql += " ORDER BY p.last_seen DESC LIMIT ?"
        args.append(limit)
        rows = self._conn.execute(sql, args).fetchall()
        return [{"id": r["id"], "label": r["label"] or f"#{r['id']}",
                 "named": bool(r["label"]), "samples": r["samples"],
                 "seen_count": r["seen_count"], "last_seen": r["last_seen"],
                 "first_seen": r["created_at"], "crop": r["crop"],
                 "person_type": r["person_type"] or ("known" if r["label"] else "unknown"),
                 "department": r["department"], "position": r["position"],
                 "phone": r["phone"], "notes": r["notes"],
                 "last_camera": r["last_camera"]} for r in rows]

    def count_persons(self, named: bool | None = None, only_ids=None) -> int:
        """Сколько людей под этот отбор. Нужно вкладкам: цифру на них нельзя брать из
        общей статистики, когда список ограничен филиалом."""
        sql = "SELECT COUNT(*) c FROM face_persons p WHERE 1=1"
        args: list = []
        if named is True:
            sql += " AND p.label IS NOT NULL AND p.label<>''"
        elif named is False:
            sql += " AND (p.label IS NULL OR p.label='')"
        if only_ids is not None:
            ids = [int(i) for i in only_ids]
            if not ids:
                return 0
            sql += f" AND p.id IN ({','.join('?' * len(ids))})"
            args += ids
        return int(self._conn.execute(sql, args).fetchone()["c"])

    def update_person(self, person_id: int, **fields):
        """Operator-editable card fields. Name goes through rename() so that typing an
        existing name still merges the two entries."""
        allowed = ("person_type", "department", "position", "phone", "notes")
        sets, vals = [], []
        for k in allowed:
            if k in fields:
                sets.append(f"{k}=?")
                v = fields[k]
                vals.append(str(v).strip() or None if v is not None else None)
        out = {"ok": True, "person_id": person_id}
        if sets:
            with self._lock:
                self._conn.execute(
                    f"UPDATE face_persons SET {', '.join(sets)} WHERE id=?",
                    (*vals, person_id))
                self._conn.commit()
        if "name" in fields:
            out.update(self.rename(person_id, fields["name"]))
        return out

    def person_samples(self, person_id: int):
        """Every capture kept for this person, newest first.

        After a merge these come from what used to be two entries, which is exactly what
        makes the list worth looking at: it shows the different angles the person is now
        known by.
        """
        rows = self._conn.execute(
            "SELECT id, crop_file, frame_file, bbox, camera_id, face_score, det_score,"
            " similarity, created_at FROM face_samples WHERE person_id=?"
            " ORDER BY created_at DESC", (person_id,)).fetchall()
        return [{"id": r["id"], "crop": r["crop_file"], "frame": r["frame_file"],
                 "bbox": [int(v) for v in r["bbox"].split(",")] if r["bbox"] else None,
                 "camera_id": r["camera_id"], "face_score": r["face_score"],
                 "det_score": r["det_score"], "similarity": r["similarity"],
                 "created_at": r["created_at"]} for r in rows]

    def crop_path(self, name: str) -> Path | None:
        p = (CROP_DIR / name).resolve()
        return p if p.is_file() and CROP_DIR.resolve() in p.parents else None

    def merge(self, src_id: int, dst_id: int):
        """Fold one person into another, keeping every sample from both.

        Both sets of vectors stay in the gallery on purpose: they are what the same face
        looks like from different angles, and keeping both is what stops the split from
        happening again.
        """
        if src_id == dst_id:
            return {"ok": True, "merged": src_id, "into": dst_id, "samples": 0}
        with self._lock:
            self._conn.execute("UPDATE face_samples SET person_id=? WHERE person_id=?",
                               (dst_id, src_id))
            # Sighting counts add up; the earlier first-seen wins.
            self._conn.execute(
                "UPDATE face_persons SET"
                "  seen_count = seen_count + COALESCE((SELECT seen_count FROM face_persons"
                "                                      WHERE id=?), 0),"
                "  last_seen  = MAX(COALESCE(last_seen,0),"
                "                   COALESCE((SELECT last_seen FROM face_persons WHERE id=?), 0))"
                " WHERE id=?", (src_id, src_id, dst_id))
            self._conn.execute("DELETE FROM face_persons WHERE id=?", (src_id,))

            # Филиалы тоже складываются: две карточки были одним человеком, значит и
            # ходил он по обоим объектам. Сперва переносим те филиалы, которых у цели
            # ещё нет, потом дописываем счётчики к общим, а остаток убираем.
            self._conn.execute(
                "UPDATE OR IGNORE person_branches SET person_id=? WHERE person_id=?",
                (dst_id, src_id))
            self._conn.execute(
                "UPDATE person_branches AS d SET"
                "  first_seen = MIN(COALESCE(d.first_seen, 1e18),"
                "                   COALESCE((SELECT s.first_seen FROM person_branches s"
                "                             WHERE s.person_id=? AND s.branch_id=d.branch_id),"
                "                            1e18)),"
                "  last_seen  = MAX(COALESCE(d.last_seen, 0),"
                "                   COALESCE((SELECT s.last_seen FROM person_branches s"
                "                             WHERE s.person_id=? AND s.branch_id=d.branch_id),"
                "                            0)),"
                "  seen_count = d.seen_count"
                "               + COALESCE((SELECT s.seen_count FROM person_branches s"
                "                           WHERE s.person_id=? AND s.branch_id=d.branch_id), 0)"
                " WHERE d.person_id=?", (src_id, src_id, src_id, dst_id))
            self._conn.execute("DELETE FROM person_branches WHERE person_id=?", (src_id,))
            self._conn.execute("UPDATE branch_alerts SET person_id=? WHERE person_id=?",
                               (dst_id, src_id))

            # Векторы объединяются все — ради этого слияние и делается. А вот снимков
            # у объединённого человека станет два, по одному от каждой карточки:
            # оставляем тот, где лицо чётче, второй убираем и с диска тоже.
            keep = self._conn.execute(
                "SELECT id FROM face_samples WHERE person_id=? AND crop_file IS NOT NULL"
                " ORDER BY det_score DESC LIMIT 1", (dst_id,)).fetchone()
            if keep is not None:
                extra = [r["crop_file"] for r in self._conn.execute(
                    "SELECT crop_file FROM face_samples WHERE person_id=?"
                    " AND crop_file IS NOT NULL AND id<>?", (dst_id, int(keep["id"])))]
                self._conn.execute(
                    "UPDATE face_samples SET crop_file=NULL, frame_file=NULL"
                    " WHERE person_id=? AND id<>?", (dst_id, int(keep["id"])))
                self._remove_files(extra)

            self._conn.commit()
            owners = self._owner_view()
            owners[owners == src_id] = dst_id
            self._forget_named()
            total = self._conn.execute(
                "SELECT COUNT(*) c FROM face_samples WHERE person_id=?",
                (dst_id,)).fetchone()["c"]
        return {"ok": True, "merged": src_id, "into": dst_id, "samples": total}

    def delete_unnamed(self) -> dict:
        """Убрать всех, кому оператор так и не дал имени.

        Безымянные карточки — это сырьё: система заводит их на каждое незнакомое лицо,
        чтобы человеку было кого назвать. Что не назвали за день, то и не нужно, а
        накапливается их сотнями. Названные люди не затрагиваются.
        """
        with self._lock:
            ids = [int(r["id"]) for r in self._conn.execute(
                "SELECT id FROM face_persons WHERE label IS NULL OR label=''")]
            if not ids:
                return {"ok": True, "persons": 0, "samples": 0, "files": 0}

            marks = ",".join("?" * len(ids))
            files = [r[k] for r in self._conn.execute(
                f"SELECT crop_file, frame_file FROM face_samples"
                f" WHERE person_id IN ({marks})", ids)
                for k in ("crop_file", "frame_file") if r[k]]
            samples = self._conn.execute(
                f"SELECT COUNT(*) c FROM face_samples WHERE person_id IN ({marks})",
                ids).fetchone()["c"]

            self._conn.execute(
                f"DELETE FROM face_samples WHERE person_id IN ({marks})", ids)
            self._conn.execute(
                f"DELETE FROM face_persons WHERE id IN ({marks})", ids)
            for table in ("person_clothing", "person_clothing_seen",
                          "person_branches", "branch_alerts"):
                self._conn.execute(
                    f"DELETE FROM {table} WHERE person_id IN ({marks})", ids)
            self._conn.commit()

            gone_files = self._remove_files(files)

            # Матрицу в памяти пересобираем из того, что осталось: удалять из неё по
            # одному дороже и легче ошибиться.
            keep = ~np.isin(self._owner_view(), np.array(ids, dtype=np.int64))
            self._set_gallery(self._emb_view()[keep], self._owner_view()[keep])
            self._forget_named()

        logger.info(f"face store: removed {len(ids)} unnamed people, "
                    f"{samples} samples, {gone_files} files")
        return {"ok": True, "persons": len(ids), "samples": samples,
                "files": gone_files}

    def delete(self, person_id: int):
        with self._lock:
            # Снимки на диске — тоже данные человека. Имена файлов знает только эта
            # таблица, поэтому собираем их ДО удаления строк, иначе они останутся
            # лежать навсегда, и никто уже не свяжет их с человеком.
            files = [r[k] for r in self._conn.execute(
                "SELECT crop_file, frame_file FROM face_samples WHERE person_id=?",
                (person_id,)) for k in ("crop_file", "frame_file") if r[k]]
            removed = self._remove_files(files)
            self._conn.execute("DELETE FROM face_samples WHERE person_id=?", (person_id,))
            self._conn.execute("DELETE FROM face_persons WHERE id=?", (person_id,))
            # Одежда — тоже данные этого человека. Оставлять её висеть незачем: карточки
            # уже нет, а запись продолжала бы участвовать в проверке имён.
            for table in ("person_clothing", "person_clothing_seen",
                          "person_branches", "branch_alerts"):
                self._conn.execute(f"DELETE FROM {table} WHERE person_id=?", (person_id,))
            self._conn.commit()
            keep = self._owner_view() != person_id
            self._set_gallery(self._emb_view()[keep], self._owner_view()[keep])
            self._forget_named()
        logger.info(f"face store: deleted person {person_id}, {removed} image files")
        return {"ok": True, "deleted": person_id, "files_removed": removed}

    # --- филиалы ----------------------------------------------------------
    def note_branch(self, person_id: int, branch_id: str, camera_id: str = "",
                    now: float | None = None, window: float = 43200.0,
                    repeat: float = 900.0) -> dict | None:
        """Отметить человека в филиале и сказать, не был ли он только что в другом.

        Проверка делается ДО записи: нас интересует, где человека видели в прошлый раз,
        а не сейчас. Окно нужно, потому что «был в двух филиалах» без времени — это про
        любого сотрудника за всю историю; тревога же про то, что он ходит между
        объектами прямо сегодня.

        Повтор придушен: человек стоит перед камерой минутами, и без этого одна и та же
        тревога писалась бы по разу на каждое узнавание.
        """
        if not branch_id:
            return None
        now = time.time() if now is None else now
        with self._lock:
            prev = self._conn.execute(
                "SELECT branch_id, last_seen FROM person_branches"
                " WHERE person_id=? AND branch_id<>? AND last_seen>=?"
                " ORDER BY last_seen DESC LIMIT 1",
                (person_id, branch_id, now - window)).fetchone()

            self._conn.execute(
                "INSERT INTO person_branches"
                " (person_id, branch_id, first_seen, last_seen, seen_count, last_camera)"
                " VALUES (?,?,?,?,1,?)"
                " ON CONFLICT(person_id, branch_id) DO UPDATE SET"
                "   last_seen=excluded.last_seen,"
                "   seen_count=seen_count+1,"
                "   last_camera=excluded.last_camera",
                (person_id, branch_id, now, now, camera_id or None))

            alert = None
            if prev is not None:
                recent = self._conn.execute(
                    "SELECT created_at FROM branch_alerts"
                    " WHERE person_id=? AND branch_id=? AND other_branch=?"
                    " ORDER BY id DESC LIMIT 1",
                    (person_id, branch_id, prev["branch_id"])).fetchone()
                if recent is None or now - recent["created_at"] >= repeat:
                    cur = self._conn.execute(
                        "INSERT INTO branch_alerts (person_id, branch_id, other_branch,"
                        " other_seen, camera_id, created_at) VALUES (?,?,?,?,?,?)",
                        (person_id, branch_id, prev["branch_id"],
                         prev["last_seen"], camera_id or None, now))
                    alert = {"id": int(cur.lastrowid), "person_id": person_id,
                             "branch_id": branch_id,
                             "other_branch": prev["branch_id"],
                             "other_seen": prev["last_seen"], "created_at": now}
            self._conn.commit()
        return alert

    def branches_of(self, person_ids) -> dict:
        """Где видели каждого из этих людей. Одним запросом на всю страницу: по
        человеку на запрос это две сотни обращений на каждое обновление списка."""
        ids = [int(p) for p in person_ids]
        if not ids:
            return {}
        out: dict[int, list] = {}
        marks = ",".join("?" * len(ids))
        for r in self._conn.execute(
                f"SELECT person_id, branch_id, first_seen, last_seen, seen_count"
                f" FROM person_branches WHERE person_id IN ({marks})"
                f" ORDER BY last_seen DESC", ids):
            out.setdefault(int(r["person_id"]), []).append(
                {"branch": r["branch_id"], "first_seen": r["first_seen"],
                 "last_seen": r["last_seen"], "count": r["seen_count"]})
        return out

    def persons_in_branch(self, branch_id: str) -> set:
        return {int(r["person_id"]) for r in self._conn.execute(
            "SELECT person_id FROM person_branches WHERE branch_id=?", (branch_id,))}

    def branch_alerts(self, branch_id: str = "", include_acked: bool = False,
                      limit: int = 50) -> list:
        sql = ("SELECT a.*, p.label FROM branch_alerts a"
               " LEFT JOIN face_persons p ON p.id = a.person_id WHERE 1=1")
        args: list = []
        if branch_id:
            # Тревога видна с обеих сторон: и там, где человека застали сейчас, и там,
            # откуда он пришёл, — иначе второй пост о его перемещении не узнает.
            sql += " AND (a.branch_id=? OR a.other_branch=?)"
            args += [branch_id, branch_id]
        if not include_acked:
            sql += " AND a.acked=0"
        sql += " ORDER BY a.id DESC LIMIT ?"
        args.append(limit)
        return [{"id": r["id"], "person_id": r["person_id"],
                 "label": r["label"] or f"#{r['person_id']}",
                 "named": bool(r["label"]),
                 "branch": r["branch_id"], "other_branch": r["other_branch"],
                 "other_seen": r["other_seen"], "camera_id": r["camera_id"],
                 "created_at": r["created_at"], "acked": bool(r["acked"])}
                for r in self._conn.execute(sql, args)]

    def ack_alerts(self, ids=None, branch_id: str = "") -> int:
        with self._lock:
            if ids:
                marks = ",".join("?" * len(ids))
                cur = self._conn.execute(
                    f"UPDATE branch_alerts SET acked=1 WHERE id IN ({marks})",
                    [int(i) for i in ids])
            elif branch_id:
                cur = self._conn.execute(
                    "UPDATE branch_alerts SET acked=1"
                    " WHERE acked=0 AND (branch_id=? OR other_branch=?)",
                    (branch_id, branch_id))
            else:
                cur = self._conn.execute("UPDATE branch_alerts SET acked=1 WHERE acked=0")
            self._conn.commit()
            return cur.rowcount

    # --- одежда на день ---------------------------------------------------
    def set_clothing(self, person_id: int, day: str, upper: str, lower: str,
                     face_sim: float, kp_conf: float, camera_id: str) -> bool:
        """Записать одежду на день. Один раз: повторный вызов ничего не меняет."""
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO person_clothing"
            " (person_id, day, upper, lower, face_sim, kp_conf, camera_id, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (person_id, day, upper or None, lower or None,
             face_sim, kp_conf, camera_id, time.time()))
        self._conn.commit()
        return cur.rowcount > 0

    def add_clothing_seen(self, person_id: int, day: str, upper: str, lower: str,
                          face_sim: float, kp_conf: float, camera_id: str) -> list:
        """Записать удачную засечку и вернуть все засечки этого человека за день."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO person_clothing_seen"
                " (person_id, day, upper, lower, face_sim, kp_conf, camera_id, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (person_id, day, upper, lower, face_sim, kp_conf, camera_id, time.time()))
            self._conn.commit()
        return self.clothing_seen(person_id, day)

    def clothing_seen(self, person_id: int, day: str) -> list:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM person_clothing_seen WHERE person_id=? AND day=?"
            " ORDER BY created_at", (person_id, day))]

    def clothing_seen_counts(self, day: str) -> dict:
        return {int(r["person_id"]): int(r["n"]) for r in self._conn.execute(
            "SELECT person_id, COUNT(*) n FROM person_clothing_seen"
            " WHERE day=? GROUP BY person_id", (day,))}

    def replace_clothing(self, person_id: int, day: str, upper: str, lower: str,
                         face_sim: float, kp_conf: float, camera_id: str) -> bool:
        """Переписать одежду дня на новую — человек переоделся.

        Отдельно от set_clothing, который пишет только once: смена одежды среди дня
        бывает (снял пиджак, надел халат), и без этого запись за утро весь оставшийся
        день отменяла бы верные распознавания.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE person_clothing SET upper=?, lower=?, face_sim=?, kp_conf=?,"
                " camera_id=?, created_at=? WHERE person_id=? AND day=?",
                (upper, lower, face_sim, kp_conf, camera_id, time.time(),
                 person_id, day))
            self._conn.commit()
            return cur.rowcount > 0

    def clothing_seen_after(self, person_id: int, day: str, after_ts: float,
                            min_face_sim: float) -> list:
        """Засечки, сделанные ПОСЛЕ признания цвета и с уверенным лицом.

        По ним и решается, не переоделся ли человек: старые засечки к новому наряду
        отношения не имеют.
        """
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM person_clothing_seen"
            " WHERE person_id=? AND day=? AND created_at > ? AND face_sim >= ?"
            " ORDER BY created_at", (person_id, day, after_ts, min_face_sim))]

    def clothing_of_day(self, person_id: int, day: str):
        r = self._conn.execute(
            "SELECT * FROM person_clothing WHERE person_id=? AND day=?",
            (person_id, day)).fetchone()
        return dict(r) if r else None

    def clothing_day_map(self, day: str) -> dict:
        """Вся дневная память разом — читается редко, а спрашивается на каждом
        распознавании, поэтому в web_server она лежит в кэше."""
        return {int(r["person_id"]): dict(r) for r in self._conn.execute(
            "SELECT * FROM person_clothing WHERE day=?", (day,))}

    def prune_clothing(self, keep_days: int = 7) -> int:
        """Старше недели память не нужна: она уже не про сегодняшнюю одежду."""
        with self._lock:
            gone = 0
            for table in ("person_clothing", "person_clothing_seen"):
                cur = self._conn.execute(
                    f"DELETE FROM {table} WHERE day < date('now', ?)",
                    (f"-{int(keep_days)} day",))
                gone += cur.rowcount
            self._conn.commit()
            return gone

    def stats(self):
        p = self._conn.execute("SELECT COUNT(*) c FROM face_persons").fetchone()["c"]
        s = self._conn.execute("SELECT COUNT(*) c FROM face_samples").fetchone()["c"]
        named = self._conn.execute(
            "SELECT COUNT(*) c FROM face_persons WHERE label IS NOT NULL").fetchone()["c"]
        return {"persons": p, "named": named, "samples": s, "threshold": self.threshold}


store = FaceStore()
