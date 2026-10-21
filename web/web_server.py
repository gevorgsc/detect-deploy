"""
web_server.py — сайт с распознаванием скелета И объектов (стол/стул/телефон/мышь/
клавиатура) на видео с камер через Hikvision SDK.

Каждая камера подключается через SDK в СВОЁМ ПРОЦЕССЕ (camera_worker.run_camera_worker) —
так надёжнее (если декодер одной камеры упадёт, остальные и сам сайт продолжат работать).
Эти процессы отдают ТОЛЬКО сырые JPEG-кадры через multiprocessing.Queue.

Распознавание (pose-модель для скелета + детектор объектов) выполняется ЗДЕСЬ, в основном
процессе — CUDA/YOLO вместе с потоками Hikvision SDK в одном процессе на Windows падает без
трейсбека, поэтому SDK и модели разнесены по процессам.

На каждую камеру — своя очередь кадров и свой поток с СОБСТВЕННЫМИ экземплярами моделей.
(Батчинг обеих камер в один forward pass пробовали — при всего 2 камерах он не даёт
выигрыша: всё сериализуется в одном потоке и GPU простаивает между вызовами сильнее, чем
при параллельных потоках. Раздельные потоки грузят GPU лучше при таком количестве камер.)
FP16 (half) — заметно быстрее на RTX, ресурсов достаточно.

Запуск (из корня проекта):
    pip install -r requirements.txt
    python web/web_server.py
    -> http://localhost:8000
"""
import asyncio
import atexit
import json
import logging
import signal
from datetime import datetime
import multiprocessing as mp
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import (FileResponse, HTMLResponse, RedirectResponse,
                               StreamingResponse)
from starlette.concurrency import run_in_threadpool

# camera_sdk.py / camera_worker.py лежат в соседней папке ../camera
CAMERA_DIR = (Path(__file__).resolve().parent.parent / "camera")
sys.path.insert(0, str(CAMERA_DIR))
from camera_worker import run_camera_worker, SHM_SLOTS, SHM_SLOT_BYTES

# skeleton.py / detect_objects.py (отрисовка) лежат в соседней папке ../recognition
RECOGNITION_DIR = (Path(__file__).resolve().parent.parent / "recognition")
sys.path.insert(0, str(RECOGNITION_DIR))
from skeleton import draw_white_skeleton
from detect_objects import (draw_objects, OBJECT_CLASSES, OBJECT_CONF, OBJECT_CONF_MIN,
                            filter_by_class_conf, PhoneBoxTracker, ObjectBoxTracker)
from activity_classifier import (classify_posture, classify_activity, draw_activity_label,
                                 person_pos_scale, ObjectSlotMemory, WorkingContactMemory,
                                 _ARM_SIDES, _finger_rect, KP_CONF, FINGER_EXTEND,
                                 _polygons_overlap, _box_corners)
from analytics_store import store
from face_store import store as face_store
from floor_map import (floor_map, foot_point, solve_homography, FLOORPLAN_DIR,
                       MIN_CALIB_POINTS, align_fit, sanitize_plan, zones_from_plan,
                       poly_centre)
from person_memory import track_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = CAMERA_DIR / "config" / "cameras.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    _CFG = json.load(f)
CAMERAS = _CFG["cameras"]
# Филиалы — именованные группы камер. Всё остальное (лица, события) остаётся общим:
# один и тот же человек может прийти в разные филиалы, и узнавать его нужно везде.
BRANCHES = _CFG.get("branches") or []
# План этажа читает те же списки: калибровка камер лежит в камере, план и зоны — в
# филиале. Один файл конфига, один источник правды.
floor_map.configure(CAMERAS, BRANCHES)


def _save_config() -> None:
    data = {"branches": BRANCHES, "cameras": CAMERAS}
    CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    floor_map.rebuild()


def _branch_cameras(branch_id: str | None) -> list:
    """Камеры филиала. Пустой филиал означает «все» — так ведут себя страницы,
    открытые без выбора, и так же выглядит система, пока филиалов ещё нет."""
    if not branch_id:
        return list(CAMERAS)
    return [c for c in CAMERAS if c.get("branch") == branch_id]


def _camera_branch(camera_id: str) -> str:
    for c in CAMERAS:
        if c["id"] == camera_id:
            return c.get("branch") or ""
    return ""


def _branch_name(branch_id: str) -> str:
    for b in BRANCHES:
        if b["id"] == branch_id:
            return b.get("name") or branch_id
    return branch_id or "—"


def _note_branch(person_id, camera_id: str, now: float) -> None:
    """Отметить человека в филиале этой камеры и, если он только что был в другом,
    завести тревогу. Вызывается из горячего цикла, поэтому молча переживает любую
    ошибку: филиалы — учёт, а не распознавание, и ронять из-за них кадр нельзя."""
    if person_id is None:
        return
    branch = _camera_branch(camera_id)
    if not branch:
        return
    try:
        alert = face_store.note_branch(int(person_id), branch, camera_id, now,
                                       window=CROSS_BRANCH_WINDOW)
    except Exception as exc:                                   # noqa: BLE001
        logger.debug(f"[branch] note failed for {person_id}: {exc}")
        return
    if alert:
        who = face_store.label_of(int(person_id))
        logger.info(
            f"[branch] {who} seen at {_branch_name(branch)} ({camera_id}); "
            f"was at {_branch_name(alert['other_branch'])} "
            f"{time.strftime('%H:%M:%S', time.localtime(alert['other_seen']))}")


def _prefer_engine(pt_path: Path) -> str:
    """TensorRT engine (.engine), если экспортирован — иначе PyTorch .pt.
    Engine собран из того же .pt с half=True тем же imgsz — точность та же,
    просто быстрее исполняется на GPU."""
    engine_path = pt_path.with_suffix(".engine")
    return str(engine_path if engine_path.exists() else pt_path)


POSE_MODEL = _prefer_engine(RECOGNITION_DIR / "yolo11s-pose.pt")  # лёгкая pose-модель YOLO11 (s) — быстрее yolo26x-pose
# (высота, ширина). Кадр 16:9, а квадратный вход заставлял letterbox добивать
# 1280x720 серым полем до 1280x1280 — ~44% того, что считала модель, было пустотой.
# 736 — это 720, округлённое вверх до кратного stride=32. Движки пересобраны под
# этот размер (dynamic=False, форма входа зашита внутрь .engine).
POSE_IMGSZ = (736, 1280)

# Детектор объектов (стол/стул/телефон/мышь/клавиатура). Рядом с FP16-движком может
# лежать INT8-вариант, собранный на калибровочной выборке С ЭТИХ ЖЕ камер
# (yolo26x.int8.engine). Переключается переменной окружения OBJ_ENGINE=int8 —
# так два движка можно сравнивать на живой системе, ничего не правя в коде и не
# подменяя файлы. Значение по умолчанию — FP16, то есть проверенный боевой вариант.
_OBJ_INT8 = RECOGNITION_DIR / "yolo26x.int8.engine"
if os.environ.get("OBJ_ENGINE", "").lower() == "int8" and _OBJ_INT8.exists():
    OBJ_MODEL = str(_OBJ_INT8)
else:
    OBJ_MODEL = _prefer_engine(RECOGNITION_DIR / "yolo26x.pt")
OBJ_CLASS_IDS = list(OBJECT_CLASSES.keys())
CHAIR_ID, MOUSE_ID, KEYBOARD_ID, PHONE_ID = 56, 64, 66, 67
MONITOR_ID, LAPTOP_ID, NOTEBOOK_ID = 62, 63, 73
# Что считается рабочим предметом: коснулся — значит работает. Клавиатура и мышь были
# здесь с самого начала, ноутбук и бумаги добавлены на тех же правах. Работа за столом
# это не только печатать: писать в тетради, читать документ, считать на ноутбуке —
# то же самое, и раньше всё это молча числилось бездельем.
WORK_ITEM_IDS = (KEYBOARD_ID, MOUSE_ID, LAPTOP_ID, NOTEBOOK_ID)
# 1920 вместо 3200 — при 3200 память едва тянула один комплект моделей на камеру
# (~4.8GB), второй воркер не влезал. На 1920 комплект ~2GB — есть место под параллелизм.
OBJ_IMGSZ = (1088, 1920)  # 1080 -> кратное 32; см. комментарий к POSE_IMGSZ
# Частота детекторов задаётся в Гц, а не «каждый N-й кадр»: fps потока плавает
# (при 3200x1800 было ~5, после ужатия до 1920x1080 стало ~8.8), и счётчик кадров
# молча менял бы реальную частоту вместе с ним. По времени — стабильно.
# Кадры между запусками закрывают трекеры: SkeletonFlowTracker для скелетов,
# ObjectBoxTracker/PhoneBoxTracker для объектов.
POSE_TARGET_HZ = 6.0   # pose-модель — 6 раз в секунду
OBJ_TARGET_HZ = 1.0    # детектор объектов — 1 раз в секунду
# Порог прореживания считается ВЫЧИТАНИЕМ половины кадрового интервала, а не
# умножением на допуск, — ровно как _min_period в camera_worker.py.
#
# Детектор может сработать только в момент прихода кадра, а кадры идут сеткой:
# при 20 fps через 0.05 с, при 25 fps через 0.04 с. Порог, попавший НА узел сетки,
# превращается в подбрасывание монетки: кадр, опоздавший на пару миллисекунд,
# отбрасывается целиком, и следующий шанс наступает через целый интервал.
#
# Прежний множитель 0.95 на 5 Гц работал (0.19 против узла 0.20), но на 6 Гц дал
# 0.158 — это впритык к узлу 0.16 у камеры 25 fps и уже за узлом 0.15 у камеры
# 20 fps. Замер: при target 6.0 выходило 4.89-5.17 Гц.
# Вычитание половины интервала (0.025 с при 20 fps) уводит порог с узла гарантированно:
#   6 Гц -> 0.1417 -> при 20 fps срабатывает на 0.15 (6.67 Гц), при 25 fps на 0.16 (6.25).
SOURCE_FPS_FLOOR = 20.0   # самая медленная камера; половина её интервала — запас
def _detector_period(hz: float) -> float:
    return max(1.0 / hz - 0.5 / SOURCE_FPS_FLOOR, 0.0)

POSE_MIN_PERIOD = _detector_period(POSE_TARGET_HZ)
OBJ_MIN_PERIOD = _detector_period(OBJ_TARGET_HZ)

# Детектор оружия. Устроен так же, как детектор объектов: целый кадр, свой движок,
# частота задаётся в Гц. Никаких вырезок по людям — кадр идёт в сеть как есть.
WEAPON_MODEL = _prefer_engine(RECOGNITION_DIR / "weapon.pt")
# 1088x1920 — ровно размер кадра (1080 округлено вверх до кратного stride=32), масштаб 1.00.
# Кадр не растягивается и не ужимается, ровно как в OBJ_IMGSZ.
#
# Цена этого решения измерена, и её стоит знать. На 250 кадрах 1920x1080 с реального
# видеонаблюдения (доля кадров, где оружие найдено):
#   целый кадр -> 1920x1088 (масштаб 1.00)   28.8%   <- выбрано
#   целый кадр -> 1280 (ужат в 0.67)         27.6%
#   целый кадр -> 640                        20.4%
#   вырезка по человеку -> 1280              37.2%
# То есть вырезки давали на 8.4 пункта больше: пистолет в кадре 1920 занимает ~48 px, а в
# растянутой вырезке ~160 px, и мелкий уровень признаков сети (stride=8) видит его гораздо
# увереннее. Обратите внимание, что масштаб 1.00 почти не отличается от ужатого 1280 —
# упирается не в ресайз, а в размер цели.
# Решение принято сознательно: сначала простой и предсказуемый конвейер, качество добираем
# следующим обучением модели на нормальных данных (в том числе с этих камер), а не
# усложнением инференса.
WEAPON_IMGSZ = (1088, 1920)
# Порог ниже, чем у остальных детекторов: пропущенное оружие стоит несравнимо дороже
# лишней рамки на экране. Кривая recall-confidence у этой модели почти плоская от 0.05
# до 0.5, так что снижение порога добавляет мало ложных срабатываний.
WEAPON_CONF = 0.25
# Классы — ПО ИМЕНИ, а не по индексу. Сейчас модель обучена на одном классе (handgun),
# после дообучения на ножах индексы сдвинутся (handgun=0, knife=1), и сравнение с
# числом сломалось бы молча, без ошибки. Имя переживает переобучение.
WEAPON_CLASS_NAMES = {"handgun", "knife"}
# Кадры РАЗНЫХ камер, попавшие в батчер одновременно, уходят на GPU одним прогоном.
# Больше числа камер накопиться неоткуда — каждая камера отдаёт по одному кадру за проход.
# Значение должно совпадать с batch=, с которым собран weapon.engine.
WEAPON_MAX_BATCH = max(2, len(CAMERAS))
WEAPON_COLOR = (0, 0, 255)   # BGR: красный
# Своя частота, как у двух других детекторов. Вровень с позой: оружие достают и пускают
# в ход быстрее, чем за секунду, и на 1 Гц момент можно не застать.
WEAPON_TARGET_HZ = 6.0
WEAPON_MIN_PERIOD = _detector_period(WEAPON_TARGET_HZ)

# --- какой детектор оружия работает ----------------------------------------------
# "trained" — обученная weapon.pt/engine, цифры выше.
# "yoloe"   — open-vocabulary: классы задаются ТЕКСТОМ, без обучения вообще.
#             Включено временно, чтобы посмотреть его вживую на камерах.
# Меняется одной строкой или переменной окружения WEAPON_BACKEND.
#
# ИТОГ ПРОВЕРКИ: yoloe отключён. Он и хуже (см. цифры ниже), и дороже — 264-357 мс/с
# против 127 мс/с у обученной модели, даже после сборки движка: yoloe-11l — это
# сегментационная сеть, она тяжелее, и часть графа не ушла в fp16 (op Range).
WEAPON_BACKEND = os.environ.get("WEAPON_BACKEND", "trained").lower()

# Формулировка запроса решает БОЛЬШЕ, чем размер модели. Замер на 250 кадрах реального
# видеонаблюдения (доля кадров, где оружие найдено / доля ЧИСТЫХ кадров, где он сработал):
#   "handgun", "knife"                        16.8%  /  58.8%   при conf 0.05
#   "handgun, pistol, gun, knife, weapon"     17.6%  /   -
#   "a gun in a person's hand"                36.4%  /   -
#   "small handgun ... far from the camera"   60.4%  /  93.5%   при conf 0.05
#   то же самое                                0.8%  /   4.2%   при conf 0.25
# Обученная модель для сравнения:             28.8%  /   4.2%   при conf 0.25
# То есть высокий recall у yoloe достигается тем, что он рисует рамку почти на каждом
# кадре. При равной доле ложных срабатываний он проигрывает обученной модели в 36 раз.
YOLOE_WEIGHTS = "yoloe-11l-seg.pt"
YOLOE_PROMPTS = ["small handgun held by a person far from the camera",
                 "knife held in a hand"]
# Что писать на рамке вместо длинного запроса.
YOLOE_LABELS = ["handgun", "knife"]
YOLOE_CONF = float(os.environ.get("YOLOE_CONF", "0.05"))
# Просили гонять его на каждом 4-м кадре (при 20 fps это те же ~5 Гц).
YOLOE_EVERY_N = 4
# Собранный движок: запросы вписаны в граф на этапе экспорта (set_classes ДО export),
# дальше он работает как обычная модель с фиксированными классами — ровно как две
# другие. Без него пришлось бы гонять PyTorch, а это втрое дороже.
YOLOE_ENGINE = RECOGNITION_DIR / "weapon_yoloe.engine"

# Детектор лиц — InsightFace antelopev2 (SCRFD-10G + glintr100/ArcFace), четвёртая модель.
# Не ultralytics: это onnxruntime, поэтому и грузится, и вызывается иначе, чем три другие.
#
# РАБОТАЕТ НЕ ПО КАДРАМ, А ПО СОБЫТИЯМ ОТ КАМЕРЫ.
# Камеры DS-2CD2346G2-ISU/SL (.55 и .197) умеют Face Capture: сами находят лицо, сами его
# вырезают и присылают сюда multipart'ом — вырезку (3-8 КБ), полный кадр и JSON с faceId,
# faceRect, возрастом, полом, очками, маской и своей оценкой качества faceScore.
# Раз камера уже сделала детекцию, гонять по всему кадру SCRFD незачем: раньше это стоило
# 150-240 мс/с на камеру и работало вхолостую, даже когда в кадре никого не было.
# Здесь остаётся только распознавание — на картинке 128x140, а не 1920x1080.
#
# Камеры DS-2CD3361G2E-LIU (.216 и .141) Face Capture НЕ поддерживают (faceSnap у них нет,
# httpHosts всего один слот). На них распознавания лиц нет вовсе - сознательно.
#
# ВАЖНО про GPU: onnxruntime-gpu ищет cublas/cudnn на путях DLL и, не найдя их, МОЛЧА
# уходит на CPU — замер был 722 мс на кадр вместо 67. Ставить CUDA toolkit не нужно:
# torch 2.11+cu128 везёт с собой ровно те версии (cublas64_12, cublasLt64_12, cudnn64_9),
# достаточно показать onnxruntime его lib-каталог перед импортом. См. _load_shared_models.
FACE_ENABLED = os.environ.get("FACE_ENABLED", "1") != "0"
# Куда складывать сырые события faceSnap от камеры, пока разбираемся с их форматом.
# Пустая строка — не сохранять.
FACE_EVENT_DUMP = os.environ.get("FACE_EVENT_DUMP", "")
FACE_MODEL = os.environ.get("FACE_MODEL", "antelopev2")
# 160: сеть больше не ищет лицо в кадре. Голова уже найдена — она в верхней части рамки
# человека, которую дал pose. По этой вырезке отрабатывает только выравнивание (пять точек)
# и распознавание. Стоимость сети зависит от det_size, а не от размера картинки, поэтому
# 640 здесь был бы платой за работу, которой нет.
FACE_DET_SIZE = int(os.environ.get("FACE_DET_SIZE", "160"))
# Доля роста человека сверху, в которой ищем голову.
FACE_HEAD_RATIO = float(os.environ.get("FACE_HEAD_RATIO", "0.33"))
# Как часто пробовать узнать ЕЩЁ БЕЗЫМЯННЫЙ трек — 2 Гц.
FACE_RETRY_SEC = float(os.environ.get("FACE_RETRY_SEC", "0.5"))
# Узнанный трек тоже перепроверяем, раз в две секунды: трек держится по близости рамок, и
# когда двое расходятся рядом, он может перескочить на соседа вместе с чужим именем.
# Перепроверка это ловит — имя следует за лицом, а не за рамкой.
FACE_VERIFY_SEC = float(os.environ.get("FACE_VERIFY_SEC", "2.0"))
# Неузнанного заводим в галерею, чтобы оператору было кого назвать. Реже, чем пробуем
# узнать: человек стоит секундами, и 2 Гц дали бы два десятка почти одинаковых снимков.
FACE_ENROLL_SEC = float(os.environ.get("FACE_ENROLL_SEC", "3.0"))
# Мутную вырезку в галерею не кладём: эмбеддинг с неё всё равно ни с чем не сойдётся,
# а место и время займёт.
FACE_ENROLL_MIN_DET = float(os.environ.get("FACE_ENROLL_MIN_DET", "0.55"))
# Написать имя над головой — заявление в кадре, и ошибка в нём видна всем. Поэтому порог
# на ИМЯ выше, чем порог склейки снимков в галерее (FACE_MATCH_THRESHOLD, 0.40): 0.40
# годится, чтобы сложить два снимка в одну карточку, но на 0.43 подписывать человека
# чужим именем нельзя — именно так «rafo» и оказался на чужой голове.
FACE_NAME_MIN_SIM = float(os.environ.get("FACE_NAME_MIN_SIM", "0.50"))
# Сколько перепроверок подряд должны НЕ подтвердить имя, чтобы его снять. Считаем только
# те, где лицо реально нашлось: если человек отвернулся, лица нет, и снимать имя не за что.
# 2 промаха при 1 Гц — это ~2 секунды чужого имени вместо бесконечности.
FACE_DROP_AFTER_MISS = int(os.environ.get("FACE_DROP_AFTER_MISS", "2"))
# Сколько имя остаётся закреплённым за камерой без подтверждения. Проверка идёт раз в
# секунду, так что трёх секунд хватает, чтобы человек ушёл из кадра и имя освободилось,
# и мало, чтобы оно там застряло.
NAME_CLAIM_TTL = float(os.environ.get("FACE_NAME_CLAIM_TTL", "3.0"))
# За какое время назад «он был в другом филиале» ещё новость. Без окна тревога сводится
# к «этот человек когда-либо бывал на двух объектах» — верно про половину сотрудников и
# бесполезно. Двенадцать часов — это рабочий день: перемещение внутри смены.
CROSS_BRANCH_WINDOW = float(os.environ.get("CROSS_BRANCH_WINDOW", "43200"))
# Порог детектора лиц (SCRFD внутри antelopev2). Было 0.35 — ниже дефолтных 0.5, как в
# работающей системе face_analytics на .179, чтобы находить мелкие лица. Поднято до 0.60
# по решению: пусть лиц находится меньше, но каждое найденное — уверенно.
FACE_DET_THRESH = float(os.environ.get("FACE_DET_THRESH", "0.45"))
# Камера сама решает, когда прислать лицо, поэтому своей частоты у этого детектора нет.
# Ниже этой оценки качества (та, что камера ставит сама, 0..100) вырезку не считаем:
# на плохом кадре эмбеддинг всё равно ни с чем не совпадёт, а время займёт.
FACE_MIN_SCORE = int(os.environ.get("FACE_MIN_SCORE", "40"))
FACE_COLOR = (0, 200, 255)   # BGR: оранжевый — не путать с красным у оружия

INFER_DEVICE = "0"   # GPU
INFER_HALF = True    # FP16 — заметно быстрее на RTX, ресурсов достаточно

WORKERS_PER_CAMERA = 1  # 2 воркера на imgsz=1920 дали МЕНЬШЕ fps, чем 1 (конкуренция за GPU
# перевесила выигрыш от лёгкого imgsz) — возвращаем 1, разница видна ниже в измерениях.

# OpenCV по умолчанию раскидывает КАЖДУЮ операцию на все 24 логических ядра и ждёт
# их spin-wait'ом. На наших размерах кадра это почти чистые накладные расходы —
# замер на этой машине (1920x1080, 24 ядра):
#   calcOpticalFlowPyrLK  24 потока -> 3.5 ms wall / 24.5 ms-core
#                          4 потока -> 3.7 ms wall /  7.8 ms-core
#   cvtColor BGR2GRAY     24 потока -> 7.3 ms-core, 1 поток -> 0.5 ms-core
# Скорость та же, а ядер съедается втрое меньше.
CV_THREADS = 4
cv2.setNumThreads(CV_THREADS)

# Качество JPEG для картинки в браузер. Этот кадр никуда, кроме <img>, не идёт —
# модели работают с оригиналом до кодирования, поэтому 90 здесь избыточно.
STREAM_JPEG_QUALITY = 75

# Сбор калибровочной выборки для INT8. По умолчанию ВЫКЛЮЧЕН и ничего не стоит.
# Включается переменной окружения:  set CALIB_DIR=...\_calib\images
#
# Кадры сохраняются ДО отрисовки скелета и рамок и в том же виде, в каком их видит
# модель: калибровка INT8 подбирает масштабы квантования под распределение входа,
# и выборка с нарисованными поверх белыми линиями отравила бы эти масштабы.
# PNG, а не JPEG — артефактов сжатия в боевом тракте нет, значит и здесь им не место.
# Снимать кадры отдельным процессом через SDK не вышло: HCNetSDK вне этого
# процесса падает нативно на декодировании (см. camera_worker.py), поэтому берём
# их там, где путь уже заведомо рабочий.
CALIB_DIR = os.environ.get("CALIB_DIR") or None
CALIB_EVERY = int(os.environ.get("CALIB_EVERY", "15"))  # каждый N-й кадр камеры
if CALIB_DIR:
    Path(CALIB_DIR).mkdir(parents=True, exist_ok=True)
    logger.warning(f"CALIB MODE: every {CALIB_EVERY}-th clean frame -> {CALIB_DIR}")

app = FastAPI(title="Skeleton Viewer")


@app.middleware("http")
async def _no_cache(request: Request, call_next):
    """Страницы и ответы API браузеру кэшировать нельзя.

    Разметка и скрипты лежат прямо в HTML, и без явного запрета браузер по своему
    усмотрению показывает старую копию: страница меняется, а у оператора на экране
    всё по-прежнему, и понять это со стороны сервера невозможно — запрос до него
    просто не доходит.
    """
    resp = await call_next(request)
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("text/html") or ctype.startswith("application/json"):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
    return resp

_latest_jpeg: dict[str, bytes] = {}
# Последний кадр камеры как BGR-массив. Хранится ссылка на уже посчитанный массив —
# это ничего не стоит, зато снимки лиц берутся отсюда напрямую, и JPEG для браузера
# можно не кодировать, когда на стрим никто не смотрит.
_latest_frame: dict[str, np.ndarray] = {}
# Кадр без рисунка — для совмещения с планом этажа: на _latest_frame уже лежат рамки,
# скелеты и подписи, а под ними не видно ни угла пола, ни края стола, то есть ровно
# того, по чему человек и наводит каркас. Копию держим одну на камеру и обновляем
# редко: 6 МБ memcpy раз в две секунды заметно дешевле, чем копия на каждом кадре.
_clean_frame: dict[str, np.ndarray] = {}
_clean_at: dict[str, float] = {}
CLEAN_SNAP_SEC = 2.0
_stream_viewers: dict[str, int] = {c["id"]: 0 for c in CAMERAS}
_viewers_lock = threading.Lock()
# То же самое число, но видимое процессу камеры: mp.Value живёт в общей памяти.
# Заполняется в __main__ перед стартом воркеров; до этого пусто, и camera_worker
# с viewer_flag=None просто работает на полной частоте.
_viewer_flags: dict = {}

# Модели загружаются ОДИН раз на процесс, а не по экземпляру на камеру. Раньше каждый
# поток камеры делал свой YOLO(...), и TensorRT выделял заново и веса, и — главное —
# activation-буфер (855 + 680 MiB на камеру). Видеопамять росла линейно, и потолок
# упирался в неё примерно на 13 камерах, раньше чем в GPU или CPU. Один общий экземпляр
# делает расход по видеопамяти постоянным. Веса, imgsz и точность те же — результат тот же.
_pose_model = None
_obj_model = None
_weapon_model = None
_face_model = None
_pose_batcher = None
_obj_batcher = None
_weapon_batcher = None
_face_batcher = None
# id -> имя, только для классов из WEAPON_CLASS_NAMES. Строится при загрузке модели из
# её собственных names, поэтому переобученная модель с другим набором классов подхватится
# сама, без правок кода.
_weapon_names: dict[int, str] = {}

# Сколько кадров разных камер класть в один запуск. Замер на этой машине (ms на кадр):
#   batch=1 -> 7.56    batch=2 -> 6.46    batch=4 -> 5.93    batch=8 -> 5.90
# После четырёх прирост кончается, поэтому 4.
POSE_MAX_BATCH = 4
# Детектор объектов остался статическим batch=1: сборка динамического движка для него
# требует ~17 ГБ видеопамяти на этапе оптимизации (у карты 16). Потеря небольшая —
# объекты считаются 1 раз в секунду на камеру, поза 5 раз, время уходит на позу.
OBJ_MAX_BATCH = 1
# Лица: пакет ArcFace. Замер: 1 лицо 9.8 мс, 8 лиц 14.4 мс, 16 лиц 19.1 мс — после
# восьми рост ещё есть, но очередь редко набирает больше: восьми хватает, чтобы
# камеры не ждали друг друга, и не создаёт задержку на сборе пачки.
FACE_MAX_BATCH = int(os.environ.get("FACE_MAX_BATCH", "8"))


class _BatchedInference:
    """Складывает вызовы модели от разных камер в один запуск на GPU.

    Батч набирается ЖАДНО, без искусственного ожидания: поток берёт первый запрос
    блокирующе, а дальше добирает только то, что УЖЕ стоит в очереди. При одной-двух
    камерах это вырождается в батч из одного кадра и ничего не замедляет; когда камер
    много, следующие запросы успевают накопиться за время предыдущего запуска, и батч
    растёт сам. Никакого параметра «сколько ждать» подбирать не нужно.

    Заодно это единственное место, откуда трогают модель, — отдельная блокировка для
    защиты контекста TensorRT больше не нужна.
    """

    def __init__(self, name: str, run, max_batch: int):
        self._name = name
        self._run = run              # run(frames) -> список результатов, по одному на кадр
        self._max_batch = max_batch
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name=f"batch-{name}").start()

    def submit(self, frame):
        req = {"frame": frame, "done": threading.Event(), "out": None, "err": None}
        self._q.put(req)
        req["done"].wait()
        if req["err"] is not None:
            raise req["err"]
        return req["out"]

    def _loop(self):
        while True:
            batch = [self._q.get()]
            while len(batch) < self._max_batch:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            try:
                outs = self._run([r["frame"] for r in batch])
                for r, o in zip(batch, outs):
                    r["out"] = o
            except Exception as e:
                logger.error(f"[{self._name}] batch of {len(batch)} failed: {e}")
                for r in batch:
                    r["err"] = e
            finally:
                for r in batch:
                    r["done"].set()


def _pose_batch(frames: list) -> list:
    """Один запуск позы на пачке кадров. Возвращает готовые numpy-массивы, а не Results:
    ultralytics переиспользует внутренние буферы предиктора, и Results, пережившие
    следующий вызов, показывали бы уже чужие данные."""
    out = []
    for r in _pose_model(frames, imgsz=POSE_IMGSZ, conf=0.35,
                         device=INFER_DEVICE, half=INFER_HALF, verbose=False):
        kpts = (r.keypoints.data.cpu().numpy()
                if r.keypoints is not None and len(r.keypoints) else np.empty((0, 17, 3)))
        boxes = (r.boxes.xyxy.cpu().numpy()
                 if r.boxes is not None and len(r.boxes) else np.empty((0, 4)))
        out.append((kpts, boxes))
    return out


def _obj_batch(frames: list) -> list:
    out = []
    # Сеть считает с самым низким порогом из набора, а дальше каждый класс отсеивается
    # своим: одним порогом на всех либо теряются бумаги, либо лезет мусор от стульев.
    for r in _obj_model(frames, classes=OBJ_CLASS_IDS, imgsz=OBJ_IMGSZ,
                        conf=OBJECT_CONF_MIN,
                        device=INFER_DEVICE, half=INFER_HALF, verbose=False):
        if r.boxes is not None and len(r.boxes):
            out.append(filter_by_class_conf(
                r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy()))
        else:
            out.append((np.empty((0, 4)), np.empty((0,)), np.empty((0,))))
    return out


def _weapon_batch(frames: list) -> list:
    conf = YOLOE_CONF if WEAPON_BACKEND == "yoloe" else WEAPON_CONF
    results = _weapon_model.predict(frames, imgsz=WEAPON_IMGSZ, conf=conf,
                                    device=INFER_DEVICE, half=INFER_HALF, verbose=False)
    out = []
    for r in results:
        if r.boxes is not None and len(r.boxes):
            out.append((r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                        r.boxes.conf.cpu().numpy()))
        else:
            out.append((np.empty((0, 4)), np.empty((0,)), np.empty((0,))))
    return out


_face_lock = threading.Lock()


def _head_crop(frame, person_kpts, person_box):
    """Вырезка головы по верхней части рамки человека.

    Границы берём по точкам лица из pose (нос, глаза, уши): они точнее, чем просто
    «верхняя треть рамки», когда человек наклонился или сидит. Если точки не видны —
    откатываемся на долю от роста.
    """
    x1, y1, x2, y2 = [float(v) for v in person_box]
    pts = [person_kpts[i][:2] for i in range(5) if person_kpts[i][2] >= KP_CONF]
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w = max(max(xs) - min(xs), 1.0)
        h = max(max(ys) - min(ys), 1.0)
        pad = max(w, h) * 1.1
        hx1, hy1 = min(xs) - pad, min(ys) - pad * 1.2
        hx2, hy2 = max(xs) + pad, max(ys) + pad * 1.4
    else:
        hx1, hx2 = x1, x2
        hy1, hy2 = y1, y1 + (y2 - y1) * FACE_HEAD_RATIO

    H, W = frame.shape[:2]
    hx1, hy1 = max(0, int(hx1)), max(0, int(hy1))
    hx2, hy2 = min(W, int(hx2)), min(H, int(hy2))
    if hx2 - hx1 < 24 or hy2 - hy1 < 24:
        return None, None
    return frame[hy1:hy2, hx1:hx2], (hx1, hy1)


# Индексы COCO-скелета, по которым режется одежда.
KP_L_SH, KP_R_SH, KP_L_HIP, KP_R_HIP = 5, 6, 11, 12
KP_L_KNEE, KP_R_KNEE, KP_L_ANK, KP_R_ANK = 13, 14, 15, 16
CLOTH_MIN_PIXELS = 200
# Сколько ждать между входами камер в HCNetSDK — и на старте, и при перезапуске.
CAMERA_LOGIN_GAP = float(os.environ.get("CAMERA_LOGIN_GAP", "5.0"))

# --- одежда: цвет по скелету ---------------------------------------------------
# Раньше место замера подсказывала сеть PAR. От неё отказались: её точки почти
# постоянны (y = 0.345 и 0.656 от рамки, замерено на 40 вырезках) — она обучена на
# стоящих в полный рост пешеходах. У сидящего за столом рамка кончается на краю стола,
# и «66% высоты» приходятся на стол, а «34%» — на шею и лицо. Кожа по тону почти
# неотличима от оранжевого, поэтому людям и записывался оранжевый цвет рубашки,
# которой на них нет. Скелет знает, где плечи, бёдра и колени, у КАЖДОГО отдельно.
CLOTH_ENABLED = os.environ.get("CLOTH_ENABLED", "1") != "0"
CLOTH_TARGET_HZ = float(os.environ.get("CLOTH_HZ", "1.0"))
CLOTH_MIN_PERIOD = _detector_period(CLOTH_TARGET_HZ)
CLOTH_VOTES = int(os.environ.get("CLOTH_VOTES", "5"))   # сколько замеров держим на трек
CLOTH_TTL = float(os.environ.get("CLOTH_TTL", "3.0"))
# Сжатие четырёхугольника торса к центру — чтобы по краям не захватить фон.
CLOTH_SHRINK = float(os.environ.get("CLOTH_SHRINK", "0.22"))
# Насколько опустить верхнюю кромку торса от линии плеч: там шея и открытая кожа.
CLOTH_NECK_TRIM = float(os.environ.get("CLOTH_NECK_TRIM", "0.18"))
# Вдоль конечности берём среднюю часть: у суставов пояс, сгиб и тень.
CLOTH_LIMB_FROM, CLOTH_LIMB_TO = 0.30, 0.85
CLOTH_LIMB_WIDTH = 0.20         # толщина полосы в долях длины конечности
# Своя, СТРОГАЯ планка уверенности точек — общая KP_CONF = 0.30 годится, чтобы
# нарисовать скелет, но не чтобы решать «вот это штаны». Поза дорисовывает ноги и
# тогда, когда они закрыты столом: точки есть, но уверенность низкая, и полоса ляжет
# по столешнице. Порог выбран по замеру распределения (см. CLOTH_PROBE).
# Одна планка и для показа, и для записи. Раньше показ шёл по более мягкой (0.60), и
# у сидящего за столом полоса вдоль ноги ложилась по столешнице: точки коленей поза
# дорисовывает, и уверенность у них как раз около этой границы. Если сустав не виден
# отчётливо — цвет этой части не считаем вообще, а честно отдаём "unknown".
CLOTH_KP_CONF = float(os.environ.get("CLOTH_KP_CONF", "0.80"))
# Мерить и ЗАПОМИНАТЬ — разные решения, и планка у них разная. Показать цвет можно и
# по среднему кадру, а в память человека он попадёт только с очень хорошего: память
# потом отменяет имена, и мусор в ней дороже, чем её медленное наполнение.
# Лицо должно быть узнано уверенно, а суставы — видны отчётливо.
# Тот же порог, при котором одежда вообще не имеет права возражать. Это не совпадение:
# память набирается ТОЛЬКО из кадров, где решало лицо. Иначе вышел бы круг — одежда
# пропустила имя, имя записало одежду, ошибка закрепилась сама собой.
CLOTH_LEARN_FACE_SIM = float(os.environ.get("CLOTH_LEARN_FACE_SIM", "0.60"))
CLOTH_LEARN_KP_CONF = float(os.environ.get("CLOTH_LEARN_KP_CONF", "0.80"))
# Насколько свежим должно быть подтверждение лица: одежду смотрим раз в секунду, лицо
# два раза, и связывать замер с распознаванием минутной давности нельзя.
CLOTH_LEARN_MAX_AGE = float(os.environ.get("CLOTH_LEARN_MAX_AGE", "2.0"))
# Какие точки должны быть уверенны разом. Только тело: плечи, бёдра, колени,
# щиколотки. Именно они означают «стоит прямо, в полный рост», и именно они закрывают
# собой рубашку и штаны. Уши и глаза сюда не входят: к одежде они отношения не имеют,
# а блокировали всё — замер на 1688 кадрах показал, что левое ухо чётко видно лишь в
# 31% случаев (волосы), и условие «все 17 точек» не выполнилось НИ РАЗУ.
CLOTH_LEARN_KP_SET = (KP_L_SH, KP_R_SH, KP_L_HIP, KP_R_HIP,
                      KP_L_KNEE, KP_R_KNEE, KP_L_ANK, KP_R_ANK)
# Сколько удачных засечек подряд должны сойтись, чтобы цвет был признан за человеком
# на день. Одна засечка может прийтись на неудачный свет или на чужого, кого лицо
# приняло за своего; три совпавшие — уже не совпадение.
CLOTH_CONFIRM_TIMES = int(os.environ.get("CLOTH_CONFIRM_TIMES", "3"))
# Минимальный промежуток между засечками. Без него три засечки набирались бы из трёх
# соседних кадров одной и той же секунды — это не «три раза за день», а один раз.
CLOTH_CONFIRM_GAP = float(os.environ.get("CLOTH_CONFIRM_GAP", "60.0"))
# Человек может переодеться среди дня — снять пиджак, надеть халат. Тогда утренняя
# запись весь остаток дня отменяла бы ВЕРНЫЕ распознавания. Поэтому цвет дня можно
# переписать, но требования выше, чем при первой записи: лицо должно быть узнано
# очень уверенно, и новые цвета должны сойтись столько же раз.
CLOTH_RECONFIRM_SIM = float(os.environ.get("CLOTH_RECONFIRM_SIM", "0.70"))

# --- короткая привязка «этот человек — вот в этой одежде, вот здесь» ------------
# Отдельно от дневной записи. Одна и та же рубашка на разных камерах и в разных позах
# меряется по-разному (замерено: тон одного человека 111 на cam_01 и 18.5 на cam_02),
# поэтому общий «цвет дня» для удержания имени — плохая опора, а снимок, взятый прямо
# здесь и прямо сейчас, — хорошая.
# С каким «весом» одежда претендует на имя. Заведомо ниже любого порога распознавания
# лица: одежда — доказательство слабое (в комнате несколько человек в тёмном верхе), и
# отнимать имя у трека, который видит лицо, она права не имеет. Занять свободное имя —
# имеет: для того эта дорожка и нужна, когда человек отвернулся.
CLOTH_NAME_SIM = float(os.environ.get("CLOTH_NAME_SIM", "0.05"))
REF_MIN_SIM = float(os.environ.get("REF_MIN_SIM", "0.60"))
REF_TTL = float(os.environ.get("REF_TTL", "180.0"))
# Насколько быстро человек может перемещаться, в РОСТАХ в секунду. В пикселях мерить
# нельзя: у камеры человек проходит за секунду впятеро больше пикселей, чем вдали.
# Это и есть защита от подмены: сосед в такой же рубашке стоит в другом месте, и
# «переход» к нему получается мгновенным скачком, а не ходьбой.
REF_SPEED_HEIGHTS = float(os.environ.get("REF_SPEED_HEIGHTS", "2.5"))
# Вес имени, выданного по одежде. Намеренно почти ноль: в споре за человека такое имя
# уступает любому распознаванию по лицу, даже слабому. Одежда лишь удерживает имя,
# пока лица не видно, и никогда не перебивает лицо.
CLOTHES_NAME_WEIGHT = 0.01

# --- запись событий ------------------------------------------------------------
# Логи вернулись в цикл, но пишем ПО ИЗМЕНЕНИЮ состояния, а не каждый кадр. Прежняя
# версия звала хранилище на каждого человека в каждом кадре — около 300 commit'ов в
# секунду, и это съедало 500-700 мс из секунды против 98-138 мс на все три сети.
EVENTS_ENABLED = os.environ.get("EVENTS_ENABLED", "1") != "0"
# Смена состояния засчитывается, только если продержалась столько секунд. Метки
# working/on_phone мигают от кадра к кадру: рука на миг опустилась — и отрезок
# закрылся бы и открылся снова. Без этой выдержки вернулись бы те же сотни записей,
# а отрезки длиной в треть секунды всё равно никому не нужны.
EVENT_DEBOUNCE_SEC = float(os.environ.get("EVENT_DEBOUNCE_SEC", "2.0"))
# Как часто закрывать отрезки треков, которых больше нет в кадре.
EVENT_SWEEP_SEC = float(os.environ.get("EVENT_SWEEP_SEC", "5.0"))
# Сколько трек может не попадаться на глаза, прежде чем считать, что он ушёл. Человека
# на секунду заслонил коллега или отвернулся так, что поза его не собрала, — трек при
# этом жив. Без этой отсрочки отрезок присутствия рвался на куски: у одного трека
# набиралось 14 отрезков на 374 секунды, и маршрут по ним читать было невозможно.
EVENT_TRACK_GRACE = float(os.environ.get("EVENT_TRACK_GRACE", "30.0"))
# Сколько дней держать след на плане. Строк он даёт больше всего, а отматывают
# обычно вчерашний день, не прошлый месяц.
TRAIL_KEEP_DAYS = float(os.environ.get("TRAIL_KEEP_DAYS", "7"))
# Сколько человек может ничего не делать, прежде чем это станет записью. Значение
# живёт в базе и меняется с сайта; здесь только начальное.
IDLE_MINUTES_DEFAULT = float(os.environ.get("IDLE_MINUTES", "5"))
# Насколько длинной должна быть работа, чтобы обнулить счётчик простоя. Короткая
# вспышка — рука на секунду легла на мышь — счётчик не сбрасывает: иначе человека,
# который раз в две минуты трогает клавиатуру, список не покажет никогда.
WORK_RESET_SEC = float(os.environ.get("WORK_RESET_SEC", "10"))
# Диагностика: пишет уверенность точек ног в разбивке по позе, чтобы этот порог
# можно было выбрать по данным, а не на глаз.
CLOTH_PROBE = os.environ.get("CLOTH_PROBE", "") == "1"

# --- одежда как проверка имени -------------------------------------------------
# Распознавание лица иногда ошибается и подписывает чужого человека именем из базы.
# Одежда — независимая от лица улика: если человек весь день в белой рубашке, а перед
# камерой стоит кто-то в чёрной, значит лицо совпало по ошибке, и имя писать нельзя.
CLOTHES_VETO = os.environ.get("CLOTHES_VETO", "1") != "0"
# Очень уверенное совпадение по лицу отменяет возражение одежды. Без этого человек,
# снявший днём куртку, попадал в ловушку: цвет изменился -> имя не пишем -> учиться
# новому цвету не на чем -> имя не появится до конца дня. Лицо надёжнее цвета, и там,
# где оно уверено, спорить нечем. Порог выбран по замеру: все отмены пришлись на
# 0.517..0.639, то есть на ту самую середину, где и живут ошибки распознавания.
CLOTHES_VETO_MAX_SIM = float(os.environ.get("CLOTHES_VETO_MAX_SIM", "0.60"))
CLOTHES_CACHE_SEC = 10.0
# Решение принимаем не по двенадцати оттенкам, а по шести широким группам. Двенадцать
# нужны оператору в поиске («синяя рубашка»), но для проверки они слишком дробны: шум
# измерения размазывался по семи корзинам, ни одна не набирала веса, и вето переставало
# срабатывать там, где было нужнее всего. Группы держат тот же самый физический свитер
# в одной корзине, даже когда тон гуляет от камеры к камере.
COLOUR_GROUPS = {
    "black": "dark", "gray": "gray", "white": "light",
    # Бежевый в отдельной группе: он честно путается и со светлым, и с коричневым,
    # но с синим — никогда. Свалив его в "warm", мы получили бы вето между бежевым
    # свитером и им же, измеренным как белый.
    "beige": "beige",
    "red": "warm", "orange": "warm", "brown": "warm",
    "yellow": "warm", "pink": "warm",
    "blue": "cool", "cyan": "cool", "purple": "cool",
    "green": "green",
}
# Группы, которые честно путаются между собой: тёмное с серым, серое со светлым.
# Дальше этого не уступаем — чёрная рубашка и оранжевая различаются надёжно.
CLOTHES_NEIGHBOURS = {
    "dark": {"gray"},
    "gray": {"dark", "light", "beige"},
    "light": {"gray", "beige"},
    "beige": {"light", "gray", "warm"},
    "warm": {"beige"},
    "cool": set(),
    "green": set(),
}


def _colour_group(colour: str) -> str:
    return COLOUR_GROUPS.get(colour, "unknown")


# Названия цветов. Границы по тону заданы в шкале OpenCV (H 0..179, то есть половина
# градуса на единицу). Сначала отсекаем то, у чего тона нет вовсе: очень тёмное, очень
# светлое и серое — на них H это шум.
COLOUR_NAMES = ("black", "white", "gray", "beige", "red", "orange", "brown", "yellow",
                "green", "cyan", "blue", "purple", "pink")


# Насколько площадка должна быть однородной, чтобы её цвету верить. Если точка легла
# на край одежды, в неё попадает и рубашка, и стена за плечом, и медиана окажется
# посередине — цвета, которого на человеке нет. Такие замеры и растаскивали память по
# семи цветам вместо одного.
COLOUR_UNIFORM_MIN = float(os.environ.get("COLOUR_UNIFORM_MIN", "0.55"))
# Ниже этой насыщенности считаем, что тона нет вовсе — только чёрный/серый/белый.
# Замерено на живых вырезках: 56% замеров попадали под прежний порог 35 и теряли цвет.
ACHROMATIC_S = float(os.environ.get("ACHROMATIC_S", "22"))
# Верхняя граница «приглушённого» тёплого тона: выше — уже честный оранжевый или жёлтый.
MUTED_WARM_S = float(os.environ.get("MUTED_WARM_S", "75"))

# Кожа. Открытые плечи, шея и руки попадают в четырёхугольник торса, а по тону кожа
# лежит ровно там же, где оранжевый и коричневый: у человека в ярко-розовой футболке
# замер выдавал orange, потому что медиану тянуло к руке, а не к ткани. Диапазон
# намеренно узкий по насыщенности — у оранжевой ткани она заметно выше, чем у кожи.
SKIN_H = (3, 22)
SKIN_S = (35, 150)
SKIN_V = (70, 245)
# Если кожи в выборке меньше этой доли, она погоды не делает и убирать её незачем.
SKIN_DROP_FROM = 0.20


def _drop_skin(px):
    """Убрать пиксели кожи, если их заметная часть и ткани после этого хватает."""
    skin = ((px[:, 0] >= SKIN_H[0]) & (px[:, 0] <= SKIN_H[1])
            & (px[:, 1] >= SKIN_S[0]) & (px[:, 1] <= SKIN_S[1])
            & (px[:, 2] >= SKIN_V[0]) & (px[:, 2] <= SKIN_V[1]))
    share = float(skin.mean())
    if share < SKIN_DROP_FROM:
        return px
    rest = px[~skin]
    return rest if len(rest) >= CLOTH_MIN_PIXELS else None


def _hsv_to_bgr(h, s, v) -> tuple:
    px = np.uint8([[[int(h) % 180, int(np.clip(s, 0, 255)), int(np.clip(v, 0, 255))]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def _colour_from_pixels(px) -> tuple[str, float, tuple | None]:
    """Название цвета, доверие к нему и САМ измеренный цвет в BGR.

    Имя нужно для поиска и проверок — там необходимы разряды, а не оттенки. А вот
    показывать на экране надо именно измеренное: тёмная бирюзовая футболка честно
    называется cyan, но нарисованная эталонным ярким циановым квадратом выглядит
    ошибкой. Квадрат измеренного цвета совпадает с тканью, и промах сразу виден.

    Медиана, а не среднее: на одежде бывают блики и складки, и одно яркое пятно
    утягивает среднее в сторону, которой на ткани нет.
    """
    if px is None or len(px) < CLOTH_MIN_PIXELS:
        return "unknown", 0.0, None
    px = _drop_skin(px)
    if px is None:
        return "unknown", 0.0, None   # под кожей ткани не осталось — врать не будем
    h = float(np.median(px[:, 0]))
    s = float(np.median(px[:, 1]))
    v = float(np.median(px[:, 2]))

    # Пёстрая выборка — это не пёстрая рубашка, а промах мимо неё. Такой замер лучше
    # выбросить, чем записать в память человека цвет, которого он не носил.
    dh = np.abs(px[:, 0].astype(np.int16) - int(h))
    dh = np.minimum(dh, 180 - dh)                     # тон — угол, 179 и 0 рядом
    close = (dh <= 15) & (np.abs(px[:, 1].astype(np.int16) - int(s)) <= 60) \
                      & (np.abs(px[:, 2].astype(np.int16) - int(v)) <= 60)
    if float(close.mean()) < COLOUR_UNIFORM_MIN:
        return "unknown", 0.0, None

    bgr = _hsv_to_bgr(h, s, v)

    def out(name, conf):
        return name, conf, bgr

    # Уверенность падает у краёв диапазона: почти чёрное и почти серое легко
    # перепутать, и честнее сказать это, чем выдать точный на вид ответ.
    if v < 45:
        return out("black", min(1.0, (45 - v) / 45 + 0.4))

    # Порог «цвета нет вовсе» был 35 — и съедал настоящие цвета. Замер на живых
    # вырезках: бежевая рубашка меряется как H=25 S=28 V=155, то есть тон у неё
    # совершенно определённый, а насыщенность просто невысокая. Под прежним порогом
    # она называлась серой, и таких замеров было больше половины.
    if s < ACHROMATIC_S:
        return out("white", 0.5 + v / 510) if v > 190 else out("gray", 0.5)

    conf = min(1.0, 0.45 + s / 255 * 0.55)

    # Приглушённые тёплые тона — бежевый, кремовый, песочный. На одежде их очень
    # много, и раньше все они уходили в серый или белый, теряя различие между
    # человеком в бежевом свитере и человеком в белой рубашке.
    if 6 <= h < 33 and s < MUTED_WARM_S:
        return out("beige", conf) if v >= 120 else out("brown", conf)

    if h < 8 or h >= 172:
        return out("red", conf)
    if h < 22:
        return out("brown", conf) if v < 130 else out("orange", conf)
    if h < 33:
        return out("yellow", conf)
    if h < 78:
        return out("green", conf)
    if h < 100:
        return out("cyan", conf)
    if h < 130:
        return out("blue", conf)
    if h < 155:
        return out("purple", conf)
    return out("pink", conf)


def _roi(frame, x1, y1, x2, y2):
    """Прямоугольник, обрезанный по кадру, и ПУСТАЯ маска ровно его размера.

    Маска раньше заводилась на весь кадр — два мегабайта нулей на каждый замер, по
    два замера на человека, до шести раз в секунду. Ради четырёхугольника в двести
    пикселей. Теперь маска размером с сам четырёхугольник, а точки в неё рисуются
    со сдвигом на его угол.
    """
    H, W = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return x1, y1, x2, y2, np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)


def _masked_hsv(frame, mask, x1, y1, x2, y2):
    """HSV-пиксели кадра под маской. Маска, а не прямоугольник: прямоугольник вокруг
    двух ног состоит наполовину из пола между ними, а вокруг разведённых рук — из
    стены за человеком. Считаем ровно по телу. Маска — размером с ROI (см. _roi)."""
    sub = mask > 0
    if int(sub.sum()) < CLOTH_MIN_PIXELS:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    return hsv[sub]


def _upper_pixels(frame, kpts):
    """Ткань на торсе: четырёхугольник плечи-бёдра, сжатый внутрь.

    Верхнюю часть срезаем: у самых плеч в вырезку попадают шея и открытая кожа, а её
    тон почти совпадает с оранжевым и коричневым — именно так кожа и записывалась
    человеку в память как цвет рубашки.
    """
    need = (KP_L_SH, KP_R_SH, KP_L_HIP, KP_R_HIP)
    if not all(kpts[i][2] >= CLOTH_KP_CONF for i in need):
        return None, 0.0
    quad = np.array([kpts[KP_L_SH][:2], kpts[KP_R_SH][:2],
                     kpts[KP_R_HIP][:2], kpts[KP_L_HIP][:2]], dtype=np.float32)
    centre = quad.mean(axis=0)
    quad = centre + (quad - centre) * (1.0 - CLOTH_SHRINK)
    # Сдвигаем весь четырёхугольник вниз от шеи на долю его высоты.
    height = float(quad[:, 1].max() - quad[:, 1].min())
    quad[0][1] += height * CLOTH_NECK_TRIM
    quad[1][1] += height * CLOTH_NECK_TRIM

    pts = quad.astype(np.int32)
    weakest = min(float(kpts[i][2]) for i in need)
    roi = _roi(frame, pts[:, 0].min(), pts[:, 1].min(),
               pts[:, 0].max() + 1, pts[:, 1].max() + 1)
    if roi is None:
        return None, weakest
    x1, y1, x2, y2, mask = roi
    cv2.fillConvexPoly(mask, pts - np.array([x1, y1], dtype=np.int32), 255)
    return _masked_hsv(frame, mask, x1, y1, x2, y2), weakest


def _lower_pixels(frame, kpts):
    """Ткань на ногах: полосы вдоль бедра и голени, по каждой видимой ноге.

    Именно вдоль линий, а не по общей рамке: между ногами всегда пол или стул.
    Начинаем не от самого бедра — там пояс и край рубашки.
    """
    limbs = [(KP_L_HIP, KP_L_KNEE), (KP_R_HIP, KP_R_KNEE),
             (KP_L_KNEE, KP_L_ANK), (KP_R_KNEE, KP_R_ANK)]
    # Сначала собираем полосы, потом рисуем: размер маски известен только когда
    # известны все полосы.
    strokes, xs, ys, confs = [], [], [], []
    for a, b in limbs:
        # Обе точки конечности должны быть уверенными: у сидящего за столом поза
        # дорисовывает колени по догадке, и полоса легла бы по столешнице.
        if kpts[a][2] < CLOTH_KP_CONF or kpts[b][2] < CLOTH_KP_CONF:
            continue
        pa = np.array(kpts[a][:2], dtype=np.float32)
        pb = np.array(kpts[b][:2], dtype=np.float32)
        length = float(np.linalg.norm(pb - pa))
        if length < 12:
            continue
        # Отступаем от суставов: у бедра пояс, у колена сгиб и тень.
        p1 = pa + (pb - pa) * CLOTH_LIMB_FROM
        p2 = pa + (pb - pa) * CLOTH_LIMB_TO
        thick = max(3, int(length * CLOTH_LIMB_WIDTH))
        strokes.append((p1, p2, thick))
        xs += [p1[0], p2[0]]
        ys += [p1[1], p2[1]]
        confs += [float(kpts[a][2]), float(kpts[b][2])]
    if not confs:
        return None, 0.0
    pad = 10
    roi = _roi(frame, min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    if roi is None:
        return None, min(confs)
    x1, y1, x2, y2, mask = roi
    off = np.array([x1, y1], dtype=np.float32)
    for p1, p2, thick in strokes:
        cv2.line(mask, tuple((p1 - off).astype(int)), tuple((p2 - off).astype(int)),
                 255, thick)
    return _masked_hsv(frame, mask, x1, y1, x2, y2), min(confs)


def _cloth_measure(frame, kpts):
    """Цвет верха и низа по скелету. Кадр — ДО отрисовки, иначе меряем белые линии.

    Уверенность точек входит в уверенность цвета: замер на еле видимых суставах не
    должен весить столько же, сколько замер на чётко видимых. Слабые отсеются на
    голосовании порогом 0.4, даже если формально прошли планку.
    """
    up_px, up_kp = _upper_pixels(frame, kpts)
    lo_px, lo_kp = _lower_pixels(frame, kpts)
    up_name, up_conf, up_bgr = _colour_from_pixels(up_px)
    lo_name, lo_conf, lo_bgr = _colour_from_pixels(lo_px)
    return {"upper": up_name, "upper_conf": up_conf * up_kp, "upper_kp": up_kp,
            "upper_bgr": up_bgr,
            "lower": lo_name, "lower_conf": lo_conf * lo_kp, "lower_kp": lo_kp,
            "lower_bgr": lo_bgr}


# Кто во что одет прямо сейчас: (камера, трек) -> запись. Живёт только в памяти —
# поиск отвечает на вопрос «где он сейчас», истории мы не ведём.
_appearance: dict[tuple, dict] = {}
_appearance_lock = threading.Lock()


def _appearance_update(cam_id, track_id, name, seen):
    """Копим замеры по треку и публикуем самый частый.

    По одному кадру цвет неустойчив — замер это показал: у одного и того же человека
    тон гулял на 40 градусов. Голосование за несколько секунд убирает выбросы, которые
    дают блик, тень и мелькнувший на площадке фон.
    """
    key = (cam_id, track_id)
    now = time.time()
    with _appearance_lock:
        rec = _appearance.get(key)
        if rec is None:
            rec = {"upper": [], "lower": [], "name": None}
            _appearance[key] = rec
        for slot in ("upper", "lower"):
            if seen[slot] != "unknown" and seen[f"{slot}_conf"] >= 0.4:
                rec[slot].append(seen[slot])
                del rec[slot][:-CLOTH_VOTES]
                # Настоящий цвет ткани — для показа. Имя останется разрядом, но
                # квадрат на экране должен совпадать с рубашкой, а не с эталоном.
                if seen.get(f"{slot}_bgr"):
                    rec[f"{slot}_bgr"] = seen[f"{slot}_bgr"]
        rec["name"] = name
        rec["t"] = now
        rec["upper_now"] = max(set(rec["upper"]), key=rec["upper"].count) if rec["upper"] else "unknown"
        rec["lower_now"] = max(set(rec["lower"]), key=rec["lower"].count) if rec["lower"] else "unknown"


def _appearance_snapshot():
    now = time.time()
    with _appearance_lock:
        for key in [k for k, r in _appearance.items() if now - r["t"] > CLOTH_TTL]:
            del _appearance[key]
        return [{"camera_id": cam, "track_id": tid, "name": r["name"],
                 "upper": r["upper_now"], "lower": r["lower_now"],
                 "upper_votes": len(r["upper"]), "lower_votes": len(r["lower"]),
                 "age": round(now - r["t"], 2)}
                for (cam, tid), r in _appearance.items()]


_clothes_cache: dict = {"day": None, "at": 0.0, "map": {}}
_clothes_lock = threading.Lock()


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _clothes_memory() -> dict:
    """Дневная память по всем людям. Читается на каждом распознавании, а меняется
    медленно, поэтому держим её в кэше и перечитываем раз в несколько секунд."""
    now = time.time()
    day = _today()
    with _clothes_lock:
        if _clothes_cache["day"] != day or now - _clothes_cache["at"] > CLOTHES_CACHE_SEC:
            # Новый день — память начинается заново: вчерашняя рубашка сегодня врёт.
            _clothes_cache["map"] = face_store.clothing_day_map(day)
            _clothes_cache["day"] = day
            _clothes_cache["at"] = now
        return _clothes_cache["map"]


def _clothes_forget_cache() -> None:
    with _clothes_lock:
        _clothes_cache["at"] = 0.0


_cloth_last_seen: dict[int, float] = {}     # person_id -> когда была засечка
_cloth_seen_lock = threading.Lock()


def _clothes_confirm(person_id, label, cam_id, seen, face_sim, kp_conf, t) -> bool:
    """Копим удачные засечки и, когда несколько подряд сошлись, признаём цвет за днём.

    Почему не по первой же засечке: она может прийтись на неудачный свет или на
    чужого человека, которого лицо приняло за своего. Такая ошибка попала бы в память
    и потом весь день отменяла бы ВЕРНЫЕ распознавания — то есть навредила бы ровно
    там, ради чего память и заводится.
    """
    day = _today()
    current = _clothes_memory().get(person_id)
    changing = False
    if current and current.get("upper"):
        # На сегодня цвет уже признан. Возвращаемся к нему только если человек, судя
        # по кадру, ПЕРЕОДЕЛСЯ, и лицо при этом узнано очень уверенно.
        if face_sim < CLOTH_RECONFIRM_SIM:
            return False
        if (_same_colour(current["upper"], seen["upper"])
                and _same_colour(current.get("lower") or "", seen["lower"])):
            return False                         # одет как записано, менять нечего
        changing = True

    with _cloth_seen_lock:
        if t - _cloth_last_seen.get(person_id, 0.0) < CLOTH_CONFIRM_GAP:
            return False                         # слишком быстро после прошлой
        _cloth_last_seen[person_id] = t

    rows = face_store.add_clothing_seen(person_id, day, seen["upper"], seen["lower"],
                                        face_sim, kp_conf, cam_id)
    if changing:
        # Считаем только засечки ПОСЛЕ признания и только уверенные: старые относятся
        # к прежнему наряду и к вопросу «переоделся ли» отношения не имеют.
        rows = face_store.clothing_seen_after(person_id, day,
                                              float(current["created_at"] or 0),
                                              CLOTH_RECONFIRM_SIM)
    last = rows[-CLOTH_CONFIRM_TIMES:]
    logger.info(f"[cloth] {cam_id} {label} "
                f"{'CHANGE ' if changing else ''}sighting {len(rows)}: "
                f"{seen['upper']} shirt, {seen['lower']} trousers "
                f"(face {face_sim:.3f}, body keypoints >= {kp_conf:.2f})")
    if len(last) < CLOTH_CONFIRM_TIMES:
        return True

    # «Одно и то же» — с точностью до группы цветов: один и тот же свитер на разных
    # камерах меряется чуть по-разному, и требовать побуквенного совпадения нельзя.
    for slot in ("upper", "lower"):
        first = last[0][slot]
        if not all(_same_colour(first, r[slot]) for r in last[1:]):
            logger.info(f"[cloth] {label} sightings disagree on {slot}: "
                        f"{[r[slot] for r in last]} — not recorded yet")
            return True

    upper = max((r["upper"] for r in last), key=[r["upper"] for r in last].count)
    lower = max((r["lower"] for r in last), key=[r["lower"] for r in last].count)

    if changing:
        if face_store.replace_clothing(person_id, day, upper, lower,
                                       face_sim, kp_conf, cam_id):
            _clothes_forget_cache()
            _drop_refs_for(person_id)     # старая привязка описывает прежний наряд
            logger.info(f"[cloth] {label} CHANGED CLOTHES: "
                        f"{current['upper']}/{current.get('lower')} -> "
                        f"{upper}/{lower} ({CLOTH_CONFIRM_TIMES} sightings agree, "
                        f"face >= {CLOTH_RECONFIRM_SIM})")
    elif face_store.set_clothing(person_id, day, upper, lower,
                                 face_sim, kp_conf, cam_id):
        _clothes_forget_cache()
        logger.info(f"[cloth] {label} CONFIRMED for today: {upper} shirt, "
                    f"{lower} trousers ({CLOTH_CONFIRM_TIMES} sightings agree)")
    return True


# Кто из треков попадался за последнее окно — ОБЩЕЕ на камеру, а не на поток.
# Кадры одной камеры разбираются несколькими потоками, каждый видит лишь часть, и
# локальный набор приводил к тому, что подметание в одном потоке закрывало отрезки
# треков, которые в это же время вёл соседний: в базе отрезок присутствия рвался и
# начинался заново каждые несколько секунд.
_seen_tracks: dict[str, dict] = {}      # камера -> {трек: когда видели последний раз}
_sweep_at: dict[str, float] = {}
_sweep_lock = threading.Lock()


def _mark_seen(cam_id: str, track_id: str, t: float) -> None:
    with _sweep_lock:
        _seen_tracks.setdefault(cam_id, {})[track_id] = t


def _sweep_due(cam_id: str, t: float):
    """Пора ли закрывать отрезки ушедших. Возвращает набор ещё присутствующих или None.

    Присутствующим считается тот, кого видели за последние EVENT_TRACK_GRACE секунд,
    а не только в последнем окне: короткий провал в детекции — это не уход.
    """
    with _sweep_lock:
        if t - _sweep_at.get(cam_id, 0.0) < EVENT_SWEEP_SEC:
            return None
        _sweep_at[cam_id] = t
        seen = _seen_tracks.setdefault(cam_id, {})
        for tid in [k for k, ts in seen.items() if t - ts > EVENT_TRACK_GRACE]:
            del seen[tid]
        return set(seen)


def _record_state(cam_id, track, posture, activities, t, seg_state, seg_pending):
    """Записать состояние трека — но только когда оно ИЗМЕНИЛОСЬ и устоялось.

    Состояние трека это тройка: где он (presence), в какой позе (posture) и что делает
    (activity). Пустой набор действий — это не «нет данных», а idle: человек в кадре и
    ничего не делает, и именно это оператор потом и спрашивает.
    """
    acts = sorted(a for a in activities) or ["idle"]
    state = (posture, tuple(acts))
    prev = seg_state.get(track.track_id)
    if prev == state:
        seg_pending.pop(track.track_id, None)
        return

    # Ждём, пока новое состояние продержится: метки мигают, и без выдержки на каждое
    # мигание открывался бы и закрывался отрезок.
    pending = seg_pending.get(track.track_id)
    if pending is None or pending[0] != state:
        seg_pending[track.track_id] = (state, t)
        return
    if t - pending[1] < EVENT_DEBOUNCE_SEC:
        return

    seg_pending.pop(track.track_id, None)
    seg_state[track.track_id] = state
    # Дела записываем на человека только с той камеры, за которой закреплено его имя.
    # Иначе рабочее время считалось бы дважды: тот же человек попадает в кадр соседней
    # камеры, второй трек тоже помнит person_id — и «работал» ему шло с обеих. Трек,
    # не держащий имени, остаётся в базе безымянным: как присутствие он правдив, а вот
    # чьи это минуты — решает одна камера.
    pid = track.person_id if _holds_name(track.person_id, cam_id, track.track_id) else None
    try:
        store.record_observation(cam_id, track.track_id, pid, posture, acts, ts=t)
        # Присутствие ведём отдельным типом: из него потом складываются маршруты и
        # рабочее время, и оно не должно зависеть от того, менялась ли поза.
        store.record_presence(cam_id, track.track_id, pid, ts=t)
    except Exception as e:
        logger.error(f"[events] write failed: {e}")


def _look_unchanged(cam_id: str, track_id: str, snapshot: dict) -> bool:
    """Одет ли человек так же, как в момент, когда имя подтвердилось.

    Это ответ на вопрос «а тот ли это вообще человек». Лицо не читается по десятку
    причин — отвернулся, наклонился, тень, — и снимать из-за этого верное имя незачем.
    А вот если трек перескочил на соседа, одежда меняется скачком, и это видно.

    Возражаем только при ПОЛОЖИТЕЛЬНОМ признаке смены: не измерили цвет — значит
    доказательств нет, и решает прежняя логика промахов.
    """
    was = snapshot.get(track_id)
    if not was:
        return False
    with _appearance_lock:
        rec = _appearance.get((cam_id, track_id))
        now_up = rec.get("upper_now") if rec else None
        now_lo = rec.get("lower_now") if rec else None
    if not now_up or now_up == "unknown":
        return False
    if not _same_colour(was[0], now_up):
        return False
    # Низ проверяем, только если он измерен и тогда, и сейчас: ноги видны меньше чем
    # в половине кадров, и его отсутствие ничего не доказывает.
    if was[1] and was[1] != "unknown" and now_lo and now_lo != "unknown":
        return _same_colour(was[1], now_lo)
    return True


# (камера, person_id) -> где и в чём человека уверенно узнали в последний раз.
_refs: dict[tuple, dict] = {}
_refs_lock = threading.Lock()


def _remember_ref(cam_id, person_id, box, upper, lower, sim, t) -> None:
    """Запомнить: вот этого человека узнали здесь, в такой одежде, в этот момент."""
    if not upper or upper == "unknown":
        return
    x1, y1, x2, y2 = [float(v) for v in box]
    with _refs_lock:
        _refs[(cam_id, person_id)] = {
            "upper": upper, "lower": lower, "sim": sim, "t": t,
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "h": max(y2 - y1, 1.0),
        }


def _drop_refs_for(person_id) -> None:
    with _refs_lock:
        for key in [k for k in _refs if k[1] == person_id]:
            del _refs[key]


def _name_from_ref(cam_id, track_id, box, t):
    """Вернуть имя треку, который его потерял или только начался.

    Одежды мало: в одной комнате бывают двое в одинаковых рубашках, и по цвету имя
    ушло бы к соседу ровно в тот момент, когда узнанный человек отвернулся. Поэтому
    спрашиваем ещё и место: от точки, где человека узнали, он мог уйти не дальше, чем
    успел бы дойти. Расстояние меряем в РОСТАХ, а не в пикселях — у камеры и вдали
    цена пикселя разная.
    """
    with _appearance_lock:
        rec = _appearance.get((cam_id, track_id))
        if not rec:
            return None, None
        up, lo = rec.get("upper_now"), rec.get("lower_now")
        settled = len(rec["upper"]) >= CLOTH_VOTES
    if not settled or up in (None, "unknown"):
        return None, None

    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy, h = (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1.0)

    with _refs_lock:
        for key in [k for k, v in _refs.items() if t - v["t"] > REF_TTL]:
            del _refs[key]
        candidates = [(pid, dict(v)) for (cam, pid), v in _refs.items() if cam == cam_id]

    hits = []
    for pid, ref in candidates:
        if not _same_colour(ref["upper"], up):
            continue
        if (ref.get("lower") and ref["lower"] != "unknown"
                and lo and lo != "unknown" and not _same_colour(ref["lower"], lo)):
            continue
        reach = REF_SPEED_HEIGHTS * max(ref["h"], h) * max(t - ref["t"], 0.5)
        if np.hypot(cx - ref["cx"], cy - ref["cy"]) > reach:
            continue          # столько за это время не пройти — это другой человек
        hits.append(pid)

    if len(hits) != 1:
        return None, None     # двое подходят — молчим, чужое имя хуже, чем никакого
    pid = hits[0]
    if not _claim_name(pid, cam_id, track_id, CLOTHES_NAME_WEIGHT, t):
        return None, None
    label = face_store.label_of(pid)
    if not label or label.startswith("#"):
        return None, None
    return pid, label


def _clothes_unique_map() -> dict:
    """Кто сегодня однозначно опознаётся по одежде.

    Человека можно узнать по одежде только если больше НИКТО из признанных сегодня не
    одет так же. Чёрная футболка с чёрными брюками бывает у половины офиса, и назвать
    по ней имя — значит подписать первого встречного. Такие люди из карты выпадают:
    лучше не показать имя, чем показать чужое.
    """
    people = [(pid, rec["upper"], rec["lower"])
              for pid, rec in _clothes_memory().items()
              if rec.get("upper") and rec.get("lower")]
    out = {}
    for pid, up, lo in people:
        clash = any(other != pid and _same_colour(up, oup) and _same_colour(lo, olo)
                    for other, oup, olo in people)
        if not clash:
            out[pid] = (up, lo)
    return out


def _name_from_clothes(cam_id: str, track_id: str, t: float):
    """Опознать по одежде трек, которому лицо имени не дало.

    Нужно ровно для одного случая: человека уже узнали в лицо, он отвернулся, и лицо
    больше не читается. Одежда за минуту не меняется, значит это тот же человек, и имя
    снимать не за что. Своё имя такой трек получает с самым слабым весом, поэтому любое
    распознавание по лицу — хоть на этой камере, хоть на соседней — его перебьёт.
    """
    if not CLOTHES_VETO:
        return None, None
    with _appearance_lock:
        rec = _appearance.get((cam_id, track_id))
        if not rec:
            return None, None
        up, lo = rec.get("upper_now"), rec.get("lower_now")
        # Требуем полную историю голосов: по одному-двум замерам имя не выдаём.
        settled = (len(rec["upper"]) >= CLOTH_VOTES and len(rec["lower"]) >= CLOTH_VOTES)
    if not settled or up in (None, "unknown") or lo in (None, "unknown"):
        return None, None

    hits = [pid for pid, (rup, rlo) in _clothes_unique_map().items()
            if _same_colour(rup, up) and _same_colour(rlo, lo)]
    if len(hits) != 1:
        return None, None

    pid = hits[0]
    if not _claim_name(pid, cam_id, track_id, CLOTHES_NAME_WEIGHT, t):
        return None, None          # этого человека прямо сейчас видит камера получше
    label = face_store.label_of(pid)
    if not label or label.startswith("#"):
        return None, None
    return pid, label


def _same_colour(recorded: str, seen: str) -> bool:
    """Один ли это цвет — с точностью до группы и соседней с ней."""
    a, b = _colour_group(recorded), _colour_group(seen)
    if a == "unknown" or b == "unknown":
        return True                       # сравнивать не с чем — не возражаем
    return a == b or b in CLOTHES_NEIGHBOURS.get(a, set())


def _clothes_verdict(person_id: int, cam_id: str, track_id: str, face_sim: float):
    """Подтверждает ли одежда, что это тот самый человек.

    Возвращает (ok, причина). Возражаем только когда есть на что опереться: набрана
    дневная память и текущий цвет верха измерен. Низ в решении не участвует — ноги
    видны меньше чем в половине кадров, и вето по ним било бы по сидящим за столом.
    """
    if not CLOTHES_VETO or person_id is None:
        return True, ""
    if face_sim >= CLOTHES_VETO_MAX_SIM:
        # Лицо уверено — верим ему и заодно запомним новую одежду.
        return True, ""
    mem = _clothes_memory().get(person_id)
    if not mem or not mem.get("upper"):
        return True, ""

    with _appearance_lock:
        rec = _appearance.get((cam_id, track_id))
        now_colour = rec.get("upper_now") if rec else None
    if not now_colour or now_colour == "unknown":
        return True, ""

    if _same_colour(mem["upper"], now_colour):
        return True, ""
    return False, f"wearing {now_colour}, recorded today as {mem['upper']}"


# Один человек — одно имя на всех камерах сразу. Обработка камер идёт потоками одного
# процесса, поэтому хватает обычного словаря: person_id -> (камера, трек, похожесть, когда).
# person_id -> ((камера, трек), когда, похожесть). РОВНО ОДИН держатель на человека:
# имя не может быть написано в двух местах сразу, ни на одной камере дважды, ни на двух
# камерах — даже если это действительно один и тот же человек.
_name_owner: dict[int, tuple] = {}
_name_owner_lock = threading.Lock()


def _claim_name(person_id: int, cam_id: str, track_id: str, sim: float, t: float) -> bool:
    """Можно ли подписать этот трек этим именем.

    Имя живёт РОВНО В ОДНОМ месте: один трек на одной камере, тот, кто видит лицо
    лучше всех. Не «обычно один», а один всегда.

    Прежняя версия допускала исключение: если одежда на двух треках сходилась с
    дневной записью человека, имя оставалось на обоих — считалось, что человек и
    правда попал в поле зрения двух камер сразу. На практике это и породило двух
    «rafo» в одном кадре: в помещении несколько человек в тёмном верхе и синих
    джинсах, цвет у них сходится, и подтверждением это быть перестаёт. Одежда
    по-прежнему может ОТМЕНИТЬ имя (это её сильная сторона — независимая улика
    против), но выдавать второе имя больше не может.

    Проигравший трек ничего не теряет: он остаётся в кадре под своим номером, а имя
    вернётся к нему в ту же секунду, как он увидит лицо лучше соперника.
    """
    me = (cam_id, track_id)
    evicted = None
    with _name_owner_lock:
        cur = _name_owner.get(person_id)
        stale = cur is not None and t - cur[1] > NAME_CLAIM_TTL
        fresh = False
        if cur is None or stale or cur[0] == me or sim > cur[2]:
            if cur is not None and cur[0] != me:
                # Отдаём отрезок и у протухшего владельца тоже. Протухло — значит его
                # лицо не подтверждалось три секунды, а вовсе не обязательно что он
                # ушёл: чаще он просто отвернулся и всё ещё стоит в кадре с открытым
                # отрезком. Оставить отрезок за ним — это и есть те самые минуты с
                # двух камер сразу.
                evicted = cur[0]
            fresh = cur is None or cur[0] != me
            _name_owner[person_id] = (me, t, sim)
            ok = True
        else:
            ok = False

    # Смена хозяина мгновенна, а отрезок — промежуток, и обе стороны надо разрезать
    # ровно по этому моменту.
    #
    # Проигравший: он узнает о проигрыше только на своей следующей проверке лица —
    # через секунду, а то и позже. Всё это время его незакрытый отрезок продолжал бы
    # идти человеку поверх отрезка победителя: имя в кадре уже одно, а минуты капали с
    # двух камер. Отнимаем сразу. Только у живого: если владелец просто пропал из
    # кадра (истёк TTL), его отрезок закроет обычная развёртка, и это честная история.
    #
    # Победитель: его отрезок начался раньше, когда человек за ним ещё не числился.
    # Без разреза минуты пошли бы человеку задним числом — как раз за тот промежуток,
    # который уже записан за проигравшим.
    if EVENTS_ENABLED and (evicted is not None or fresh):
        try:
            if evicted is not None:
                store.release_track(*evicted, ts=t)
            if fresh:
                store.switch_track_person(cam_id, track_id, person_id, ts=t)
        except Exception as exc:                                # noqa: BLE001
            logger.error(f"[events] handover: {exc}")
    return ok


def _holds_name(person_id, cam_id: str, track_id: str) -> bool:
    if person_id is None:
        return False
    with _name_owner_lock:
        cur = _name_owner.get(person_id)
        return bool(cur) and cur[0] == (cam_id, track_id)


def _release_name(person_id, cam_id: str, track_id: str) -> None:
    if person_id is None:
        return
    with _name_owner_lock:
        cur = _name_owner.get(person_id)
        if cur and cur[0] == (cam_id, track_id):
            _name_owner.pop(person_id, None)


def _face_batch(crops: list) -> list:
    """Лица по пачке вырезок голов — одним проходом ArcFace на всех сразу.

    Раньше каждая камера звала _face_model.get(crop) под общим замком: детектор, потом
    ArcFace на одном лице, и следующая камера ждёт. Замерено на этой машине:
      SCRFD (160)                    5.9 мс на вырезку — по одной, не пакетуется
      ArcFace, одно лицо             9.8 мс
      ArcFace, восемь лиц пакетом   14.4 мс  -> 1.8 мс на лицо
    То есть ArcFace на одном лице — это почти целиком накладные расходы запуска, и
    пакет из восьми стоит как полтора одиночных. Поэтому детектор гоняем по каждой
    вырезке, а лица выравниваем и отправляем в ArcFace ОДНИМ вызовом.

    Замок при этом остаётся, но держит его один поток на всех — камеры больше не
    стоят друг за другом, они кладут вырезку в очередь и получают ответ.
    """
    det = _face_model.models["detection"]
    rec = _face_model.models["recognition"]
    from insightface.utils import face_align

    aligned, scores, where = [], [], []
    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            continue
        bboxes, kpss = det.detect(crop, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            continue
        j = int(np.argmax(bboxes[:, 4]))
        aligned.append(face_align.norm_crop(crop, landmark=kpss[j]))
        scores.append(float(bboxes[j, 4]))
        where.append(i)

    out: list = [None] * len(crops)
    if not aligned:
        return out
    feats = np.asarray(rec.get_feat(aligned), dtype=np.float32).reshape(len(aligned), -1)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.maximum(norms, 1e-6)
    for k, i in enumerate(where):
        out[i] = {"embedding": feats[k], "det_score": scores[k]}
    return out


def _recognise_head(crop):
    """Эмбеддинг по вырезке головы. Возвращает None, если лица там не разобрать."""
    if crop is None or crop.size == 0:
        return None
    return _face_batcher.submit(crop)


def _faces_in_jpeg(jpeg: bytes):
    """Все лица на присланной картинке: эмбеддинг, оценка и вырезка по каждому.

    Работаем по ПОЛНОМУ кадру из события (backgroundImage, 1280x720), а не по вырезке,
    которую камера сделала сама. Камера на одно событие отдаёт ровно одно лицо — то,
    которое сочла лучшим, — и если в кадре стоят трое, двое просто теряются. Именно на
    этом мы и не добирали людей. Работающая система на 172.16.11.179 делает так же:
    в её обработчике alarm'а saved_image перезаписывается по всем image-частям и в итоге
    указывает на backgroundImage, по которому потом и гоняется детектор.

    Полный кадр здесь стоит недорого: он приходит только по событию от камеры, а не
    шесть раз в секунду, и он 1280x720, а не 1920x1080.

    Модель одна на процесс и не потокобезопасна, а события могут прийти от двух камер
    одновременно — отсюда блокировка.
    """
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    with _face_lock:
        found = _face_model.get(img)

    H, W = img.shape[:2]
    out = []
    for f in found:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        pad = int(max(x2 - x1, y2 - y1) * 0.15)
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
        crop = img[cy1:cy2, cx1:cx2]
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        out.append({"embedding": f.normed_embedding,
                    "det_score": float(f.det_score),
                    "bbox": (x1, y1, x2, y2),
                    "size": (x2 - x1, y2 - y1),
                    "crop": buf.tobytes() if ok else None})
    return out


def _load_shared_models():
    from ultralytics import YOLO
    global _pose_model, _obj_model, _weapon_model, _face_model
    global _pose_batcher, _obj_batcher, _weapon_batcher, _face_batcher, _weapon_names

    logger.info(f"loading shared models (pose + objects + weapon, half={INFER_HALF}) ...")
    logger.info(f"  objects: {Path(OBJ_MODEL).name}")
    logger.info(f"  pose   : {Path(POSE_MODEL).name}")
    _pose_model = YOLO(POSE_MODEL)
    _obj_model = YOLO(OBJ_MODEL)

    warm = np.zeros((1080, 1920, 3), dtype=np.uint8)

    if WEAPON_BACKEND == "yoloe":
        if YOLOE_ENGINE.exists():
            logger.info(f"  weapon : {YOLOE_ENGINE.name}  (OPEN-VOCABULARY, TensorRT)")
            _weapon_model = YOLO(str(YOLOE_ENGINE))
        else:
            from ultralytics import YOLOE
            logger.info(f"  weapon : {YOLOE_WEIGHTS}  (OPEN-VOCABULARY, PyTorch — no engine)")
            _weapon_model = YOLOE(YOLOE_WEIGHTS)
            _weapon_model.set_classes(YOLOE_PROMPTS, _weapon_model.get_text_pe(YOLOE_PROMPTS))
        # На рамке пишем короткое имя, а не весь текстовый запрос.
        _weapon_names = dict(enumerate(YOLOE_LABELS))
        logger.info(f"  prompts: {YOLOE_PROMPTS}")
        logger.info(f"  conf   : {YOLOE_CONF}, every {YOLOE_EVERY_N}-th frame")
        _weapon_model.predict(warm, imgsz=WEAPON_IMGSZ, conf=YOLOE_CONF,
                              device=INFER_DEVICE, half=INFER_HALF, verbose=False)
    else:
        logger.info(f"  weapon : {Path(WEAPON_MODEL).name}")
        _weapon_model = YOLO(WEAPON_MODEL)
        # Имена классов берём у самой модели: так набор переживает дообучение.
        _weapon_names = {int(i): n for i, n in _weapon_model.names.items()
                         if n in WEAPON_CLASS_NAMES}
        if not _weapon_names:
            logger.warning(f"weapon model classes {dict(_weapon_model.names)} do not intersect "
                           f"WEAPON_CLASS_NAMES {sorted(WEAPON_CLASS_NAMES)} — detection is OFF")
        else:
            logger.info(f"  weapon classes: {_weapon_names}")
        _weapon_model(warm, imgsz=WEAPON_IMGSZ, device=INFER_DEVICE, half=INFER_HALF,
                      verbose=False)

    _pose_model(warm, imgsz=POSE_IMGSZ, device=INFER_DEVICE, half=INFER_HALF, verbose=False)
    _obj_model(warm, classes=OBJ_CLASS_IDS, imgsz=OBJ_IMGSZ, device=INFER_DEVICE,
               half=INFER_HALF, verbose=False)

    # --- лица -------------------------------------------------------------
    if FACE_ENABLED:
        # DLL-каталог torch добавляем ДО импорта onnxruntime: иначе он не найдёт
        # cublas/cudnn и молча уйдёт на CPU (722 мс на кадр вместо 67).
        try:
            import torch
            os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
        except Exception as e:
            logger.warning(f"could not add torch DLL dir: {e}")
        from insightface.app import FaceAnalysis
        _face_model = FaceAnalysis(name=FACE_MODEL,
                                   allowed_modules=["detection", "recognition"],
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        _face_model.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE),
                            det_thresh=FACE_DET_THRESH)
        used = _face_model.models["detection"].session.get_providers()
        logger.info(f"  faces  : {FACE_MODEL} det={FACE_DET_SIZE} providers={used}")
        if "CUDAExecutionProvider" not in used:
            logger.warning("faces on CPU — ~700 ms per frame instead of ~67")
        _face_model.get(warm)
    else:
        logger.info("  faces  : DISABLED (FACE_ENABLED=0)")

    # --- одежда -----------------------------------------------------------
    if CLOTH_ENABLED:
        logger.info(f"  clothes: from pose keypoints, {CLOTH_TARGET_HZ} Hz per person "
                    f"(no extra model)")
    else:
        logger.info("  clothes: DISABLED (CLOTH_ENABLED=0)")

    _pose_batcher = _BatchedInference("pose", _pose_batch, POSE_MAX_BATCH)
    _obj_batcher = _BatchedInference("objects", _obj_batch, OBJ_MAX_BATCH)
    _weapon_batcher = _BatchedInference("weapon", _weapon_batch, WEAPON_MAX_BATCH)
    if FACE_ENABLED:
        # Замок внутри: путь событий от камеры (_faces_in_jpeg) трогает те же сессии
        # ONNX, и пусть они не пересекаются. Контенции от камер тут уже нет —
        # в очередь кладут, за замок не борются.
        def _face_batch_locked(crops):
            with _face_lock:
                return _face_batch(crops)
        _face_batcher = _BatchedInference("face", _face_batch_locked, FACE_MAX_BATCH)
    logger.info(f"shared models ready (pose batch<={POSE_MAX_BATCH}, "
                f"objects batch<={OBJ_MAX_BATCH}, weapon batch<={WEAPON_MAX_BATCH}; "
                f"faces are event-driven, not per-frame)")
_connected: dict[str, bool] = {c["id"]: False for c in CAMERAS}
_last_error: dict[str, str] = {c["id"]: "" for c in CAMERAS}
UNKNOWN_FACE_DIR = Path(__file__).resolve().parent.parent / "data" / "unknown_faces"
UNKNOWN_FACE_DIR.mkdir(parents=True, exist_ok=True)


def _person_hand_rects(person_kpts, with_wrist=False):
    rects = []
    for elbow_idx, wrist_idx in _ARM_SIDES:
        if person_kpts[elbow_idx, 2] < KP_CONF or person_kpts[wrist_idx, 2] < KP_CONF:
            continue
        elbow = person_kpts[elbow_idx, :2]
        wrist = person_kpts[wrist_idx, :2]
        fingers = wrist + (wrist - elbow) * FINGER_EXTEND
        rect = _finger_rect(wrist, fingers)
        if rect is not None:
            rect = np.asarray(rect, dtype=np.float32)
            rects.append((wrist_idx, rect) if with_wrist else rect)
    return rects


def _collect_hand_rects(kpts):
    rects = []
    for person_kpts in kpts:
        rects.extend(_person_hand_rects(person_kpts))
    return rects


def _expand_box(box, pad_ratio=0.35, min_pad=18.0):
    x1, y1, x2, y2 = [float(v) for v in box]
    pad = max(min_pad, max(x2 - x1, y2 - y1) * pad_ratio)
    return np.array([x1 - pad, y1 - pad, x2 + pad, y2 + pad], dtype=np.float32)


def _weapon_from_output(weapon_out):
    """Разбирает выход детектора оружия, оставляя только классы из WEAPON_CLASS_NAMES."""
    boxes, names, confs = [], [], []
    if weapon_out is None:
        return boxes, names, confs
    out_boxes, out_cls, out_conf = weapon_out
    for box, cls_id, conf in zip(out_boxes, out_cls, out_conf):
        name = _weapon_names.get(int(cls_id))
        if name is None:      # класс есть в модели, но не в WEAPON_CLASS_NAMES
            continue
        boxes.append([float(v) for v in box])
        names.append(name)
        confs.append(float(conf))
    return boxes, names, confs


def _draw_weapons(frame, boxes, names, confs):
    """Красная рамка — единственный цвет, которым в кадре не рисуется ничего другого:
    скелет белый, объекты белые, подписи треков тоже. Оружие должно быть видно сразу."""
    for box, name, conf in zip(boxes, names, confs):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), WEAPON_COLOR, 3)
        text = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), WEAPON_COLOR, -1)
        cv2.putText(frame, text, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _person_phone_wrists(person_kpts, phone_boxes):
    if len(phone_boxes) == 0:
        return []
    expanded_boxes = [_expand_box(box) for box in phone_boxes]
    phone_wrists = []
    for wrist_idx, rect in _person_hand_rects(person_kpts, with_wrist=True):
        if any(_polygons_overlap(rect, _box_corners(box)) for box in expanded_boxes):
            phone_wrists.append(wrist_idx)
    return phone_wrists


# BGR для квадратиков рядом с именем. Показываем именно то, что система записала,
# а не то, что кажется глазу: если квадратик не совпадает с рубашкой на картинке —
# значит промахнулась точка замера, и это видно сразу, без открывания страниц.
SWATCH_BGR = {
    "black": (25, 25, 25), "gray": (136, 136, 136), "white": (238, 238, 238),
    "beige": (150, 190, 214),
    "red": (51, 51, 211), "orange": (32, 128, 232), "brown": (43, 90, 138),
    "yellow": (50, 200, 216), "green": (58, 157, 58), "cyan": (200, 183, 58),
    "blue": (216, 111, 58), "purple": (200, 82, 138), "pink": (176, 122, 224),
}
SWATCH_PX = 16


def _swatch_css(name: str) -> str:
    """Тот же цвет, что рисуется над головой в кадре, но для страницы.

    Один источник на оба места: если на видео «beige» одного оттенка, а в списке
    выбора другого, доверять списку перестают — а он именно про цвет.
    """
    bgr = SWATCH_BGR.get(name)
    if not bgr:
        return "#444"
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def _draw_track_label(frame, person_box, text, upper=None, lower=None):
    """Подпись над головой: имя и, рядом, два квадратика — цвет верха и цвет низа.

    upper/lower — либо имя цвета, либо уже готовый BGR измеренного цвета. Второе
    предпочтительнее: эталонный квадрат яркого циана рядом с тёмной бирюзовой
    футболкой читается как ошибка, хотя имя цвета верное.
    """
    swatches = [c if isinstance(c, tuple) else SWATCH_BGR.get(c)
                for c in (upper, lower)]
    swatches = [c for c in swatches if c]
    if not text and not swatches:
        return
    text = text or ""
    x1, y1, x2, _ = [int(v) for v in person_box]
    frame_h, frame_w = frame.shape[:2]
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)

    extra = len(swatches) * (SWATCH_PX + 4)
    box_w = tw + 10 + extra
    lx = int(np.clip((x1 + x2 - box_w) / 2, 0, max(frame_w - box_w - 2, 0)))
    top = max(0, y1 - th - base - 38)
    bottom = min(frame_h - 1, top + th + base + 8)
    rx = min(lx + box_w, frame_w - 1)

    cv2.rectangle(frame, (lx, top), (rx, bottom), (20, 20, 20), -1)
    cv2.rectangle(frame, (lx, top), (rx, bottom), (255, 255, 255), 1)
    if text:
        cv2.putText(frame, text, (lx + 5, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2)

    sx = lx + 5 + tw + 4
    sy = top + (bottom - top - SWATCH_PX) // 2
    for colour in swatches:
        if sx + SWATCH_PX >= frame_w:
            break
        cv2.rectangle(frame, (sx, sy), (sx + SWATCH_PX, sy + SWATCH_PX),
                      colour, -1)
        cv2.rectangle(frame, (sx, sy), (sx + SWATCH_PX, sy + SWATCH_PX),
                      (255, 255, 255), 1)
        sx += SWATCH_PX + 4


class SkeletonFlowTracker:
    """Тянет keypoints между запусками pose-модели оптическим потоком.

    Работает по ЯРКОСТНОЙ плоскости кадра (gray), а не по BGR: она приходит из
    камеры готовой — это ровно плоскость Y кадра YV12, и лежит в начале слота.
    Раньше здесь был cvtColor(frame, BGR2GRAY), то есть кадр сначала разворачивали
    в цвет, а потом цвет из него же и убирали.

    Y хранится в узком диапазоне (16..235), поэтому численно он не равен выходу
    BGR2GRAY, а связан с ним аффинно (y = 0.859*x + 16). Для LK это безразлично:
    смещение получается из отношения градиентов, и общий множитель сокращается.
    Единственное следствие — err на выходе потока меньше на те же ~14%, то есть
    порог FLOW_ERR_LIMIT стал мягче примерно как 70 в прежней шкале.
    """
    MAX_POINT_JUMP = 120.0
    FLOW_ERR_LIMIT = 60.0
    CONF_DECAY = 0.90
    MIN_BOX_POINTS = 4

    def __init__(self):
        self.prev_gray = None
        self.kpts = np.empty((0, 17, 3), dtype=np.float32)
        self.boxes = np.empty((0, 4), dtype=np.float32)

    def reset(self, gray, kpts, boxes):
        self.prev_gray = gray
        self.kpts = np.asarray(kpts, dtype=np.float32).copy()
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()

    def update(self, gray):
        if self.prev_gray is None or len(self.kpts) == 0:
            return np.empty((0, 17, 3), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

        tracked = self.kpts.copy()
        boxes = self.boxes.copy()
        valid_mask = self.kpts[:, :, 2] >= KP_CONF
        points = self.kpts[:, :, :2][valid_mask].astype(np.float32)
        if len(points) == 0:
            self.prev_gray = gray
            return tracked, boxes

        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            points.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        status = status.reshape(-1).astype(bool) if status is not None else np.zeros((len(points),), dtype=bool)
        err = err.reshape(-1) if err is not None else np.full((len(points),), 999.0, dtype=np.float32)
        next_points = next_points.reshape(-1, 2) if next_points is not None else points.copy()
        jumps = np.linalg.norm(next_points - points, axis=1)
        good = status & (err <= self.FLOW_ERR_LIMIT) & (jumps <= self.MAX_POINT_JUMP)

        person_idx, point_idx = np.where(valid_mask)
        h, w = gray.shape[:2]
        for n, (pi, ki) in enumerate(zip(person_idx, point_idx)):
            if good[n]:
                tracked[pi, ki, :2] = np.clip(next_points[n], [0, 0], [w - 1, h - 1])
                tracked[pi, ki, 2] = min(1.0, tracked[pi, ki, 2] * 0.98)
            else:
                tracked[pi, ki, 2] *= self.CONF_DECAY

        for pi in range(len(tracked)):
            old_valid = valid_mask[pi]
            new_valid = tracked[pi, :, 2] >= KP_CONF
            both = old_valid & new_valid
            if np.count_nonzero(both) >= self.MIN_BOX_POINTS and pi < len(boxes):
                deltas = tracked[pi, both, :2] - self.kpts[pi, both, :2]
                delta = np.median(deltas, axis=0)
                boxes[pi] = boxes[pi] + np.array([delta[0], delta[1], delta[0], delta[1]], dtype=np.float32)
                boxes[pi, [0, 2]] = np.clip(boxes[pi, [0, 2]], 0, w - 1)
                boxes[pi, [1, 3]] = np.clip(boxes[pi, [1, 3]], 0, h - 1)

        self.prev_gray = gray
        self.kpts = tracked
        self.boxes = boxes
        return tracked.copy(), boxes.copy()


def _process_camera(cam_id: str, shms: list, free_q: mp.Queue, filled_q: mp.Queue,
                    worker_idx: int = 0):
    """Один из WORKERS_PER_CAMERA параллельных потоков на камеру, каждый со своими
    экземплярами моделей — камеры и воркеры внутри камеры считаются параллельно.

    Поза, объекты и оружие — независимые вычисления на одном и том же кадре, но раньше
    вызывались один за другим (GPU часть времени простаивала). Гоняем их ПАРАЛЛЕЛЬНО
    в трёх потоках внутри воркера — GPU получает работу от всех вызовов одновременно."""
    executor = ThreadPoolExecutor(max_workers=3)
    km_memory = ObjectSlotMemory()      # память позиции мыши/клавиатуры для ЭТОЙ камеры
    working_memory = WorkingContactMemory()
    skeleton_tracker = SkeletonFlowTracker()
    object_box_tracker = ObjectBoxTracker()
    phone_box_tracker = PhoneBoxTracker()  # keeps phone boxes stable when detector flickers
    # Счётчики фактической частоты. Целевая задаётся в POSE_TARGET_HZ/OBJ_TARGET_HZ, но это
    # ВЕРХНЯЯ граница: если цикл не успевает, детекторы идут реже, и понять это по кадрам
    # в браузере нельзя — картинка просто чуть менее плавная. Для планирования числа камер
    # важна именно достигнутая частота, поэтому меряем её и пишем в лог.
    stat_t0 = time.monotonic()
    stat_frames = stat_pose = stat_obj = stat_weapon = stat_people = 0
    stat_face_tries = stat_face_hits = stat_face_new = 0
    stat_face_drop = stat_face_yield = stat_face_keep = 0
    stat_ms_models = stat_ms_jpeg = stat_ms_track = stat_ms_frame = 0.0
    face_try: dict[str, float] = {}     # track_id -> когда последний раз опрашивали
    face_enroll: dict[str, float] = {}  # track_id -> когда последний раз клали в галерею
    face_miss: dict[str, int] = {}      # track_id -> сколько проверок подряд не подтвердили имя
    par_try: dict[str, float] = {}      # track_id -> когда последний раз смотрели одежду
    prev_track_ids: set = set()         # треки прошлого кадра — решают, нужна ли копия кадра
    clothes_veto_log: dict[str, float] = {}   # чтобы не писать одно и то же каждую секунду
    face_sim_at: dict[str, tuple] = {}        # track_id -> (похожесть лица, когда)
    clothes_name_log: dict[str, float] = {}   # чтобы не писать про одежду каждый кадр
    track_look: dict[str, tuple] = {}         # track_id -> одежда на момент подтверждения
    look_keep_log: dict[str, float] = {}
    track_best: dict[str, tuple] = {}         # track_id -> (person_id, лучшая похожесть)
    seg_state: dict[str, tuple] = {}          # track_id -> записанное состояние
    seg_pending: dict[str, tuple] = {}        # track_id -> (новое состояние, с какого момента)
    stat_par = stat_cloth_veto = stat_cloth_learn = stat_cloth_name = 0
    last_pose_t = 0.0   # monotonic-время последнего запуска каждого детектора;
    last_obj_t = 0.0    # 0.0 => на первом кадре все отрабатывают сразу
    last_weapon_t = 0.0
    calib_seen = 0      # считаем кадры только для CALIB_DIR (см. константу)
    frame_no = 0        # для «каждый N-й кадр» у yoloe

    # Кадр уходит в общий батчер и ждёт своей доли результата: кадры других камер,
    # подошедшие в это же время, поедут на GPU вместе с ним одним запуском.
    def run_pose(frame):
        return _pose_batcher.submit(frame)

    def run_obj(frame):
        return _obj_batcher.submit(frame)

    def run_weapon(frame):
        return _weapon_batcher.submit(frame)

    while cam_id not in _cams_removed:
        try:
            _, slot, w, h = filled_q.get(timeout=1)  # camera_worker кладёт номер слота
        except Exception:
            continue
        _frame_t0 = time.perf_counter()
        try:
            # Разворачиваем YV12 из общей памяти в BGR. cvtColor тут же копирует данные
            # наружу, в собственный массив кадра, поэтому слот можно освободить сразу —
            # camera_worker сможет писать в него следующий кадр, пока мы считаем модели.
            try:
                yv12 = np.ndarray((h * 3 // 2, w), dtype=np.uint8, buffer=shms[slot].buf)
                frame = cv2.cvtColor(yv12, cv2.COLOR_YUV2BGR_YV12)
                # Первые h строк слота — плоскость яркости, то есть готовый ч/б кадр.
                # Копия обязательна: слот освобождается строкой ниже и будет переписан.
                # 2 МБ memcpy (~0.2 ms) дешевле, чем cvtColor BGR2GRAY на каждом кадре,
                # и заодно снимает конвертацию со всех патчей в трекерах объектов.
                gray = yv12[:h].copy()
            finally:
                try:
                    free_q.put_nowait(slot)
                except Exception:
                    pass
            _now = time.monotonic()
            if _now - _clean_at.get(cam_id, 0.0) >= CLEAN_SNAP_SEC:
                _clean_at[cam_id] = _now
                _clean_frame[cam_id] = frame.copy()

            # Строго ДО отрисовки: модели видят именно этот кадр.
            if CALIB_DIR:
                calib_seen += 1
                if calib_seen % CALIB_EVERY == 0:
                    cv2.imwrite(f"{CALIB_DIR}/{cam_id}_{calib_seen:06d}.png", frame)

            now = time.monotonic()
            # Что включено на ЭТОЙ камере. Читаем каждый кадр, а не один раз на старте:
            # оператор переключает галочки на сайте, и это должно действовать сразу.
            on = _cam_models(next((c for c in CAMERAS if c["id"] == cam_id), {}))
            run_pose_detector = on["pose"] and now - last_pose_t >= POSE_MIN_PERIOD
            run_object_detector = on["objects"] and now - last_obj_t >= OBJ_MIN_PERIOD
            # yoloe гоняем по счётчику кадров (просили каждый 4-й), обученную модель —
            # по времени, как остальные детекторы.
            frame_no += 1
            if not on["weapon"]:
                run_weapon_detector = False
            elif WEAPON_BACKEND == "yoloe":
                run_weapon_detector = frame_no % YOLOE_EVERY_N == 0
            else:
                run_weapon_detector = now - last_weapon_t >= WEAPON_MIN_PERIOD
            if run_pose_detector:
                last_pose_t = now
                stat_pose += 1
            if run_object_detector:
                last_obj_t = now
                stat_obj += 1
            if run_weapon_detector:
                last_weapon_t = now
                stat_weapon += 1
            stat_frames += 1
            if now - stat_t0 >= 30.0:
                dt = now - stat_t0
                logger.info(
                    f"[{cam_id}] achieved: frames {stat_frames/dt:.1f}/s | "
                    f"pose {stat_pose/dt:.2f} Hz (target {POSE_TARGET_HZ}) | "
                    f"objects {stat_obj/dt:.2f} Hz (target {OBJ_TARGET_HZ}) | "
                    f"weapon {stat_weapon/dt:.2f} Hz (target {WEAPON_TARGET_HZ}) | "
                    f"face {stat_face_tries/dt:.2f} tries/s, {stat_face_hits} named, "
                    f"{stat_face_new} enrolled, {stat_face_drop} dropped, "
                    f"{stat_face_yield} yielded, {stat_face_keep} kept | "
                    f"clothes {stat_par/dt:.2f}/s, "
                    f"{stat_cloth_learn} sightings, {stat_cloth_veto} vetoed, "
                    f"{stat_cloth_name} named by clothes")
                # Куда уходит секунда работы потока камеры. Доли считаем от РЕАЛЬНОГО
                # времени: всё, что здесь не названо, — декодирование YV12, оптический
                # поток, отрисовка, JPEG и ожидание кадра.
                busy = (stat_ms_models + stat_ms_jpeg + stat_ms_track) / dt
                logger.info(
                    f"[{cam_id}] budget/s: detectors {stat_ms_models/dt:6.0f} ms | "
                    f"tracks+db {stat_ms_track/dt:6.0f} ms | jpeg {stat_ms_jpeg/dt:6.0f} ms | "
                    f"BUSY {busy:5.0f} ms ({busy/10:4.1f}%) | "
                    f"per frame {stat_ms_frame/max(stat_frames,1):5.1f} ms | "
                    f"people/frame {stat_people/max(stat_frames,1):4.1f}")
                stat_t0 = now
                stat_frames = stat_pose = stat_obj = stat_weapon = stat_people = 0
                stat_face_tries = stat_face_hits = stat_face_new = 0
                stat_face_drop = stat_face_yield = stat_par = 0
                stat_face_keep = 0
                look_keep_log = {k: v for k, v in look_keep_log.items() if now - v < 60}
                stat_cloth_veto = stat_cloth_learn = stat_cloth_name = 0
                clothes_name_log = {k: v for k, v in clothes_name_log.items()
                                    if now - v < 60}
                clothes_veto_log = {k: v for k, v in clothes_veto_log.items()
                                    if now - v < 60}
                face_sim_at = {k: v for k, v in face_sim_at.items() if now - v[1] < 60}
                stat_ms_models = stat_ms_jpeg = stat_ms_track = stat_ms_frame = 0.0
                face_try = {k: v for k, v in face_try.items() if now - v < 60}
                face_enroll = {k: v for k, v in face_enroll.items() if now - v < 60}
                face_miss = {k: v for k, v in face_miss.items() if k in face_try}

            # Три детектора — независимые вычисления на одном и том же кадре, поэтому
            # уходят в модели ПАРАЛЛЕЛЬНО и все до любой отрисовки: рисуем мы поверх
            # этого же массива, и скелет попал бы им на вход.
            _t = time.perf_counter()
            pose_future = executor.submit(run_pose, frame) if run_pose_detector else None
            obj_future = executor.submit(run_obj, frame) if run_object_detector else None
            weapon_future = executor.submit(run_weapon, frame) if run_weapon_detector else None
            pose_out = pose_future.result() if pose_future is not None else None
            obj_out = obj_future.result() if obj_future is not None else None
            weapon_out = weapon_future.result() if weapon_future is not None else None
            stat_ms_models += (time.perf_counter() - _t) * 1000

            kpts = np.empty((0, 17, 3))
            person_boxes = np.empty((0, 4))
            if pose_out is not None and len(pose_out[0]):
                kpts, person_boxes = pose_out  # [n,17,3], [n,4]
                skeleton_tracker.reset(gray, kpts, person_boxes)
            else:
                kpts, person_boxes = skeleton_tracker.update(gray)
            weapon_boxes, weapon_names, weapon_confs = _weapon_from_output(weapon_out)

            # Головы вырезаем ДО отрисовки: ниже на этот же кадр лягут белый скелет и
            # подписи, и они попали бы прямо в вырезку, которую увидит сеть. Копия нужна
            # по той же причине — срез смотрит в исходный массив.
            head_crops = []
            if FACE_ENABLED and run_pose_detector and len(person_boxes) and len(kpts):
                for i in range(min(len(person_boxes), len(kpts))):
                    c, _off = _head_crop(frame, kpts[i], person_boxes[i])
                    head_crops.append(c.copy() if c is not None else None)

            # Цвет одежды — по той же причине до отрисовки: белый скелет проходит ровно
            # по груди и бёдрам, то есть по тем самым местам, которые мы и меряем.
            # Копия кадра целиком: маска торса и полосы вдоль ног живут в координатах
            # кадра, а резать под каждого человека отдельно дороже одного memcpy.
            # Но копируем не на каждом кадре позы, а только когда хоть один замер в этом
            # кадре действительно случится: замер по человеку идёт раз в CLOTH_MIN_PERIOD,
            # а поза — шесть раз в секунду, и пять копий из шести шли в никуда. Треки
            # этого кадра ещё не известны, поэтому смотрим на треки прошлого и на
            # «людей стало больше» — новый трек всегда должен замер.
            clean_frame = None
            if CLOTH_ENABLED and on["clothes"] and run_pose_detector and len(kpts):
                due = (len(kpts) > len(prev_track_ids)
                       or any(t - par_try.get(tid, 0.0) >= CLOTH_MIN_PERIOD
                              for tid in prev_track_ids))
                if due:
                    clean_frame = frame.copy()

            if len(kpts):
                draw_white_skeleton(frame, kpts)
            hand_rects = _collect_hand_rects(kpts)

            obj_boxes = np.empty((0, 4))
            obj_cls_ids = np.empty((0,))
            confs = np.empty((0,))
            if obj_out is not None:
                obj_boxes, obj_cls_ids, confs = obj_out
            if run_object_detector:
                raw_phone_mask = obj_cls_ids == PHONE_ID
                non_phone_boxes, non_phone_cls_ids, non_phone_confs = object_box_tracker.update(
                    obj_boxes[~raw_phone_mask],
                    obj_cls_ids[~raw_phone_mask],
                    confs[~raw_phone_mask],
                    gray,
                    detector_ran=True,
                )
                if np.any(raw_phone_mask):
                    obj_boxes = np.concatenate([non_phone_boxes, obj_boxes[raw_phone_mask]], axis=0)
                    obj_cls_ids = np.concatenate([non_phone_cls_ids, obj_cls_ids[raw_phone_mask]], axis=0)
                    confs = np.concatenate([non_phone_confs, confs[raw_phone_mask]], axis=0)
                else:
                    obj_boxes, obj_cls_ids, confs = non_phone_boxes, non_phone_cls_ids, non_phone_confs
            else:
                obj_boxes, obj_cls_ids, confs = object_box_tracker.update(
                    np.empty((0, 4)),
                    np.empty((0,)),
                    np.empty((0,)),
                    gray,
                    detector_ran=False,
                )
            obj_boxes, obj_cls_ids, confs = phone_box_tracker.update(obj_boxes, obj_cls_ids, confs, hand_rects, gray)
            if len(obj_boxes):
                draw_objects(frame, obj_boxes, obj_cls_ids, confs)

            # поза + активность на каждого человека — геометрия поверх keypoints и рамок объектов
            _t = time.perf_counter()
            t = time.time()
            km_boxes = obj_boxes[np.isin(obj_cls_ids, WORK_ITEM_IDS)]
            km_memory.update(km_boxes, t)  # обновляем память ОДИН раз за кадр, не на человека

            frame_track_ids = set()
            if len(person_boxes) and len(kpts):
                chair_boxes = obj_boxes[obj_cls_ids == CHAIR_ID]
                phone_label_boxes = obj_boxes[obj_cls_ids == PHONE_ID]
                for i in range(min(len(person_boxes), len(kpts))):
                    track = track_memory.observe(cam_id, person_boxes[i], t)
                    frame_track_ids.add(track.track_id)
                    posture = classify_posture(kpts[i], person_boxes[i], chair_boxes,
                                               frame_shape=frame.shape)
                    pos, scale = person_pos_scale(kpts[i])
                    activity = classify_activity(
                        kpts[i], pos, scale, km_boxes, km_memory, t,
                        phone_wrists=_person_phone_wrists(kpts[i], phone_label_boxes),
                        working_memory=working_memory,
                    )
                    track.status = {
                        "posture": posture,
                        "activities": sorted(activity),
                        "updated_at": t,
                    }
                    # Безымянный трек опрашиваем чаще, узнанный — реже, но опрашиваем:
                    # имя должно следовать за лицом, а не залипать на рамке.
                    interval = FACE_VERIFY_SEC if track.name else FACE_RETRY_SEC
                    if (FACE_ENABLED and on["face"] and run_pose_detector
                            and t - face_try.get(track.track_id, 0.0) >= interval):
                        face_try[track.track_id] = t
                        crop = head_crops[i] if i < len(head_crops) else None
                        got = _recognise_head(crop)
                        stat_face_tries += 1
                        if got is not None:
                            pid, sim = face_store.match(got["embedding"])
                            label = face_store.label_of(pid) if pid is not None else None
                            has_name = bool(label) and not label.startswith("#")
                            # Поставить имя и удержать его — разные решения, и порог у них
                            # разный. Ставим только уверенно (FACE_NAME_MIN_SIM), а держим,
                            # пока лицо продолжает узнаваться КАК ТОТ ЖЕ человек: у одного
                            # и того же лица похожесть гуляет от кадра к кадру, и по
                            # единому порогу верное имя мигало бы (nina 0.565 -> 0.468).
                            confirmed = (pid is not None and pid == track.person_id
                                         and has_name)
                            named = confirmed or (has_name and sim >= FACE_NAME_MIN_SIM)

                            # Одежда — независимая от лица улика, и она имеет право
                            # отменить имя: чужого человека можно принять за своего по
                            # лицу, но не по лицу И одежде сразу.
                            if named:
                                ok_cloth, why = _clothes_verdict(pid, cam_id,
                                                                 track.track_id, sim)
                                if not ok_cloth:
                                    if track.name:
                                        _release_name(track.person_id, cam_id,
                                                      track.track_id)
                                        if EVENTS_ENABLED:
                                            try:
                                                store.release_track(cam_id,
                                                                    track.track_id)
                                            except Exception as e:
                                                logger.error(f"[events] release: {e}")
                                        track_best.pop(track.track_id, None)
                                        track.name = None
                                        track.person_id = None
                                    if t - clothes_veto_log.get(track.track_id, 0.0) > 5:
                                        clothes_veto_log[track.track_id] = t
                                        logger.info(
                                            f"[cloth] {cam_id} {track.track_id} "
                                            f"NOT {label} — {why} (face {sim:.3f})")
                                    stat_cloth_veto += 1
                                    named = False

                            if named and not _claim_name(pid, cam_id, track.track_id,
                                                        sim, t):
                                # Этого человека прямо сейчас лучше видит другая камера.
                                # Подписываем там, а здесь имя не показываем.
                                if track.name:
                                    logger.info(
                                        f"[face] {cam_id} {track.track_id} yields "
                                        f"{track.name} to a better view ({sim:.3f})")
                                    stat_face_yield += 1
                                    # Отдали имя — отдаём и незакрытые отрезки, иначе
                                    # они шли бы человеку поверх отрезков победителя.
                                    if EVENTS_ENABLED:
                                        try:
                                            store.release_track(cam_id, track.track_id)
                                        except Exception as e:
                                            logger.error(f"[events] release: {e}")
                                    track_best.pop(track.track_id, None)
                                    track.name = None
                                    track.person_id = None
                                named = False

                            if named:
                                face_miss.pop(track.track_id, None)
                                if track.name != label:
                                    if track.name:
                                        logger.info(
                                            f"[face] {cam_id} {track.track_id} "
                                            f"{track.name} -> {label} ({sim:.3f})")
                                    else:
                                        logger.info(
                                            f"[face] {cam_id} {track.track_id} -> {label} "
                                            f"(similarity {sim:.3f})")
                                    stat_face_hits += 1
                                track.person_id = pid
                                track.name = label
                                _note_branch(pid, cam_id, t)

                                # За кем числятся дела этого трека. Имя на треке может
                                # смениться: сперва лицо совпало слабо и не с тем, через
                                # секунду — уверенно и с настоящим. Держим лучшее за всё
                                # время трека и, когда оно улучшилось, переписываем на
                                # него ВСЕ отрезки трека, включая уже закрытые.
                                best = track_best.get(track.track_id)
                                if best is None or sim > best[1]:
                                    track_best[track.track_id] = (pid, sim)
                                    if EVENTS_ENABLED and (best is None or best[0] != pid):
                                        try:
                                            moved = store.reattribute_track(
                                                cam_id, track.track_id, pid)
                                            store.record_identity_event(
                                                pid, cam_id, track.track_id, sim)
                                            if best is not None:
                                                logger.info(
                                                    f"[events] {cam_id} {track.track_id} "
                                                    f"re-attributed to {label} "
                                                    f"({best[1]:.3f} -> {sim:.3f}), "
                                                    f"{moved} segments moved")
                                        except Exception as e:
                                            logger.error(f"[events] {e}")
                                # Насколько уверенно узнали — нужно ниже, чтобы решить,
                                # можно ли по этому кадру запоминать одежду.
                                face_sim_at[track.track_id] = (sim, t)
                                # Во что он был одет в момент, когда имя подтвердилось.
                                # Дальше по этому снимку решаем, тот же это человек или
                                # трек перескочил на соседа.
                                with _appearance_lock:
                                    arec = _appearance.get((cam_id, track.track_id))
                                    if arec and arec.get("upper_now") != "unknown":
                                        track_look[track.track_id] = (arec["upper_now"],
                                                                      arec.get("lower_now"))
                                        # Уверенное узнавание — ставим метку «этот
                                        # человек, в этой одежде, вот здесь». По ней
                                        # имя вернётся, если трек порвётся.
                                        if sim >= REF_MIN_SIM:
                                            _remember_ref(cam_id, pid, person_boxes[i],
                                                          arec["upper_now"],
                                                          arec.get("lower_now"), sim, t)
                            elif track.name:
                                # Лицо видно, но это НЕ тот, кем трек подписан. Молча оставить
                                # имя нельзя: до этого снять его было нечем, и чужое имя висело
                                # над головой до конца трека.
                                #
                                # Но перед тем как снимать, спрашиваем одежду: если человек
                                # одет ровно так же, как в момент, когда имя ставили, значит
                                # перед камерой он же, а лицо просто не прочиталось — отвернулся,
                                # моргнул, ушёл в тень. Снимать имя тут не за что.
                                if _look_unchanged(cam_id, track.track_id, track_look):
                                    face_miss.pop(track.track_id, None)
                                    if t - look_keep_log.get(track.track_id, 0.0) > 15:
                                        look_keep_log[track.track_id] = t
                                        logger.info(
                                            f"[face] {cam_id} {track.track_id} keeps "
                                            f"{track.name}: clothing unchanged "
                                            f"(face said {label or '-'} {sim:.3f})")
                                        stat_face_keep += 1
                                    miss = 0
                                else:
                                    miss = face_miss.get(track.track_id, 0) + 1
                                    face_miss[track.track_id] = miss
                                if miss >= FACE_DROP_AFTER_MISS:
                                    logger.info(
                                        f"[face] {cam_id} {track.track_id} dropped "
                                        f"{track.name} after {miss} misses "
                                        f"(best {label or '-'} {sim:.3f})")
                                    _release_name(track.person_id, cam_id, track.track_id)
                                    if EVENTS_ENABLED:
                                        try:
                                            store.release_track(cam_id, track.track_id)
                                        except Exception as e:
                                            logger.error(f"[events] release: {e}")
                                    track_best.pop(track.track_id, None)
                                    track.name = None
                                    track.person_id = None
                                    face_miss.pop(track.track_id, None)
                                    track_look.pop(track.track_id, None)
                                    stat_face_drop += 1

                            if not track.name and (got["det_score"] >= FACE_ENROLL_MIN_DET
                                    and t - face_enroll.get(track.track_id, 0.0) >= FACE_ENROLL_SEC):
                                # Не узнали (или человек в галерее ещё без имени) —
                                # кладём снимок, чтобы оператор мог его назвать.
                                face_enroll[track.track_id] = t
                                ok, buf = cv2.imencode(".jpg", crop,
                                                       [cv2.IMWRITE_JPEG_QUALITY, 92])
                                r = face_store.add_sighting(
                                    got["embedding"], det_score=got["det_score"],
                                    camera_id=cam_id,
                                    crop=buf.tobytes() if ok else None)
                                stat_face_new += 1
                                # Безымянных отмечаем тоже — общая база держится именно
                                # на них: имени нет, а «в каком филиале видели» есть.
                                _note_branch(r["person_id"], cam_id, t)
                                logger.info(
                                    f"[face] {cam_id} {track.track_id} "
                                    f"{'NEW' if r['is_new'] else 'added to'} {r['label']} "
                                    f"(similarity {r['similarity']:.3f}, "
                                    f"det {got['det_score']:.2f})")

                    # Имя рисуем, только если оно закреплено ЗА ЭТИМ треком. Решать это
                    # в момент распознавания мало: свою очередь трек ждёт до секунды, и
                    # проигравший всё это время продолжал бы подписываться чужим именем —
                    # в кадре получалось два «rafo» сразу. Здесь же имя пропадает у
                    # проигравшего в тот же кадр, а остаётся у того, кто видит лицо лучше.
                    # Имя, закреплённое ЗА ЭТИМ треком, — то же, что попадёт в кадр.
                    # Брать сырой track.name нельзя: проигравший спор трек ещё держит
                    # имя до своей проверки, и поиск по одежде показывал бы «karen на
                    # cam_03», когда в кадре karen подписан на cam_04.
                    shown_name = None
                    by_clothes = False
                    if track.name and _holds_name(track.person_id, cam_id,
                                                  track.track_id):
                        shown_name = track.name

                    # Лица не видно — спрашиваем одежду. Человек отвернулся к камере
                    # спиной, но переодеться за минуту он не мог, значит это он же.
                    # Сначала свежая метка на этой камере (одежда И место), потом уже
                    # дневная запись: метка точнее, потому что цвет в ней измерен здесь
                    # же и только что, а не на соседней камере при другом свете.
                    # Одежда возвращает имя тоже ЧЕРЕЗ ЗАКРЕПЛЕНИЕ, а не в обход. Иначе
                    # получалось два «rafo»: одному треку имя дало лицо, второму — тот
                    # же цвет одежды, и оба его рисовали. Похожесть здесь заведомо
                    # низкая (CLOTH_NAME_SIM): человека, чьё лицо видно, одежда с
                    # чужого трека не перебьёт, а вот на свободное имя встанет.
                    for finder, why in ((_name_from_ref, "clothing + position"),
                                        (_name_from_clothes, "clothing (face not visible)")):
                        if shown_name is not None:
                            break
                        cpid, clabel = (finder(cam_id, track.track_id, person_boxes[i], t)
                                        if finder is _name_from_ref
                                        else finder(cam_id, track.track_id, t))
                        if not clabel:
                            continue
                        if not _claim_name(cpid, cam_id, track.track_id,
                                           CLOTH_NAME_SIM, t):
                            continue          # это имя сейчас держит тот, кто видит лицо
                        shown_name, by_clothes = clabel, True
                        track.person_id, track.name = cpid, clabel
                        if t - clothes_name_log.get(track.track_id, 0.0) > 10:
                            clothes_name_log[track.track_id] = t
                            logger.info(f"[cloth] {cam_id} {track.track_id} -> {clabel} "
                                        f"by {why}")
                            stat_cloth_name += 1

                    if (clean_frame is not None and i < len(kpts)
                            and t - par_try.get(track.track_id, 0.0) >= CLOTH_MIN_PERIOD):
                        par_try[track.track_id] = t
                        seen = _cloth_measure(clean_frame, kpts[i])
                        if seen is not None:
                            _appearance_update(cam_id, track.track_id, shown_name, seen)
                            stat_par += 1
                            # Засечку делаем только с очень удачного кадра: точки тела
                            # уверенны разом и лицо узнано уверенно — то есть человек
                            # стоит прямо, в полный рост. По такому кадру и рубашка, и
                            # штаны видны целиком. Но одной засечки мало: цвет признаём
                            # за человеком, лишь когда несколько подряд сошлись.
                            fs = face_sim_at.get(track.track_id)
                            kp_min = float(np.min(kpts[i][list(CLOTH_LEARN_KP_SET), 2]))
                            if CLOTH_PROBE and shown_name:
                                fs_txt = f"{fs[0]:.2f}" if fs else "none"
                                allk = ",".join(f"{c:.2f}" for c in kpts[i][:, 2])
                                logger.info(
                                    f"[gate] {cam_id} {shown_name} face={fs_txt} "
                                    f"up={seen['upper']} lo={seen['lower']} kp={allk}")
                            if (shown_name and track.person_id is not None
                                    and fs is not None
                                    and fs[0] >= CLOTH_LEARN_FACE_SIM
                                    and t - fs[1] <= CLOTH_LEARN_MAX_AGE
                                    and kp_min >= CLOTH_LEARN_KP_CONF
                                    and seen["upper"] != "unknown"
                                    and seen["lower"] != "unknown"):
                                if _clothes_confirm(track.person_id, shown_name, cam_id,
                                                    seen, fs[0], kp_min, t):
                                    stat_cloth_learn += 1

                    if CLOTH_PROBE and i < len(kpts):
                        k = kpts[i]
                        logger.info(
                            "[kp] {} posture={} hip={:.2f}/{:.2f} knee={:.2f}/{:.2f} "
                            "ankle={:.2f}/{:.2f} sh={:.2f}/{:.2f}".format(
                                cam_id, posture,
                                k[KP_L_HIP][2], k[KP_R_HIP][2],
                                k[KP_L_KNEE][2], k[KP_R_KNEE][2],
                                k[KP_L_ANK][2], k[KP_R_ANK][2],
                                k[KP_L_SH][2], k[KP_R_SH][2]))

                    # Цвета берём из той же сглаженной записи, по которой работает и
                    # проверка имени: в кадре должно быть видно ровно то, чем система
                    # руководствуется, иначе спорить с ней не о чем.
                    with _appearance_lock:
                        arec = _appearance.get((cam_id, track.track_id))
                        # Показываем измеренный цвет, а не эталон разряда: на экране
                        # квадрат должен совпадать с тканью.
                        cloth_up = (arec.get("upper_bgr") or arec.get("upper_now")) if arec else None
                        cloth_lo = (arec.get("lower_bgr") or arec.get("lower_now")) if arec else None
                    _draw_track_label(frame, person_boxes[i],
                                      shown_name or track.external_id,
                                      cloth_up, cloth_lo)

                    if EVENTS_ENABLED:
                        _mark_seen(cam_id, track.track_id, t)
                        _record_state(cam_id, track, posture, activity, t,
                                      seg_state, seg_pending)
                    draw_activity_label(frame, person_boxes[i], posture, activity)

                    # Точка на плане этажа. Считается только у откалиброванной камеры;
                    # зону пишем на человека и только с камеры, держащей его имя —
                    # по той же причине, что и остальные отрезки.
                    if floor_map.calibrated(cam_id):
                        fx, fy = foot_point(kpts[i], person_boxes[i], frame.shape)
                        uv = floor_map.project(cam_id, fx, fy)
                        if uv is not None:
                            owner = (track.person_id if _holds_name(
                                track.person_id, cam_id, track.track_id) else None)
                            floor_map.update_live(cam_id, track.track_id, uv,
                                                  shown_name, owner, t)
                            if EVENTS_ENABLED and owner is not None:
                                try:
                                    _br = _camera_branch(cam_id)
                                    floor_map.observe_zone(
                                        store, _br, owner, uv, t, EVENT_DEBOUNCE_SEC)
                                    # След пишет тоже только держатель имени: иначе
                                    # одного человека вели бы две камеры сразу и
                                    # точка на плане дёргалась бы между ними.
                                    floor_map.record_point(store, _br, owner, uv, t)
                                except Exception as e:      # noqa: BLE001
                                    logger.error(f"[map] zone write failed: {e}")

            # Отрезки тех, кто ушёл из кадра, закрываем не сразу, а раз в несколько
            # секунд: закрывать на каждом кадре — это снова запись в базу на каждом
            # кадре, ровно то, из-за чего аналитику когда-то и отключили.
            if EVENTS_ENABLED:
                seen = _sweep_due(cam_id, t)
                if seen is not None:
                    try:
                        store.close_stale_tracks(cam_id, seen, ts=t)
                        floor_map.sweep(store, t, EVENT_TRACK_GRACE, EVENT_SWEEP_SEC)
                    except Exception as e:
                        logger.error(f"[events] sweep failed: {e}")
                    for gone in [k for k in seg_state if k not in seen]:
                        seg_state.pop(gone, None)
                        seg_pending.pop(gone, None)
                        track_best.pop(gone, None)

            stat_ms_track += (time.perf_counter() - _t) * 1000
            stat_people += len(person_boxes)
            prev_track_ids = frame_track_ids

            # Красное рисуем последним, поверх скелета, объектов и подписей: если оружие
            # в кадре есть, оно не должно оказаться под чужой рамкой.
            if weapon_boxes:
                _draw_weapons(frame, weapon_boxes, weapon_names, weapon_confs)

            _latest_frame[cam_id] = frame

            # JPEG нужен только живому <img> в браузере. Кодировать его в пустоту, когда
            # вкладку никто не открыл, — чистая потеря (~0.08 ядра на камеру, а камер
            # предполагается много). Снимки лиц берутся не отсюда, а из _latest_frame.
            if _stream_viewers.get(cam_id, 0) > 0:
                _t = time.perf_counter()
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
                if ok:
                    _latest_jpeg[cam_id] = buf.tobytes()
                stat_ms_jpeg += (time.perf_counter() - _t) * 1000
            stat_ms_frame += (time.perf_counter() - _frame_t0) * 1000
        except Exception as e:
            logger.error(f"[{cam_id}#{worker_idx}] inference error: {e}")


def _drain_status(status_queue: mp.Queue):
    while True:
        try:
            cam_id, ok, err = status_queue.get(timeout=1)
            _connected[cam_id] = ok
            _last_error[cam_id] = err
        except Exception:
            continue


_stop_event = None          # mp.Event, создаётся в main до запуска процессов
_procs_info: list[dict] = []
# Камеры, которые оператор удалил. Потоки обработки смотрят сюда и выходят сами:
# у каждой камеры их WORKERS_PER_CAMERA штук, и убивать их нечем — только попросить.
_cams_removed: set = set()


def stop_camera(cam_id: str, timeout: float = 10.0) -> str:
    """Погасить одну камеру, не трогая остальные.

    Обязательно через её собственный stop-event, а не terminate(): убитый процесс
    оставляет на камере открытый логин в SDK, и именно из-за этого раньше приходилось
    перезагружать машину.
    """
    info = next((i for i in _procs_info if i["cam_id"] == cam_id), None)
    if info is None:
        return "not running"

    _cams_removed.add(cam_id)          # потоки обработки увидят и выйдут
    ev = info.get("stop")
    if ev is not None:
        ev.set()
    info["proc"].join(timeout=timeout)
    how = "closed cleanly"
    if info["proc"].is_alive():
        info["proc"].terminate()
        how = "had to terminate"

    # Слоты общей памяти держал этот процесс — освобождаем, иначе они останутся
    # висеть до конца работы программы.
    for s in info["shms"]:
        try:
            s.close()
            s.unlink()
        except Exception:
            pass

    _procs_info.remove(info)
    _viewer_flags.pop(cam_id, None)
    _connected.pop(cam_id, None)
    _latest_frame.pop(cam_id, None)
    _clean_frame.pop(cam_id, None)
    _clean_at.pop(cam_id, None)
    _latest_jpeg.pop(cam_id, None)
    logger.info(f"[camera] {cam_id} stopped ({how})")
    return how


def shutdown_cameras(timeout: float = 12.0) -> str:
    """Попросить процессы камер закрыть сессии в SDK и дождаться их.

    Ради этого всё и затевалось: убитый снаружи процесс оставляет на камере открытый
    логин, и после нескольких таких случаев камера начинает ронять любого, кто к ней
    подключается, — лечится только перезагрузкой машины. Здесь воркеры выходят сами,
    вызвав StopRealPlay и Logout.
    """
    if _stop_event is None:
        return "no camera processes"
    _stop_event.set()
    # У каждой камеры свой event — общий она уже не слушает.
    for info in _procs_info:
        if info.get("stop") is not None:
            info["stop"].set()
    deadline = time.monotonic() + timeout
    for info in _procs_info:
        left = max(0.5, deadline - time.monotonic())
        info["proc"].join(timeout=left)

    clean = [i["cam_id"] for i in _procs_info if not i["proc"].is_alive()]
    stuck = [i["cam_id"] for i in _procs_info if i["proc"].is_alive()]
    for info in _procs_info:
        if info["proc"].is_alive():
            info["proc"].terminate()      # не вышел сам — иначе никак
    # Закрываем отрезки ЗДЕСЬ же. Треки не пропали из кадра — исчезли те, кто за ними
    # следил, и развёртка, которая обычно закрывает отрезки, больше не сработает.
    # Оставленный открытым отрезок в отчёте считается до конца суток: человек ушёл
    # в шесть вечера, программу выключили — и назавтра у него в «работал» стоят часы,
    # которых не было.
    if EVENTS_ENABLED:
        try:
            n = store.close_open_segments()
            if n:
                logger.info(f"[shutdown] closed {n} open activity segments")
        except Exception as e:                                   # noqa: BLE001
            logger.error(f"[shutdown] closing segments: {e}")
        # Последние секунды следа ещё в буфере — без этого при каждой остановке
        # у дня пропадал бы хвост.
        try:
            floor_map.flush_trail(store)
        except Exception as e:                                   # noqa: BLE001
            logger.error(f"[shutdown] flushing trail: {e}")

    msg = f"closed cleanly: {clean or '-'}; had to terminate: {stuck or '-'}"
    logger.info(f"[shutdown] {msg}")
    return msg


# Какие модели вообще можно включать на камере. Не всякой камере нужно всё: на дальнем
# плане лица не читаются, а оружие имеет смысл искать не везде — а каждая включённая
# модель это доля секунды GPU на кадр.
MODEL_KEYS = ("pose", "objects", "weapon", "face", "clothes")


def _cam_models(cam: dict) -> dict:
    """Что включено на этой камере. Камера без списка получает всё — так вели себя
    все камеры до появления настройки, и менять их поведение молча нельзя."""
    chosen = cam.get("models")
    if not isinstance(chosen, dict):
        return {k: True for k in MODEL_KEYS}
    return {k: bool(chosen.get(k, True)) for k in MODEL_KEYS}


_status_queue = None      # заполняется в main, нужен при добавлении камеры на ходу


def start_camera(cam: dict) -> dict:
    """Поднять одну камеру: общая память, процесс чтения и потоки обработки.

    Вынесено из main, чтобы камеру можно было добавить, не перезапуская программу:
    оператору незачем править cameras.json и терять все текущие треки ради этого.
    """
    cam_id = cam["id"]

    # Кольцо слотов в общей памяти на камеру. free_q — какие слоты свободны для
    # записи, filled_q — какие заполнены и ждут обработки. Через очереди ходят
    # только номера слотов, сами кадры между процессами не копируются вообще.
    # Блоки создаёт и держит открытыми этот процесс: Windows освобождает общую
    # память, когда закрыт последний хэндл.
    shms = [shared_memory.SharedMemory(create=True, size=SHM_SLOT_BYTES)
            for _ in range(SHM_SLOTS)]
    # Обе очереди ограничены числом слотов: даже если после аварийного перезапуска
    # воркера в free_q попадёт дубликат, кольцо не сможет раздать больше буферов,
    # чем существует.
    free_q: mp.Queue = mp.Queue(maxsize=SHM_SLOTS)
    filled_q: mp.Queue = mp.Queue(maxsize=SHM_SLOTS)
    for i in range(SHM_SLOTS):
        free_q.put(i)

    # Флаг «на эту камеру кто-то смотрит» — создаём ДО старта процесса, иначе воркеру
    # его уже не передать: общая память фиксируется в момент fork/spawn.
    viewer_flag = mp.Value("i", 0)
    _viewer_flags[cam_id] = viewer_flag

    # Свой stop-event на камеру: общий годится для выключения всей программы, но
    # удалить одну камеру им нельзя — погасли бы все.
    stop = mp.Event()
    _cams_removed.discard(cam_id)
    p = mp.Process(target=run_camera_worker,
                   args=(cam, [s.name for s in shms], free_q, filled_q, _status_queue,
                         viewer_flag, stop),
                   daemon=True)
    p.start()
    info = {"cam_id": cam_id, "cam": cam, "proc": p, "shms": shms, "stop": stop,
            "free_q": free_q, "filled_q": filled_q, "viewer_flag": viewer_flag}
    _procs_info.append(info)
    logger.info(f"Started worker process for {cam_id} (pid={p.pid})")

    for w in range(WORKERS_PER_CAMERA):
        threading.Thread(target=_process_camera,
                         args=(cam_id, shms, free_q, filled_q, w), daemon=True).start()
    return info


def _watchdog(procs_info: list[dict], status_queue: mp.Queue):
    """Hikvision SDK иногда роняет процесс камеры нативным крашем (access violation
    в python311.dll — не питоновское исключение, поэтому без трейсбека и без
    возможности перехватить try/except). Вместо борьбы с самим крашем — следим за
    процессами и перезапускаем упавшие автоматически, чтобы видео само возвращалось.

    Перезапускаем строго ПО ОДНОМУ и с той же паузой, что и при старте. Раньше все
    упавшие поднимались в одном проходе, то есть логинились в SDK одновременно — ровно
    то, чего избегает запуск. Одна общая помеха роняла две камеры, watchdog поднимал их
    вместе, одновременный логин ронял их снова, и это уже не прекращалось: в логе
    четыре смерти в пределах 25 мс, и так по кругу каждые шесть секунд."""
    while True:
        time.sleep(3)
        for info in list(procs_info):
            p = info["proc"]
            if p.is_alive():
                continue
            cam_id = info["cam_id"]
            if cam_id in _cams_removed:
                continue          # камеру убрали намеренно, поднимать не надо
            logger.warning(f"[{cam_id}] worker process died (exitcode={p.exitcode}), restarting...")
            _connected[cam_id] = False

            # Упавший воркер мог держать слот, взятый из free_q, — он бы пропал
            # навсегда, и кольцо постепенно осталось бы без буферов. Пересобираем
            # его целиком: чистим обе очереди и заново объявляем все слоты свободными.
            for q in (info["free_q"], info["filled_q"]):
                while True:
                    try:
                        q.get_nowait()
                    except Exception:
                        break
            for i in range(SHM_SLOTS):
                try:
                    info["free_q"].put_nowait(i)
                except Exception:
                    pass

            if _stop_event.is_set():
                return          # выключаемся — поднимать камеру обратно незачем
            new_p = mp.Process(target=run_camera_worker,
                               args=(info["cam"], [s.name for s in info["shms"]],
                                     info["free_q"], info["filled_q"], status_queue,
                                     info.get("viewer_flag"), _stop_event),
                               daemon=True)
            new_p.start()
            info["proc"] = new_p
            logger.info(f"[{cam_id}] worker process restarted (pid={new_p.pid})")
            # Одна камера за проход: дать ей войти в SDK в одиночку, остальные
            # подождут следующего круга.
            time.sleep(CAMERA_LOGIN_GAP)
            break


def _sync_viewer_flag(cam_id: str):
    """Передаёт процессу камеры, смотрит ли кто-нибудь её стрим. Вызывать под
    _viewers_lock — счётчик и флаг должны меняться вместе."""
    flag = _viewer_flags.get(cam_id)
    if flag is not None:
        flag.value = 1 if _stream_viewers.get(cam_id, 0) > 0 else 0


@app.get("/stream/{cam_id}")
def stream(cam_id: str):
    # Генератор ОБЯЗАН быть async. Синхронный отдаётся Starlette в пул потоков (по умолчанию
    # 40 на весь процесс), и поток держится всё время, пока открыт стрим. Хуже того, обрыв
    # соединения синхронный генератор замечает только в момент yield — а yield здесь бывает
    # лишь когда появился НОВЫЙ кадр. У камеры, которая не отдаёт кадров (не отвечает по сети,
    # выключена, ошибка логина), _latest_jpeg не заполняется никогда, генератор навсегда
    # остаётся в sleep, и поток не возвращается в пул. Каждое открытие страницы подвешивало
    # ещё один; на 40 пул кончался, и переставали отвечать ВСЕ ручки, включая /api/status,
    # хотя обработка камер продолжала идти в своих потоках.
    # async-версия не занимает поток вообще, а разрыв соединения Starlette отслеживает сам
    # и закрывает генератор — finally отрабатывает даже если ни одного кадра так и не было.
    async def generate():
        boundary = b"--frame\r\n"
        last = None
        # Пока счётчик больше нуля, _process_camera кодирует JPEG для этой камеры.
        with _viewers_lock:
            _stream_viewers[cam_id] = _stream_viewers.get(cam_id, 0) + 1
            _sync_viewer_flag(cam_id)
        try:
            while True:
                jpeg = _latest_jpeg.get(cam_id)
                if jpeg and jpeg is not last:
                    last = jpeg
                    yield (boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.03)
        finally:
            with _viewers_lock:
                _stream_viewers[cam_id] = max(0, _stream_viewers.get(cam_id, 0) - 1)
                _sync_viewer_flag(cam_id)
            if _stream_viewers.get(cam_id, 0) == 0:
                _latest_jpeg.pop(cam_id, None)  # чтобы новый зритель не увидел старый кадр

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
def index():
    """Выбор филиала — вход в систему.

    Раньше здесь была ещё одна копия страницы камер. Теперь первый вопрос другой:
    с каким объектом работаем. Внутри филиала всё то же самое, что и было.
    """
    return _page("Detect — branches", _nav("branches"), _BRANCH_BODY, _BRANCH_JS)


_BRANCH_BODY = """
<div class="home">
  <!-- Слои сцены, снизу вверх: свечение -> радар -> сетка пола -> строка развёртки
       -> зерно -> виньетка. Ни одной картинки и ни одной библиотеки: система живёт
       в локальной сети, где интернета может не быть. -->
  <div class="fx" aria-hidden="true">
    <span class="fx-aurora"></span>
    <span class="fx-radar"></span>
    <span class="fx-floor"></span>
    <span class="fx-grain"></span>
    <span class="fx-vign"></span>
  </div>

  <!-- Ни заголовка, ни подписи: единственное, что здесь происходит, — выбор объекта,
       и объекты нарисованы прямо под курсором. Их состояние карточка показывает
       сама, поэтому строка «1 site · 4 cameras · all live» только повторяла бы её. -->
  <div id="add-branch" class="panel home-form" hidden>
    <div class="find">
      <input id="b-name" placeholder="Site name" style="width:220px;">
      <input id="b-note" placeholder="address or note (optional)" style="width:280px;">
      <button onclick="addBranch()" style="width:auto;">Create</button>
      <button onclick="showAdd()" style="width:auto;">Cancel</button>
      <span id="b-msg" class="muted"></span>
    </div>
  </div>

  <div class="muted home-orphans" id="home-orphans"></div>
  <div id="branch-list" class="home-grid"></div>
</div>
"""

_BRANCH_JS = """
function showAdd() {
  const p = document.getElementById('add-branch');
  p.hidden = !p.hidden;
  if (!p.hidden) document.getElementById('b-name').focus();
}

const hesc = v => String(v === undefined || v === null ? '' : v)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// Полоски связи: сколько камер объекта сейчас на связи. Число рядом всё равно есть,
// полоски — чтобы состояние читалось раньше, чем прочитан текст.
function bars(live, total) {
  const n = Math.max(total, 1);
  let out = '';
  for (let i = 0; i < n; i++) {
    out += `<i class="${i < live ? 'on' : ''}" style="--i:${i}"></i>`;
  }
  return `<span class="sig">${out}</span>`;
}

async function loadBranches() {
  const b = await fetch('/api/branches').then(r => r.json()).catch(() => null);
  if (!b) return;

  const box = document.getElementById('branch-list');

  // Опрос идёт раз в пять секунд, а меняется тут почти никогда. Перерисовывать
  // разметку каждый раз значит запускать заново анимацию появления — карточки
  // мигали, будто страница перезагружается. Поэтому список строится, только когда
  // изменился САМ НАБОР объектов; всё остальное правится на месте.
  const key = b.branches.map(x => x.id + ':' + hesc(x.name) + ':' + hesc(x.note)
                                  + ':' + x.cameras).join('|');
  if (key !== box.dataset.key) {
    box.dataset.key = key;
    box.innerHTML = b.branches.map((x, n) => `
      <article class="site" data-tilt data-id="${x.id}" style="--n:${n}">
        <a href="/live?branch=${x.id}" class="site-face">
          <span class="site-sheen" aria-hidden="true"></span>
          <span class="site-top">
            <span class="site-dot"></span>
            <span class="site-state"></span>
          </span>
          <span class="site-name">${hesc(x.name)}</span>
          <span class="site-note">${hesc(x.note) || '&nbsp;'}</span>
          <span class="site-foot">
            <span class="sig"></span>
            <span class="site-count"></span>
          </span>
          <span class="site-enter">Open →</span>
        </a>
        <button class="site-del" title="Remove this site"
                onclick="delBranch('${x.id}', '${hesc(x.name)}')">✕</button>
      </article>`).join('')
      + `<button class="site site-new" onclick="showAdd()"
                 style="--n:${b.branches.length}">
           <span class="site-plus">+</span>
           <span class="site-newtext">Add site</span>
         </button>`;
    tilt();
  }

  // Живое состояние: пишем, только если оно действительно другое. Присваивание
  // того же самого текста браузер бы пережил, а вот перезапись .sig подняла бы
  // анимацию полосок на каждом опросе.
  for (const x of b.branches) {
    const card = box.querySelector(`.site[data-id="${x.id}"]`);
    if (!card) continue;
    const dot = card.querySelector('.site-dot');
    const cls = 'site-dot ' + (x.live ? 'on' : 'off');
    if (dot.className !== cls) dot.className = cls;
    const state = x.live ? 'ONLINE' : 'OFFLINE';
    const st = card.querySelector('.site-state');
    if (st.textContent !== state) st.textContent = state;
    const cnt = `<b>${x.live}</b>/${x.cameras} cameras`;
    const ce = card.querySelector('.site-count');
    if (ce.innerHTML !== cnt) ce.innerHTML = cnt;
    const sg = card.querySelector('.sig');
    const sig = bars(x.live, x.cameras);
    if (sg.outerHTML !== sig) sg.outerHTML = sig;
  }

  const orph = document.getElementById('home-orphans');
  const want = (!b.branches.length && b.unassigned.length)
    ? `${b.unassigned.length} camera(s) are running and waiting to be put into a site.`
    : '';
  if (orph.textContent !== want) orph.textContent = want;
}

// Наклон карточки вслед за курсором. Это не украшение ради украшения: карточка —
// вход на объект, и физический отклик показывает, какая из них сейчас под рукой,
// раньше, чем сработает наведение. Считаем на pointermove без перерисовки разметки,
// поэтому браузеру остаётся только композитинг.
function tilt() {
  const soft = matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.querySelectorAll('[data-tilt]').forEach(card => {
    if (card._tilted) return;
    card._tilted = true;
    if (soft) return;                       // кому-то от такого движения плохо
    card.addEventListener('pointermove', ev => {
      const r = card.getBoundingClientRect();
      const px = (ev.clientX - r.left) / r.width - 0.5;
      const py = (ev.clientY - r.top) / r.height - 0.5;
      card.style.setProperty('--rx', (-py * 9).toFixed(2) + 'deg');
      card.style.setProperty('--ry', (px * 12).toFixed(2) + 'deg');
      card.style.setProperty('--mx', ((px + 0.5) * 100).toFixed(1) + '%');
      card.style.setProperty('--my', ((py + 0.5) * 100).toFixed(1) + '%');
    });
    card.addEventListener('pointerleave', () => {
      card.style.setProperty('--rx', '0deg');
      card.style.setProperty('--ry', '0deg');
    });
  });
}

async function addBranch() {
  const msg = document.getElementById('b-msg');
  const name = document.getElementById('b-name').value.trim();
  const note = document.getElementById('b-note').value.trim();
  if (!name) { msg.innerHTML = '<span class="bad">enter a name</span>'; return; }
  const r = await fetch('/api/branches', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, note}),
  }).then(r => r.json()).catch(() => ({ok: false, error: 'request failed'}));
  if (!r.ok) { msg.innerHTML = `<span class="bad">${r.error}</span>`; return; }
  document.getElementById('b-name').value = '';
  document.getElementById('b-note').value = '';
  msg.innerHTML = '<span class="ok">created</span>';
  showAdd();
  loadBranches();
}

async function delBranch(id, name) {
  if (!confirm(`Remove branch "${name}"?\\n\\n`
             + `Its cameras are logged out and removed with it — a branch is the site, `
             + `and the cameras belong to it. You will add cameras again for the next `
             + `branch.\\n\\n`
             + `Faces, names and recorded activity are not touched.`)) return;
  const r = await fetch('/api/branches/' + id, {method: 'DELETE'})
    .then(r => r.json()).catch(() => ({ok: false}));
  if (!r.ok) { alert(r.error || 'failed'); return; }
  loadBranches();
}

setInterval(loadBranches, 5000);
loadBranches();
"""


# Один стиль на обе страницы: камеры и лица разъехались по разным адресам, но выглядеть
# должны как один сайт. Обычная строка, а не f-строка, — поэтому фигурные скобки одинарные.
_CSS = """
/* Ширина считается вместе с рамкой и отступами. Без этого input с width:100% внутри
   сетки шире своей ячейки ровно на padding и border — соседние поля наезжали друг на
   друга рамками и в карточке человека, и в форме добавления камеры. */
*, *::before, *::after { box-sizing:border-box; }
/* hidden должен прятать. Браузер задаёт для него display:none самым слабым правилом,
   и любой наш класс с display его перебивает — так «3/7/30» оставались на экране на
   вкладке дня, хотя элементу выставлен hidden. */
[hidden] { display:none !important; }
/* Палитра одним местом. Раньше каждый цвет был вписан в правило руками, и «сделать
   чуть холоднее» означало пройти по сотне мест. Смысл закреплён за именем, а не за
   числом: known — зелёный, unknown — янтарный, тревога — красный, и так везде. */
:root {
  --bg:#0b0f14; --bg-2:#111823; --line:#1e2a38; --line-2:#2a3a4d;
  --ink:#e8eef5; --ink-2:#9db0c4; --ink-3:#6b8299;
  --accent:#4d9fe0; --accent-dim:#2b5f8f;
  --known:#3ddc84; --unknown:#f5c451; --alarm:#ff6b6b;
}
body { background:var(--bg); color:var(--ink); font-family:Arial,sans-serif;
       margin:0; padding:20px; }
h1 { font-size:1.2rem; margin:0 0 16px; }
h2 { font-size:1rem; margin:0 0 10px; }
/* Сетка, а не свободное изменение размеров: оператор ставит «сколько камер в ряд»,
   ширина считается сама и кадры остаются одного размера. Тянуть каждую плитку за
   угол — это ещё и разъезжающиеся пропорции 16:9, и мусор в пустотах между ними. */
.grid { display:grid; gap:20px; align-items:start;
        grid-template-columns:repeat(var(--cols,1), minmax(0,1fr));
        max-width:calc(var(--cols,1) * 740px); }
/* Плитка камеры. Оформление идёт ВОКРУГ кадра и никогда поверх него: на картинке
   уже нарисованы рамки, скелеты и подписи, и любая наша полоса поверх закрыла бы
   именно то, ради чего сюда смотрят. */
.cam { background:linear-gradient(158deg,#141b24f2,#0f151de6); border-radius:14px;
       padding:12px; min-width:0; border:1px solid var(--line);
       box-shadow:0 16px 34px -20px #000c, inset 0 1px 0 #ffffff0d;
       transition:border-color .3s, box-shadow .3s;
       animation:rise .6s cubic-bezier(.2,.8,.25,1) backwards;
       animation-delay:calc(var(--n,0) * 80ms); }
.cam:hover { border-color:var(--line-2); box-shadow:0 22px 44px -22px #000e; }
.cam h3 { display:flex; align-items:center; gap:8px; margin:0 0 10px;
          font-size:15px; font-weight:600; }
/* Уголки видоискателя вокруг кадра — тем же языком, что рамка детектора внутри. */
.cam .shot { position:relative; }
.cam .shot::before { content:""; position:absolute; inset:-4px; pointer-events:none;
  z-index:2; border-radius:8px;
  background:
    linear-gradient(var(--cc) 0 0) 0 0 / 14px 2px no-repeat,
    linear-gradient(var(--cc) 0 0) 0 0 / 2px 14px no-repeat,
    linear-gradient(var(--cc) 0 0) 100% 0 / 14px 2px no-repeat,
    linear-gradient(var(--cc) 0 0) 100% 0 / 2px 14px no-repeat,
    linear-gradient(var(--cc) 0 0) 0 100% / 14px 2px no-repeat,
    linear-gradient(var(--cc) 0 0) 0 100% / 2px 14px no-repeat,
    linear-gradient(var(--cc) 0 0) 100% 100% / 14px 2px no-repeat,
    linear-gradient(var(--cc) 0 0) 100% 100% / 2px 14px no-repeat;
  --cc:#2c3a4b; transition:--cc .3s; }
.cam:hover .shot::before { --cc:var(--accent-dim); }
.cam.found .shot::before { --cc:var(--known); }
.cam img { width:100%; min-height:180px; border-radius:6px; background:#000;
           display:block; }
.cam-head { cursor:grab; user-select:none; }
.cam-head:active { cursor:grabbing; }
.cam-head .grip { color:var(--ink-3); margin-right:2px; letter-spacing:-2px;
                  transition:color .2s; }
.cam:hover .cam-head .grip { color:var(--accent); }
.cam.dragging { opacity:.35; }
.cam.drop-to { outline:2px dashed #44617f; outline-offset:3px; }
/* Переключатель раскладки прижат вправо и намеренно тихий: его трогают раз в день,
   а смотрят на кадры. Сегменты слиты в одну плашку, чтобы не спорить с заголовком. */
.live-title { display:flex; align-items:center; gap:10px; }
.layout-pick { display:inline-flex; align-items:center; margin-left:auto;
               background:#161616; border:1px solid #2e2e2e; border-radius:7px;
               overflow:hidden; }
.layout-pick .lbl { color:#5c5c5c; font-size:11px; padding:0 7px; line-height:1; }
.layout-pick button { padding:3px 9px; font-size:11px; font-weight:400; line-height:16px;
                      color:#8f8f8f; background:transparent; border:0; border-radius:0;
                      border-left:1px solid #2e2e2e; }
.layout-pick button:hover { background:#202020; color:#e0e0e0; }
.layout-pick button.on { background:#243b55; color:#fff; }
/* Где человека видели. Тот филиал, что открыт сейчас, подсвечен: в общей базе иначе
   не отличить «был у нас» от «был у соседей». */
/* Порог бездействия и выбор дня — одна строка без подложки. Панель здесь была рамкой
   вокруг одного поля ввода: фон обещает раздел, а внутри настройка в два символа. */
.idle-bar { display:flex; align-items:center; gap:10px; margin:0 0 14px; }
.idle-bar h2 { margin:0 6px 0 0; }
.idle-bar select { padding:7px 8px; }
/* Единица измерения живёт внутри поля: рядом с ним она читалась как отдельный элемент
   интерфейса, хотя это подпись к числу. */
.unit-field { position:relative; display:inline-block; }
.unit-field input { width:104px; padding-right:40px; }
.unit-field input::-webkit-outer-spin-button,
.unit-field input::-webkit-inner-spin-button { -webkit-appearance:none; margin:0; }
.unit-field input { -moz-appearance:textfield; }
.unit-field .unit { position:absolute; right:11px; top:50%; transform:translateY(-50%);
                    font-size:11px; color:#777; pointer-events:none; }
/* Вкладки внутри базы. Тоньше верхней шапки: это деление одной страницы, а не
   переход в другую часть системы. */
/* Строка вкладок стоит выше содержимого страницы, и это не косметика: из неё
   раскрывается список филиалов. Пока у неё был тот же z-index, что у подписи и
   у сетки карточек, её собственный z-index:50 ничего не решал — он действует
   ВНУТРИ её слоя, а сам слой лежал вровень с теми, кто идёт следом в разметке.
   Список открывался и оказывался под ними: сквозь него читался текст, а нижние
   строки закрывали карточки. Ниже шапки (5), чтобы та осталась сверху. */
.subnav { position:relative; z-index:4;
          display:flex; align-items:center; gap:6px; margin:0 0 12px; }
.subnav a { color:#9a9a9a; text-decoration:none; font-size:13px; padding:5px 13px;
            border-radius:7px; border:1px solid transparent; }
.subnav a:hover { background:#1e1e1e; color:#ddd; }
.subnav a.active { background:#243b55; border-color:#44617f; color:#fff; }
.subnav .cnt { margin-left:7px; font-size:11px; color:#7d8b99; }
.subnav a.active .cnt { color:#a9c6e8; }
.subnav select { padding:5px 8px; font-size:13px; }
/* Управление вкладки стоит рядом с её названием, а не у края окна: оно относится
   к вкладке, и тянуться к нему через весь экран незачем. Выбор дня и выбор периода
   занимают одно и то же место, потому что одновременно нужен только один из них. */
.work-controls { margin-left:10px; padding-left:14px;
                 border-left:1px solid var(--line);
                 display:flex; align-items:center; gap:8px; }
.work-controls .layout-pick { margin-left:0; }
.br-tags { margin-top:5px; display:flex; flex-wrap:wrap; gap:4px; }
.br-tag { font-size:10px; letter-spacing:.3px; padding:1px 7px; border-radius:9px;
          background:#20262e; color:#9fb3c8; border:1px solid #2c3540; }
.br-tag.here { background:#14432a; color:#7ee2a8; border-color:#1f6b42; }
/* Уведомление о человеке сразу в двух филиалах. Висит до тех пор, пока его не убрали:
   оператор может отойти, а вопрос задают потом. */
.toasts { position:fixed; right:18px; bottom:18px; z-index:400; display:flex;
          flex-direction:column; gap:10px; max-width:360px; }
.toast { display:flex; gap:8px; align-items:flex-start; background:#241c14;
         border:1px solid #6b4a1f; border-left:4px solid #e0a33a; border-radius:9px;
         padding:10px 12px; font-size:13px; box-shadow:0 6px 20px #0009; }
.toast-h { font-size:10px; letter-spacing:.6px; color:#e0a33a; margin-bottom:4px; }
.toast-body { min-width:0; }
.toast-x { flex:none; background:transparent; border:0; padding:0 2px;
           font-size:16px; line-height:1; color:#a08a63; }
.toast-x:hover { background:transparent; color:#fff; }
/* Панель — тот же материал, что карточки: слегка прозрачное стекло с тонким
   светом по верхней кромке. Иначе на страницах со сценой она выглядела глухой
   заплаткой поверх сияния. */
.panel { background:linear-gradient(158deg,#141b24e6,#0f151dd9);
         border:1px solid var(--line); border-radius:14px; padding:14px;
         margin-top:18px; box-shadow:0 16px 34px -22px #000c, inset 0 1px 0 #ffffff0d;
         animation:rise .55s cubic-bezier(.2,.8,.25,1) backwards;
         animation-delay:calc(var(--n,0) * 70ms); }
.panel h2 { display:flex; align-items:center; gap:10px; }
/* Управление — из той же палитры, что карточки. Раньше поля были почти чёрными на
   почти чёрном, и границу между «можно ввести» и «просто фон» приходилось искать. */
input,select,button { background:#0e141c; color:var(--ink);
                      border:1px solid var(--line-2); border-radius:8px; padding:8px;
                      transition:border-color .2s, background .2s; }
input:focus,select:focus { outline:none; border-color:var(--accent-dim);
                           background:#111a24; }
input::placeholder { color:var(--ink-3); }
button { cursor:pointer; border-color:#3a5b7d;
         background:linear-gradient(180deg,#26405c,#1b2c40); }
button:hover { background:linear-gradient(180deg,#2f4f72,#213850); }
.muted { color:var(--ink-2); font-size:12px; }
.ok { color:var(--known); }
.warn { color:var(--unknown); }
.bad { color:var(--alarm); }
/* Живая точка дышит: неподвижный зелёный кружок ничем не отличается от нарисованного,
   а тут он значит «поток идёт прямо сейчас». */
/* Кольцо расходится масштабом отдельного слоя, а не растущей тенью: тень браузер
   перерисовывает на каждом кадре, а слой ему достаточно растянуть. Точек на
   странице столько же, сколько камер, так что мелочь набегает. */
.dot { display:inline-block; position:relative; width:9px; height:9px;
       border-radius:50%; background:#3a4a5c; flex:none; }
.dot.on { background:var(--known); }
.dot.on::after { content:""; position:absolute; inset:0; border-radius:50%;
                 border:2px solid #3ddc84; will-change:transform, opacity;
                 animation:ping 2.4s ease-out infinite; }
/* Шапка выше декоративных слоёв. Без этого фиксированный .fx на входной странице
   накрывает вкладки, и по ним нельзя щёлкнуть — видно всё, не нажимается ничего. */
.nav { position:relative; z-index:5; display:flex; gap:8px; align-items:center;
       margin-bottom:16px; }
.nav a { color:var(--ink-2); text-decoration:none; padding:7px 14px; border-radius:9px;
         border:1px solid var(--line); background:#111823; font-size:14px;
         transition:background .2s, color .2s, border-color .2s; }
.nav a:hover { background:#18222f; color:var(--ink); border-color:var(--line-2); }
.nav a.active { background:linear-gradient(180deg,#22405f,#18304a);
                border-color:#3a5b7d; color:#fff;
                box-shadow:0 0 0 1px #3a5b7d40, 0 6px 16px -8px #000; }
.nav .sp { flex:1; }
.find { display:flex; gap:8px; align-items:center; }
.find input { width:200px; }
.cam.found { outline:2px solid #2ecc71; outline-offset:3px; }
.cam.found h3 { color:#4ade80; }
.cam .here { display:none; font-size:11px; font-weight:600; letter-spacing:.4px;
             background:#14432a; color:#4ade80; padding:2px 8px; border-radius:10px;
             margin-left:6px; }
.cam.found .here { display:inline-block; }
/* ---------- Карточка человека ------------------------------------------
   Страница базы — это про узнавание, поэтому и язык у неё биометрический:
   уголки кадра вокруг снимка, как у рамки детектора, и проход луча по лицу при
   наведении. Названный и безымянный различаются не подписью, а цветом и краем:
   в списке на четыреста карточек подпись читать некогда.
   Без backdrop-filter: карточек сотни, и размытие фона под каждой съело бы
   больше, чем весь остальной интерфейс вместе.                             */
#faces { display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr));
         gap:14px; perspective:1400px; }
/* В базе бывает под шестьсот карточек, а на экране помещается десяток. Браузеру
   разрешено не рисовать те, что за пределами окна, — он всё равно их не показывает.
   contain-intrinsic-size говорит, какого они примерно размера, чтобы полоса
   прокрутки не прыгала, пока до них не дошли. Это самая крупная экономия здесь:
   она снимает не эффект, а работу, которая и так была не видна. */
#faces .person-card { animation:rise .55s cubic-bezier(.2,.8,.25,1) backwards;
                      content-visibility:auto; contain-intrinsic-size:auto 260px; }

/* Уголки кадра. Чистый CSS: две линии в углу через градиенты, без разметки. */
.person-photo::before { content:""; position:absolute; inset:-3px; border-radius:10px;
  pointer-events:none; z-index:2;
  background:
    linear-gradient(var(--fr) 0 0) 0 0 / 11px 2px no-repeat,
    linear-gradient(var(--fr) 0 0) 0 0 / 2px 11px no-repeat,
    linear-gradient(var(--fr) 0 0) 100% 0 / 11px 2px no-repeat,
    linear-gradient(var(--fr) 0 0) 100% 0 / 2px 11px no-repeat,
    linear-gradient(var(--fr) 0 0) 0 100% / 11px 2px no-repeat,
    linear-gradient(var(--fr) 0 0) 0 100% / 2px 11px no-repeat,
    linear-gradient(var(--fr) 0 0) 100% 100% / 11px 2px no-repeat,
    linear-gradient(var(--fr) 0 0) 100% 100% / 2px 11px no-repeat;
  --fr:#3a4a5c; transition:--fr .3s; }
.person-card.known .person-photo::before { --fr:#2e7f52; }
.person-card:hover .person-photo::before { --fr:var(--accent); }

/* Луч проходит по лицу один раз при наведении — ровно то, что делает система,
   когда решает, кто это. */
.person-photo::after { content:""; position:absolute; inset:0; border-radius:8px;
  pointer-events:none; z-index:3; opacity:0;
  background:linear-gradient(180deg, transparent, #7fc4ff00 40%,
             #bfe4ff99 50%, #7fc4ff00 60%, transparent);
  background-size:100% 200%; }
.person-head:hover .person-photo::after { opacity:1; animation:facescan 1.1s ease-in-out; }
@keyframes facescan { from { background-position:0 -100%; }
                      to   { background-position:0 200%; } }

.person-card { position:relative; background:#141b24cc; border:1px solid var(--line);
               border-left:4px solid #3a4a5c; border-radius:12px; padding:12px;
               transition:transform .28s cubic-bezier(.2,.8,.25,1),
                          box-shadow .28s, border-color .28s; }
.person-card:hover { transform:translateY(-3px);
                     border-color:var(--accent-dim);
                     box-shadow:0 18px 34px -18px #000e; }
.person-card.known { border-left-color:var(--known); }
/* Бегущий по контуру луч убран. Он крутил градиент через переменную --a, а без
   @property её нельзя менять плавно: в Firefox и Safari он вообще не двигался, а в
   хромоподобных перерисовывал контур каждый кадр — и так на каждой карточке под
   курсором. Осталась подсветка края: тот же смысл «карточка под рукой», ценой
   одного перехода прозрачности. */
.person-card::after { content:""; position:absolute; inset:-1px; border-radius:12px;
  opacity:0; transition:opacity .3s; pointer-events:none;
  box-shadow:0 0 0 1px #7fc4ff55, 0 0 22px -4px #4d9fe045; }
.person-card:hover::after { opacity:1; }

.person-head { display:flex; gap:12px; cursor:pointer; }
.person-info { min-width:0; }
.person-photo { position:relative; flex:none; width:88px; height:88px; }
.person-photo img { width:88px; height:88px; object-fit:cover; border-radius:8px;
                    display:block; filter:saturate(.9) contrast(1.05); }
.shot-badge { position:absolute; top:4px; right:4px; background:#000a; color:#fff;
              font-size:11px; padding:1px 6px; border-radius:10px; z-index:4; }
.person-name { font-size:17px; font-weight:600; margin-bottom:4px; letter-spacing:-.2px; }
.tag { display:inline-block; font-size:10px; letter-spacing:.5px; padding:2px 7px;
       border-radius:10px; background:#1d2733; color:var(--ink-2); margin-right:4px; }
.tag-known { background:#0f3b26; color:var(--known); box-shadow:0 0 0 1px #1f6b42 inset; }
.tag-unknown { background:#332c16; color:var(--unknown);
               box-shadow:0 0 0 1px #6b5a1f inset; }
.person-seen { display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:12px;
               margin-top:10px; padding-top:8px; border-top:1px solid var(--line); }
.person-edit { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:10px; }
.person-edit input { width:100%; }
.card-buttons { display:grid; grid-template-columns:1fr auto; gap:6px; margin-top:8px; }
.person-card button { width:100%; }
button.danger { background:#4a1f22; border-color:#7a3339; color:#ffb3b3; }
button.danger:hover { background:#63262b; }
/* Всплывающее окно с кадрами человека. Прокручивается только оно само, страница под
   ним стоит на месте: список людей обновляется каждые три секунды, и уехавшая под
   окном страница возвращала бы оператора не туда, откуда он открыл карточку. */
.modal { position:fixed; inset:0; z-index:500; background:#000c;
         display:flex; align-items:center; justify-content:center; padding:24px; }
/* display:flex по классу перебивает браузерное правило для hidden, и без этой строки
   окно висело бы поверх страницы всегда. */
.modal[hidden] { display:none; }
.modal-box { background:#141414; border:1px solid #333; border-radius:12px;
             width:min(1100px, 100%); max-height:100%; display:flex;
             flex-direction:column; box-shadow:0 20px 60px #000b; }
.modal-head { display:flex; align-items:center; gap:12px; padding:14px 18px;
              border-bottom:1px solid #2a2a2a; }
.modal-head h3 { margin:0; }
.modal-x { margin-left:auto; width:auto; background:transparent; border:0; color:#999;
           font-size:22px; line-height:1; padding:0 4px; }
.modal-x:hover { background:transparent; color:#fff; }
.modal-body { overflow-y:auto; padding:16px 18px; }
body.modal-open { overflow:hidden; }
/* align-content:start обязателен: сетка внутри прокручиваемого блока с заданной
   высотой иначе растягивает свои строки на всю высоту, и карточки разъезжались,
   а текст в них обрезался рамкой. */
.shot-grid { display:grid; gap:14px; align-content:start;
             grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); }
.detail-h { margin:20px 0 8px; font-size:13px; letter-spacing:.4px; color:#cfcfcf; }
.shot-card { background:#181818; border:1px solid #2c2c2c; border-radius:10px; overflow:hidden; }
.shot-frame { width:100%; display:block; }
.shot-meta { display:flex; gap:10px; align-items:center; padding:8px 10px; }
.shot-thumb { width:52px; height:52px; object-fit:cover; border-radius:6px; flex:none; }
.person-cloth { margin-top:9px; padding-top:8px; border-top:1px solid #2c2c2c;
                font-size:12px; color:#aaa; }
.sw { display:inline-flex; align-items:center; gap:4px; margin:2px 6px 2px 0; }
.sw i { width:12px; height:12px; border-radius:3px; border:1px solid #0006; display:block; }
.sw.dim { opacity:.42; }
.add-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; }
.add-grid label { display:flex; flex-direction:column; gap:4px; font-size:12px; color:#aaa; }
.add-grid input { width:100%; }
.checks { display:flex; flex-wrap:wrap; gap:14px; margin-top:8px; }
.chk { display:flex; align-items:center; gap:6px; font-size:13px; color:#ddd; }
.chk input { width:auto; }
button.row-del { padding:4px 12px; font-size:12px; }
/* ---------- Вход в систему ----------------------------------------------
   Страница выбора объекта — единственная, куда попадают ДО работы, и её задача
   не украшать, а отвечать на один вопрос: где сейчас всё в порядке, а где нет.
   Поэтому объём здесь не поверх смысла, а вместо подписи: карточка живого объекта
   светится и приподнята, мёртвого — вдавлена и погашена; состояние читается
   раньше, чем прочитан текст.
   Всё на CSS, без сторонних библиотек: система работает в локальной сети, где
   интернета может не быть вовсе.                                            */
.home { position:relative; min-height:calc(100vh - 74px); }

/* Все декоративные слои — в одном контейнере на всю ширину окна, за содержимым.
   fixed, а не absolute: сцена не должна ездить вместе со списком объектов. */
.fx { position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
.fx > span { position:absolute; display:block; }

/* Северное сияние: три пятна холодного света, которые медленно дышат и расходятся.
   Разная длительность у каждого — чтобы рисунок не повторялся заметным циклом. */
/* Без filter:blur. Размытие целого экрана пересчитывается на каждом шаге движения,
   и на слабой машине именно оно съедало кадры. Градиенты и так мягкие — достаточно
   растянуть переходы, картинка та же, а рисовать нечего.
   will-change обещает браузеру, что двигаться будет только transform: он кладёт слой
   на видеокарту и дальше просто его сдвигает, не перерисовывая. */
.fx-aurora { inset:-25%; opacity:.5; will-change:transform;
  background:
    radial-gradient(42% 48% at 22% 26%, #1f5c9e4a, #1f5c9e1c 45%, transparent 72%),
    radial-gradient(38% 44% at 78% 20%, #2b7f8f40, #2b7f8f18 45%, transparent 72%),
    radial-gradient(50% 54% at 55% 88%, #143a6b45, #143a6b1a 48%, transparent 76%);
  animation:aurora 26s ease-in-out infinite alternate; }
@keyframes aurora {
  0%   { transform:translate3d(0,0,0) scale(1); }
  50%  { transform:translate3d(-2.5%, 2%, 0) scale(1.08); }
  100% { transform:translate3d(2%, -1.5%, 0) scale(1.03); }
}

/* Луч радара. Здесь он не украшение: страница — выбор объекта наблюдения, и
   вращающийся сектор говорит про систему то же, что зелёная точка про камеру. */
/* Без mix-blend-mode: смешивание заставляет пересобирать всё, что лежит ниже, на
   каждом кадре вращения. Тот же вид даёт обычная непрозрачность. Луч заодно вдвое
   меньше — 170vmax это была поверхность больше экрана втрое. */
.fx-radar { top:50%; left:50%; width:96vmax; height:96vmax; margin:-48vmax;
  opacity:.34; will-change:transform;
  background:conic-gradient(from 0deg, #4d9fe000 0deg, #4d9fe000 300deg,
                            #4d9fe016 348deg, #7fc4ff30 358deg, #4d9fe000 360deg);
  mask-image:radial-gradient(closest-side, #000 32%, transparent 74%);
  -webkit-mask-image:radial-gradient(closest-side, #000 32%, transparent 74%);
  animation:radar 18s linear infinite; }
@keyframes radar { to { transform:rotate(360deg); } }

/* Уходящая в перспективу сетка — «пол» под карточками. Даёт сцене глубину,
   от которой отталкивается наклон карточек; без неё они висят в пустоте. */
.fx-floor { inset:auto 0 0 0; height:56vh; perspective:340px; opacity:.5; }
/* Сетка едет transform'ом, а не background-position: сдвиг фона — это перерисовка
   всей поверхности, сдвиг слоя — нет. Отсюда и лишние 78px высоты: слой уезжает
   ровно на клетку и возвращается, шва не видно. */
.fx-floor::before { content:""; position:absolute; inset:-60% -50% -20% -50%;
  background:
    repeating-linear-gradient(90deg, #2a3a4d 0 1px, transparent 1px 78px),
    repeating-linear-gradient(0deg,  #2a3a4d 0 1px, transparent 1px 78px);
  transform:rotateX(76deg); transform-origin:50% 100%; will-change:transform;
  animation:floor 22s linear infinite;
  mask-image:linear-gradient(to top, #000 0%, transparent 78%);
  -webkit-mask-image:linear-gradient(to top, #000 0%, transparent 78%); }
@keyframes floor {
  from { transform:rotateX(76deg) translate3d(0,0,0); }
  to   { transform:rotateX(76deg) translate3d(0,78px,0); }
}

/* Зерно. Градиенты без него выглядят пластиковыми — плёночный шум делает их
   похожими на снятое камерой. Рисуется один раз, движется только позицией. */
/* Зерно теперь НЕПОДВИЖНО и без смешивания. Дрожание давало плёночный характер,
   но это пять перерисовок всего экрана в секунду — самая дорогая мелочь из всех,
   какие тут были. Текстура на месте, а тратить на неё нечего. */
.fx-grain { inset:0; opacity:.035;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)'/%3E%3C/svg%3E"); }

/* Виньетка собирает взгляд в центр, где стоят карточки. */
.fx-vign { inset:0;
  background:radial-gradient(120% 90% at 50% 40%, transparent 45%, #05070ac9 100%); }

/* Облегчённый режим. Оптимизация оптимизацией, но машины бывают очень разные, и
   решать должен тот, кто сидит за этой машиной, а не мы отсюда. Одним классом на
   <body> снимается вся сцена и всё движение; вёрстка и цвета остаются. */
body.lite .fx { display:none; }
body.lite .site, body.lite .person-card, body.lite .cam, body.lite .panel,
body.lite .sig i.on { animation:none; }
body.lite .dot.on::after, body.lite .site-dot.on::after { display:none; }
body.lite .site { transform:none !important; transition:none; }
body.lite .person-photo::after, body.lite .site-sheen { display:none; }
body.lite .person-card, body.lite .cam, body.lite .panel, body.lite .site-face {
  box-shadow:none; }
.lite-btn { width:auto; padding:5px 11px; font-size:12px; background:#111823;
            border:1px solid var(--line); color:var(--ink-3); }
.lite-btn:hover { background:#18222f; color:var(--ink); }
body.lite .lite-btn { border-color:var(--accent-dim); color:var(--accent); }

/* Тихий вариант сцены — для страниц со списками. Там сцена должна быть фоном, а
   не событием: сияние глуше, зерна почти нет. */
.fx-calm .fx-aurora { opacity:.3; }
.fx-calm .fx-grain { opacity:.03; }
.fx-calm .fx-vign {
  background:radial-gradient(130% 100% at 50% 30%, transparent 55%, #05070aa8 100%); }

/* Содержимое страниц со сценой — над ней. Перечислены КОНКРЕТНЫЕ блоки, а не
   «.fx ~ div»: тот селектор задевал и всплывающее окно карточки человека, причём
   с большей силой, чем .modal, — position менялся с fixed на relative, и окно
   вместо наложения поверх страницы падало вниз документа. Клик по человеку при
   этом срабатывал, а показывать было нечего. */
.fx ~ h1, .fx ~ .panel, .fx ~ .grid, .fx ~ .idle-bar, .fx ~ #idle-now,
.fx ~ #faces, .fx ~ #faces-hint, .fx ~ .home-grid, .fx ~ .home-orphans {
  position:relative; z-index:2; }
/* И на всякий случай: окно всегда поверх и всегда fixed, что бы ни решили
   соседние правила. */
#face-detail.modal { position:fixed; z-index:500; }

.home-form { position:relative; z-index:2; margin-bottom:18px; }
.home-orphans { position:relative; z-index:2; margin-bottom:12px; }

.home-grid { position:relative; z-index:2; display:grid; gap:22px;
             grid-template-columns:repeat(auto-fill, minmax(290px, 1fr));
             perspective:1100px; }

/* Карточка объекта. Наклон и подъём считает JS через переменные --rx/--ry,
   CSS отвечает только за то, как это выглядит. */
.site { position:relative; transform-style:preserve-3d; border:0; padding:0;
        background:none; border-radius:16px;
        transform:rotateX(var(--rx,0deg)) rotateY(var(--ry,0deg));
        transition:transform .5s cubic-bezier(.2,.7,.3,1);
        animation:rise .7s cubic-bezier(.2,.8,.25,1) backwards;
        animation-delay:calc(var(--n, 0) * 90ms); }
/* Карточки не появляются разом: очередь в 90 мс даёт глазу пройти по ним слева
   направо и заодно прячет момент, когда список перерисовывается. */
@keyframes rise {
  from { opacity:0; transform:translateY(26px) rotateX(-9deg) scale(.97); }
}
.site:hover { transition:transform .08s linear; }
.site-face { display:flex; flex-direction:column; gap:6px; min-height:196px;
             padding:18px 18px 16px; border-radius:16px; text-decoration:none;
             color:inherit; position:relative; overflow:hidden;
             background:linear-gradient(158deg, #1b2531f2, #131a24f2);
             border:1px solid #2c3a4b;
             /* backdrop-filter убран: он размывает всё, что лежит ПОД карточкой, и
                делает это постоянно, для каждой карточки. Фон под ними и так почти
                непрозрачный — разницы не видно, а на слабой машине это были кадры. */
             box-shadow:0 18px 40px -18px #000c, inset 0 1px 0 #ffffff10;
             transition:box-shadow .35s, border-color .35s; }
.site:hover .site-face { border-color:#3f6288;
             box-shadow:0 34px 60px -22px #000f, 0 0 0 1px #3f628855,
                        inset 0 1px 0 #ffffff1a; }
/* Блик идёт за курсором — это и есть ощущение стекла под наклоном. */
.site-sheen { position:absolute; inset:0; pointer-events:none; opacity:0;
              transition:opacity .3s;
              background:radial-gradient(340px 240px at var(--mx,50%) var(--my,50%),
                #7fb3ff26, transparent 68%); }
.site:hover .site-sheen { opacity:1; }

.site-top { display:flex; align-items:center; gap:8px; }
.site-dot { width:8px; height:8px; border-radius:50%; flex:none; }
.site-dot { position:relative; }
.site-dot.on { background:#3ddc84; }
.site-dot.on::after { content:""; position:absolute; inset:0; border-radius:50%;
                      border:2px solid #3ddc84; will-change:transform, opacity;
                      animation:ping 2.4s ease-out infinite; }
.site-dot.off { background:#5a3a3a; box-shadow:0 0 8px #ff585840; }
@keyframes ping {
  from { transform:scale(1);   opacity:.55; }
  70%, to { transform:scale(3); opacity:0; }
}
.site-state { font-size:9px; letter-spacing:1.8px; color:#6f8aa6; }
.site-name { font-size:22px; font-weight:650; letter-spacing:-.2px; color:#eef4fa;
             transform:translateZ(26px); }
.site-note { font-size:12px; color:#7f93a8; min-height:16px;
             transform:translateZ(14px); }
.site-foot { margin-top:auto; display:flex; align-items:center; gap:10px;
             transform:translateZ(20px); }
.site-count { font-size:12px; color:#8fa3b8; }
.site-count b { color:#dce8f4; }
.site-enter { position:absolute; right:18px; bottom:16px; font-size:12px;
              color:#7fb3ff; opacity:0; transform:translateX(-6px);
              transition:opacity .3s, transform .3s; }
.site:hover .site-enter { opacity:1; transform:translateX(0); }

/* Полоски связи: по одной на камеру. Загораются по очереди — глаз ловит движение
   и понимает, что это живое состояние, а не нарисованная иконка. */
.sig { display:inline-flex; align-items:flex-end; gap:3px; height:16px; }
.sig i { width:4px; border-radius:2px; background:#2f3d4d; height:7px;
         display:block; }
.sig i:nth-child(2){height:10px} .sig i:nth-child(3){height:13px}
.sig i:nth-child(n+4){height:16px}
.sig i.on { background:linear-gradient(180deg,#5ef0a0,#28a866);
            box-shadow:0 0 8px #3ddc8455;
            animation:sigin .5s backwards; animation-delay:calc(var(--i) * .07s); }
@keyframes sigin { from { transform:scaleY(.2); opacity:0; } }

.site-del { position:absolute; top:10px; right:10px; width:auto; padding:2px 7px;
            font-size:12px; line-height:1.2; border-radius:8px; color:#8a6a6a;
            background:#2a1e1e88; border:1px solid #4a2f2f; opacity:0;
            transform:translateZ(30px); transition:opacity .25s; }
.site:hover .site-del { opacity:1; }
.site-del:hover { background:#4a1f22; color:#ffb3b3; }

/* «Добавить объект» — та же карточка, но пустая: место, куда объект встанет. */
.site-new { display:flex; flex-direction:column; align-items:center;
            justify-content:center; gap:8px; min-height:196px; cursor:pointer;
            border:1px dashed #33445a; border-radius:16px; background:#141b2480;
            color:#6f8aa6; }
.site-new:hover { border-color:#4a7099; color:#9dc0e4; background:#18212d90; }
.site-plus { font-size:30px; line-height:1; }
.site-newtext { font-size:13px; }

/* Кому от движения плохо — тот получает ту же картинку, но неподвижную.
   Гасим ВСЁ, что двигалось, а не только заметное: зерно и радар укачивают
   сильнее медленного сияния. */
@media (prefers-reduced-motion: reduce) {
  .fx-aurora, .fx-radar, .fx-floor::before, .fx-grain, .person-card::after,
  .site, .sig i.on, #faces .person-card, .cam { animation:none; }
  .dot.on::after, .site-dot.on::after { display:none; }
  .site, .person-card, .cam { transition:none; }
  .panel { animation:none; }
  .person-photo::after { display:none; }
}

.branch-card { width:min(320px,100%); }
.branch-open { display:block; text-decoration:none; color:inherit; }
.branch-open:hover .branch-name { color:#7fb3ff; }
.branch-name { font-size:19px; font-weight:600; margin-bottom:4px; }
.picker { position:relative; display:inline-block; }
/* Половина экрана и прокрутка: список имён растёт вместе с базой, и без потолка
   он однажды закрыл собой всю страницу. */
.picker-list { display:none; position:absolute; top:calc(100% + 6px); left:0;
               min-width:100%; max-height:45vh; overflow-y:auto; z-index:50;
               background:var(--bg-2); border:1px solid var(--line-2);
               border-radius:10px; padding:4px;
               box-shadow:0 16px 34px -10px #000e; }
.picker-item { padding:7px 11px; cursor:pointer; white-space:nowrap; font-size:14px;
               color:var(--ink); border-radius:7px; }
.picker-item:hover { background:#1d2c3d; }
/* Полоса прокрутки тоже часть окна: системная светлая на тёмном списке видна
   первой, раньше самих имён. */
.picker-list::-webkit-scrollbar { width:10px; }
.picker-list::-webkit-scrollbar-track { background:transparent; }
.picker-list::-webkit-scrollbar-thumb { background:#2a3a4d; border-radius:6px;
                                        border:3px solid var(--bg-2); }
.picker-list::-webkit-scrollbar-thumb:hover { background:#3a5171; }
.picker-list { scrollbar-width:thin; scrollbar-color:#2a3a4d transparent; }
/* Выбранная строка помечена галочкой слева, а не только цветом: в тёмном списке
   подсветка и наведение курсора выглядят одинаково. */
.picker-item.on { color:var(--accent); }
.picker-item.on::before { content:"✓"; margin-right:7px; }
.picker-item:not(.on)::before { content:""; margin-right:19px; }
.pick-btn { display:inline-flex; align-items:center; gap:8px; width:auto;
            padding:5px 10px; font-size:13px; border-radius:8px;
            background:#141b24; border:1px solid var(--line-2); color:var(--ink); }
.pick-btn:hover { background:#1a2532; border-color:var(--accent-dim); }
.pick-caret { color:var(--ink-3); font-size:10px; }

/* Фильтр стоит СРАЗУ ЗА вкладками, а не у правого края окна. Он сужает тот же
   список, что и вкладки, значит и рука тянется к нему тем же движением; отброшенный
   на другой конец широкого экрана, он оказывался в полутора тысячах пикселей от
   того, чем управляет. Тонкая черта слева отделяет «что показывать» от «чьё».
   Цвета — те же переменные, что у карточек, иначе выпадашка выглядит деталью из
   другого интерфейса. */
.db-pick { margin-left:10px; padding-left:14px;
           border-left:1px solid var(--line); }
.db-pick .picker-list { left:0; right:auto; min-width:190px;
                        background:var(--bg-2); border-color:var(--line-2);
                        border-radius:10px; padding:4px;
                        box-shadow:0 16px 34px -10px #000e; }
.db-pick .picker-item { color:var(--ink); border-radius:7px; padding:8px 11px; }
.db-pick .picker-item:hover { background:#1d2c3d; }
.db-pick .picker-item.on { background:#16304a; }

/* Выбор цвета одежды. Квадратик слева — тот же цвет, что рисуется над головой
   в кадре: список и видео обязаны говорить об одном и том же одним и тем же. */
.cpick .picker-list { left:0; right:auto; min-width:170px; padding:4px;
                      background:var(--bg-2); border-color:var(--line-2);
                      border-radius:10px; box-shadow:0 16px 34px -10px #000e; }
.cpick .picker-item { display:flex; align-items:center; gap:9px; color:var(--ink);
                      border-radius:7px; padding:7px 10px; }
.cpick .picker-item::before { content:none; }
.cpick .picker-item:hover { background:#1d2c3d; }
.cpick .picker-item.on { background:#16304a; color:var(--accent); }
.cdot { width:13px; height:13px; border-radius:4px; flex:none; display:block;
        border:1px solid #0008; box-shadow:0 0 0 1px #ffffff14; }
.cdot.any { background:linear-gradient(135deg,#3a4a5c 0 50%,#22303f 50% 100%); }
.cpick .pick-btn { gap:7px; }
.cloth-now { color:#e8e8e8; }
.cloth-row { margin-top:4px; }
.table-wrap { overflow:auto; max-height:420px; margin-top:10px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { color:var(--ink-3); font-size:11px; letter-spacing:.6px; text-transform:uppercase;
     font-weight:600; }
th,td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; }
tr:last-child td { border-bottom:0; }
tbody tr:hover td, table tr:hover td { background:#141c2680; }
table tr:first-child:hover td { background:transparent; }
/* Полоса прокрутки таблиц — как у списков. */
.table-wrap::-webkit-scrollbar { width:10px; height:10px; }
.table-wrap::-webkit-scrollbar-track { background:transparent; }
.table-wrap::-webkit-scrollbar-thumb { background:#2a3a4d; border-radius:6px;
                                       border:3px solid #101720; }
.table-wrap { scrollbar-width:thin; scrollbar-color:#2a3a4d transparent; }
th { color:#bbb; font-weight:600; background:#202020; position:sticky; top:0; }
"""


def _nav(active: str, tail: str = "") -> str:
    def link(href, label, key):
        cls = ' class="active"' if key == active else ""
        return f'<a href="{href}"{cls}>{label}</a>'
    lite = ('<button class="lite-btn" id="lite-btn" onclick="toggleLite()"'
            ' title="turn the background and animation off on a slow machine">'
            'Lite</button>')
    if active in ("branches", "database"):
        # Вход в систему — две вкладки и больше ничего. Филиалы — это места, база —
        # это люди; ни то ни другое не вложено в другое, поэтому они равны в шапке, а
        # не «главная и ссылка на ней». Остальные страницы сюда не попадают: они всегда
        # про конкретный объект, а объект здесь ещё не выбран.
        return ('<div class="nav">'
                + link("/", "Branches", "branches")
                + link("/database", "Database", "database")
                + f'<div class="sp"></div>{tail}{lite}</div>')
    return ('<div class="nav">'
            + link("/", "← Branches", "branches")
            + link("/live", "Live", "live")
            + link("/faces", "Faces", "faces")
            + link("/activity", "Activity", "activity")
            + link("/map", "Map", "map")
            + link("/dashboard", "Cameras", "cameras")
            + f'<div class="sp"></div><span id="branch-tag" class="muted"></span>'
            + f'{tail}{lite}'
            + '</div>')


# Выбранный филиал ходит между страницами в адресе. Ссылки в шапке его подхватывают,
# иначе переход на «Лица» молча выбрасывал бы оператора из филиала обратно во «все».
_BRANCH_KEEP_JS = """
// Облегчённый режим запоминается в браузере: он про эту машину, а не про систему.
// Ставим класс ДО первой отрисовки, иначе на слабой машине сцена успеет мигнуть.
(function () {
  try { if (localStorage.getItem('lite') === '1') document.body.classList.add('lite'); }
  catch (e) {}
})();

function toggleLite() {
  const on = document.body.classList.toggle('lite');
  try { localStorage.setItem('lite', on ? '1' : '0'); } catch (e) {}
  const b = document.getElementById('lite-btn');
  if (b) b.textContent = on ? 'Lite on' : 'Lite';
}

(function () {
  const b = document.getElementById('lite-btn');
  if (b && document.body.classList.contains('lite')) b.textContent = 'Lite on';
})();

(function () {
  const b = new URLSearchParams(location.search).get('branch') || '';
  if (b) {
    document.querySelectorAll('.nav a').forEach(a => {
      if (a.getAttribute('href') !== '/') a.href = a.getAttribute('href') + '?branch=' + b;
    });
    // У вкладок внутри страницы адрес уже может нести свой параметр, поэтому склейка
    // не «?», а по обстоятельствам: «?show=unnamed?branch=…» браузер не разберёт.
    document.querySelectorAll('a[data-keep-branch]').forEach(a => {
      const href = a.getAttribute('href');
      a.href = href + (href.includes('?') ? '&' : '?') + 'branch=' + b;
    });
    fetch('/api/branches').then(r => r.json()).then(d => {
      const one = (d.branches || []).find(x => x.id === b);
      const tag = document.getElementById('branch-tag');
      if (one && tag) tag.textContent = one.name;
    }).catch(() => {});
  }
})();
"""


def _page(title: str, nav: str, body: str, script: str) -> str:
    return (f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>{title}</title>\n'
            f"<style>{_CSS}</style></head>\n<body>\n{nav}\n{body}\n"
            f"<script>\n{_BRANCH_KEEP_JS}\n{script}\n</script>\n</body></html>")


# Перезапуск застрявшего MJPEG нужен обеим страницам с картинками.
_STREAM_JS = """
const lastFrame = {};
document.querySelectorAll('img[id^="img-"]').forEach(img => {
  const camId = img.id.slice(4);
  lastFrame[camId] = Date.now();
  img.addEventListener('load', () => { lastFrame[camId] = Date.now(); });
});

function retryImg(camId) {
  setTimeout(() => {
    const img = document.getElementById('img-' + camId);
    if (img) img.src = '/stream/' + camId + '?t=' + Date.now();
  }, 2000);
}

setInterval(() => {
  const now = Date.now();
  document.querySelectorAll('img[id^="img-"]').forEach(img => {
    const camId = img.id.slice(4);
    if (now - (lastFrame[camId] || now) > 8000) {
      lastFrame[camId] = now;
      img.src = '/stream/' + camId + '?t=' + now;
    }
  });
}, 5000);

async function refreshStatus() {
  const status = await fetch('/api/status').then(r => r.json());
  for (const id in status) {
    const el = document.getElementById('dot-' + id);
    if (el) el.className = 'dot ' + (status[id] ? 'on' : '');
  }
}
setInterval(refreshStatus, 3000);
refreshStatus();
"""


# Один человек — сразу два филиала. Всё внутри функции: страницы, куда это добавляется,
# уже объявляют свои myBranch и esc, и второе объявление на верхнем уровне уронило бы
# разбор ВСЕГО скрипта, а не только этого куска.
_ALERT_JS = """
(function () {
  const mine = new URLSearchParams(location.search).get('branch') || '';
  const E = v => String(v == null ? '' : v)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const clock = ts => ts
    ? new Date(ts * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})
    : '';
  const live = new Map();
  let host = null;

  function box() {
    if (!host) {
      host = document.createElement('div');
      host.className = 'toasts';
      document.body.appendChild(host);
    }
    return host;
  }

  async function ack(ids) {
    ids.forEach(id => {
      const el = live.get(id);
      if (el) el.remove();
      live.delete(id);
    });
    try {
      await fetch('/api/branch-alerts/ack', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ids: ids}),
      });
    } catch (e) {}
  }

  function show(a) {
    // Тревога всегда о перемещении между двумя объектами, но смотрят на неё с любого
    // из них. Поэтому «здесь» и «там» определяются по открытому филиалу, а не по тому,
    // какой из них записан в тревоге первым.
    const here = (!mine || a.branch === mine)
      ? [a.branch_name, a.other_branch_name] : [a.other_branch_name, a.branch_name];
    const el = document.createElement('div');
    el.className = 'toast';
    el.innerHTML =
      `<div class="toast-body">
         <div class="toast-h">Same person at two branches</div>
         <div><b>${E(a.label)}</b>${a.named ? '' : ' <span class="muted">(no name yet)</span>'}
              — <b>${E(here[0])}</b> now</div>
         <div class="muted">was at <b>${E(here[1])}</b> at ${clock(a.other_seen)}
              ${a.camera_id ? '· ' + E(a.camera_id) : ''}</div>
       </div>`;
    const x = document.createElement('button');
    x.className = 'toast-x';
    x.textContent = '×';
    x.title = 'dismiss';
    x.onclick = () => ack([a.id]);
    el.appendChild(x);
    box().appendChild(el);
    live.set(a.id, el);
  }

  async function poll() {
    const q = new URLSearchParams({branch: mine});
    const d = await fetch('/api/branch-alerts?' + q).then(r => r.json())
      .catch(() => null);
    if (!d) return;
    const open = new Set();
    // Показываем от старых к новым, чтобы свежая тревога оказалась внизу столбика,
    // где её и ждут.
    for (const a of (d.alerts || []).slice().reverse()) {
      open.add(a.id);
      if (!live.has(a.id)) show(a);
    }
    for (const id of [...live.keys()]) {
      if (!open.has(id)) { live.get(id).remove(); live.delete(id); }
    }
  }

  setInterval(poll, 5000);
  poll();
})();
"""


# Раскладка плиток. Порядок и число камер в ряду живут в localStorage: это настройка
# рабочего места, а не системы, — у охранника на посту и у директора экраны разные,
# и хранить общий на сервере значило бы, что один переставил, а у другого поехало.
_LAYOUT_JS = """
(function () {
  const grid = document.getElementById('live-grid');
  if (!grid) return;
  const key = 'live-layout:' + (grid.dataset.branch || '_all');
  const tilesOf = () => Array.from(grid.querySelectorAll('.cam'));

  function cols() { return Number(grid.style.getPropertyValue('--cols')) || 1; }

  function save() {
    try {
      localStorage.setItem(key, JSON.stringify(
        {cols: cols(), order: tilesOf().map(el => el.dataset.cam)}));
    } catch (e) {}
  }

  window.setCols = function (n) {
    grid.style.setProperty('--cols', n);
    document.querySelectorAll('.layout-pick button').forEach(b => {
      b.classList.toggle('on', Number(b.dataset.cols) === n);
    });
    save();
  };

  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(key) || 'null'); } catch (e) {}
  if (saved && Array.isArray(saved.order)) {
    const byId = {};
    tilesOf().forEach(el => { byId[el.dataset.cam] = el; });
    saved.order.forEach(id => { if (byId[id]) grid.appendChild(byId[id]); });
  }
  setCols(saved && saved.cols ? saved.cols : 1);

  // Плитка становится перетаскиваемой только когда взяли за заголовок: иначе
  // выделение текста и клики внутри кадра превращаются в попытку перетащить.
  let dragged = null, target = null, after = false;

  grid.addEventListener('mousedown', ev => {
    const tile = ev.target.closest('.cam');
    if (tile && ev.target.closest('.cam-head')) tile.draggable = true;
  });

  function cleanup() {
    tilesOf().forEach(el => {
      el.draggable = false;
      el.classList.remove('dragging', 'drop-to');
    });
    dragged = null; target = null;
  }
  document.addEventListener('mouseup', () => { if (!dragged) cleanup(); });

  grid.addEventListener('dragstart', ev => {
    const tile = ev.target.closest('.cam');
    if (!tile || !tile.draggable) { ev.preventDefault(); return; }
    dragged = tile;
    tile.classList.add('dragging');
    ev.dataTransfer.effectAllowed = 'move';
    try { ev.dataTransfer.setData('text/plain', tile.dataset.cam); } catch (e) {}
  });

  // Двигаем плитку один раз, на отпускании. Если переставлять прямо во время
  // перетаскивания, каждый шаг мыши вынимает <img> из документа и mjpeg-поток
  // стартует заново — на четырёх камерах это сплошное мигание.
  grid.addEventListener('dragover', ev => {
    if (!dragged) return;
    ev.preventDefault();
    ev.dataTransfer.dropEffect = 'move';
    const over = ev.target.closest('.cam');
    if (target && target !== over) target.classList.remove('drop-to');
    if (!over || over === dragged) { target = null; return; }
    target = over;
    over.classList.add('drop-to');
    const r = over.getBoundingClientRect();
    after = cols() > 1 ? (ev.clientX - r.left) > r.width / 2
                       : (ev.clientY - r.top) > r.height / 2;
  });

  grid.addEventListener('drop', ev => {
    ev.preventDefault();
    if (dragged && target && target !== dragged) {
      grid.insertBefore(dragged, after ? target.nextSibling : target);
    }
    cleanup();
    save();
  });

  grid.addEventListener('dragend', () => { cleanup(); });
})();
"""


@app.get("/live", response_class=HTMLResponse)
def live_page(branch: str = ""):
    """Живое видео с разметкой. Отдельно от настройки камер: смотрят на него часами,
    а список камер трогают раз в месяц, и держать их на одной странице значило
    заставлять оператора листать настройки, чтобы увидеть кадр."""
    if not branch and BRANCHES:
        return RedirectResponse("/", status_code=303)
    shown = _branch_cameras(branch)
    # draggable="false" на кадре — не мелочь: картинку браузер тащит сам, и вместо
    # перестановки плитки начинается перетаскивание изображения в соседнюю вкладку.
    stream_cards = "\n".join(
        f'''<div class="cam" id="cam-{c["id"]}" data-cam="{c["id"]}" style="--n:{n}">
              <h3 class="cam-head" title="drag to move">
                  <span class="grip">⠿</span>
                  {c["name"]} <span id="dot-{c["id"]}" class="dot"></span>
                  <span class="here">HERE</span></h3>
              <div class="shot">
                <img id="img-{c["id"]}" src="/stream/{c["id"]}" draggable="false"
                     onerror="retryImg('{c["id"]}')">
              </div>
            </div>'''
        for n, c in enumerate(shown)
    )
    # Свой список вместо <select>: у встроенного выпадающую часть рисует система, и
    # на тёмной странице она выглядит чужой. А главное — здесь выбирают ЦВЕТ, и
    # квадратик рядом с названием отвечает на вопрос быстрее слова «beige».
    def colour_pick(key: str, label: str) -> str:
        rows = "".join(
            f'<div class="picker-item" data-c="{c}">'
            f'<i class="cdot" style="background:{_swatch_css(c)}"></i>{c}</div>'
            for c in COLOUR_NAMES)
        return (f'<span class="picker cpick" id="{key}-pick">'
                f'<input type="hidden" id="find-{key}">'
                f'<button type="button" class="pick-btn" id="{key}-btn">'
                f'<i class="cdot" id="{key}-dot" style="background:transparent"></i>'
                f'<span id="{key}-label">{label}: any</span>'
                f'<span class="pick-caret">▾</span></button>'
                f'<div class="picker-list" id="{key}-list">'
                f'<div class="picker-item" data-c=""><i class="cdot any"></i>any</div>'
                f'{rows}</div></span>')
    find = ('<div class="find">'
            '<span class="picker">'
            '<input id="find-name" placeholder="Find by name" autocomplete="off">'
            '<div id="name-list" class="picker-list"></div>'
            '</span>'
            + colour_pick("upper", "shirt")
            + colour_pick("lower", "trousers")
            + '<button onclick="clearFind()" style="padding:7px 12px;">Clear</button>'
            + '</div>')
    layout = ('<div class="layout-pick" title="cameras per row">'
              '<span class="lbl">▦</span>'
              + "".join(f'<button data-cols="{n}" onclick="setCols({n})">{n}</button>'
                        for n in (1, 2, 3, 4))
              + '</div>')
    body = (f'<h1 class="live-title">Live analytics '
            f'<span id="find-status" class="muted"></span>{layout}</h1>\n'
            f'<div class="fx fx-calm" aria-hidden="true">'
            f'<span class="fx-aurora"></span><span class="fx-vign"></span></div>'
            f'<div class="grid" id="live-grid" data-branch="{branch or "_all"}">'
            f'{stream_cards}</div>'
            if shown else
            '<h1>Live analytics</h1>'
            '<div class="muted">No cameras in this branch yet — add one on the '
            'Cameras page.</div>')
    return _page("Detect — live", _nav("live", find), body,
                 _STREAM_JS + _FIND_JS + _LAYOUT_JS + _ALERT_JS)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(branch: str = ""):
    # Пока филиалы не заведены, система работает как раньше — со всеми камерами.
    # Как только они появились, работать «вообще» уже не с чем: надо выбрать объект.
    if not branch and BRANCHES:
        return RedirectResponse("/", status_code=303)
    checks = "".join(
        f'<label class="chk"><input type="checkbox" id="m-{k}" checked> {k}</label>'
        for k in MODEL_KEYS)
    body = (f'''
<div class="fx fx-calm" aria-hidden="true">
  <span class="fx-aurora"></span><span class="fx-vign"></span>
</div>
<h1>Cameras</h1>
<div class="panel">
  <h2>Cameras</h2>
  <div id="cam-list"></div>
  <button onclick="toggleAdd()" id="add-btn" style="margin-top:12px; width:auto;">
    + Add camera</button>
  <div id="add-form" style="display:none; margin-top:12px;">
    <div class="add-grid">
      <label>IP address<input id="c-ip" placeholder="172.16.11.20"></label>
      <label>Password<input id="c-pass" type="password" placeholder="camera password"></label>
      <label>Name<input id="c-name" placeholder="optional"></label>
      <label>User<input id="c-user" value="admin"></label>
      <label>Port<input id="c-port" type="number" value="8000"></label>
      <label>Channel<input id="c-chan" type="number" value="1"></label>
    </div>
    <div class="muted" style="margin-top:10px;">Which models should run on this camera</div>
    <div class="checks">{checks}</div>
    <div style="margin-top:12px; display:flex; gap:8px; align-items:center;">
      <button onclick="addCamera()" id="add-go" style="width:auto;">Connect and add</button>
      <span id="add-msg" class="muted"></span>
    </div>
    <div class="muted" style="margin-top:8px;">The camera is contacted over the
    Hikvision SDK, exactly like the ones already running. It is written to the config
    only after it answers.</div>
  </div>
</div>''')
    return _page("Detect — cameras", _nav("cameras"), body, _CAMERA_JS)


_CAMERA_JS = """
function toggleAdd() {
  const f = document.getElementById('add-form');
  const shown = f.style.display !== 'none';
  f.style.display = shown ? 'none' : 'block';
  document.getElementById('add-btn').textContent = shown ? '+ Add camera' : 'Cancel';
}

const myBranch = new URLSearchParams(location.search).get('branch') || '';

async function loadCameras() {
  const [d, b] = await Promise.all([
    fetch('/api/cameras').then(r => r.json()),
    fetch('/api/branches').then(r => r.json()),
  ]).catch(() => [null, null]);
  if (!d) return;
  const all = (b && b.branches) || [];
  const opts = all.map(x => `<option value="${x.id}">${x.name}</option>`).join('');
  const nameOf = {};
  all.forEach(x => { nameOf[x.id] = x.name; });
  // В филиале — ТОЛЬКО его камеры. Раньше сюда же попадали ничьи, чтобы их можно было
  // подобрать, и получалось вот что: удалил филиал, завёл новый — а в нём уже стоят
  // камеры прежнего. Новый объект должен начинаться с пустого места; камеры к нему
  // добавляют заново. Осиротеть камере теперь тоже негде: вместе с филиалом удаляются
  // и они.
  const list = myBranch ? d.cameras.filter(c => c.branch === myBranch) : d.cameras;

  // Внутри филиала филиал не выбирают: он уже выбран, страницу открыли из него, и
  // список из одного правильного пункта только предлагал бы ошибиться.
  const branchCell = c => myBranch
    ? `<span class="br-tag here">${nameOf[c.branch] || c.branch}</span>`
    : `<select onchange="setBranch('${c.id}', this.value)" data-branch-of="${c.id}">
         <option value="">— none —</option>${opts}
       </select>`;

  document.getElementById('cam-list').innerHTML = `
    <table>
      <tr><th>Camera</th><th>Address</th><th>State</th><th>Branch</th>
          ${d.model_keys.map(k => `<th>${k}</th>`).join('')}<th></th></tr>
      ${list.map(c => `<tr${c.branch === myBranch || !myBranch ? '' : ' style="opacity:.55"'}>
        <td>${c.name}<div class="muted">${c.id}</div></td>
        <td>${c.ip}:${c.port} ch${c.channel}</td>
        <td>${c.connected ? '<span class="ok">live</span>'
                          : '<span class="bad">down</span>'}</td>
        <td>${branchCell(c)}</td>
        ${d.model_keys.map(k => `<td><input type="checkbox" ${c.models[k] ? 'checked' : ''}
             onchange="setModels('${c.id}', this)"
             data-cam="${c.id}" data-model="${k}"></td>`).join('')}
        <td style="text-align:right;"><button class="danger row-del"
            onclick="removeCamera('${c.id}', '${c.name}', '${c.ip}')">Remove</button></td>
      </tr>`).join('')}
    </table>`;

  // Значения списков ставим после вставки: в разметке они сбрасываются на первый пункт.
  list.forEach(c => {
    const sel = document.querySelector(`[data-branch-of="${c.id}"]`);
    if (sel) sel.value = c.branch || '';
  });
}

async function setBranch(camId, branchId) {
  const r = await fetch(`/api/cameras/${camId}/branch`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({branch: branchId}),
  }).then(r => r.json()).catch(() => ({ok: false}));
  if (!r.ok) alert(r.error || 'failed');
  loadCameras();
}

async function removeCamera(camId, name, ip) {
  if (!confirm(`Remove ${name} (${ip})?\n\n`
             + `The camera will be logged out of properly and taken out of the config. `
             + `Its video and detections stop; already recorded events and faces stay.\n\n`
             + `You can add it back with the same IP later.`)) return;
  const r = await fetch('/api/cameras/' + camId, {method: 'DELETE'})
    .then(r => r.json()).catch(() => ({ok: false, error: 'request failed'}));
  if (!r.ok) { alert(r.error || 'failed'); return; }
  location.reload();
}

async function setModels(camId, el) {
  const boxes = document.querySelectorAll(`input[data-cam="${camId}"]`);
  const models = {};
  boxes.forEach(b => models[b.dataset.model] = b.checked);
  const r = await fetch(`/api/cameras/${camId}/models`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({models}),
  }).then(r => r.json()).catch(() => ({ok: false}));
  if (!r.ok) { el.checked = !el.checked; alert(r.error || 'failed'); }
}

async function addCamera() {
  const msg = document.getElementById('add-msg');
  const btn = document.getElementById('add-go');
  const models = {};
  document.querySelectorAll('.checks input').forEach(
    b => models[b.id.slice(2)] = b.checked);

  const body = {
    ip: document.getElementById('c-ip').value.trim(),
    password: document.getElementById('c-pass').value,
    name: document.getElementById('c-name').value.trim(),
    username: document.getElementById('c-user').value.trim(),
    port: document.getElementById('c-port').value,
    channel: document.getElementById('c-chan').value,
    branch: myBranch,          // добавляем сразу в тот филиал, где стоим
    models,
  };
  if (!body.ip) { msg.innerHTML = '<span class="bad">enter the IP</span>'; return; }

  btn.disabled = true;
  msg.textContent = 'connecting to the camera…';
  const r = await fetch('/api/cameras', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  }).then(r => r.json()).catch(() => ({ok: false, error: 'request failed'}));
  btn.disabled = false;

  if (!r.ok) {
    msg.innerHTML = `<span class="bad">${r.error}</span>`;
    return;
  }
  msg.innerHTML = `<span class="ok">${r.camera.id} added</span> — reloading`;
  setTimeout(() => location.reload(), 1200);
}

setInterval(loadCameras, 5000);
loadCameras();
"""


# Поиск человека по имени: оператор набирает имя и попадает на ту камеру, где человек
# сейчас. Источник — та же таблица закреплений, что рисует имя над головой, поэтому
# страница и кадр не могут разойтись. Слежение продолжается: человек переходит в другую
# комнату — подсветка уходит за ним.
_FIND_JS = """
let findName = '';
let findCam = null;

// Свой список вместо <datalist>: у встроенного нельзя ограничить высоту — он тянулся
// до низа страницы, — и после выбора имени он сжимается до одной строки, так что взять
// другого человека можно было только очистив поле.
let allNames = [];
const nameInput = document.getElementById('find-name');
const nameList = document.getElementById('name-list');

fetch('/api/names').then(r => r.json())
  .then(d => { allNames = d.names || []; })
  .catch(() => {});

function drawNames(filter) {
  const f = (filter || '').trim().toLowerCase();
  const shown = f ? allNames.filter(n => n.toLowerCase().includes(f)) : allNames;
  nameList.innerHTML = shown.length
    ? shown.map(n => `<div class="picker-item" data-name="${n}">${n}</div>`).join('')
    : '<div class="picker-item muted">no match</div>';
  nameList.style.display = 'block';
}

function hideNames() { nameList.style.display = 'none'; }

// По клику показываем ВЕСЬ список и выделяем текст: набранное имя заменится само,
// и очищать поле, чтобы выбрать другого, больше не нужно.
nameInput.addEventListener('focus', () => { nameInput.select(); drawNames(''); });
nameInput.addEventListener('click', () => drawNames(''));

nameList.addEventListener('mousedown', e => {
  const item = e.target.closest('.picker-item');
  if (!item || !item.dataset.name) return;
  e.preventDefault();
  nameInput.value = item.dataset.name;
  findName = item.dataset.name.toLowerCase();
  findCam = null;
  hideNames();
  nameInput.blur();
  followPerson();
});

document.addEventListener('mousedown', e => {
  if (!e.target.closest('.picker')) hideNames();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') hideNames();
});

nameInput.addEventListener('input', e => {
  findName = e.target.value.trim().toLowerCase();
  findCam = null;
  drawNames(e.target.value);
  if (!findName) clearHighlight();
  followPerson();
});

// Выбор цвета. Значение по-прежнему лежит в поле find-upper/find-lower, поэтому
// весь поиск ниже не заметил подмены <select> на свой список.
function colourPick(key, label) {
  const btn = document.getElementById(key + '-btn');
  const list = document.getElementById(key + '-list');
  const val = document.getElementById('find-' + key);
  if (!btn) return;
  const close = () => { list.style.display = 'none'; };

  btn.addEventListener('click', ev => {
    ev.stopPropagation();
    document.querySelectorAll('.cpick .picker-list').forEach(
      l => { if (l !== list) l.style.display = 'none'; });
    list.style.display = list.style.display === 'block' ? 'none' : 'block';
  });
  list.addEventListener('click', ev => {
    const item = ev.target.closest('.picker-item');
    if (!item) return;
    setColour(key, label, item.dataset.c);
    close();
    findCam = null;
    followPerson();
  });
  document.addEventListener('click', close);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
  val.dataset.label = label;
}

function setColour(key, label, colour) {
  const val = document.getElementById('find-' + key);
  const dot = document.getElementById(key + '-dot');
  const txt = document.getElementById(key + '-label');
  const list = document.getElementById(key + '-list');
  val.value = colour || '';
  txt.textContent = colour ? `${label}: ${colour}` : `${label}: any`;
  const src = list.querySelector(`.picker-item[data-c="${colour}"] .cdot`);
  dot.style.background = colour && src ? src.style.background : 'transparent';
  dot.classList.toggle('any', !colour);
  list.querySelectorAll('.picker-item').forEach(
    i => i.classList.toggle('on', i.dataset.c === (colour || '')));
}

colourPick('upper', 'shirt');
colourPick('lower', 'trousers');

function clearFind() {
  hideNames();
  document.getElementById('find-name').value = '';
  setColour('upper', 'shirt', '');
  setColour('lower', 'trousers', '');
  findName = '';
  findCam = null;
  clearHighlight();
  document.getElementById('find-status').textContent = '';
}

function clearHighlight() {
  document.querySelectorAll('.cam.found').forEach(c => c.classList.remove('found'));
}

function highlight(cams) {
  clearHighlight();
  for (const c of cams) {
    const card = document.getElementById('cam-' + c);
    if (card) card.classList.add('found');
  }
  // Прокручиваем только при смене камеры: иначе страница дёргалась бы каждую секунду.
  const first = cams[0] || null;
  if (first && findCam !== first) {
    const card = document.getElementById('cam-' + first);
    if (card) card.scrollIntoView({behavior: 'smooth', block: 'center'});
  }
  findCam = first;
}

async function followPerson() {
  const status = document.getElementById('find-status');
  const upper = document.getElementById('find-upper').value;
  const lower = document.getElementById('find-lower').value;

  if (!findName && !upper && !lower) {
    clearHighlight();
    findCam = null;
    status.textContent = '';
    return;
  }

  // Имя точнее одежды, поэтому если оно задано — оно и решает.
  if (findName) {
    const d = await fetch('/api/whereis').then(r => r.json()).catch(() => null);
    if (!d) return;
    const hit = (d.people || []).find(p => p.label.toLowerCase() === findName);
    if (!hit) {
      clearHighlight();
      findCam = null;
      status.innerHTML = `— <span class="warn">${findName}</span> is not on any camera right now`;
      return;
    }
    highlight([hit.camera_id]);
    status.innerHTML = `— <span class="ok">${hit.label}</span> on ${hit.camera_id}` +
                       ` (match ${(hit.similarity * 100).toFixed(0)}%)`;
    return;
  }

  const q = new URLSearchParams({upper, lower});
  const d = await fetch('/api/appearance?' + q).then(r => r.json()).catch(() => null);
  if (!d) return;

  const want = [upper ? upper + ' shirt' : null, lower ? lower + ' trousers' : null]
    .filter(Boolean).join(' + ');

  if (!d.matches.length) {
    clearHighlight();
    findCam = null;
    status.innerHTML = `— nobody in <span class="warn">${want}</span> right now`
                     + ` <span class="muted">(${d.people.length} people on camera)</span>`;
    return;
  }

  highlight(d.cameras);
  const who = d.matches.map(m => (m.name ? m.name + ' ' : '') + '@' + m.camera_id).join(', ');
  status.innerHTML = `— <span class="ok">${d.matches.length}</span> in ${want}: ${who}`;
}

setInterval(followPerson, 1000);
"""


# Страница лиц отдельно от камер: видео и галерея нужны разным людям в разное время, а
# четыре MJPEG-потока рядом со списком мешали и грузили канал впустую.
def _faces_body(global_base: bool, show: str = "named") -> str:
    """Одна и та же страница на два адреса.

    /database — вся база; /faces?branch=… — только те, кого поймали камеры этого
    филиала. Разделять было бы двумя копиями редактора карточек, а он один и тот же:
    человек-то общий, филиал лишь отвечает, где его видели.

    В базе к тому же две вкладки. Названные и безымянные — разная работа: первых
    ищут и правят, вторых просматривают пачкой и либо называют, либо стирают. В одном
    списке безымянных всегда на порядок больше, и названные в них просто тонут.
    """
    title = "People database" if global_base else "Faces"
    unnamed = show == "unnamed"
    base = "/database" if global_base else "/faces"
    # Ни подписи под заголовком, ни строки со статистикой: счётчики стоят прямо на
    # вкладках («Named 27», «Unnamed 491»), а что такое названный и безымянный,
    # видно по самим карточкам.
    hint = ""

    def tab(query, label, key):
        cls = ' class="active"' if key == show else ""
        # Филиал живёт в адресе, и вкладка обязана его нести: без этого щелчок по
        # «Unnamed» молча выбрасывал бы оператора из филиала в общую базу.
        return (f'<a href="{base}{query}" data-keep-branch{cls}>{label}'
                f'<span class="cnt" id="cnt-{key}"></span></a>')

    # Фильтр по филиалу — только в общей базе. На странице филиала он был бы вторым
    # ответом на тот же вопрос: филиал там уже выбран адресом.
    # Свой список, а не <select>. У встроенного выпадающая часть рисуется системой:
    # синяя подсветка Windows поверх тёмной страницы, и сделать с ней нельзя ничего.
    # Такой же собственный список уже стоит на «живой» странице — берём его, а не
    # заводим второй вид выпадашки на тот же вопрос.
    pick = ('<span class="picker db-pick" id="db-pick">'
            '<button type="button" class="pick-btn" id="db-pick-btn"'
            ' title="show only people seen at one branch">'
            '<span id="db-pick-label">All branches</span>'
            '<span class="pick-caret">▾</span></button>'
            '<div class="picker-list" id="db-pick-list"></div>'
            '</span>' if global_base else "")
    tabs = ('<div class="subnav">'
            + tab("", "Named", "named")
            + tab("?show=unnamed", "Unnamed", "unnamed")
            + pick + '</div>')
    purge = ('<button class="danger" onclick="purgeUnnamed()"'
             ' style="width:auto; padding:6px 14px;">Delete all unnamed</button>'
             if unnamed and global_base else "")
    # Та же сцена, что на входе, но тише: радар и сетка пола сюда не идут. Эта
    # страница листается, а неподвижный пол под уезжающим списком выглядит поломкой.
    scene = ('<div class="fx fx-calm" aria-hidden="true">'
             '<span class="fx-aurora"></span>'
             '<span class="fx-grain"></span>'
             '<span class="fx-vign"></span>'
             '</div>')
    return f"""
{scene}
<h1 class="live-title">{title}{purge}</h1>
{tabs}
{f'<div class="muted" id="faces-hint">{hint}</div>' if hint else ''}
<div id="faces" style="margin-top:12px;"></div>
<div id="face-detail" class="modal" hidden onclick="detailBackdrop(event)">
  <div class="modal-box" role="dialog" aria-modal="true">
    <div class="modal-head">
      <h3 id="detail-title"></h3>
      <button class="modal-x" onclick="closeDetail()" title="close">×</button>
    </div>
    <div id="detail-shots" class="modal-body"></div>
  </div>
</div>
"""

_FACES_JS = """
function fmtTime(ts) {
  if (!ts) return '-';
  return new Date(ts * 1000).toLocaleTimeString();
}

function fmtDay(ts) {
  if (!ts) return '-';
  const d = new Date(ts * 1000);
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}

function fmtAgo(ts) {
  if (!ts) return '-';
  const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  if (s < 86400) return Math.floor(s / 3600) + 'h ago';
  return Math.floor(s / 86400) + 'd ago';
}

// Что оператор уже набрал, но ещё не сохранил: id -> {поле: текст}.
const drafts = {};

document.addEventListener('input', e => {
  const m = /^f-(name|pos|phone)-(\\d+)$/.exec(e.target.id || '');
  if (!m) return;
  (drafts[m[2]] = drafts[m[2]] || {})[m[1]] = e.target.value;
});

function typing() {
  const a = document.activeElement;
  return a && a.tagName === 'INPUT' && /^f-(name|pos|phone)-\\d+$/.test(a.id || '');
}

const esc = v => String(v === undefined || v === null ? '' : v)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// Снимок и текст — РАЗНЫЕ куски карточки, и обновляются они порознь.
// «last 16s ago» меняется на каждом опросе, а снимок — раз в несколько дней. Пока
// они лежали в одной строке, любое изменение времени пересоздавало <img>, браузер
// заново тянул картинку, и вся страница моргала раз в три секунды.
function photoHTML(p) {
  return (p.crop ? `<img src="/api/faces/crop/${p.crop}" alt="face">`
                 : '<div class="muted" style="padding:20px;">no image</div>')
       + `<span class="shot-badge">${p.samples}</span>`;
}

function infoHTML(p) {
  return `
    <div class="person-name ${p.named ? 'ok' : 'warn'}">${esc(p.label)}</div>
    <span class="tag ${p.named ? 'tag-known' : 'tag-unknown'}">
      ${p.named ? (p.person_type === 'visitor' ? 'VISITOR' : 'KNOWN') : 'UNKNOWN'}
    </span>
    <div class="muted" style="margin-top:6px;">${p.seen_count} seen · last ${fmtAgo(p.last_seen)}</div>
    <div class="muted">FACE ID #${p.id}${p.last_camera ? ' · ' + esc(p.last_camera) : ''}</div>
    ${branchHTML(p)}`;
}

// Где этого человека видели. Только в общей базе: внутри филиала метка стояла бы на
// каждой карточке и всегда одна и та же — она отвечала бы на вопрос, который задан
// самим адресом страницы.
function branchHTML(p) {
  if (myBranch) return '';
  const bs = p.branches || [];
  if (!bs.length) return '';
  return '<div class="br-tags">' + bs.map(b =>
    `<span class="br-tag${b.branch === myBranch ? ' here' : ''}" `
    + `title="${esc(b.count)} sightings, last ${fmtDay(b.last_seen)}">`
    + `${esc(b.name)}</span>`).join('') + '</div>';
}

function seenHTML(p) {
  return `<div><span class="muted">FIRST SEEN</span><br>${fmtDay(p.first_seen)}</div>
          <div><span class="muted">LAST SEEN</span><br>${fmtDay(p.last_seen)}</div>`;
}

// Цвет для квадратика рядом с названием: читать «brown x51» глазами дольше, чем увидеть.
const SWATCH = {
  black:'#111', gray:'#888', white:'#eee', beige:'#d6bd96', red:'#d33', orange:'#e88020',
  brown:'#8a5a2b', yellow:'#d8c832', green:'#3a9d3a', cyan:'#3ab7c8',
  blue:'#3a6fd8', purple:'#8a52c8', pink:'#e07ab0', unknown:'#444',
};
const GROUP_OF = {
  black:'dark', gray:'gray', white:'light', beige:'beige', red:'warm', orange:'warm', brown:'warm',
  yellow:'warm', pink:'warm', blue:'cool', cyan:'cool', purple:'cool', green:'green',
};

function swatch(colour, text, dim) {
  return `<span class="sw${dim ? ' dim' : ''}" title="${colour}">`
       + `<i style="background:${SWATCH[colour] || '#444'}"></i>${text}</span>`;
}

// Память об одежде за сегодня. Показываем и то, что зачтено, и то, что отброшено:
// иначе не понять, почему человеку отказали в имени.
function clothHTML(p, mem, now) {
  if (!p.named) return '';
  const parts = [];

  if (now) {
    const bits = [];
    if (now.upper !== 'unknown') bits.push(swatch(now.upper, 'shirt'));
    if (now.lower !== 'unknown') bits.push(swatch(now.lower, 'trousers'));
    parts.push(`<div class="cloth-row cloth-now">NOW on ${now.camera_id}: `
             + (bits.join('') || 'colour not readable') + '</div>');
  }

  if (!mem || !mem.confirmed) {
    const n = mem ? mem.sightings : 0;
    const need = mem ? mem.sightings_needed : 3;
    const shots = (mem && mem.seen || [])
      .map(s => swatch(s.upper, '') + swatch(s.lower, '') + ' ').join('');
    parts.push(`<div class="cloth-row muted">not confirmed today — ${n} of ${need} `
             + `sightings ${shots}</div>`);
    return parts.join('');
  }

  parts.push('<div class="cloth-row">CONFIRMED TODAY: '
           + swatch(mem.upper, 'shirt') + swatch(mem.lower, 'trousers') + '</div>');
  return parts.join('');
}

function makeCard(p) {
  const el = document.createElement('div');
  el.className = 'person-card';
  el.dataset.pid = p.id;
  el.innerHTML = `
    <div class="person-head" onclick="openDetail(${p.id})" title="Show every shot of this person">
      <div class="person-photo"></div>
      <div class="person-info"></div>
    </div>
    <div class="person-seen"></div>
    <div class="person-cloth"></div>
    <div class="person-edit">
      <input id="f-name-${p.id}"  placeholder="Name">
      <input id="f-pos-${p.id}"   placeholder="Position">
      <input id="f-phone-${p.id}" placeholder="Phone">
    </div>
    <div class="card-buttons">
      <button onclick="savePerson(${p.id})">Save</button>
      <button class="danger" onclick="deletePerson(${p.id})">Delete</button>
    </div>
    <div id="face-msg-${p.id}" class="muted"></div>`;
  return el;
}

// Пишем разметку, только когда она изменилась. Один и тот же innerHTML, присвоенный
// заново, для браузера не «то же самое», а «разобрать заново»: узлы пересоздаются,
// картинки перезапрашиваются, анимации начинаются с нуля.
function setHTML(el, html) {
  if (el && el.innerHTML !== html) el.innerHTML = html;
}

// Поле обновляем, только если оператор его не держит и ничего в нём не набрал:
// иначе обновление затрёт наполовину введённое имя.
function setField(id, field, value) {
  const el = document.getElementById(`f-${field}-${id}`);
  if (!el || el === document.activeElement) return;
  if ((drafts[id] || {})[field] !== undefined) return;
  const v = value || '';
  if (el.value !== v) el.value = v;
}

// Раньше здесь заменялся весь innerHTML списка. Пока список перерисовывался, карточка
// исчезала прямо под курсором: оператор целился в поле «Name», в этот момент камера
// ловила нового человека, список перестраивался — и клик попадал уже в другой элемент.
// Поэтому карточки теперь живут постоянно, а обновляется в них только текст.
function renderFaces(persons, clothing, live) {
  const box = document.getElementById('faces');
  const seen = new Set();

  for (const p of persons) {
    seen.add(String(p.id));
    let card = box.querySelector(`.person-card[data-pid="${p.id}"]`);
    if (!card) {
      card = makeCard(p);
      box.appendChild(card);
    }
    card.classList.toggle('known', !!p.named);
    // Присваивать innerHTML вслепую нельзя. Внутри .person-head лежит <img> со
    // снимком: перезапись пересоздаёт его, браузер заново берёт картинку и карточка
    // моргает — раз в три секунды, на всём списке сразу. Пишем, только если текст
    // действительно другой.
    setHTML(card.querySelector('.person-photo'), photoHTML(p));
    setHTML(card.querySelector('.person-info'), infoHTML(p));
    setHTML(card.querySelector('.person-seen'), seenHTML(p));
    setHTML(card.querySelector('.person-cloth'),
            clothHTML(p, clothing[p.id], live[(p.label || '').toLowerCase()]));
    setField(p.id, 'name',  p.named ? p.label : '');
    setField(p.id, 'pos',   p.position);
    setField(p.id, 'phone', p.phone);
  }

  for (const card of [...box.children]) {
    if (!seen.has(card.dataset.pid)) card.remove();
  }

  // Порядок меняем только когда никто ничего не набирает: перестановка карточек
  // тоже уводит поле из-под курсора.
  //
  // И только если он ДЕЙСТВИТЕЛЬНО другой. appendChild уже стоящего в списке узла —
  // это перемещение, а перемещение перезапускает на элементе CSS-анимации. Раньше
  // цикл гонял так все карточки каждые три секунды, и весь список заново проигрывал
  // своё появление: это и выглядело как перезагрузка страницы.
  if (!typing()) {
    const want = persons.map(p => String(p.id));
    const have = [...box.children].map(c => c.dataset.pid);
    if (want.length !== have.length || want.some((id, i) => id !== have[i])) {
      for (const p of persons) {
        const card = box.querySelector(`.person-card[data-pid="${p.id}"]`);
        if (card) box.appendChild(card);
      }
    }
  }

  if (!box.children.length) {
    box.innerHTML = '<div class="muted">'
      + ((myBranch || dbBranch)
         ? 'Nobody in this branch under this tab yet.'
         : 'No faces captured yet')
      + '</div>';
  } else if (box.firstElementChild && !box.firstElementChild.dataset.pid) {
    box.firstElementChild.remove();
  }
}

// Что показывать, решает адрес, а не переключатель: /database — вся база, филиал —
// только свои. Переключателя нет намеренно: два адреса и так отвечают на разные вопросы,
// а тумблер поверх них означал бы, что страница филиала иногда показывает чужих.
const myBranch = new URLSearchParams(location.search).get('branch') || '';
// Вкладка: названные или безымянные. Одинаково и в базе, и в филиале — работа с ними
// разная везде, а безымянных всегда на порядок больше.
const faceShow =
  new URLSearchParams(location.search).get('show') === 'unnamed' ? 'unnamed' : 'named';

// Фильтр по филиалу в общей базе. На странице филиала его нет — там филиал задан
// адресом. Выбор запоминается: это привычка рабочего места, а не разовый вопрос.
let dbBranch = '';
try { dbBranch = localStorage.getItem('db-branch') || ''; } catch (e) {}
let branchOptsDone = false;

function fillBranchPick(list) {
  const box = document.getElementById('db-pick-list');
  if (!box || branchOptsDone) return;
  // Рисуем один раз: перерисовка каждые три секунды закрывала бы раскрытый список
  // прямо под курсором.
  branchOptsDone = true;
  if (!list.some(b => b.id === dbBranch)) dbBranch = '';

  const rows = [{id: '', name: 'All branches'}].concat(list);
  box.innerHTML = rows.map(b =>
    `<div class="picker-item${b.id === dbBranch ? ' on' : ''}" data-b="${b.id}">`
    + `${esc(b.name)}</div>`).join('');
  const label = document.getElementById('db-pick-label');
  const cur = rows.find(b => b.id === dbBranch);
  if (cur) label.textContent = cur.name;

  const btn = document.getElementById('db-pick-btn');
  const close = () => { box.style.display = 'none'; };
  btn.addEventListener('click', ev => {
    ev.stopPropagation();
    box.style.display = box.style.display === 'block' ? 'none' : 'block';
  });
  box.addEventListener('click', ev => {
    const item = ev.target.closest('.picker-item');
    if (!item) return;
    dbBranch = item.dataset.b;
    try { localStorage.setItem('db-branch', dbBranch); } catch (e) {}
    box.querySelectorAll('.picker-item').forEach(
      i => i.classList.toggle('on', i.dataset.b === dbBranch));
    label.textContent = item.textContent;
    close();
    document.getElementById('faces').innerHTML = '';
    refreshFaces();
  });
  document.addEventListener('click', close);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
}

async function refreshFaces() {
  // Один параметр вместо двух: пустой филиал и означает «все». Отдельный scope только
  // повторял бы это же другим словом и позволял им разойтись.
  const q = new URLSearchParams({branch: myBranch || dbBranch});
  q.set('named', faceShow === 'named' ? '1' : '0');
  const [faces, cloth, appear] = await Promise.all([
    fetch('/api/faces?' + q).then(r => r.json()),
    fetch('/api/clothing').then(r => r.json()).catch(() => ({people: []})),
    fetch('/api/appearance').then(r => r.json()).catch(() => ({people: []})),
  ]);

  fillBranchPick(faces.branches || []);

  // Цифры на вкладках — по тому же отбору, что и список. Взять их из общей статистики
  // значило бы написать в филиале «27 named», когда сюда заходили трое.
  const c = faces.counts || {};
  const cn = document.getElementById('cnt-named');
  const cu = document.getElementById('cnt-unnamed');
  if (cn && cn.textContent !== String(c.named || 0)) cn.textContent = c.named || 0;
  if (cu && cu.textContent !== String(c.unnamed || 0)) cu.textContent = c.unnamed || 0;

  const byId = {};
  for (const c of (cloth.people || [])) byId[c.person_id] = c;
  const live = {};
  for (const a of (appear.people || [])) if (a.name) live[a.name.toLowerCase()] = a;

  renderFaces(faces.persons || [], byId, live);
  if (openPersonId !== null) await loadDetail(openPersonId);
}

// Enter в любом из четырёх полей = нажать Save для этой карточки.
document.addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  const m = /^f-(?:name|pos|phone)-(\\d+)$/.exec(e.target.id || '');
  if (m) { e.preventDefault(); savePerson(Number(m[1])); }
});

async function purgeUnnamed() {
  const s = await fetch('/api/faces').then(r => r.json()).catch(() => null);
  const n = s ? (s.stats.persons - s.stats.named) : '?';
  if (!confirm(`Delete all ${n} cards that still have no name?\n\n`
             + `Their face vectors and images are removed for good. `
             + `Named people are not touched.\n\n`
             + `Anyone still around will be picked up again as a new card.`)) return;
  const r = await fetch('/api/faces/purge-unnamed', {method: 'POST'})
    .then(r => r.json()).catch(() => ({ok: false}));
  if (!r.ok) { alert('failed'); return; }
  alert(`Removed ${r.persons} people, ${r.samples} vectors, ${r.files} images.`);
  location.reload();
}

// Удаление необратимо и уносит с собой снимки с диска, поэтому спрашиваем и называем,
// кого именно удаляем: карточки перестраиваются на ходу, и промахнуться легко.
async function deletePerson(id) {
  const card = document.querySelector(`.person-card[data-pid="${id}"]`);
  const name = card ? card.querySelector('.person-name').textContent.trim() : ('#' + id);
  const shots = card ? card.querySelector('.shot-badge').textContent.trim() : '?';
  if (!confirm(`Delete ${name} for good?\n\n`
             + `${shots} face samples and their images will be removed from disk, `
             + `along with today's clothing record. This cannot be undone.\n\n`
             + `The person will be picked up again as a new unnamed card.`)) return;

  const msg = document.getElementById('face-msg-' + id);
  if (msg) msg.innerHTML = 'deleting…';
  const r = await fetch('/api/faces/' + id, {method: 'DELETE'})
    .then(r => r.json()).catch(() => ({ok: false}));
  if (!r.ok) {
    if (msg) msg.innerHTML = '<span class="bad">delete failed</span>';
    return;
  }
  delete drafts[id];
  if (card) card.remove();
  await refreshFaces();
}

async function savePerson(id) {
  const body = {
    name:       (document.getElementById('f-name-'  + id).value || '').trim(),
    position:   (document.getElementById('f-pos-'   + id).value || '').trim(),
    phone:      (document.getElementById('f-phone-' + id).value || '').trim(),
  };
  const msg = document.getElementById('face-msg-' + id);
  const res = await fetch('/api/faces/' + id + '/edit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!data.ok) {
    msg.innerHTML = '<span class="bad">failed</span>';
  } else if (data.merged_from) {
    msg.innerHTML = `<span class="ok">merged into ${data.label}</span> — ${data.samples} shots kept`;
  } else {
    msg.innerHTML = '<span class="ok">saved</span>';
  }
  delete drafts[id];
  if (document.activeElement) document.activeElement.blur();
  await refreshFaces();
}

let openPersonId = null;

// Всплывающее окно, а не блок под списком. Раньше клик по человеку уводил страницу
// вниз, к отдельному разделу: список оставался наверху, и чтобы вернуться к нему,
// приходилось прокручивать обратно. Теперь снимки открываются поверх, а список
// остаётся на месте — закрыл окно и продолжил с той же карточки.
async function openDetail(id) {
  openPersonId = id;
  const box = document.getElementById('face-detail');
  box.hidden = false;
  document.body.classList.add('modal-open');
  document.getElementById('detail-shots').innerHTML = '<div class="muted">loading…</div>';
  await loadDetail(id);
}

function closeDetail() {
  openPersonId = null;
  document.getElementById('face-detail').hidden = true;
  document.body.classList.remove('modal-open');
}

// Клик по затемнению закрывает, клик внутри окна — нет.
function detailBackdrop(ev) {
  if (ev.target.id === 'face-detail') closeDetail();
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && openPersonId !== null) closeDetail();
});

// Снимок хранится ОДИН на человека — самый чёткий, остальные стираются, чтобы диск не
// рос. Значит из сорока засечек картинка есть у одной, и рисовать сорок карточек, из
// которых тридцать девять пустые, бессмысленно: окно было стеной из «no image».
// Поэтому картинки — карточками, а все засечки без картинки — строчками таблицы:
// ничего не потеряно, но видно то, что действительно есть.
async function loadDetail(id) {
  const d = await fetch('/api/faces/' + id + '/samples').then(r => r.json());
  const all = d.samples || [];
  const shots = all.filter(s => s.frame || s.crop);

  document.getElementById('detail-title').textContent =
    `${d.label} — ${all.length} sighting${all.length === 1 ? '' : 's'}`
    + `, ${shots.length} with an image`;

  const cards = shots.map(s => `
    <div class="shot-card">
      ${s.frame ? `<img src="/api/faces/crop/${s.frame}" alt="frame" class="shot-frame">`
                : `<img src="/api/faces/crop/${s.crop}" alt="shot" class="shot-frame">`}
      <div class="shot-meta">
        ${s.crop ? `<img src="/api/faces/crop/${s.crop}" alt="face" class="shot-thumb">` : ''}
        <div>
          <div><strong>${s.camera_id || '-'}</strong> · ${fmtDay(s.created_at)}</div>
          <div class="muted">
            ${s.similarity != null ? `matched ${(Number(s.similarity) * 100).toFixed(0)}%`
                                   : '<span class="warn">first shot</span>'}
            · det ${Number(s.det_score || 0).toFixed(2)}
            ${s.bbox ? ` · ${s.bbox[2] - s.bbox[0]}x${s.bbox[3] - s.bbox[1]} px` : ''}
          </div>
        </div>
      </div>
    </div>`).join('');

  const rows = all.map(s => `
    <tr>
      <td>${s.camera_id || '-'}</td>
      <td>${fmtDay(s.created_at)}</td>
      <td>${s.similarity != null ? (Number(s.similarity) * 100).toFixed(0) + '%'
                                 : '<span class="warn">first</span>'}</td>
      <td>${Number(s.det_score || 0).toFixed(2)}</td>
      <td>${s.frame || s.crop ? 'yes' : '<span class="muted">—</span>'}</td>
    </tr>`).join('');

  document.getElementById('detail-shots').innerHTML = all.length ? `
    ${cards ? `<div class="shot-grid">${cards}</div>`
            : '<div class="muted">No image kept for this person yet.</div>'}
    <h4 class="detail-h">Every sighting <span class="muted">${all.length}</span></h4>
    <div class="table-wrap">
      <table><tr><th>Camera</th><th>When</th><th>Match</th><th>Det</th><th>Image</th></tr>
      ${rows}</table>
    </div>` : '<div class="muted">No sightings recorded.</div>';
}

setInterval(refreshFaces, 3000);
refreshFaces();
"""


@app.get("/faces", response_class=HTMLResponse)
def faces_page(branch: str = "", show: str = "named"):
    if not branch and BRANCHES:
        return RedirectResponse("/", status_code=303)
    show = "unnamed" if show == "unnamed" else "named"
    return _page("Detect — faces", _nav("faces"), _faces_body(False, show),
                 _FACES_JS + _ALERT_JS)


@app.get("/database", response_class=HTMLResponse)
def database_page(show: str = "named"):
    """Общая база: все люди системы, названные и безымянные, с пометкой филиалов.

    Живёт рядом с выбором филиала, а не внутри него: она не про объект, а про людей,
    и вопрос «кого мы вообще знаем» задают до того, как выберут, с чем работать.
    """
    show = "unnamed" if show == "unnamed" else "named"
    return _page("Detect — people database", _nav("database"),
                 _faces_body(True, show), _FACES_JS + _ALERT_JS)


# Бездействие и перемещения. Порог оператор задаёт сам: «сколько человек может ничего
# не делать, прежде чем это станет записью» — вопрос не технический, а рабочий, и
# зашивать в код ответ на него нельзя.
_ACTIVITY_BODY = """
<div class="fx fx-calm" aria-hidden="true">
  <span class="fx-aurora"></span><span class="fx-vign"></span>
</div>
<h1>Activity</h1>
<div class="idle-bar">
  <h2>Idle alert</h2>
  <span class="unit-field" title="minutes of doing nothing before it counts as idle">
    <input id="idle-min" type="number" min="0.5" step="0.5">
    <span class="unit">min</span>
  </span>
  <span id="idle-saved" class="muted"></span>
</div>
<div id="idle-now"></div>

<!-- Одно окно на оба вопроса. Раньше это были две панели с почти одинаковыми
     таблицами: «за день» и «за период». Стоя рядом, они читались как повтор, хотя
     отвечают на разное — «что было сегодня» и «сколько наработал вообще». -->
<div class="panel" style="--n:0">
  <h2 class="live-title">Work time
    <span id="work-note" class="muted" style="font-weight:400;"></span>
  </h2>
  <div class="subnav" id="work-tabs">
    <a href="#" data-tab="day" onclick="setWorkTab('day'); return false;">One day</a>
    <a href="#" data-tab="period" onclick="setWorkTab('period'); return false;">3 / 7 / 30 days</a>
    <span class="work-controls">
      <select id="day-pick"></select>
      <span class="layout-pick" id="period-pick" title="how many days back, today included">
        <button data-days="3" onclick="setPeriod(3)">3</button>
        <button data-days="7" onclick="setPeriod(7)">7</button>
        <button data-days="30" onclick="setPeriod(30)">30</button>
      </span>
    </span>
  </div>
  <div id="summary" class="table-wrap"></div>
  <div id="period" class="table-wrap"></div>
</div>

<div class="panel" style="--n:1">
  <h2>Idle episodes today <span id="idle-count" class="muted"></span></h2>
  <div id="idle-today" class="table-wrap"></div>
</div>

<div class="panel" style="--n:2">
  <h2>Records
    <button class="danger" onclick="clearActivity()"
            style="width:auto; float:right; padding:6px 14px;">Clear records</button>
  </h2>
  <div id="routes" style="margin-top:10px;"></div>
</div>
"""

_ACTIVITY_JS = """
const myBranch = new URLSearchParams(location.search).get('branch') || '';

// Таблицы перерисовываются раз в пять секунд, а меняются раз в несколько минут.
// Присвоенный заново innerHTML — это заново разобранная разметка: пропадает
// выделенный мышью текст и сбрасывается прокрутка внутри таблицы. Пишем, только
// когда действительно другое.
function setPanel(id, html) {
  const el = document.getElementById(id);
  if (el && el.innerHTML !== html) el.innerHTML = html;
}

function fmtClock(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}

function mins(sec) {
  if (sec == null) return '—';
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return m ? `${m}m ${s}s` : `${s}s`;
}

// Часы и минуты: за смену набегают часы, и «412.5 min» читать неудобно.
function hm(minutes) {
  if (!minutes) return '—';
  const h = Math.floor(minutes / 60), m = Math.round(minutes % 60);
  return h ? `${h}h ${m}m` : `${m}m`;
}

// Порог сохраняется сам, как только набран: раньше он жил только в поле ввода, и
// уход со страницы стирал набранное, хотя человек считал, что уже задал число.
let saveTimer = null;
document.addEventListener('input', e => {
  if (e.target.id !== 'idle-min') return;
  const msg = document.getElementById('idle-saved');
  msg.textContent = 'saving…';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const v = parseFloat(e.target.value);
    if (!(v > 0)) { msg.innerHTML = '<span class="bad">enter a number</span>'; return; }
    const r = await fetch('/api/idle/threshold', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({minutes: v}),
    }).then(r => r.json()).catch(() => ({ok: false}));
    msg.innerHTML = r.ok ? '<span class="ok">saved</span>'
                         : `<span class="bad">${r.error || 'failed'}</span>`;
  }, 600);
});

let thresholdFocused = false;
document.addEventListener('focusin', e => {
  if (e.target.id === 'idle-min') thresholdFocused = true;
});
document.addEventListener('focusout', e => {
  if (e.target.id === 'idle-min') thresholdFocused = false;
});

let pickedDay = '';
document.addEventListener('change', e => {
  if (e.target.id === 'day-pick') { pickedDay = e.target.value; refresh(); }
});

// Две вкладки одного окна: конкретный день и итог за 3/7/30 дней. Обе показывают
// рабочее время, но отвечают на разные вопросы, поэтому и таблицы разные — у периода
// есть «в среднем за день» и «дней на работе», которых у одного дня быть не может.
let workTab = 'day';
try { workTab = localStorage.getItem('work-tab') === 'period' ? 'period' : 'day'; }
catch (e) {}

function setWorkTab(t) {
  workTab = t;
  try { localStorage.setItem('work-tab', t); } catch (e) {}
  document.querySelectorAll('#work-tabs a').forEach(a => {
    a.classList.toggle('active', a.dataset.tab === t);
  });
  // Управление показываем то, которое к вкладке относится: выбор дня на вкладке
  // периода означал бы, что он на что-то там влияет.
  const day = document.getElementById('day-pick');
  const per = document.getElementById('period-pick');
  if (day) day.hidden = t !== 'day';
  if (per) per.hidden = t !== 'period';
  document.getElementById('summary').hidden = t !== 'day';
  document.getElementById('period').hidden = t !== 'period';
  refresh();
}

let periodDays = 7;
try { periodDays = Number(localStorage.getItem('period-days')) || 7; } catch (e) {}

function setPeriod(n) {
  periodDays = n;
  try { localStorage.setItem('period-days', n); } catch (e) {}
  refreshPeriod();
}

async function refreshPeriod() {
  document.querySelectorAll('#period-pick button').forEach(b => {
    b.classList.toggle('on', Number(b.dataset.days) === periodDays);
  });
  const d = await fetch('/api/period?days=' + periodDays)
    .then(r => r.json()).catch(() => null);
  if (!d) return;
  const range = document.getElementById('work-note');
  if (range && workTab === 'period') {
    const have = d.days_with_records;
    range.innerHTML = `— ${d.from} … ${d.to}`
      + (have < d.days
         ? ` · <span class="warn">only ${have} of these ${d.days} days`
           + ` ${have === 1 ? 'has' : 'have'} records</span>`
         : ` · ${have} days with records`);
  }

  const p = d.people || [];
  setPanel('period', p.length ? `
    <table>
      <tr><th>Person</th><th>Worked</th><th>Per day</th><th>Days</th>
          <th>On phone</th><th>Idle</th><th>Idle spells</th></tr>
      ${p.map(x => `<tr>
        <td>${x.label}</td>
        <td class="ok"><b>${hm(x.working_min)}</b></td>
        <td>${hm(x.avg_working_min)}</td>
        <td>${x.days}</td>
        <td>${hm(x.phone_min)}</td>
        <td class="warn">${hm(x.idle_min)}</td>
        <td>${x.idle_episodes ? `<span class="bad">${x.idle_episodes}</span>` : '—'}</td>
      </tr>`).join('')}
    </table>`
    : `<div class="muted">Nothing recorded in the last ${d.days} days.</div>`);
}

async function loadDays() {
  const d = await fetch('/api/days').then(r => r.json()).catch(() => null);
  if (!d) return;
  const days = d.days && d.days.length ? d.days : [d.today];
  if (!days.includes(d.today)) days.unshift(d.today);
  const sel = document.getElementById('day-pick');
  if (sel.options.length !== days.length) {
    sel.innerHTML = days.map(x =>
      `<option value="${x}">${x === d.today ? x + ' (today)' : x}</option>`).join('');
    sel.value = pickedDay || d.today;
  }
}

async function clearActivity() {
  // Переносы строк пишем как \\n: этот блок — обычная строка Python, и одиночный
  // \\n превратился бы в настоящий перенос прямо посреди JS-строки в кавычках.
  // Так уже случилось однажды: скрипт переставал разбираться целиком, и страница
  // молча оставалась пустой.
  const all = confirm('Clear ALL recorded movement, idle time and recognition events?\\n\\n'
                    + 'OK = everything ever recorded.\\n'
                    + 'Cancel = only today.\\n\\n'
                    + 'Faces and names are not touched. Recording continues from now.');
  const scope = all ? 'all' : 'today';
  const r = await fetch('/api/activity/clear?scope=' + scope, {method: 'POST'})
    .then(r => r.json()).catch(() => ({ok: false}));
  if (!r.ok) { alert('failed'); return; }
  alert(`Removed ${r.segments} intervals and ${r.events} recognition events (${scope}).`);
  refresh();
}

async function refresh() {
  await loadDays();
  const q = pickedDay ? '?day=' + pickedDay : '';
  const d = await fetch('/api/idle' + q).then(r => r.json()).catch(() => null);
  if (!d) return;
  if (!thresholdFocused) document.getElementById('idle-min').value = d.minutes;

  // Когда бездействующих нет, блок пуст: строка «никто не бездействует» занимала место
  // ровно в том случае, когда сообщать нечего.
  const now = d.now || [];
  setPanel('idle-now', !d.is_today
    ? `<div class="muted">${d.day} is a past day — see the totals below.</div>`
    : now.map(p => `
    <div class="person-card" style="margin-bottom:8px;">
      <div class="person-name bad">${p.label} — doing nothing for ${p.idle_min} min</div>
      <div class="muted">on ${p.camera_id} · since ${fmtClock(p.since)}</div>
    </div>`).join(''));

  const hist = d.today || [];
  document.getElementById('idle-count').textContent = `— ${hist.length} today`;
  setPanel('idle-today', hist.length ? `
    <table><tr><th>Person</th><th>Camera</th><th>From</th><th>To</th><th>Length</th></tr>
    ${hist.map(r => `<tr><td>${r.label}</td><td>${r.camera_id}</td>
      <td>${fmtClock(r.start_ts)}</td><td>${fmtClock(r.end_ts)}</td>
      <td>${mins(r.duration_sec)}</td></tr>`).join('')}</table>`
    : '<div class="muted">No finished idle episodes yet today.</div>');

  const sm = await fetch('/api/summary' + q).then(r => r.json()).catch(() => ({people: []}));
  const sp = sm.people || [];
  const note = document.getElementById('work-note');
  if (note && workTab === 'day') note.textContent = `— ${sm.day || ''}`;
  setPanel('summary', sp.length ? `
    <table>
      <tr><th>Person</th><th>Worked</th><th>On phone</th><th>Idle</th>
          <th>Did nothing ≥ ${sm.minutes} min</th><th>Longest</th></tr>
      ${sp.map(p => `<tr>
        <td>${p.label}</td>
        <td class="ok">${hm(p.working_min)}</td>
        <td>${hm(p.phone_min)}</td>
        <td class="warn">${hm(p.idle_min)}</td>
        <td>${p.idle_episodes ? `<span class="bad">${p.idle_episodes} times</span>` : '—'}</td>
        <td>${p.longest_idle_min ? hm(p.longest_idle_min) : '—'}</td>
      </tr>`).join('')}
    </table>`
    : `<div class="muted">Nothing recorded on ${sm.day || 'that day'}.</div>`);

  // Скрытую вкладку не дёргаем каждые пять секунд: за тридцать дней это тридцать
  // запросов к базе ради таблицы, на которую никто не смотрит.
  if (workTab === 'period') await refreshPeriod();

  // Списком маршрут больше не показываем: строка «showroom 14:13–14:13 (0m)» ничего
  // не объясняет, а то же самое на плане видно сразу. Отмотка дня живёт там.
  const mapLink = '/map?branch=' + encodeURIComponent(myBranch);
  setPanel('routes', `<div class="muted">Movement is on the floor map now — `
    + `<a href="${mapLink}">open it</a> and drag the time bar to replay the day.</div>`);
}

setInterval(refresh, 5000);
setWorkTab(workTab);        // расставляет вкладки и сам вызывает refresh()
"""


@app.get("/activity", response_class=HTMLResponse)
def activity_page(branch: str = ""):
    if not branch and BRANCHES:
        return RedirectResponse("/", status_code=303)
    return _page("Detect — activity", _nav("activity"), _ACTIVITY_BODY, _ACTIVITY_JS)


# ---------- План этажа ------------------------------------------------------------
# Одна картинка плана на филиал, поверх неё SVG: зоны, покрытие камер, живые точки.
# Три режима одной страницы: смотреть, рисовать зоны, привязывать камеры. Калибровка
# — пары точек «на снимке камеры / на плане», и то и другое кликом.
_MAP_BODY = """
<style>
.map-wrap { display:flex; gap:16px; align-items:flex-start; }
.map-stage { position:relative; flex:1; min-width:0; border-radius:12px; overflow:hidden;
             background:#0d1218; border:1px solid var(--line-2); }
.map-stage img { width:100%; display:block; user-select:none; -webkit-user-drag:none; }
.map-stage img + svg { position:absolute; inset:0; width:100%; height:100%; }
.map-stage.draw { cursor:crosshair; }
.map-side { width:340px; flex:none; }
.map-list { list-style:none; margin:10px 0 0; padding:0; font-size:13px; }
.map-list li { display:flex; align-items:center; gap:8px; padding:5px 0;
               border-bottom:1px solid var(--line); }
.map-list .x { margin-left:auto; padding:2px 8px; font-size:12px; }
.map-list li.pick { cursor:pointer; padding-left:6px; border-radius:6px; }
.map-list li.pick:hover { background:#16202c; }
.map-list li.pick.on { background:#243b55; }
.map-tools { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:0 0 10px; }
.map-tools button.on { background:#243b55; }
.map-hint { color:var(--muted); font-size:13px; }
/* Своей строкой: с max-width на самом блоке он влезает в первую и жмётся к Clear. */
.cam-sliders { flex-basis:100%; width:100%; display:flex; flex-direction:column;
               gap:7px; margin:6px 0 2px; }
.cam-sliders label { display:flex; align-items:center; gap:10px; font-size:13px;
                     color:var(--muted); max-width:460px; }
.cam-sliders label > span:first-child { width:64px; flex:none; }
.cam-sliders input[type=range] { flex:1; accent-color:var(--accent); }
.cam-sliders .val { width:48px; flex:none; text-align:right; color:var(--ink);
                    font-variant-numeric:tabular-nums; }

/* Чертёж рисуется на пустом поле: клетка даёт глазу масштаб, которого без метров нет. */
.wf-wall { fill:none; stroke:#9fb4c8; stroke-width:3; stroke-linejoin:round; }
.wf-room { fill:#4d9fe01f; stroke:#4d9fe0; stroke-width:2; }
.wf-furn { fill:#f5c4511f; stroke:#f5c451; stroke-width:2; }
.wf-door { fill:none; stroke:#3ddc84; stroke-width:4; stroke-linecap:round; }
.wf-draft { stroke-dasharray:7 5; }
.zone-name { fill:#cfe3f7; font-size:13px; font-weight:600; paint-order:stroke;
             stroke:#0b0f14; stroke-width:3px; pointer-events:none; }
.cam-fov { fill:#3ddc8414; stroke:none; pointer-events:none; }
.cam-dir { stroke:#ff9f43; stroke-width:2.5; pointer-events:none; }
.cam-mk { fill:#ff9f43; stroke:#0b0f14; stroke-width:1.5; cursor:grab; }
.cam-lbl { fill:#ffd9a8; font-size:11px; paint-order:stroke; stroke:#0b0f14;
           stroke-width:3px; pointer-events:none; }
.vx { fill:#f5c451; stroke:#0b0f14; stroke-width:1.5; cursor:grab; }
.vx:hover { fill:#fff; }

/* Наложение на кадр: поставленная человеком точка — закрашенная, прикидка — полая. */
.hnd { fill:#0b0f1499; stroke:#f5c451; stroke-width:2; cursor:grab; }
.hnd.set { fill:#f5c451; }
.hnd.ok { stroke:#3ddc84; fill:#3ddc84; }
.hnd.warn { stroke:#f5c451; fill:#f5c451; }
.hnd.bad { stroke:#ff6b6b; fill:#ff6b6b; }
.hnd.out { opacity:.2; }
.dot-live { fill:#f5c451; stroke:#0b0f14; stroke-width:1.5; }
.dot-live.named { fill:#3ddc84; }
.dot-name { fill:#fff; font-size:13px; font-weight:700; paint-order:stroke;
            stroke:#0b0f14; stroke-width:3px; pointer-events:none; }
.trail { fill:none; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round;
         opacity:.55; pointer-events:none; }
.trail.coarse { stroke-dasharray:8 6; opacity:.35; }

/* Полоса времени: перемотка дня прямо под планом. */
.tl { display:flex; align-items:center; gap:10px; margin-top:10px; flex-wrap:wrap; }
.tl input[type=range] { flex:1; min-width:220px; accent-color:var(--accent); }
.tl .clock { font-variant-numeric:tabular-nums; font-size:15px; font-weight:600;
             color:var(--ink); width:92px; text-align:center; }
.tl button.live { background:#1d3b2a; }
.tl .muted { font-size:12px; }
</style>
<div class="fx fx-calm" aria-hidden="true">
  <span class="fx-aurora"></span><span class="fx-vign"></span>
</div>
<h1>Floor map</h1>
<div class="subnav" id="map-tabs">
  <a href="#" data-mode="view" onclick="setMode('view'); return false;">Live</a>
  <a href="#" data-mode="plan" onclick="setMode('plan'); return false;">Plan</a>
  <a href="#" data-mode="align" onclick="setMode('align'); return false;">Align</a>
  <span id="map-status" class="muted" style="margin-left:12px;"></span>
</div>
<div id="map-tools" class="map-tools"></div>
<div id="map-root"></div>
"""

_MAP_JS = """
const myBranch = new URLSearchParams(location.search).get('branch') || '';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

let cfg = null;            // ответ /api/map
let mode = 'view';
let plan = null;           // рабочая копия чертежа
let dirty = false;         // нарисовано, но не сохранено
let tool = 'select';
let snap90 = true;
let draft = [];            // вершины рисуемой фигуры
let history = [];          // что добавляли по порядку, для Undo
let camPick = '';          // какую камеру ставим на чертёж
let camDraft = null;       // точка камеры, ждущая направления
let selCam = null;         // камера на вкладке Align
let align = null;          // ответ /api/map/align
let adjust = {};           // id точки чертежа -> [x, y] в кадре: поправки человека
let live = [];
let drag = null;
let snapStamp = Date.now();
let snapRatio = 9 / 16;

// Перемотка дня. rt === null — живой режим, иначе момент, который показываем.
let replay = null;
let rt = null;
let playing = false;
let speed = 10;
let replayDay = '';
let selPerson = null;         // чей след показываем: выбирают в списке справа
// Хвост в живом режиме копим из самих живых опросов (они идут раз в 0.7 с).
// Из следа за день его не собрать: тот перечитывается раз в полминуты и к
// десятисекундному окну всегда опаздывает.
const liveHist = new Map();
// Короткий хвост, а не путь за весь день: длинный след за полчаса накрывает весь
// план и по нему уже не видно, куда человек идёт сейчас.
const TRAIL_SEC = 10;
const REPLAY_GAP = 45;        // дольше не видели — человека на плане нет

// План рисуется в единицах 0..1000 по ширине, высота — по пропорции листа.
const W = 1000;
const EMPTY = () => ({aspect: 1.6, walls: [], rooms: [], furniture: [], doors: [], cams: {}});
const KINDS = ['walls', 'rooms', 'furniture', 'doors'];
const KIND_LABEL = {walls: 'wall', rooms: 'room', furniture: 'table', doors: 'door'};
const CAM_DIR_LEN = 0.12;   // длина стрелки направления, в долях ширины плана

function aspect() { return (plan && plan.aspect) || 1.6; }
function PH() { return W / aspect(); }
function sx(u) { return u * W; }
function sy(v) { return v * PH(); }
function ax(x) { return x * W; }
function ay(y) { return y * W * snapRatio; }
function camName(id) { const c = (cfg.cameras || []).find(c => c.id === id); return c ? c.name : id; }

// Угол и точка считаются в изотропных единицах: по вертикали доля 0..1 — это другое
// расстояние, чем по горизонтали, и без поправки стрелка смотрела бы мимо.
function angTo(c, p) {
  const a = aspect();
  return (Math.atan2((p[1] - c[1]) / a, p[0] - c[0]) * 180 / Math.PI + 360) % 360;
}
function isoTo(c, ang, dist) {
  const a = aspect();
  return [c[0] + Math.cos(ang) * dist, (c[1] / a + Math.sin(ang) * dist) * a];
}

// Докуда смотрит камера: до края листа вдоль стрелки. Повторяет guess_far на
// сервере — нарисованный сектор обязан обещать ровно то, что ляжет в кадре.
function guessFar(pos, dirDeg) {
  const g = (cfg && cfg.guess) || {far_min: 0.2, far_max: 1.5};
  const a = aspect(), cx = pos[0], cy = pos[1] / a;
  const th = dirDeg * Math.PI / 180, dx = Math.cos(th), dy = Math.sin(th);
  const ts = [];
  if (Math.abs(dx) > 1e-6) ts.push(((dx > 0 ? 1 : 0) - cx) / dx);
  if (Math.abs(dy) > 1e-6) ts.push(((dy > 0 ? 1 / a : 0) - cy) / dy);
  const pos_ts = ts.filter(t => t > 0);
  return Math.min(g.far_max, Math.max(g.far_min, pos_ts.length ? Math.min(...pos_ts) : g.far_max));
}

function setMode(m) {
  if (mode === 'plan' && m !== 'plan' && dirty
      && !confirm('The drawing is not saved. Leave anyway?')) return;
  mode = m;
  draft = []; camDraft = null; drag = null;
  document.querySelectorAll('#map-tabs a').forEach(a => a.classList.toggle('active', a.dataset.mode === m));
  if (m === 'align' && selCam) refit(); else render();
  if (m === 'view' && !replay) loadReplay();
}

function setTool(t) { tool = t; draft = []; camDraft = null; render(); }

async function load() {
  cfg = await fetch('/api/map?branch=' + encodeURIComponent(myBranch)).then(r => r.json());
  if (!dirty) plan = cfg.drawn ? copyPlan(cfg.plan) : EMPTY();
  render();
}

function copyPlan(p) {
  const out = EMPTY();
  out.aspect = p.aspect || 1.6;
  for (const k of KINDS) out[k] = (p[k] || []).map(o => ({...o, points: o.points.map(q => [...q])}));
  for (const [k, v] of Object.entries(p.cams || {})) {
    // Угол и дальность копируем наравне с позицией: без этого после сохранения
    // страница перечитывает план, теряет их, и следующий Save стирает их в конфиге.
    const c = {pos: [...v.pos], dir: v.dir || 0};
    if (v.fov != null) c.fov = v.fov;
    if (v.far != null) c.far = v.far;
    out.cams[k] = c;
  }
  return out;
}

// ---------- отрисовка ----------
function poly(points) { return points.map(p => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' '); }
function centroid(points) {
  let x = 0, y = 0;
  for (const p of points) { x += p[0]; y += p[1]; }
  return [x / points.length, y / points.length];
}

function planSvg() {
  let s = `<defs><pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
    <path d="M50 0H0V50" fill="none" stroke="#1b2733" stroke-width="1"></path></pattern></defs>
    <rect x="0" y="0" width="${W}" height="${PH()}" fill="url(#grid)"></rect>`;
  for (const o of plan.rooms) {
    s += `<polygon class="wf-room" points="${poly(o.points)}"></polygon>`;
    const c = centroid(o.points);
    if (o.name) s += `<text class="zone-name" x="${sx(c[0])}" y="${sy(c[1])}" text-anchor="middle">${esc(o.name)}</text>`;
  }
  for (const o of plan.furniture) s += `<polygon class="wf-furn" points="${poly(o.points)}"></polygon>`;
  for (const o of plan.walls) s += `<polyline class="wf-wall" points="${poly(o.points)}"></polyline>`;
  for (const o of plan.doors) s += `<polyline class="wf-door" points="${poly(o.points)}"></polyline>`;

  const g = (cfg && cfg.guess) || {fov: 90};
  for (const [id, c] of Object.entries(plan.cams)) {
    const fov = c.fov || g.fov, far = c.far || guessFar(c.pos, c.dir);
    const th = c.dir * Math.PI / 180, hf = fov * Math.PI / 360;
    const p1 = isoTo(c.pos, th - hf, far), p2 = isoTo(c.pos, th + hf, far);
    // Стрелка всегда одной длины: дальность показывает сам сектор, а ручка, которая
    // ещё и ездит вслед за ползунком, только мешает её поймать.
    const tip = isoTo(c.pos, th, CAM_DIR_LEN);
    // Сектор нужен, пока расставляют камеры. На живом плане четыре наложенных
    // треугольника закрашивают всё поле, и следа людей под ними не видно.
    if (mode === 'plan') {
      s += `<path class="cam-fov" d="M${sx(c.pos[0])},${sy(c.pos[1])} L${sx(p1[0])},${sy(p1[1])}`
         + ` L${sx(p2[0])},${sy(p2[1])} Z"></path>`;
      s += `<line class="cam-dir" x1="${sx(c.pos[0])}" y1="${sy(c.pos[1])}" x2="${sx(tip[0])}" y2="${sy(tip[1])}"></line>`;
    }
    s += `<circle class="cam-mk" cx="${sx(c.pos[0])}" cy="${sy(c.pos[1])}" r="7" data-cam="${id}"></circle>`;
    // Конец оси разворачивает камеру. Ширину и дальность крутят ползунки: на глаз
    // за угол сектора не поймаешь, а число видно сразу.
    if (mode === 'plan' && tool === 'select') {
      s += `<circle class="cam-mk" cx="${sx(tip[0])}" cy="${sy(tip[1])}" r="6" data-caim="${id}"></circle>`;
    }
    s += `<text class="cam-lbl" x="${sx(c.pos[0])}" y="${sy(c.pos[1]) - 13}" text-anchor="middle">${esc(camName(id))}</text>`;
  }
  if (mode === 'plan' && tool === 'select') {
    for (const k of KINDS) plan[k].forEach((o, oi) => o.points.forEach((p, pi) => {
      s += `<circle class="vx" cx="${sx(p[0])}" cy="${sy(p[1])}" r="6" data-vx="${k}|${oi}|${pi}"></circle>`;
    }));
  }
  if (draft.length) {
    s += `<polyline class="wf-wall wf-draft" points="${poly(draft)}"></polyline>`;
    draft.forEach((p, i) => { s += `<circle class="vx" cx="${sx(p[0])}" cy="${sy(p[1])}" r="${i ? 5 : 8}"></circle>`; });
  }
  if (camDraft) s += `<circle class="cam-mk" cx="${sx(camDraft[0])}" cy="${sy(camDraft[1])}" r="7"></circle>`;
  if (mode === 'view') {
    // След — только у выбранного в списке. Восемь хвостов сразу превращали план
    // в клубок, по которому нельзя проследить ни одного человека.
    if (selPerson != null) {
      let tr = [], coarse = false;
      if (rt == null) {
        tr = (liveHist.get(selPerson) || []).map(q => [q[1], q[2]]);
      } else {
        const sel = replay ? replay.people.find(p => p.person_id === selPerson) : null;
        if (sel) { tr = trailOf(sel, rt); coarse = !sel.smooth; }
      }
      if (tr.length >= 2) {
        s += `<polyline class="trail${coarse ? ' coarse' : ''}" stroke="${personColour(selPerson)}"`
           + ` points="${tr.map(q => `${sx(q[0]).toFixed(1)},${sy(q[1]).toFixed(1)}`).join(' ')}"></polyline>`;
      }
    }
    if (rt == null) {
      for (const d of live) {
        if (d.u < -0.02 || d.u > 1.02 || d.v < -0.02 || d.v > 1.02) continue;
        const on = selPerson != null && d.person_id === selPerson;
        s += `<circle class="dot-live${d.name ? ' named' : ''}" cx="${sx(d.u)}" cy="${sy(d.v)}"`
           + ` r="${on ? 12 : d.name ? 9 : 6}"></circle>`;
        if (d.name) s += `<text class="dot-name" x="${sx(d.u)}" y="${sy(d.v) - 13}" text-anchor="middle">${esc(d.name)}</text>`;
      }
    } else if (replay) {
      for (const p of replay.people) {
        const pos = atTime(p, rt);
        if (!pos) continue;
        const on = p.person_id === selPerson;
        s += `<circle class="dot-live" fill="${personColour(p.person_id)}"`
           + ` cx="${sx(pos[0])}" cy="${sy(pos[1])}" r="${on ? 12 : 9}"></circle>`;
        s += `<text class="dot-name" x="${sx(pos[0])}" y="${sy(pos[1]) - 13}" text-anchor="middle">${esc(p.label)}</text>`;
      }
    }
  }
  return s;
}

// Цвет от номера человека: следы нескольких людей на одном плане иначе не различить,
// а просить оператора назначать цвета вручную не за чем.
function personColour(id) { return `hsl(${(id * 47) % 360} 70% 62%)`; }

// Где человек был в момент t. Точка следа живёт несколько секунд и гаснет, а
// отметка зоны (четвёртое число = 1) держит человека в комнате до следующей
// отметки: «был в зале с 14:04 до 15:11» — это две отметки, а не отсутствие.
// Между двумя соседними точками следа движемся линейно, иначе точка дёргается.
function atTime(p, t) {
  const a = p.pts;
  if (!a.length || t < a[0][0]) return null;
  let lo = 0, hi = a.length - 1, i = 0;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (a[m][0] <= t) { i = m; lo = m + 1; } else hi = m - 1; }
  const cur = a[i], nxt = a[i + 1];
  if (cur[1] == null) return null;
  if (cur[3] === 1) return [cur[1], cur[2]];
  if (t - cur[0] > REPLAY_GAP) return null;
  if (nxt && nxt[1] != null && nxt[3] !== 1 && nxt[0] - cur[0] <= 12) {
    const k = (t - cur[0]) / (nxt[0] - cur[0] || 1);
    return [cur[1] + (nxt[1] - cur[1]) * k, cur[2] + (nxt[2] - cur[2]) * k];
  }
  return [cur[1], cur[2]];
}

function trailOf(p, t) {
  const from = t - TRAIL_SEC;
  const out = [];
  for (const q of p.pts) {
    if (q[0] > t) break;
    if (q[1] == null) { out.length = 0; continue; }   // уходил из виду — след рвётся
    if (q[0] >= from) out.push([q[1], q[2]]);
  }
  return out;
}

function fmtClock(t) {
  const d = new Date(t * 1000);
  return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0')
       + ':' + String(d.getSeconds()).padStart(2, '0');
}

function timeline() {
  if (!replay) return `<div class="tl"><span class="muted">loading the day…</span></div>`;
  const cur = rt == null ? replay.t1 : rt;
  const shown = rt == null ? live.length : replay.people.filter(p => atTime(p, rt)).length;
  return `<div class="tl">
    <button onclick="playPause()" ${rt == null ? 'disabled' : ''}>${playing ? '❚❚' : '▶'}</button>
    <span class="clock">${fmtClock(cur)}</span>
    <input type="range" min="${replay.t0}" max="${replay.t1}" step="1" value="${cur}"
           oninput="setT(this.value)">
    <button class="${rt == null ? 'live' : ''}" onclick="goLive()">LIVE</button>
    <select onchange="speed = +this.value">${[2, 5, 10]
      .map(x => `<option value="${x}" ${x === speed ? 'selected' : ''}>×${x}</option>`).join('')}</select>
    <input type="date" value="${replay.day}" onchange="replayDay = this.value; rt = null; loadReplay();">
    <span class="muted">${shown} on the map · ${replay.people.length} people today ·
      ${replay.points} points${replay.stride > 1 ? ' · thinned ×' + replay.stride : ''}</span>
  </div>`;
}

// Ползунок времени не пересобираем на каждое движение: пересозданный <input>
// теряет захват мыши. Меняем только часы и картинку.
function setT(v) {
  rt = +v;
  const c = document.querySelector('.tl .clock');
  if (c) c.textContent = fmtClock(rt);
  const b = document.querySelector('.tl button');
  if (b) b.disabled = false;
  const l = document.querySelector('.tl button.live');
  if (l) l.classList.remove('live');
  redraw();
}

function playPause() { playing = !playing; render(); }
function goLive() { rt = null; playing = false; render(); }

async function loadReplay() {
  const q = '?branch=' + encodeURIComponent(myBranch)
          + (replayDay ? '&day=' + encodeURIComponent(replayDay) : '');
  replay = await fetch('/api/map/replay' + q).then(r => r.json()).catch(() => null);
  render();
}

// Обрезка отрезка по рамке (Лианг — Барски). Дальний угол комнаты может лежать
// в сотне ширин кадра: без обрезки браузер получает бессмысленные числа.
function clipSeg(a, b) {
  const L = -0.15, R = 1.15, T = -0.15, B = 1.15;
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const p = [-dx, dx, -dy, dy], q = [a[0] - L, R - a[0], a[1] - T, B - a[1]];
  let t0 = 0, t1 = 1;
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) { if (q[i] < 0) return null; continue; }
    const r = q[i] / p[i];
    if (p[i] < 0) { if (r > t1) return null; if (r > t0) t0 = r; }
    else { if (r < t0) return null; if (r < t1) t1 = r; }
  }
  return [[a[0] + t0 * dx, a[1] + t0 * dy], [a[0] + t1 * dx, a[1] + t1 * dy]];
}

function alignSvg() {
  if (!align || !align.ok) return '';
  const by = {};
  for (const h of align.handles) by[h.id] = h;
  let s = '';
  for (const sh of align.shapes) {
    const cls = sh.kind === 'rooms' ? 'wf-room' : sh.kind === 'furniture' ? 'wf-furn'
              : sh.kind === 'doors' ? 'wf-door' : 'wf-wall';
    const pts = sh.pts.map(id => by[id]);
    if (pts.some(h => !h || h.behind || h.off)) {
      // Хоть один угол вне кадра — рисуем рёбра поштучно. Замкнуть фигуру через
      // оставшиеся углы нельзя: получится стена, которой в комнате нет, и человек
      // будет наводить каркас на неё.
      const n = pts.length, last = sh.closed ? n : n - 1;
      for (let i = 0; i < last; i++) {
        const a = pts[i], b = pts[(i + 1) % n];
        if (!a || !b || a.behind || b.behind) continue;
        const c = clipSeg(a.img, b.img);
        if (c) s += `<line class="${cls}" fill="none" x1="${ax(c[0][0])}" y1="${ay(c[0][1])}"`
                  + ` x2="${ax(c[1][0])}" y2="${ay(c[1][1])}"></line>`;
      }
    } else {
      const tag = sh.closed ? 'polygon' : 'polyline';
      s += `<${tag} class="${cls}" points="${pts.map(h => `${ax(h.img[0])},${ay(h.img[1])}`).join(' ')}"></${tag}>`;
    }
  }
  for (const h of align.handles) {
    if (h.behind || h.off) continue;      // тащить можно только то, что видно
    const cls = h.err == null ? (h.set ? ' set' : '')
              : h.err < 0.02 ? ' ok' : h.err < 0.05 ? ' warn' : ' bad';
    s += `<circle class="hnd${cls}" cx="${ax(h.img[0])}" cy="${ay(h.img[1])}"`
       + ` r="8" data-hnd="${h.id}"></circle>`;
  }
  return s;
}

function hint() {
  if (tool === 'select') return 'Drag any corner to fix the drawing. On a camera: the big dot is where it hangs, '
    + 'the dot ahead of it turns it. Pick it in the list to set how wide and how far it sees.';
  if (tool === 'wall') return 'Click along the wall; Finish ends the line.';
  if (tool === 'room') return 'Click around the place and close it on the first point — this is what becomes a zone.';
  if (tool === 'table') return 'Click two opposite corners of the table.';
  if (tool === 'door') return 'Click both sides of the doorway.';
  if (tool === 'camera') return camPick ? 'Click where the camera hangs, then click where it looks.'
                                        : 'Pick a camera in the list first.';
  return '';
}

// Ползунки выбранной камеры. Перерисовываем от них только чертёж, не всю панель:
// пересобранный <input> теряет захват мыши, и ползунок бросало бы после первого
// же движения.
function camSliders() {
  const c = camPick && plan.cams[camPick];
  if (!c) return '';
  const g = (cfg && cfg.guess) || {fov: 90, far_min: 0.2, far_max: 1.5};
  const fov = c.fov || g.fov, far = c.far || guessFar(c.pos, c.dir);
  return `<div class="cam-sliders">
    <label><span>Width</span>
      <input type="range" min="10" max="170" step="1" value="${fov}"
             oninput="setCamFov(this.value)">
      <span class="val" id="fov-val">${Math.round(fov)}°</span></label>
    <label><span>Distance</span>
      <input type="range" min="${g.far_min}" max="${g.far_max}" step="0.01" value="${far.toFixed(2)}"
             oninput="setCamFar(this.value)">
      <span class="val" id="far-val">${far.toFixed(2)}</span></label>
    <label><span></span>
      <button id="save-cam" onclick="savePlan()" ${dirty ? '' : 'disabled'}>
        ${dirty ? 'Save plan •' : 'Save plan'}</button>
      <span class="map-hint">saves the whole drawing</span></label>
  </div>`;
}

function markDirty() {
  dirty = true;
  for (const id of ['save-plan', 'save-cam']) {
    const b = document.getElementById(id);
    if (b) { b.disabled = false; b.textContent = 'Save plan •'; }
  }
}

function setCamFov(v) {
  const c = plan.cams[camPick];
  if (!c) return;
  c.fov = +v;
  const el = document.getElementById('fov-val');
  if (el) el.textContent = Math.round(c.fov) + '°';
  markDirty();
  redraw();
}

function setCamFar(v) {
  const c = plan.cams[camPick];
  if (!c) return;
  c.far = +v;
  const el = document.getElementById('far-val');
  if (el) el.textContent = c.far.toFixed(2);
  markDirty();
  redraw();
}

// Уже стоящую камеру выбирают, чтобы покрутить ползунки, а не поставить заново:
// переключать на инструмент «камера» в этом случае — значит уронить её при
// следующем щелчке по плану.
function pickPlanCam(v) {
  camPick = v;
  camDraft = null;
  if (v && !plan.cams[v]) setTool('camera'); else render();
}

function toolbar() {
  if (mode === 'plan') {
    const b = (t, label) => `<button class="${tool === t ? 'on' : ''}" onclick="setTool('${t}')">${label}</button>`;
    let s = b('select', 'Move') + b('wall', 'Wall') + b('room', 'Room') + b('table', 'Table') + b('door', 'Door');
    s += `<select onchange="pickPlanCam(this.value)"><option value="">— camera —</option>`
       + cfg.cameras.map(c => `<option value="${c.id}" ${c.id === camPick ? 'selected' : ''}>`
         + `${esc(c.name)} (${c.id})${plan.cams[c.id] ? ' ✓' : ''}</option>`).join('')
       + `</select>`;
    s += `<button class="${snap90 ? 'on' : ''}" onclick="snap90 = !snap90; render();">Snap 90°</button>`;
    if (draft.length) s += `<button onclick="finishShape()">Finish</button>`
                         + `<button onclick="draft = []; render();">Cancel</button>`;
    s += `<button onclick="undoLast()">Undo</button>`;
    s += `<button id="save-plan" style="margin-left:auto;" onclick="savePlan()" ${dirty ? '' : 'disabled'}>`
       + `${dirty ? 'Save plan •' : 'Save plan'}</button>`;
    s += `<button class="danger" onclick="clearPlan()" style="padding:5px 10px; font-size:12px;">Clear</button>`;
    s += camSliders();
    s += `<div class="map-hint" style="flex-basis:100%;">${hint()}</div>`;
    return s;
  }
  if (mode === 'align') {
    let s = `<select onchange="pickCam(this.value)"><option value="">— camera —</option>`
      + cfg.cameras.map(c => `<option value="${c.id}" ${c.id === selCam ? 'selected' : ''}>`
        + `${esc(c.name)} (${c.id})${c.calibrated ? ' ✓' : c.placed ? '' : ' — not on the plan'}</option>`).join('')
      + `</select>`;
    if (!selCam) return s + `<span class="map-hint">Pick a camera: its frame comes up with the drawing laid over it.</span>`;
    const n = Object.keys(adjust).length, need = cfg.min_points;
    const bad = !!(align && align.warn);
    s += `<button onclick="refit(true)" ${n && !bad ? '' : 'disabled'}>Fit rest</button>`;
    s += `<button onclick="newSnapshot()">New snapshot</button>`;
    s += `<button onclick="adjust = {}; refit(true);" ${n ? '' : 'disabled'}>Reset</button>`;
    s += `<button onclick="saveAlign()" ${n >= need && !bad ? '' : 'disabled'}>Save</button>`;
    // Четырёх точек хватает математически, но если они сбились в один угол кадра,
    // гомография шаткая: за пределами этого угла ошибка растёт быстро.
    s += `<span class="map-hint">` + (align && !align.ok ? esc(align.error)
        : bad ? `<span class="bad">${esc(align.warn)}</span>`
        : n >= need ? `Fitted on ${n} points. Drag moves that point only — `
          + `press Fit rest to move the others onto the fit.`
        : `Drag each corner onto what you actually see — spread them across the frame, `
          + `near and far. ${n} of ${need} set.`) + `</span>`;
    return s;
  }
  return `<span class="map-hint">Green — named, amber — not recognised. `
       + `Positions come only from aligned cameras.</span>`;
}

function sideList() {
  if (mode === 'plan') {
    let rows = '';
    for (const k of KINDS) {
      plan[k].forEach((o, i) => {
        const label = o.name ? esc(o.name) : KIND_LABEL[k] + ' ' + (i + 1);
        rows += `<li><span class="${k === 'rooms' ? '' : 'muted'}">${label}</span>`
             + (k === 'rooms' ? `<button class="x" onclick="renameRoom(${i})">rename</button>` : '')
             + `<button class="x danger" onclick="dropShape('${k}', ${i})">✕</button></li>`;
      });
    }
    for (const id of Object.keys(plan.cams)) {
      rows += `<li><span class="muted">📹 ${esc(camName(id))}</span>`
           + `<button class="x danger" onclick="dropCam('${id}')">✕</button></li>`;
    }
    return `<ul class="map-list">${rows || '<li class="muted">Nothing drawn yet.</li>'}</ul>`;
  }
  if (mode === 'align') {
    if (!align || !align.ok) return '';
    const set = align.handles.filter(h => h.set);
    const rows = set.map(h => `<li>${esc(KIND_LABEL[h.kind] || h.kind)} `
      + (h.err == null ? '<span class="muted">set</span>'
         : `<span class="${h.err < 0.02 ? 'ok' : h.err < 0.05 ? 'warn' : 'bad'}">±${(h.err * 100).toFixed(1)}%</span>`)
      + `<button class="x" onclick="delete adjust['${h.id}']; refit(false);">✕</button></li>`).join('');
    return `<div class="map-hint">Fit: ${esc(align.fit)}</div>`
         + `<ul class="map-list">${rows || '<li class="muted">No points set yet.</li>'}</ul>`;
  }
  let out = '';
  const cams = cfg.cameras.filter(c => !c.calibrated);
  if (cams.length) out += `<div class="map-hint">Not aligned yet: `
    + `${cams.map(c => esc(c.name)).join(', ')}.</div>`;
  if (replay && replay.people.length) {
    out += `<div class="map-hint">Click a name to follow their path.</div>`;
    out += `<ul class="map-list">` + replay.people.map(p =>
      `<li class="pick${p.person_id === selPerson ? ' on' : ''}" onclick="pickPerson(${p.person_id})">`
      + `<span style="display:inline-block; width:10px; height:10px; border-radius:50%;`
      + ` background:${personColour(p.person_id)};"></span> ${esc(p.label)}`
      + `<span class="muted" style="margin-left:auto;">${p.smooth ? '' : 'by room'}</span></li>`
    ).join('') + `</ul>`;
  }
  return out;
}

function pickPerson(id) {
  selPerson = selPerson === id ? null : id;
  render();
}

function layout(side) {
  if (mode === 'align') {
    // Камера ещё не выбрана — кадра нет и просить его не за чем.
    if (!selCam) return `<div class="map-wrap"><div class="map-stage" id="stage"`
      + ` style="padding:40px; text-align:center;">`
      + `<span class="map-hint">Pick a camera above.</span></div></div>`;
    return `<div class="map-wrap">
      <div class="map-stage" id="stage">
        <img id="snap" src="/snapshot/${selCam}?clean=1&t=${snapStamp}" onerror="this.alt='no frame yet'">
        <svg viewBox="0 0 ${W} ${W * snapRatio}" preserveAspectRatio="none" id="align-svg"></svg>
      </div>
      <div class="map-side"><div id="side-list">${side}</div></div></div>`;
  }
  // У чертежа нет картинки, которая задала бы высоту, — её задаёт сам svg.
  return `<div class="map-wrap">
    <div style="flex:1; min-width:0;">
      <div class="map-stage${mode === 'plan' && tool !== 'select' ? ' draw' : ''}" id="stage">
        <svg viewBox="0 0 ${W} ${PH()}" id="plan-svg" style="width:100%; height:auto; display:block;"></svg>
      </div>
      <div id="tl-row">${mode === 'view' ? timeline() : ''}</div>
    </div>
    <div class="map-side"><div id="side-list">${side}</div></div></div>`;
}

function render() {
  if (!cfg || !plan) return;
  const root = document.getElementById('map-root');
  const placed = cfg.cameras.filter(c => c.placed).length;
  const done = cfg.cameras.filter(c => c.calibrated).length;
  document.getElementById('map-status').textContent =
    `${placed} of ${cfg.cameras.length} cameras on the plan · ${done} aligned · ${(cfg.zones || []).length} zones`;
  document.getElementById('map-tools').innerHTML = toolbar();

  const side = sideList();
  // Картинку кадра не пересоздаём при каждой правке: моргает и теряет позицию.
  const key = `${mode}|${selCam}|${mode === 'plan' && tool !== 'select'}`;
  if (root.dataset.key !== key) {
    root.dataset.key = key;
    root.innerHTML = layout(side);
    const st = document.getElementById('stage');
    st.onmousedown = onDown;
    st.onclick = stageClick;
    if (mode === 'align') fixSnapAspect();
  } else {
    const sl = document.getElementById('side-list');
    if (sl) sl.innerHTML = side;
  }
  // Полосу времени трогаем, только когда изменилось что-то кроме самого момента:
  // пересобранный на каждое движение ползунок вырывался бы из-под курсора.
  const tl = document.getElementById('tl-row');
  if (tl) {
    const tk = `${mode}|${rt == null}|${playing}|${speed}|`
             + (replay ? replay.day + ':' + replay.points : '');
    if (tl.dataset.k !== tk) {
      tl.dataset.k = tk;
      tl.innerHTML = mode === 'view' ? timeline() : '';
    }
  }
  redraw();
}

function redraw() {
  const ps = document.getElementById('plan-svg');
  if (ps) { ps.setAttribute('viewBox', `0 0 ${W} ${PH()}`); ps.innerHTML = planSvg(); }
  const as = document.getElementById('align-svg');
  if (as) as.innerHTML = alignSvg();
}

// Кадр не обязан быть 16:9 — подгоняем viewBox под реальные пропорции после загрузки.
function fixSnapAspect() {
  const img = document.getElementById('snap');
  const svg = document.getElementById('align-svg');
  if (!img || !svg) return;
  const apply = () => {
    if (!img.naturalWidth) return;
    snapRatio = img.naturalHeight / img.naturalWidth;
    svg.setAttribute('viewBox', `0 0 ${W} ${W * snapRatio}`);
    redraw();
  };
  if (img.complete) apply(); else img.onload = apply;
}

function norm(e, el) {
  const r = el.getBoundingClientRect();
  return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
          Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
}

function stageEl() {
  return document.getElementById(mode === 'align' ? 'snap' : 'plan-svg');
}

// ---------- мышь ----------
function onDown(e) {
  const d = e.target.dataset || {};
  if (mode === 'plan' && tool === 'select') {
    if (d.vx) drag = {t: 'vx', ref: d.vx.split('|')};
    else if (d.cam) drag = {t: 'cam', id: d.cam};
    else if (d.caim) drag = {t: 'caim', id: d.caim};
  } else if (mode === 'align' && d.hnd && align && align.ok) {
    // Щелчок без движения — тоже ответ: значит, точка уже стоит где надо.
    const h = align.handles.find(x => x.id === d.hnd);
    if (h) { adjust[d.hnd] = [...h.img]; h.set = true; drag = {t: 'hnd', id: d.hnd}; }
  }
  if (drag) e.preventDefault();
}

function onMove(e) {
  if (!drag) return;
  const p = norm(e, stageEl());
  if (drag.t === 'vx') { const [k, oi, pi] = drag.ref; plan[k][+oi].points[+pi] = p; dirty = true; }
  else if (drag.t === 'cam') { plan.cams[drag.id].pos = p; dirty = true; }
  else if (drag.t === 'caim') { plan.cams[drag.id].dir = angTo(plan.cams[drag.id].pos, p); dirty = true; }
  else if (drag.t === 'hnd') {
    adjust[drag.id] = p;
    const h = align.handles.find(x => x.id === drag.id);
    if (h) { h.img = p; h.behind = h.off = h.out = false; }
  }
  redraw();
}

function onUp() {
  const was = drag;
  drag = null;
  if (!was) return;
  if (was.t === 'hnd') refit(false); else render();
}

function allVertices() {
  const out = [];
  for (const k of KINDS) for (const o of plan[k]) for (const p of o.points) out.push(p);
  return out;
}

function snapPoint(p) {
  // Притягиваем к уже нарисованному углу: стены должны сходиться, а не почти сходиться.
  let best = null, bd = 14;
  for (const v of allVertices()) {
    const d = Math.hypot((v[0] - p[0]) * W, (v[1] - p[1]) * PH());
    if (d < bd) { bd = d; best = v; }
  }
  if (best) return [best[0], best[1]];
  if (snap90 && draft.length) {
    const q = draft[draft.length - 1];
    const dx = (p[0] - q[0]) * W, dy = (p[1] - q[1]) * PH();
    if (Math.abs(dy) < Math.abs(dx) * 0.22) return [p[0], q[1]];
    if (Math.abs(dx) < Math.abs(dy) * 0.22) return [q[0], p[1]];
  }
  return p;
}

function stageClick(e) {
  if (mode !== 'plan' || tool === 'select' || drag) return;
  const raw = norm(e, stageEl());
  if (tool === 'camera') {
    if (!camPick) return;
    if (!camDraft) camDraft = raw;
    else { plan.cams[camPick] = {pos: camDraft, dir: angTo(camDraft, raw)}; camDraft = null; dirty = true; }
    render();
    return;
  }
  const p = snapPoint(raw);
  // Щелчок по первой вершине замыкает комнату.
  if (tool === 'room' && draft.length >= 3
      && Math.hypot((p[0] - draft[0][0]) * W, (p[1] - draft[0][1]) * PH()) < 12) { finishShape(); return; }
  draft.push(p);
  if ((tool === 'table' || tool === 'door') && draft.length === 2) { finishShape(); return; }
  render();
}

// ---------- правка чертежа ----------
function finishShape() {
  const need = {wall: 2, room: 3, table: 2, door: 2}[tool] || 99;
  if (draft.length < need) return;
  let kind = 'walls', pts = draft.slice(), name = '';
  if (tool === 'room') {
    kind = 'rooms';
    const nm = prompt('Name of this place (Hall, Kitchen, Desk 3):');
    if (nm === null || !nm.trim()) { draft = []; render(); return; }
    name = nm.trim();
  } else if (tool === 'table') {
    kind = 'furniture';
    const [a, b] = draft;
    pts = [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]];
  } else if (tool === 'door') {
    kind = 'doors';
    pts = draft.slice(0, 2);
  }
  const o = {id: kind[0] + Date.now().toString(36), points: pts};
  if (name) o.name = name;
  plan[kind].push(o);
  history.push(kind);
  draft = []; dirty = true;
  render();
}

function undoLast() {
  if (draft.length) { draft.pop(); render(); return; }
  const k = history.pop();
  if (!k || !plan[k].length) return;
  plan[k].pop();
  dirty = true;
  render();
}

function dropShape(kind, i) { plan[kind].splice(i, 1); dirty = true; render(); }
function dropCam(id) { delete plan.cams[id]; dirty = true; render(); }

function renameRoom(i) {
  const name = prompt('New name:', plan.rooms[i].name || '');
  if (name === null || !name.trim()) return;
  plan.rooms[i].name = name.trim();
  dirty = true;
  render();
}

async function savePlan() {
  const r = await fetch('/api/map/plan?branch=' + encodeURIComponent(myBranch), {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({plan}),
  }).then(r => r.json()).catch(() => ({ok: false}));
  if (!r.ok) { alert(r.error || 'failed'); return; }
  if (r.warning) alert(r.warning);
  dirty = false;
  await load();
}

async function clearPlan() {
  if (!confirm('Remove the drawing, all zones and camera alignment for this branch?')) return;
  const r = await fetch('/api/map/plan?branch=' + encodeURIComponent(myBranch), {method: 'DELETE'})
    .then(r => r.json()).catch(() => ({ok: false}));
  if (!r.ok) { alert(r.error || 'failed'); return; }
  plan = EMPTY(); dirty = false; selCam = null; align = null; adjust = {}; history = [];
  await load();
}

// ---------- совмещение с кадром ----------
// apply=false — пересчитать только ошибки, каркас оставить как лежит. Человек
// ставит точку и хочет видеть, что сдвинулась ровно она: если после каждой правки
// весь чертёж переползает на новое решение, целиться становится не во что.
// Разложить остальные точки по новой гомографии — отдельное действие, кнопка Fit.
async function refit(apply = true) {
  if (!selCam) { render(); return; }
  const keep = {};
  if (!apply && align && align.ok) {
    for (const h of align.handles) keep[h.id] = {img: h.img, out: h.out};
  }
  align = await fetch('/api/map/align/' + selCam, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({adjust}),
  }).then(r => r.json()).catch(() => ({ok: false, error: 'request failed'}));
  if (align.ok) {
    for (const h of align.handles) {
      const k = keep[h.id];
      if (k && !h.set) { h.img = k.img; h.out = k.out; }
    }
  }
  render();
}

async function pickCam(id) {
  selCam = id || null;
  adjust = {}; align = null; snapStamp = Date.now();
  if (!selCam) { render(); return; }
  await refit();
  // Уже сохранённая калибровка — это те же точки чертежа: узнаём их по координате
  // на плане и возвращаем человеку ровно там, где он их оставил.
  const c = cfg.cameras.find(x => x.id === selCam);
  if (c && c.points.length && align && align.ok) {
    for (const p of c.points) {
      const h = align.handles.find(h => Math.abs(h.map[0] - p.map[0]) < 1e-6
                                     && Math.abs(h.map[1] - p.map[1]) < 1e-6);
      if (h) adjust[h.id] = [...p.img];
    }
    if (Object.keys(adjust).length) await refit();
  }
}

function newSnapshot() {
  snapStamp = Date.now();
  const img = document.getElementById('snap');
  if (img) img.src = `/snapshot/${selCam}?clean=1&t=${snapStamp}`;
}

async function saveAlign() {
  const by = {};
  for (const h of (align && align.handles) || []) by[h.id] = h;
  const points = Object.entries(adjust).filter(([id]) => by[id])
    .map(([id, xy]) => ({img: xy, map: by[id].map}));
  const r = await fetch('/api/map/calib/' + selCam, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({points}),
  }).then(r => r.json()).catch(() => ({ok: false}));
  if (!r.ok) { alert(r.error || 'failed'); return; }
  await load();
  await refit();
}

async function pollLive() {
  if (mode !== 'view' || !cfg || rt != null) return;     // в перемотке живое не нужно
  live = await fetch('/api/map/live?branch=' + encodeURIComponent(myBranch)).then(r => r.json()).catch(() => []);
  const now = Date.now() / 1000;
  for (const d of live) {
    if (d.person_id == null) continue;
    let a = liveHist.get(d.person_id);
    if (!a) liveHist.set(d.person_id, a = []);
    a.push([now, d.u, d.v]);
  }
  for (const [pid, a] of liveHist) {
    while (a.length && now - a[0][0] > TRAIL_SEC) a.shift();
    if (!a.length) liveHist.delete(pid);
  }
  redraw();
}

document.addEventListener('mousemove', onMove);
document.addEventListener('mouseup', onUp);
setInterval(pollLive, 700);
setInterval(() => { if (mode === 'view' && rt == null) load(); }, 15000);
// Догружаем след, пока смотрим живое: иначе отмотать назад можно будет только до
// того момента, когда страницу открыли.
setInterval(() => { if (mode === 'view' && rt == null) loadReplay(); }, 30000);
setInterval(() => {
  if (!playing || rt == null || !replay) return;
  rt = Math.min(replay.t1, rt + speed * 0.25);
  const c = document.querySelector('.tl .clock');
  if (c) c.textContent = fmtClock(rt);
  const s = document.querySelector('.tl input[type=range]');
  if (s) s.value = rt;
  if (rt >= replay.t1) { playing = false; render(); return; }
  redraw();
}, 250);

load().then(() => setMode(cfg && cfg.drawn ? 'view' : 'plan'));
"""


@app.get("/map", response_class=HTMLResponse)
def map_page(branch: str = ""):
    if not branch and BRANCHES:
        return RedirectResponse("/", status_code=303)
    return _page("Detect — floor map", _nav("map"), _MAP_BODY, _MAP_JS)


@app.get("/api/map")
def api_map(branch: str = ""):
    return floor_map.describe(branch)


@app.get("/api/map/live")
def api_map_live(branch: str = ""):
    return floor_map.live(branch)


# Больше этого в один ответ не кладём: дальше браузеру тяжелее рисовать, чем нам
# считать, а разглядеть разницу на плане уже нельзя.
REPLAY_MAX_POINTS = 40000


@app.get("/api/map/replay")
def api_map_replay(branch: str = "", day: str = "", person_id: int | None = None):
    """След людей за день — то, чем время отматывают назад на плане.

    Точки отдаём как есть: они прорежены ещё при записи, по движению. Там, где
    следа нет (день раньше, чем его начали писать), достраиваем из отрезков зон —
    по отметке в центре зоны на вход и на выход. Движение тогда ступенчатое, зато
    день не пустой, а зоны писались задолго до следа.
    """
    start, end, day_text = _day_bounds(day)
    # Хвост сегодняшнего дня ещё лежит в буфере и в базу попадёт через секунды —
    # для отмотки назад это не важно, а для «последних двух минут» важно.
    floor_map.flush_trail(store)
    by_person: dict[int, list] = {}
    for r in store.positions(branch, start, end, person_id):
        by_person.setdefault(int(r["person_id"]), []).append(
            [round(float(r["ts"]), 1), r["u"], r["v"]])

    stride = 1
    total = sum(len(v) for v in by_person.values())
    if total > REPLAY_MAX_POINTS:
        stride = total // REPLAY_MAX_POINTS + 1
        for pid, pts in by_person.items():
            if len(pts) > 2 * stride:
                by_person[pid] = pts[::stride] + [pts[-1]]

    zones = {z["name"]: poly_centre(z["points"])
             for z in ((floor_map.branch(branch) or {}).get("zones") or [])}
    now = time.time()
    segs: dict[int, list] = {}
    if zones:
        for s in store.zone_routes(start, branch, person_id, limit=20000, day_end=end):
            c = zones.get(s["label"])
            if c is None:
                continue
            segs.setdefault(int(s["person_id"]), []).append(
                (round(float(s["start_ts"]), 1),
                 round(float(s["end_ts"] if s["end_ts"] else min(end, now)), 1),
                 round(c[0], 4), round(c[1], 4)))

    # Зоны и след склеиваем по времени, а не «или то, или это»: след появился
    # сегодня в обед, и человек, у которого есть и то и другое, иначе терял бы всё
    # утро — на плане он возникал бы из ниоткуда в момент первой записанной точки.
    coarse: dict[int, list] = {}
    for pid, lst in segs.items():
        lst.sort()
        cut = by_person[pid][0][0] if pid in by_person else None
        pts = []
        for i, (a, b, cx, cy) in enumerate(lst):
            if cut is not None:
                if a >= cut:
                    break
                b = min(b, cut)
            # Третьим числом помечаем отметку зоны: она держит человека на месте до
            # следующей, тогда как точка следа живёт несколько секунд и гаснет.
            pts.append([a, cx, cy, 1])
            if b > a:
                pts.append([b, cx, cy, 1])
            nxt = lst[i + 1][0] if i + 1 < len(lst) else None
            # Между отрезками человека не было видно. Без явной отметки «здесь его
            # нет» проигрыватель держал бы его в прошлой комнате до следующего входа.
            # Перед началом следа её не ставим: там эстафету принимает след.
            if cut is not None and b >= cut - 1.0:
                continue
            if nxt is None or nxt - b > 1.0:
                pts.append([round(b + 0.1, 1), None, None])
        if pts:
            coarse[pid] = pts

    out = []
    for pid in set(by_person) | set(coarse):
        label = face_store.label_of(pid)
        if not label or label.startswith("#"):
            continue
        pts = coarse.get(pid, []) + by_person.get(pid, [])
        pts.sort(key=lambda p: p[0])
        out.append({"person_id": pid, "label": label,
                    "smooth": pid in by_person, "pts": pts})
    out.sort(key=lambda p: p["label"].lower())
    # Полосу времени начинаем с первой записи дня, а не с полуночи: иначе девять
    # десятых ползунка — это пустая ночь, и весь рабочий день жмётся в хвостик.
    first = min((p["pts"][0][0] for p in out if p["pts"]), default=None)
    t0 = max(start, first - 60) if first is not None else start
    return {"day": day_text, "t0": t0, "day_start": start, "t1": min(end, now),
            "now": now, "people": out, "points": sum(len(p["pts"]) for p in out),
            "stride": stride, "keep_days": TRAIL_KEEP_DAYS}


@app.get("/floorplans/{name}")
def floorplan_file(name: str):
    p = FLOORPLAN_DIR / Path(name).name
    if not p.exists():
        return Response(status_code=404)
    return FileResponse(str(p))


@app.get("/snapshot/{cam_id}")
def snapshot(cam_id: str, clean: int = 0):
    """Последний кадр камеры одной картинкой — для калибровки нужен кадр, а не поток.

    clean=1 отдаёт кадр без рамок и скелетов: наводя каркас плана, человек смотрит
    на пол и углы, а не на то, что распознали модели.
    """
    frame = _clean_frame.get(cam_id) if clean else None
    if frame is None:
        frame = _latest_frame.get(cam_id)
    if frame is None:
        return Response(status_code=404)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return Response(status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/map/floorplan")
async def api_map_floorplan(request: Request, branch: str = ""):
    """Картинка плана телом запроса как есть: multipart здесь ни к чему."""
    b = next((x for x in BRANCHES if x["id"] == branch), None)
    if b is None:
        return {"ok": False, "error": "no such branch"}
    data = await request.body()
    if not data or len(data) > 30 * 1024 * 1024:
        return {"ok": False, "error": "empty or too large (max 30 MB)"}
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "not a readable image"}
    h, w = img.shape[:2]
    ext = "png" if data[:4] == b"\x89PNG" else "jpg"
    fname = f"{branch}.{ext}"
    (FLOORPLAN_DIR / fname).write_bytes(data)
    # Старый файл с другим расширением, если план заменили png на jpg.
    for other in FLOORPLAN_DIR.glob(f"{branch}.*"):
        if other.name != fname:
            other.unlink(missing_ok=True)
    b["floorplan"] = {"file": fname, "width": int(w), "height": int(h),
                      "stamp": int(time.time())}
    try:
        _save_config()
    except Exception as e:
        return {"ok": False, "error": f"cannot write config: {e}"}
    logger.info(f"[map] {branch} floor plan {w}x{h} uploaded")
    return {"ok": True, "floorplan": b["floorplan"]}


@app.delete("/api/map/floorplan")
def api_map_floorplan_delete(branch: str = ""):
    """Убрать план вместе с зонами и калибровкой камер: без плана они бессмысленны."""
    b = next((x for x in BRANCHES if x["id"] == branch), None)
    if b is None:
        return {"ok": False, "error": "no such branch"}
    fp = b.pop("floorplan", None)
    b.pop("zones", None)
    cams = [c["id"] for c in CAMERAS if (c.get("branch") or "") == branch and c.pop("floor_calib", None)]
    try:
        _save_config()
    except Exception as e:
        return {"ok": False, "error": f"cannot write config: {e}"}
    if fp:
        (FLOORPLAN_DIR / fp["file"]).unlink(missing_ok=True)
    logger.info(f"[map] {branch} floor plan removed, cameras reset: {cams}")
    return {"ok": True, "cameras_reset": cams}


@app.post("/api/map/plan")
async def api_map_plan(request: Request, branch: str = ""):
    """Нарисованный план целиком. Комнаты сразу становятся зонами."""
    b = next((x for x in BRANCHES if x["id"] == branch), None)
    if b is None:
        return {"ok": False, "error": "no such branch"}
    d = await request.json()
    plan = sanitize_plan(d.get("plan") or {})
    zones = zones_from_plan(plan)
    seen, dupes = set(), []
    for z in zones:
        if z["name"].lower() in seen:
            dupes.append(z["name"])
        seen.add(z["name"].lower())
    b["plan"] = plan
    b["zones"] = zones
    try:
        _save_config()
    except Exception as e:                                       # noqa: BLE001
        return {"ok": False, "error": f"cannot write config: {e}"}
    logger.info(f"[map] {branch} plan saved: "
                f"{len(plan['walls'])} walls, {len(plan['furniture'])} furniture, "
                f"{len(zones)} rooms, {len(plan['cams'])} cameras placed")
    out = {"ok": True, "zones": len(zones)}
    # Две зоны с одним именем сливаются в отчётах в одну строку: пишем в базу имя,
    # а не id. Рисовать не мешаем, но сказать обязаны.
    if dupes:
        out["warning"] = "two places named " + ", ".join(sorted(set(dupes)))
    return out


@app.delete("/api/map/plan")
def api_map_plan_delete(branch: str = ""):
    """Убрать чертёж вместе с зонами и калибровкой: без плана они бессмысленны."""
    b = next((x for x in BRANCHES if x["id"] == branch), None)
    if b is None:
        return {"ok": False, "error": "no such branch"}
    b.pop("plan", None)
    b.pop("zones", None)
    cams = [c["id"] for c in CAMERAS
            if (c.get("branch") or "") == branch and c.pop("floor_calib", None)]
    try:
        _save_config()
    except Exception as e:                                       # noqa: BLE001
        return {"ok": False, "error": f"cannot write config: {e}"}
    logger.info(f"[map] {branch} plan removed, cameras reset: {cams}")
    return {"ok": True, "cameras_reset": cams}


@app.post("/api/map/align/{cam_id}")
async def api_map_align(cam_id: str, request: Request):
    """Куда лёг бы каркас чертежа на кадре этой камеры при нынешних поправках.

    Считает сервер, а не браузер: гомография уже решается здесь, и второй её
    реализации на JavaScript, которая разойдётся с первой, заводить не за чем.
    """
    cam = next((c for c in CAMERAS if c["id"] == cam_id), None)
    if cam is None:
        return {"ok": False, "error": "no such camera"}
    d = await request.json()
    adjust = {}
    for k, v in (d.get("adjust") or {}).items():
        try:
            adjust[str(k)] = [float(v[0]), float(v[1])]
        except (TypeError, ValueError, IndexError):
            return {"ok": False, "error": "bad point"}
    return align_fit(floor_map.plan(cam.get("branch") or ""), cam_id, adjust)


@app.post("/api/map/zones")
async def api_map_zones(request: Request, branch: str = ""):
    b = next((x for x in BRANCHES if x["id"] == branch), None)
    if b is None:
        return {"ok": False, "error": "no such branch"}
    d = await request.json()
    zones = []
    names = set()
    for z in d.get("zones") or []:
        name = str(z.get("name") or "").strip()[:60]
        pts = [[min(1.0, max(0.0, float(p[0]))), min(1.0, max(0.0, float(p[1])))]
               for p in (z.get("points") or [])]
        if not name or len(pts) < 3:
            continue
        if name.lower() in names:
            return {"ok": False, "error": f"two places named '{name}'"}
        names.add(name.lower())
        zones.append({"id": str(z.get("id") or f"z{len(zones)+1}"), "name": name,
                      "points": pts})
    b["zones"] = zones
    try:
        _save_config()
    except Exception as e:
        return {"ok": False, "error": f"cannot write config: {e}"}
    logger.info(f"[map] {branch} zones -> {[z['name'] for z in zones]}")
    return {"ok": True, "zones": zones}


@app.post("/api/map/calib/{cam_id}")
async def api_map_calib(cam_id: str, request: Request):
    """Пары точек «кадр — план». Пусто = снять камеру с плана."""
    cam = next((c for c in CAMERAS if c["id"] == cam_id), None)
    if cam is None:
        return {"ok": False, "error": "no such camera"}
    d = await request.json()
    pts = []
    for p in d.get("points") or []:
        try:
            pts.append({"img": [float(p["img"][0]), float(p["img"][1])],
                        "map": [float(p["map"][0]), float(p["map"][1])]})
        except (KeyError, TypeError, ValueError, IndexError):
            return {"ok": False, "error": "bad point"}
    errors = []
    if pts:
        H, errors = solve_homography(pts)
        if H is None:
            return {"ok": False, "error": f"need at least {MIN_CALIB_POINTS} points "
                                          "that are not all on one line"}
        cam["floor_calib"] = {"points": pts}
    else:
        cam.pop("floor_calib", None)
    try:
        _save_config()
    except Exception as e:
        return {"ok": False, "error": f"cannot write config: {e}"}
    logger.info(f"[map] {cam_id} calibrated with {len(pts)} points, "
                f"errors {[round(e, 3) for e in errors]}")
    return {"ok": True, "errors": errors, "calibrated": bool(pts)}


@app.get("/api/status")
def status():
    return _connected


@app.get("/api/branches")
def api_branches():
    out = []
    for b in BRANCHES:
        cams = _branch_cameras(b["id"])
        out.append({**b, "cameras": len(cams),
                    "live": sum(1 for c in cams if _connected.get(c["id"]))})
    loose = [c["id"] for c in CAMERAS if not c.get("branch")]
    return {"branches": out, "unassigned": loose}


@app.post("/api/branches")
async def api_add_branch(request: Request):
    d = await request.json()
    name = str(d.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "name is required"}
    if any(b["name"].lower() == name.lower() for b in BRANCHES):
        return {"ok": False, "error": f"'{name}' already exists"}
    nid = f"br_{max([int(b['id'].split('_')[-1]) for b in BRANCHES] + [0]) + 1:02d}"
    branch = {"id": nid, "name": name, "note": str(d.get("note") or "").strip()}
    BRANCHES.append(branch)
    try:
        _save_config()
    except Exception as e:
        BRANCHES.remove(branch)
        return {"ok": False, "error": f"cannot write config: {e}"}
    logger.info(f"[branch] added {nid} '{name}'")
    return {"ok": True, "branch": branch}


@app.delete("/api/branches/{branch_id}")
def api_delete_branch(branch_id: str):
    """Убрать филиал ВМЕСТЕ с его камерами.

    Раньше камеры оставались — просто становились ничьими. Выглядело безобиднее, а на
    деле оставляло их нигде: филиала у них нет, а страница камер всегда открыта внутри
    филиала. Хуже того, следующий заведённый филиал подбирал их себе, и новый объект
    начинался с чужого оборудования.

    Филиал — это объект целиком: адрес, камеры, то что они видят. Убрали объект —
    убрали и камеры. Лица, события и записи остаются: они про людей, а не про место.
    """
    b = next((x for x in BRANCHES if x["id"] == branch_id), None)
    if b is None:
        return {"ok": False, "error": "no such branch"}
    doomed = [c["id"] for c in CAMERAS if c.get("branch") == branch_id]
    for cam_id in doomed:
        try:
            stop_camera(cam_id)
        except Exception as e:                                   # noqa: BLE001
            logger.error(f"[branch] stopping {cam_id}: {e}")
    CAMERAS[:] = [c for c in CAMERAS if c.get("branch") != branch_id]
    BRANCHES.remove(b)
    try:
        _save_config()
    except Exception as e:
        return {"ok": False, "error": f"cannot write config: {e}"}
    logger.info(f"[branch] removed {branch_id} with {len(doomed)} cameras: {doomed}")
    return {"ok": True, "removed": branch_id, "cameras_removed": doomed}


@app.post("/api/cameras/{cam_id}/branch")
async def api_camera_branch(cam_id: str, request: Request):
    cam = next((c for c in CAMERAS if c["id"] == cam_id), None)
    if cam is None:
        return {"ok": False, "error": "no such camera"}
    d = await request.json()
    bid = str(d.get("branch") or "").strip()
    if bid and not any(b["id"] == bid for b in BRANCHES):
        return {"ok": False, "error": "no such branch"}
    if bid:
        cam["branch"] = bid
    else:
        cam.pop("branch", None)
    try:
        _save_config()
    except Exception as e:
        return {"ok": False, "error": f"cannot write config: {e}"}
    return {"ok": True, "camera": cam_id, "branch": bid or None}


@app.get("/api/cameras")
def api_cameras():
    return {"cameras": [{"id": c["id"], "name": c.get("name", c["id"]),
                         "ip": c["ip"], "port": c.get("port", 8000),
                         "channel": c.get("channel", 1),
                         "branch": c.get("branch"),
                         "connected": bool(_connected.get(c["id"])),
                         "models": _cam_models(c)}
                        for c in CAMERAS],
            "model_keys": list(MODEL_KEYS)}


def _probe_worker(cam: dict, q) -> None:
    """Вход в камеру и сразу выход — в ОТДЕЛЬНОМ процессе."""
    sys.path.insert(0, str(CAMERA_DIR))
    try:
        import camera_sdk
        camera_sdk.init()
        uid, _info = camera_sdk.login(cam["ip"], int(cam.get("port", 8000)),
                                      cam["username"], cam["password"])
        if uid < 0:
            q.put((False, f"camera did not accept the login "
                          f"(SDK error {camera_sdk.get_last_error()}) — "
                          f"check the address, user and password"))
            return
        camera_sdk.logout(uid)
        q.put((True, "ok"))
    except Exception as e:
        q.put((False, f"{type(e).__name__}: {e}"))


def _probe_camera(cam: dict, timeout: float = 25.0) -> tuple[bool, str]:
    """Проверить, что камера действительно отвечает, ДО того как записать её в конфиг.

    Иначе неверный пароль или опечатка в адресе попадут в cameras.json, и программа
    после перезапуска будет молча поднимать камеру, которой нет.

    Проверка идёт в отдельном процессе, и это не перестраховка. Первая версия звала
    SDK прямо здесь — и HCNetSDK оказывался загружен в главный процесс, где ему делать
    нечего: через несколько секунд после успешного входа он ронял весь сервер без
    трейсбека. Вся система построена на том, что SDK живёт только в процессах камер;
    проверка не имеет права быть исключением.
    """
    q: mp.Queue = mp.Queue(maxsize=1)
    p = mp.Process(target=_probe_worker, args=(cam, q), daemon=True)
    p.start()
    try:
        ok, why = q.get(timeout=timeout)
    except Exception:
        ok, why = False, "camera did not answer in time"
    p.join(timeout=3)
    if p.is_alive():
        p.terminate()
    return ok, why


@app.post("/api/cameras")
async def api_add_camera(request: Request):
    """Добавить камеру с сайта: подключение тем же SDK, что и у остальных."""
    d = await request.json()
    ip = str(d.get("ip", "")).strip()
    if not re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", ip):
        return {"ok": False, "error": "IP must look like 172.16.11.20"}
    if any(c["ip"] == ip for c in CAMERAS):
        return {"ok": False, "error": f"{ip} is already added"}

    models = d.get("models") or {}
    cam = {
        "id": f"cam_{max([int(c['id'].split('_')[-1]) for c in CAMERAS] + [0]) + 1:02d}",
        "name": str(d.get("name") or "").strip() or f"Camera {ip}",
        "ip": ip,
        "port": int(d.get("port") or 8000),
        "username": str(d.get("username") or "admin").strip(),
        "password": str(d.get("password") or ""),
        "channel": int(d.get("channel") or 1),
        "models": {k: bool(models.get(k, True)) for k in MODEL_KEYS},
    }
    bid = str(d.get("branch") or "").strip()
    if bid and any(b["id"] == bid for b in BRANCHES):
        cam["branch"] = bid

    ok, why = _probe_camera(cam)
    if not ok:
        logger.warning(f"[camera] {ip} refused: {why}")
        return {"ok": False, "error": why}

    # В конфиг пишем только то, что уже ответило. Файл — источник правды при
    # следующем запуске, и держать в нём непроверенное нельзя.
    CAMERAS.append(cam)
    try:
        _save_config()
    except Exception as e:
        CAMERAS.remove(cam)
        return {"ok": False, "error": f"cannot write config: {e}"}

    start_camera(cam)
    logger.info(f"[camera] added {cam['id']} {ip}, models={cam['models']}")
    return {"ok": True, "camera": {k: cam[k] for k in
                                   ("id", "name", "ip", "port", "channel", "models")}}


@app.delete("/api/cameras/{cam_id}")
def api_delete_camera(cam_id: str):
    """Убрать камеру: погасить её и вычеркнуть из конфига."""
    # Последнюю камеру раньше удалить было нельзя. Запрет ничего не защищал: система
    # без камер работает — сайт, база лиц и записи на месте, просто смотреть не на что,
    # — а вот заменить единственную камеру на другую он делал невозможным.
    cam = next((c for c in CAMERAS if c["id"] == cam_id), None)
    if cam is None:
        return {"ok": False, "error": "no such camera"}

    how = stop_camera(cam_id)
    CAMERAS[:] = [c for c in CAMERAS if c["id"] != cam_id]
    try:
        _save_config()
    except Exception as e:
        return {"ok": False, "error": f"stopped, but config not updated: {e}"}
    logger.info(f"[camera] removed {cam_id} ({cam['ip']}), {how}")
    return {"ok": True, "removed": cam_id, "detail": how}


@app.post("/api/cameras/{cam_id}/models")
async def api_camera_models(cam_id: str, request: Request):
    """Переключить модели на камере — без перезапуска."""
    cam = next((c for c in CAMERAS if c["id"] == cam_id), None)
    if cam is None:
        return {"ok": False, "error": "no such camera"}
    d = await request.json()
    models = {k: bool((d.get("models") or {}).get(k, True)) for k in MODEL_KEYS}
    cam["models"] = models
    try:
        _save_config()
    except Exception as e:
        return {"ok": False, "error": f"cannot write config: {e}"}
    logger.info(f"[camera] {cam_id} models -> {models}")
    return {"ok": True, "models": models}


def _parse_face_event(raw: bytes):
    """Разбирает multipart от камеры на (meta, вырезка лица, полный кадр).

    Камера шлёт три части: JSON с faceId/faceRect/faceScore и атрибутами, вырезанное
    лицо (3-8 КБ) и уменьшенный полный кадр (~250 КБ). Раз в секунду она же присылает
    короткое VMD-событие про движение — у него этих частей нет, и оно отсеивается тем,
    что meta не про faceCapture.
    """
    meta, face_jpg, bg_jpg = None, None, None
    for part in raw.split(b"--boundary"):
        part = part.strip(b"\r\n-")
        if not part:
            continue
        head, _, body = part.partition(b"\r\n\r\n")
        h = head.decode("utf-8", "replace")
        if "application/json" in h:
            try:
                meta = json.loads(body.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                pass
        elif "application/xml" in h or body.lstrip()[:5] == b"<?xml":
            # Событие про движение приходит XML'ем, а не JSON. Нам оно не нужно, но тип
            # надо распознать, иначе он выглядит как None и попадает в лог как незнакомый.
            m = re.search(r"<eventType>(.*?)</eventType>", body.decode("utf-8", "replace"))
            if m:
                meta = {"eventType": m.group(1)}
        elif 'name="faceImage"' in h:
            face_jpg = body
        elif 'name="backgroundImage"' in h:
            bg_jpg = body
    return meta, face_jpg, bg_jpg


@app.post("/api/camera-face-event")
@app.post("/api/camera-event")
async def camera_face_event(request: Request):
    """Лицо, пойманное самой камерой: узнаём кто это и заводим нового, если не знаем."""
    raw = await request.body()
    if FACE_EVENT_DUMP:
        try:
            d = Path(FACE_EVENT_DUMP)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{time.strftime('%H%M%S')}_{len(raw)}.bin").write_bytes(raw)
        except Exception as e:
            logger.error(f"[face-event] dump failed: {e}")

    meta, face_jpg, _bg = _parse_face_event(raw)
    etype = (meta or {}).get("eventType")
    if etype != "faceCapture":
        # Раз в секунду прилетает VMD (движение) — это норма, не шумим в лог.
        if etype != "VMD":
            logger.info(f"[face-event] ignored eventType={etype!r} ({len(raw)} bytes)")
        return {"ok": True, "skipped": f"eventType={etype}"}
    # Полный кадр предпочтительнее вырезки: камера присылает только одно лицо на событие,
    # а в кадре их может быть несколько. На вырезку откатываемся, если кадра не приложили.
    image = _bg or face_jpg
    if not image:
        logger.warning(f"[face-event] faceCapture without any image ({len(raw)} bytes)")
        return {"ok": True, "skipped": "no image attached"}

    cam_ip = str(meta.get("ipAddress") or "")
    camera_id = next((c["id"] for c in CAMERAS if c.get("ip") == cam_ip), cam_ip)
    cam_score = 0.0
    for block in meta.get("faceCapture", []):
        for f in block.get("faces", []):
            cam_score = max(cam_score, float(f.get("faceScore") or 0))

    t_recv = time.time()
    faces = await run_in_threadpool(_faces_in_jpeg, image)
    if not faces:
        logger.info(f"[face] {camera_id} event but our detector found no face "
                    f"(camera faceScore={cam_score:.0f})")
        return {"ok": True, "skipped": "no face found"}

    # По faceTime задержку не померить: часы камер не синхронны с этой машиной, замер
    # давал от -85 до +39 секунд. Судить о привязке приходится по тому, попало ли лицо
    # в чей-то трек, а не по времени.
    #
    # Кадр события и кадр конвейера — разного размера, поэтому переводим через долю.
    frame = _latest_frame.get(camera_id)
    img_wh = None
    probe = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_8)
    if probe is not None:
        img_wh = (probe.shape[1] * 8, probe.shape[0] * 8)

    results = []
    for f in faces:
        r = face_store.add_sighting(
            f["embedding"], face_score=cam_score, det_score=f["det_score"],
            camera_id=camera_id, crop=f["crop"], frame=image, bbox=f["bbox"])

        # Привязку ПРОБУЕМ всегда, даже для безымянных: вопрос «попадает ли лицо в
        # чей-то трек» отдельный от вопроса «есть ли у человека имя», и мерить его надо
        # на всех событиях, иначе данных почти нет. Имя при этом ставим только named.
        named = bool(r["label"]) and not r["label"].startswith("#")
        link = "no frame"
        if frame is not None and img_wh:
            x1, y1, x2, y2 = f["bbox"]
            sx, sy = frame.shape[1] / img_wh[0], frame.shape[0] / img_wh[1]
            face_box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            track = track_memory.bind_identity(
                camera_id=camera_id, person_id=r["person_id"],
                external_id=str(r["person_id"]),
                name=r["label"] if named else None, box=face_box)
            if track is None:
                live = sum(1 for t in track_memory.tracks if t.camera_id == camera_id)
                link = f"NO TRACK (live tracks on cam: {live})"
            else:
                link = f"-> {track.track_id}" + ("" if named else " (unnamed, not labelled)")
                r["track_id"] = track.track_id

        results.append(r)
        logger.info(
            f"[face] {camera_id} {'NEW' if r['is_new'] else 'match'} "
            f"person={r['label']} similarity={r['similarity']:.3f} "
            f"det={f['det_score']:.2f} size={f['size'][0]}x{f['size'][1]} {link}")
    return {"ok": True, "camera_id": camera_id, "faces": len(faces), "results": results}


@app.get("/api/faces")
def api_faces(limit: int = 200, branch: str = "", named: str = ""):
    """Люди. По филиалу или все сразу, названные или безымянные.

    Человек в базе один: карточка, векторы и снимок общие, — филиал лишь отвечает на
    вопрос «где его видели». Поэтому это фильтр списка, а не отдельная база: заведи
    филиалам свои базы, и один и тот же посетитель стал бы четырьмя разными людьми,
    а узнавание между объектами пропало бы.

    Оба отбора уходят в запрос: обрезать список после выборки значит показать не
    «двести названных», а «названных из двухсот последних», то есть десяток.
    """
    only = face_store.persons_in_branch(branch) if branch else None
    want = True if named == "1" else (False if named == "0" else None)
    persons = face_store.list_persons(limit, named=want, only_ids=only)
    where = face_store.branches_of([p["id"] for p in persons])
    names = {b["id"]: b.get("name") or b["id"] for b in BRANCHES}
    for p in persons:
        p["branches"] = [{**b, "name": names.get(b["branch"], b["branch"])}
                         for b in where.get(p["id"], [])]
    return {"stats": face_store.stats(), "persons": persons,
            "counts": {"named": face_store.count_persons(True, only),
                       "unnamed": face_store.count_persons(False, only)},
            "scope": branch or "all",
            "branches": [{"id": b["id"], "name": b.get("name") or b["id"]}
                         for b in BRANCHES]}


@app.get("/api/branch-alerts")
def api_branch_alerts(branch: str = "", history: int = 0, limit: int = 50):
    """Кого застали сразу в двух филиалах."""
    names = {b["id"]: b.get("name") or b["id"] for b in BRANCHES}
    out = face_store.branch_alerts(branch, include_acked=bool(history), limit=limit)
    for a in out:
        a["branch_name"] = names.get(a["branch"], a["branch"])
        a["other_branch_name"] = names.get(a["other_branch"], a["other_branch"])
    return {"alerts": out}


@app.post("/api/branch-alerts/ack")
async def api_branch_alerts_ack(request: Request):
    body = await request.json() if await request.body() else {}
    n = face_store.ack_alerts(ids=body.get("ids"), branch_id=body.get("branch", ""))
    return {"ok": True, "acked": n}


@app.get("/api/whereis")
def api_whereis():
    """Кто из названных людей виден прямо сейчас и на какой камере.

    Отвечает та же таблица закреплений, по которой рисуется имя над головой, поэтому
    страница показывает ровно то же, что и кадр: одна камера на человека, та, где лицо
    видно лучше всех.
    """
    now = time.time()
    with _name_owner_lock:
        claims = [(pid, cur[0], cur[2], cur[1]) for pid, cur in _name_owner.items()]

    people = []
    for pid, (cam, track), sim, seen_at in claims:
        if now - seen_at > NAME_CLAIM_TTL:
            continue
        label = face_store.label_of(pid)
        if not label or label.startswith("#"):
            continue
        people.append({"person_id": pid, "label": label, "camera_id": cam,
                       "track_id": track, "similarity": round(float(sim), 3),
                       "age": round(now - seen_at, 2)})
    people.sort(key=lambda p: p["label"].lower())
    return {"people": people}


@app.post("/api/shutdown")
def api_shutdown():
    """Выключение через HTTP — чтобы больше не убивать процесс снаружи.

    Именно внешнее убийство и ломало SDK: сессии на камерах оставались открытыми.
    """
    msg = shutdown_cameras()
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return {"ok": True, "detail": msg}


@app.get("/api/appearance")
def api_appearance(upper: str = "", lower: str = ""):
    """Во что одеты все, кого камеры видят прямо сейчас, и кто подходит под запрос.

    Пустое поле означает «всё равно», а не «unknown»: искать можно по одной только
    рубашке, и это главный случай — низ человека виден меньше чем в половине кадров.
    """
    people = _appearance_snapshot()
    upper, lower = upper.strip().lower(), lower.strip().lower()
    matches = [p for p in people
               if (not upper or p["upper"] == upper)
               and (not lower or p["lower"] == lower)]
    cams = sorted({p["camera_id"] for p in matches})
    return {"people": people, "matches": matches, "cameras": cams,
            "colours": list(COLOUR_NAMES)}


def _day_start(ts: float | None = None) -> float:
    d = datetime.fromtimestamp(ts or time.time())
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _day_bounds(day: str | None):
    """Границы суток по строке YYYY-MM-DD. Пустая строка — сегодня.

    Записи копятся день за днём и никуда не деваются, поэтому смотреть можно любой
    прошедший день — надо только уметь его назвать.
    """
    if not day:
        start = _day_start()
        return start, start + 86400, datetime.fromtimestamp(start).strftime("%Y-%m-%d")
    try:
        d = datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        start = _day_start()
        return start, start + 86400, datetime.fromtimestamp(start).strftime("%Y-%m-%d")
    start = d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return start, start + 86400, day


@app.get("/api/days")
def api_days():
    """Дни, за которые есть записи, — для выбора даты."""
    return {"days": store.recorded_days(),
            "today": datetime.now().strftime("%Y-%m-%d")}


def _idle_minutes() -> float:
    try:
        return float(store.get_setting("idle_minutes", IDLE_MINUTES_DEFAULT))
    except (TypeError, ValueError):
        return IDLE_MINUTES_DEFAULT


@app.get("/api/idle")
def api_idle(day: str = ""):
    """Кто из известных людей сейчас ничего не делает дольше заданного времени."""
    minutes = _idle_minutes()
    thr = minutes * 60.0
    start, end, day_text = _day_bounds(day)
    today = day_text == datetime.now().strftime("%Y-%m-%d")
    # «Сейчас» имеет смысл только для сегодняшнего дня; для прошедших остаётся история.
    api_idle_now = store.idle_now(thr, start, WORK_RESET_SEC) if today else []
    people = []
    for r in api_idle_now:
        label = face_store.label_of(r["person_id"])
        if not label or label.startswith("#"):
            continue          # безымянных не обвиняем в безделье
        people.append({**r, "label": label,
                       "idle_min": round(r["idle_sec"] / 60.0, 1)})
    people.sort(key=lambda p: -p["idle_sec"])

    history = []
    for r in store.idle_today(thr, start, day_end=end):
        label = face_store.label_of(r["person_id"])
        if label and not label.startswith("#"):
            history.append({**r, "label": label,
                            "duration_min": round((r["duration_sec"] or 0) / 60.0, 1)})
    return {"minutes": minutes, "work_reset_sec": WORK_RESET_SEC, "day": day_text,
            "is_today": today, "now": people, "today": history[:100]}


@app.post("/api/idle/threshold")
async def api_idle_threshold(request: Request):
    data = await request.json()
    try:
        minutes = float(data.get("minutes"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "minutes must be a number"}
    if not 0.1 <= minutes <= 600:
        return {"ok": False, "error": "minutes out of range"}
    store.set_setting("idle_minutes", minutes)
    logger.info(f"[idle] threshold set to {minutes} minutes")
    return {"ok": True, "minutes": minutes}


@app.get("/api/summary")
def api_summary(day: str = ""):
    """Сколько каждый работал, говорил по телефону и бездельничал за выбранный день."""
    minutes = _idle_minutes()
    start, end, day_text = _day_bounds(day)
    rows = store.daily_summary(start, minutes * 60.0, WORK_RESET_SEC, day_end=end)
    out = []
    for r in rows:
        label = face_store.label_of(r["person_id"])
        if not label or label.startswith("#"):
            continue          # безымянных не оцениваем: под номером нет человека
        out.append({**r, "label": label,
                    "working_min": round(r["working_sec"] / 60.0, 1),
                    "phone_min": round(r["phone_sec"] / 60.0, 1),
                    "idle_min": round(r["idle_sec"] / 60.0, 1),
                    "present_min": round(r["present_sec"] / 60.0, 1),
                    "longest_idle_min": round(r["longest_idle_sec"] / 60.0, 1)})
    return {"minutes": minutes, "day": day_text, "people": out}


@app.get("/api/period")
def api_period(days: int = 7):
    """Итоги за несколько дней: 3, 7, 30.

    Отсчёт от сегодня назад, включая сегодня: «за 7 дней» — это неделя, которая
    заканчивается сейчас, а не последняя полная. Оператор спрашивает «как отработали за
    неделю», стоя посреди дня, и ждёт, что сегодняшнее в ответе уже есть.
    """
    days = max(1, min(int(days or 7), 90))
    today = _day_start()
    starts = [today - 86400 * i for i in range(days)]
    rows = store.period_summary(starts, _idle_minutes() * 60.0, WORK_RESET_SEC)
    # Сколько дней из запрошенных вообще что-то содержат. Без этого числа страница
    # выглядит сломанной: нажимаешь 7, потом 30 — а цифры те же, потому что записей
    # всего за три дня. Пусть об этом говорит сама страница, а не догадка оператора.
    have = [d for d in store.recorded_days(120)
            if datetime.strptime(d, "%Y-%m-%d").timestamp() >= starts[-1]]
    out = []
    for r in rows:
        label = face_store.label_of(r["person_id"])
        if not label or label.startswith("#"):
            continue          # безымянных не оцениваем: под номером нет человека
        out.append({**r, "label": label,
                    "working_min": round(r["working_sec"] / 60.0, 1),
                    "phone_min": round(r["phone_sec"] / 60.0, 1),
                    "idle_min": round(r["idle_sec"] / 60.0, 1),
                    "present_min": round(r["present_sec"] / 60.0, 1),
                    "avg_working_min": round(r["avg_working_sec"] / 60.0, 1)})
    return {"days": days, "from": datetime.fromtimestamp(starts[-1]).strftime("%Y-%m-%d"),
            "to": datetime.fromtimestamp(today).strftime("%Y-%m-%d"),
            "days_with_records": len(have), "recorded": sorted(have), "people": out}


@app.get("/api/routes")
def api_routes(person_id: int | None = None, day: str = "", branch: str = ""):
    """Маршруты за сегодня: где человек был, по порядку.

    Если у филиала есть план этажа — маршрут идёт по ЗОНАМ плана, а не по камерам.
    Камера — точка зрения, а не место: две камеры, глядящие на один стол, давали
    «cam_03 -> cam_04 -> cam_03» по нулю минут при неподвижном человеке — имя
    перескакивало к той, что лучше видит лицо. Зона у обеих одна, и отрезок один.
    Без плана остаётся прежний маршрут по камерам.

    Соседние отрезки в ОДНОМ месте сливаем в один. Трек рвётся всякий раз, когда
    человека на секунду заслонили, и без склейки маршрут превращался в «cam_02 ->
    cam_02 -> cam_02» три десятка раз подряд — прочитать по нему, куда человек ходил,
    было невозможно.
    """
    now = time.time()
    start, end, day_text = _day_bounds(day)
    by_zones = bool(branch) and floor_map.has_plan(branch)
    if by_zones:
        rows = store.zone_routes(start, branch, person_id, limit=5000, day_end=end)
        for r in rows:
            r["place"] = r.pop("label")
    else:
        rows = store.routes(start, person_id, limit=5000, day_end=end)
        for r in rows:
            r["place"] = r["camera_id"]
    by_person: dict[int, list] = {}
    for r in rows:
        by_person.setdefault(int(r["person_id"]), []).append(r)

    out = []
    for pid, segs in by_person.items():
        label = face_store.label_of(pid)
        if not label or label.startswith("#"):
            continue
        path = []
        for s in sorted(segs, key=lambda x: x["start_ts"]):
            end = s["end_ts"]
            if path and path[-1]["place"] == s["place"]:
                # То же место — продолжение того же пребывания, а не новый переход.
                prev_end = path[-1]["to"]
                path[-1]["to"] = None if end is None else max(prev_end or 0, end)
                if prev_end is None:
                    path[-1]["to"] = None
                continue
            path.append({"place": s["place"], "camera_id": s["place"],
                         "from": s["start_ts"], "to": end})
        for p in path:
            p["seconds"] = round((p["to"] or now) - p["from"], 1)
        out.append({"person_id": pid, "label": label, "path": path})
    out.sort(key=lambda p: p["label"].lower())
    res = {"day": day_text, "people": out, "mode": "zones" if by_zones else "cameras"}
    if by_zones:
        # Что мешает маршруту быть полным: камеры без калибровки в план не попадают,
        # а без единой зоны и попадать некуда.
        res["uncalibrated"] = floor_map.uncalibrated(branch)
        res["zones"] = len((floor_map.branch(branch) or {}).get("zones") or [])
    return res


@app.get("/api/clothing")
def api_clothing(day: str = ""):
    """Во что каждый названный человек одет СЕГОДНЯ, по накопленным наблюдениям.

    Это та самая память, которой проверяется распознавание лица: показываем не только
    итоговый цвет, но и все счётчики — иначе не понять, почему кому-то отказали в имени.
    """
    day = day or _today()
    confirmed = face_store.clothing_day_map(day)
    counts = face_store.clothing_seen_counts(day)
    out = []
    for pid in set(confirmed) | set(counts):
        rec = confirmed.get(pid) or {}
        out.append({
            "person_id": pid, "label": face_store.label_of(pid),
            "upper": rec.get("upper"), "lower": rec.get("lower"),
            "face_sim": rec.get("face_sim"), "kp_conf": rec.get("kp_conf"),
            "camera_id": rec.get("camera_id"), "recorded_at": rec.get("created_at"),
            "confirmed": bool(rec),
            "sightings": counts.get(pid, 0),
            "sightings_needed": CLOTH_CONFIRM_TIMES,
            # Пока не признано — что именно уже засечено, чтобы было видно, сошлось
            # или расходится.
            "seen": ([] if rec else
                     [{"upper": r["upper"], "lower": r["lower"],
                       "camera_id": r["camera_id"], "at": r["created_at"]}
                      for r in face_store.clothing_seen(pid, day)]),
        })
    out.sort(key=lambda r: (r["confirmed"], r["recorded_at"] or r["sightings"]),
             reverse=True)
    return {"day": day, "people": out, "enabled": CLOTHES_VETO,
            "sightings_needed": CLOTH_CONFIRM_TIMES}


@app.get("/api/names")
def api_names():
    """Только имена — для подсказок в поиске, без снимков и счётчиков."""
    return {"names": sorted(
        (p["label"] for p in face_store.list_persons(1000) if p.get("named")),
        key=str.lower)}


@app.get("/api/faces/{person_id}/samples")
def api_face_samples(person_id: int):
    return {"person_id": person_id, "label": face_store.label_of(person_id),
            "samples": face_store.person_samples(person_id)}


@app.get("/api/faces/crop/{name}")
def api_face_crop(name: str):
    p = face_store.crop_path(name)
    if p is None:
        return Response(status_code=404)
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/faces/{person_id}/name")
async def api_face_rename(person_id: int, request: Request):
    data = await request.json()
    return face_store.rename(person_id, str(data.get("name") or ""))


@app.post("/api/faces/{person_id}/edit")
async def api_face_edit(person_id: int, request: Request):
    data = await request.json()
    return face_store.update_person(person_id, **data)


@app.post("/api/faces/{person_id}/merge")
async def api_face_merge(person_id: int, request: Request):
    data = await request.json()
    return face_store.merge(person_id, int(data["into"]))


@app.post("/api/activity/clear")
def api_clear_activity(scope: str = "today"):
    """Стереть записанные перемещения, простои и события распознавания."""
    res = store.clear_activity(_day_start() if scope == "today" else None)
    logger.info(f"[events] cleared {scope}: {res}")
    return {"ok": True, "scope": scope, **res}


@app.post("/api/faces/purge-unnamed")
def api_purge_unnamed():
    """Убрать все карточки без имени — и их следы в аналитике.

    Без второй половины на странице активности остались бы маршруты людей, которых
    в галерее уже нет.
    """
    before = {p["id"] for p in face_store.list_persons(100000)}
    res = face_store.delete_unnamed()
    after = {p["id"] for p in face_store.list_persons(100000)}
    res["activity_rows"] = store.forget_persons(before - after)
    return res


@app.delete("/api/faces/{person_id}")
def api_face_delete(person_id: int):
    return face_store.delete(person_id)


if __name__ == "__main__":
    # Создаём ДО запуска процессов: при spawn его нельзя передать задним числом.
    _stop_event = mp.Event()
    status_queue: mp.Queue = mp.Queue(maxsize=32)

    # Вычесть из отрезков время, когда программы не было. Штатную остановку закрывает
    # shutdown_cameras, но её могут и убить — тогда отрезок доживает до этого запуска,
    # и развёртка закроет его текущим временем, приписав человеку всю ночь.
    if EVENTS_ENABLED:
        try:
            r = store.repair_downtime()
            if r["clipped"] or r["closed"]:
                logger.info(f"[events] downtime repair: {r['gaps']} gap(s), "
                            f"{r['clipped']} segments clipped, {r['closed']} closed")
        except Exception as e:                                   # noqa: BLE001
            logger.error(f"[events] downtime repair: {e}")

    # Один раз на весь процесс, до запуска потоков камер: дальше все камеры работают
    # с этими же экземплярами.
    _load_shared_models()

    _status_queue = status_queue
    procs_info = _procs_info
    for cam in CAMERAS:
        start_camera(cam)
        # Пауза между подключениями камер — одновременный/близкий по времени логин
        # на HCNetSDK с нескольких камер роняет процесс без трейсбека (проверено:
        # каждая камера по отдельности подключается стабильно, вместе — нет).
        # Ту же паузу соблюдает watchdog при перезапуске.
        time.sleep(CAMERA_LOGIN_GAP)

    # Память об одежде старше недели уже ничего не проверяет — чистим её раз в час.
    def _prune_clothing_loop():
        while True:
            try:
                gone = face_store.prune_clothing()
                if gone:
                    logger.info(f"[cloth] pruned {gone} old clothing rows")
                    _clothes_forget_cache()
            except Exception as e:
                logger.error(f"clothing prune failed: {e}")
            try:
                gone = store.prune_positions(TRAIL_KEEP_DAYS)
                if gone:
                    logger.info(f"[map] pruned {gone} trail points older than "
                                f"{TRAIL_KEEP_DAYS} days")
            except Exception as e:
                logger.error(f"trail prune failed: {e}")
            time.sleep(3600)

    # Разовая уборка накопленного: файлы, на которые база не ссылается, и вторые
    # снимки одного человека. В отдельном потоке — тридцать тысяч файлов удаляются
    # не мгновенно, а камеры ждать этого не должны.
    def _prune_images_once():
        time.sleep(20)
        try:
            face_store.prune_images()
        except Exception as e:
            logger.error(f"image prune failed: {e}")

    threading.Thread(target=_prune_images_once, daemon=True).start()
    threading.Thread(target=_prune_clothing_loop, daemon=True).start()
    threading.Thread(target=_watchdog, args=(procs_info, status_queue), daemon=True).start()
    threading.Thread(target=_drain_status, args=(status_queue,), daemon=True).start()

    print("\n" + "=" * 40)
    print("  Skeleton Viewer running")
    print("  http://localhost:8000")
    print("  stop:  POST http://localhost:8000/api/shutdown  (or Ctrl+C)")
    print("=" * 40 + "\n")

    # Ctrl+C и «попросили завершиться» ведут к одному и тому же: закрыть сессии в SDK.
    def _on_signal(signum, _frame):
        logger.info(f"[shutdown] signal {signum}")
        shutdown_cameras()
        os._exit(0)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _on_signal)
        except (ValueError, OSError):
            pass
    atexit.register(shutdown_cameras)

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    finally:
        shutdown_cameras()
