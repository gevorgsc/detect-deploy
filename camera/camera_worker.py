"""
camera_worker.py — функция для отдельного процесса на камеру.

Hikvision SDK (HCNetSDK/PlayCtrl) не выдерживает несколько декодеров в одном
процессе на Windows (падает без трейсбека) — и точно так же падает, если в этом
же процессе ещё загружена CUDA/YOLO-модель. Поэтому этот процесс делает ТОЛЬКО
подключение через SDK и отдаёт кадры дальше. Распознавание скелета (YOLO)
выполняется отдельно, в основном процессе сайта (web_server.py).

Подключение к камере — ТОЛЬКО через Hikvision SDK (camera_sdk.py). RTSP не используется.
"""
import logging
import os
import queue
import time
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)

# Камеры отдают 3200x1800, но модели всё равно видят максимум 1920 (OBJ_IMGSZ=1920,
# POSE_IMGSZ=1280) — лишние пиксели letterbox выбрасывает.
TARGET_WIDTH = 1920

# Детекторы работают на 5 Гц (поза) и 1 Гц (объекты), а камера отдаёт ~20 fps.
# Лишние кадры нужны только для плавности картинки в браузере, но каждый из них
# стоит уменьшения и пересылки. Режем частоту ЗДЕСЬ, до всякой обработки.
# ВАЖНО: H.264-декодирование происходит раньше, внутри PlayCtrl, и этим не режется.
# Камера отдаёт ровно 20 fps, и декодер их все и распаковывает — независимо от того,
# сколько мы возьмём (замер: "decoder 20.0 fps -> sent 8.2 fps"). Значит самая
# дорогая часть, H.265-декодирование 3200x1800, оплачена в любом случае, и брать
# все кадры вместо каждого второго не добавляет к ней ничего: дорожает только то,
# что идёт ПОСЛЕ — уменьшение, cvtColor, оптический поток, отрисовка и JPEG.
# Отсюда 20: пока на камеру смотрят, отдаём каждый кадр, без прореживания вообще.
TARGET_FPS = 20.0
# РАВНО TARGET_FPS: прореживание, когда на стрим никто не смотрит, ОТКЛЮЧЕНО.
#
# Раньше здесь стояло 5.0, и рассуждение было такое: детекторам хватает 5 Гц, а
# промежуточные кадры дают только плавность картинки в браузере, ~25 ms-core каждый
# (~0.12 ядра на камеру). Для аналитики, где отрезки измеряются минутами, потери
# казались неважными.
#
# На практике важны. Поза и активность считаются НА КАЖДОМ кадре, и на редкой сетке
# короткое событие — взял телефон и положил, встал и сел — даёт единицы измерений
# вместо десятков и может не попасть в статистику вообще. Плюс замер показал, что
# на 5 fps сама частота позы падает с ~4.4 до ~3.7 Гц: шаг сетки (0.213 с) почти
# равен порогу детектора (0.19 с), и любой джиттер съедает целый цикл.
#
# Цена решения: те самые ~0.12 ядра на камеру. На 24 ядрах это не узкое место —
# упираемся мы в GPU, а он от частоты кадров не зависит, детекторы идут по времени.
# Чтобы вернуть экономию, достаточно снова поставить сюда 5.0.
IDLE_FPS = TARGET_FPS
SOURCE_FPS = 20.0  # что реально отдаёт декодер; замерено: "decoder 20.0 fps"


def _min_period(target_fps: float) -> float:
    """Порог прореживания под кадровую СЕТКУ источника.

    Кадры приходят не когда угодно, а узлами 1/SOURCE_FPS — и с джиттером. Порог,
    попавший НА узел, превращается в подбрасывание монетки: кадр, опоздавший на
    пару миллисекунд, отбрасывается целиком, а следующий шанс наступает только
    через 1/SOURCE_FPS. Так и получалось: при точном пороге 1/10 выходило 7.4 fps,
    при допуске 10% (0.09с при узлах 0.05 и 0.10) — 8.2 вместо 10.

    Поэтому ставим порог ровно ПОСЕРЕДИНЕ между узлами: полпериода сетки ниже
    желаемого. Запас до обоих соседей — 1/(2*SOURCE_FPS) = 25 мс, джиттер такого
    масштаба здесь не наблюдался.

    Следствие той же сетки: целыми кадрами достижимы только SOURCE_FPS/n, то есть
    20, 10, 6.7, 5. Промежуточные значения не существуют — 15 округлится до 10.

    Отдельный случай — «берём всё». Сетка равномерна лишь в среднем: PlayCtrl
    отдаёт кадры пачками, и два подряд бывают ближе 25 мс друг к другу. Любой
    ненулевой порог такие пары режет — с порогом 0.025с из 20 fps доходило 15.4.
    Поэтому когда просят не меньше, чем даёт источник, не фильтруем вовсе.
    """
    if target_fps >= SOURCE_FPS:
        return 0.0
    return max(1.0 / target_fps - 0.5 / SOURCE_FPS, 0.0)


_FRAME_MIN_PERIOD = _min_period(TARGET_FPS)   # 20 fps -> 0.025с, проходит каждый кадр
_IDLE_MIN_PERIOD = _min_period(IDLE_FPS)      # 5 fps  -> 0.175с, проходит каждый 4-й

FPS_LOG_PERIOD = 30.0  # как часто печатать фактическую частоту, сек

# Hikvision restarts its lock timer on every login attempt, even with the
# correct password. Do not keep retrying rejected credentials every 5 seconds.
LOGIN_LOCK_COOLDOWN = 30 * 60 + 5


def _login_retry_delay(error) -> float | None:
    code = getattr(error, "code", None)
    if code in (1, 152):
        return None  # incorrect credentials: require correction and restart
    if code == 153:
        return max(LOGIN_LOCK_COOLDOWN, getattr(error, "lock_seconds", 0) + 5)
    return 5.0

# OpenCV по умолчанию раскидывает каждую операцию на все логические ядра и ждёт их
# spin-wait'ом. На кадрах такого размера это почти чистые накладные расходы.
CV_THREADS = 4

# 0 — main stream (то, что камера пишет: 3200x1800), 1 — substream.
# Substream камера кодирует сама в низком разрешении: это единственный способ убрать
# софтверное H.264-декодирование 3200x1800 в PlayCtrl. По умолчанию 0: substream у
# Hikvision часто 704x576, и детектор на imgsz=1920 от этого потеряет в качестве.
# Переключать поштучно через "stream_type" в config/cameras.json.
DEFAULT_STREAM_TYPE = 0

# Кадр между процессами едет через общую память в формате YV12 — ровно как его отдал
# декодер, только уже уменьшенный. Раньше здесь был JPEG q90: кодирование тут плюс
# декодирование в web_server стоили ~0.3 ядра, и модель смотрела на кадр СКВОЗЬ это
# сжатие. Общая память не требует ни сериализации, ни копирования через pipe, и кадр
# доходит до модели ровно таким, каким его выдал декодер камеры.
# 12, а не 4. На четырёх камерах при полном 20 fps счётчик показывал
#   decoder 20.0 fps -> sent 8.0 fps; dropped: 0.0 by rate, 12.0 by no-slot
# при том, что GPU был занят на 15%, CPU на 3.7 ядра из 24, а цикл камеры простаивал
# 78-88% времени. То есть терялись кадры не из-за нехватки ресурсов, а из-за нехватки
# БУФЕРА: слот освобождается сразу после cvtColor, но дальше поток ещё считает
# детекторы, трекеры, отрисовку и JPEG, и на этих всплесках четырёх слотов не хватает.
# Сильнее всех страдали камеры 3200x1800 — у них заполнение слота дороже.
# Слот — это обычная оперативная память, ~3 МБ: 4 камеры по 12 слотов это 144 МБ.
SHM_SLOTS = int(os.environ.get("SHM_SLOTS", "12"))
SHM_SLOT_BYTES = TARGET_WIDTH * 1200 * 3 // 2  # запас под 16:10; YV12 = 1.5 байта/пиксель


def _target_size(w: int, h: int) -> tuple[int, int]:
    """Размер кадра после уменьшения. Высота округляется вниз до кратной 4: цветовые
    плоскости вдвое меньше по каждой оси, и их строки должны лечь в буфер целиком."""
    if w <= TARGET_WIDTH:
        return w, h
    return TARGET_WIDTH, int(round(h * TARGET_WIDTH / w)) // 4 * 4


def _write_slot(cv2, np, raw: bytes, w: int, h: int, buf) -> tuple[int, int]:
    """Уменьшает YV12-кадр ПО ПЛОСКОСТЯМ прямо в общую память.

    В YV12 плоскости лежат подряд: Y (h, w), затем V и U — каждая (h/2, w/2), то есть
    вчетверо меньше. Если сначала развернуть кадр в BGR, эта экономия теряется: цветовые
    плоскости раздуваются до полного размера, и resize приходится гнать по трём полным
    каналам вместо полутора. Замер на 3200x1800 -> 1920x1080:
        BGR сначала : 27.9 ms-core
        плоскости   : 14.1 ms-core   (расхождение картинки — 0.31/255)
    Возвращает фактический размер записанного кадра.
    """
    src = np.frombuffer(raw, dtype=np.uint8).reshape((h * 3 // 2, w))
    if w <= TARGET_WIDTH:
        dst = np.ndarray((h * 3 // 2, w), dtype=np.uint8, buffer=buf)
        dst[:] = src
        return w, h

    dw, dh = _target_size(w, h)
    dst = np.ndarray((dh * 3 // 2, dw), dtype=np.uint8, buffer=buf)

    cv2.resize(src[:h], (dw, dh), dst=dst[:dh], interpolation=cv2.INTER_AREA)
    cv2.resize(src[h:h + h // 4].reshape(h // 2, w // 2), (dw // 2, dh // 2),
               dst=dst[dh:dh + dh // 4].reshape(dh // 2, dw // 2),
               interpolation=cv2.INTER_AREA)
    cv2.resize(src[h + h // 4:].reshape(h // 2, w // 2), (dw // 2, dh // 2),
               dst=dst[dh + dh // 4:].reshape(dh // 2, dw // 2),
               interpolation=cv2.INTER_AREA)
    return dw, dh


def run_camera_worker(cam: dict, shm_names: list[str], free_q, filled_q, status_queue,
                      viewer_flag=None, stop_event=None):
    """Работает в отдельном процессе: подключается к камере через SDK и кладёт
    уменьшенные YV12-кадры в общую память, отдавая наружу только номер слота.

    viewer_flag — общий с web_server mp.Value: 1, если стрим этой камеры кто-то
    открыл в браузере. Пока он 0, частота падает до IDLE_FPS (см. комментарий там).

    stop_event — просьба завершиться. Без неё процесс убивали снаружи, и SDK оставался
    с открытой сессией на камере: StopRealPlay и Logout не вызывались никогда. После
    нескольких таких убийств камера начинала ронять любого, кто к ней подключался, и
    лечила это только перезагрузка машины. Теперь выход идёт через reader.stop().
    """
    import cv2
    import numpy as np
    from camera_sdk import HikvisionSDKFrameReader

    cv2.setNumThreads(CV_THREADS)

    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s [%(levelname)s] [{cam['id']}] %(message)s")

    cam_id = cam["id"]
    shms = [shared_memory.SharedMemory(name=n) for n in shm_names]

    def asked_to_stop():
        return stop_event is not None and stop_event.is_set()

    # A recovery launch can preserve an already observed lockout across restart.
    # Attempting login during that period would restart the device's timer.
    not_before = float(os.environ.get(f"CAMERA_LOGIN_NOT_BEFORE_{cam_id.upper()}", "0"))
    initial_delay = max(0.0, not_before - time.time())
    if initial_delay:
        message = f"waiting {initial_delay:.0f}s for camera account lockout to expire"
        logger.warning(message)
        status_queue.put((cam_id, False, message))
        if stop_event is None:
            time.sleep(initial_delay)
        elif stop_event.wait(initial_delay):
            return

    while not asked_to_stop():
        reader = HikvisionSDKFrameReader(
            ip=cam["ip"], port=cam["port"],
            username=cam["username"], password=cam["password"],
            channel=cam["channel"],
            stream_type=int(cam.get("stream_type", DEFAULT_STREAM_TYPE)),
        )
        try:
            logger.info(f"connecting to {cam['ip']}:{cam['port']} via HCNetSDK ...")
            reader.start()
            status_queue.put((cam_id, True, ""))
            logger.info("connected")
        except Exception as e:
            logger.error(f"connection failed: {e}")
            status_queue.put((cam_id, False, str(e)))
            retry_delay = _login_retry_delay(e)
            if retry_delay is None:
                logger.error("login attempts paused: correct camera credentials and restart")
                while not asked_to_stop():
                    if stop_event is None:
                        time.sleep(1)
                    else:
                        stop_event.wait(1)
                return
            if retry_delay > 5:
                logger.warning("camera user is locked; waiting %.0f seconds before retry", retry_delay)
            if stop_event is None:
                time.sleep(retry_delay)
            elif stop_event.wait(retry_delay):
                return
            continue

        last_sent_t = 0.0
        logged_size = False
        # Раз в FPS_LOG_PERIOD печатаем фактическую частоту: она теперь зависит от
        # того, смотрит ли кто-то стрим, и без этой строки переключение 10<->5
        # никак не проверить снаружи.
        fps_window_t = time.monotonic()
        fps_sent = 0
        # Считаем ещё и то, что ОТДАЛ декодер, и то, что пришлось выбросить из-за
        # занятых слотов. Иначе по одному «сколько отправили» не отличить «камера
        # столько и даёт» от «мы сами не успеваем разобрать».
        fps_recv = 0
        fps_noslot = 0
        while True:
            if asked_to_stop():
                break
            item = reader.read_raw(timeout=3)
            if item is None:
                logger.warning(f"stream lost (last_error={reader.last_error!r}), reconnecting")
                status_queue.put((cam_id, False, "stream lost"))
                break
            raw, w, h = item
            fps_recv += 1

            now = time.monotonic()
            watched = viewer_flag is None or viewer_flag.value
            if now - last_sent_t < (_FRAME_MIN_PERIOD if watched else _IDLE_MIN_PERIOD):
                continue  # выброшено до всякой обработки — самый дешёвый отказ от кадра

            out_w, out_h = _target_size(w, h)
            if not logged_size:
                logger.info(f"source {w}x{h} YV12 ({len(raw) / 1024 / 1024:.1f} MB) "
                            f"-> shared slot {out_w}x{out_h} "
                            f"({out_w * out_h * 3 / 2 / 1024 / 1024:.1f} MB)")
                logged_size = True

            # Проверяем размер ПОСЛЕ уменьшения — в слот кладётся именно он.
            if out_w * out_h * 3 // 2 > SHM_SLOT_BYTES:
                logger.error(f"frame {out_w}x{out_h} does not fit into shared slot, dropping")
                continue

            try:
                slot = free_q.get_nowait()
            except queue.Empty:
                fps_noslot += 1
                continue  # web_server не разобрал предыдущие кадры — пропускаем этот

            try:
                _write_slot(cv2, np, raw, w, h, shms[slot].buf)
            except Exception as e:
                try:
                    free_q.put_nowait(slot)
                except Exception:
                    pass
                logger.error(f"frame resize failed: {e}")
                continue

            last_sent_t = now
            filled_q.put((cam_id, slot, out_w, out_h))

            fps_sent += 1
            if now - fps_window_t >= FPS_LOG_PERIOD:
                dt = now - fps_window_t
                logger.info(f"decoder {fps_recv / dt:.1f} fps -> sent {fps_sent / dt:.1f} fps "
                            f"({'viewer' if watched else 'idle'}); "
                            f"dropped: {(fps_recv - fps_sent - fps_noslot) / dt:.1f} by rate, "
                            f"{fps_noslot / dt:.1f} by no-slot")
                fps_window_t, fps_sent, fps_recv, fps_noslot = now, 0, 0, 0

        # Закрыть сессию в SDK нужно в любом случае — и когда потеряли поток, и когда
        # нас попросили выйти. Иначе на камере остаётся висеть логин.
        try:
            reader.stop()
        except Exception as e:
            logger.error(f"reader.stop() failed: {e}")
        if asked_to_stop():
            logger.info("stop requested, SDK session closed")
            return
        time.sleep(2)
