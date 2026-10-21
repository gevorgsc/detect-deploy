"""
camera_sdk.py — обёртка над Hikvision HCNetSDK + PlayCtrl (ctypes) для логина
на камеру/регистратор и получения декодированных BGR-кадров (OpenCV).

DLL берутся из SDK/lib рядом с этим файлом (HCNetSDK.dll, HCCore.dll, hpr.dll,
PlayCtrl.dll + папка HCNetSDKCom).
"""
import ctypes
import logging
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SDK_DIR = Path(__file__).resolve().parent / "SDK" / "lib"

_hpr: ctypes.CDLL | None = None
_hcore: ctypes.CDLL | None = None
_sdk: ctypes.CDLL | None = None
_play: ctypes.CDLL | None = None
_initialized = False
_dll_dirs: list[Any] = []
_sdk_call_lock = threading.RLock()

CALLBACK = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)

NET_DVR_SYSHEAD = 1
NET_DVR_STREAMDATA = 2
STREAME_REALTIME = 0
T_YV12 = 3


class NET_DVR_DEVICEINFO_V30(ctypes.Structure):
    _fields_ = [
        ("sSerialNumber", ctypes.c_byte * 48),
        ("byAlarmInPortNum", ctypes.c_byte),
        ("byAlarmOutPortNum", ctypes.c_byte),
        ("byDiskNum", ctypes.c_byte),
        ("byDVRType", ctypes.c_byte),
        ("byChanNum", ctypes.c_byte),
        ("byStartChan", ctypes.c_byte),
        ("byAudioChanNum", ctypes.c_byte),
        ("byIPChanNum", ctypes.c_byte),
        ("byZeroChanNum", ctypes.c_byte),
        ("byMainProto", ctypes.c_byte),
        ("bySubProto", ctypes.c_byte),
        ("bySupport", ctypes.c_byte),
        ("bySupport1", ctypes.c_byte),
        ("bySupport2", ctypes.c_byte),
        ("wDevType", ctypes.c_uint16),
        ("bySupport3", ctypes.c_byte),
        ("byMultiStreamProto", ctypes.c_byte),
        ("byStartDChan", ctypes.c_byte),
        ("byStartDTalkChan", ctypes.c_byte),
        ("byHighDChanNum", ctypes.c_byte),
        ("bySupport4", ctypes.c_byte),
        ("byLanguageType", ctypes.c_byte),
        ("byVoiceInChanNum", ctypes.c_byte),
        ("byStartVoiceInChanNo", ctypes.c_byte),
        ("byRes3", ctypes.c_byte * 2),
        ("byMirrorChanNum", ctypes.c_byte),
        ("wStartMirrorChanNo", ctypes.c_uint16),
        ("byRes2", ctypes.c_byte * 2),
    ]


class NET_DVR_DEVICEINFO_V40(ctypes.Structure):
    """То, что NET_DVR_Login_V40 на самом деле заполняет.

    Раньше сюда передавали NET_DVR_DEVICEINFO_V30 — 80 байт. Но V40-логин пишет
    V40-структуру: те же 80 байт плюс ещё ~270. SDK честно заполнял их все — поверх
    того, что лежало в памяти следом. Что именно там лежало, решал случай: в одни дни
    это были свободные байты, и всё работало; после перезагрузки раскладка памяти
    менялась, следом оказывался живой объект Python — и процесс падал с access
    violation в python311.dll сразу после входа. Так выглядели «загадочные» смерти
    воркеров с кодом 3221225477.

    Хвост byRes2 взят с запасом больше, чем в заголовке SDK: разные версии SDK
    отличаются на десятки байт, и лишнее место безопасно, а недостающее — нет.
    """
    _fields_ = [
        ("struDeviceV30", NET_DVR_DEVICEINFO_V30),
        ("bySupportLock", ctypes.c_byte),
        ("byRetryLoginTime", ctypes.c_byte),
        ("byPasswordLevel", ctypes.c_byte),
        ("byProxyType", ctypes.c_byte),
        ("dwSurplusLockTime", ctypes.c_uint32),
        ("byCharEncodeType", ctypes.c_byte),
        ("bySupportDev5", ctypes.c_byte),
        ("bySupport", ctypes.c_byte),
        ("byLoginMode", ctypes.c_byte),
        ("dwOEMCode", ctypes.c_uint32),
        ("iResidualValidity", ctypes.c_int32),
        ("byResidualValidity", ctypes.c_byte),
        ("bySingleStartDTalkChan", ctypes.c_byte),
        ("bySingleDTalkChanNums", ctypes.c_byte),
        ("byPassWordResetLevel", ctypes.c_byte),
        ("bySupportStreamEncrypt", ctypes.c_byte),
        ("byMarketType", ctypes.c_byte),
        ("byRes2", ctypes.c_byte * 238),
        ("_reserve", ctypes.c_byte * 512),      # запас на версии SDK с длинным хвостом
    ]


class NET_DVR_USER_LOGIN_INFO(ctypes.Structure):
    _fields_ = [
        ("sDeviceAddress", ctypes.c_char * 129),
        ("byUseTransport", ctypes.c_byte),
        ("wPort", ctypes.c_uint16),
        ("sUserName", ctypes.c_char * 64),
        ("sPassword", ctypes.c_char * 64),
        ("cbLoginResult", ctypes.c_void_p),
        ("pUser", ctypes.c_void_p),
        ("bUseAsynLogin", ctypes.c_int32),
        ("byProxyType", ctypes.c_byte),
        ("byUseUTCTime", ctypes.c_byte),
        ("byLoginMode", ctypes.c_byte),
        ("byHttps", ctypes.c_byte),
        ("iProxyID", ctypes.c_int32),
        ("byVerifyMode", ctypes.c_byte),
        ("byRes3", ctypes.c_byte * 119),
    ]


class NET_DVR_PREVIEWINFO(ctypes.Structure):
    _fields_ = [
        ("lChannel", ctypes.c_int32),
        ("dwStreamType", ctypes.c_uint32),
        ("dwLinkMode", ctypes.c_uint32),
        ("hPlayWnd", ctypes.c_void_p),
        ("bBlocked", ctypes.c_int32),
        ("bPassbackRecord", ctypes.c_int32),
        ("byPreviewMode", ctypes.c_byte),
        ("byStreamID", ctypes.c_byte * 32),
        ("byProtoType", ctypes.c_byte),
        ("byRes1", ctypes.c_byte),
        ("byVideoCodingType", ctypes.c_byte),
        ("dwDisplayBufNum", ctypes.c_uint32),
        ("byNPQMode", ctypes.c_byte),
        ("byRes", ctypes.c_byte * 215),
    ]


REALDATACALLBACK = CALLBACK(
    None, ctypes.c_long, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_byte), ctypes.c_uint32, ctypes.c_void_p,
)


class FRAME_INFO(ctypes.Structure):
    _fields_ = [
        ("nWidth", ctypes.c_long),
        ("nHeight", ctypes.c_long),
        ("nStamp", ctypes.c_long),
        ("nType", ctypes.c_long),
        ("nFrameRate", ctypes.c_long),
        ("dwFrameNum", ctypes.c_uint32),
    ]


DECCALLBACK = CALLBACK(
    None, ctypes.c_long, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_long,
    ctypes.POINTER(FRAME_INFO), ctypes.c_long, ctypes.c_long,
)


def _add_dll_directory(path: Path):
    if hasattr(__import__("os"), "add_dll_directory") and path.exists():
        import os
        _dll_dirs.append(os.add_dll_directory(str(path)))


def _load_dll(name: str) -> ctypes.CDLL:
    dll_path = _SDK_DIR / name
    if not dll_path.exists():
        raise FileNotFoundError(f"Hikvision SDK DLL not found: {dll_path}")
    return ctypes.CDLL(str(dll_path))


def _configure_signatures():
    assert _sdk is not None
    assert _play is not None

    _sdk.NET_DVR_Init.restype = ctypes.c_bool
    _sdk.NET_DVR_Cleanup.restype = ctypes.c_bool
    _sdk.NET_DVR_GetLastError.restype = ctypes.c_uint32

    _sdk.NET_DVR_Login_V40.argtypes = [
        ctypes.POINTER(NET_DVR_USER_LOGIN_INFO),
        ctypes.POINTER(NET_DVR_DEVICEINFO_V40),
    ]
    _sdk.NET_DVR_Login_V40.restype = ctypes.c_long

    _sdk.NET_DVR_Logout.argtypes = [ctypes.c_long]
    _sdk.NET_DVR_Logout.restype = ctypes.c_bool

    _sdk.NET_DVR_RealPlay_V40.argtypes = [
        ctypes.c_long, ctypes.POINTER(NET_DVR_PREVIEWINFO),
        REALDATACALLBACK, ctypes.c_void_p,
    ]
    _sdk.NET_DVR_RealPlay_V40.restype = ctypes.c_long

    _sdk.NET_DVR_StopRealPlay.argtypes = [ctypes.c_long]
    _sdk.NET_DVR_StopRealPlay.restype = ctypes.c_bool

    _sdk.NET_DVR_SetConnectTime.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    _sdk.NET_DVR_SetConnectTime.restype = ctypes.c_bool

    _play.PlayM4_GetPort.argtypes = [ctypes.POINTER(ctypes.c_long)]
    _play.PlayM4_GetPort.restype = ctypes.c_bool
    _play.PlayM4_FreePort.argtypes = [ctypes.c_long]
    _play.PlayM4_FreePort.restype = ctypes.c_bool
    _play.PlayM4_SetStreamOpenMode.argtypes = [ctypes.c_long, ctypes.c_uint32]
    _play.PlayM4_SetStreamOpenMode.restype = ctypes.c_bool
    _play.PlayM4_OpenStream.argtypes = [
        ctypes.c_long, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint32, ctypes.c_uint32,
    ]
    _play.PlayM4_OpenStream.restype = ctypes.c_bool
    _play.PlayM4_InputData.argtypes = [
        ctypes.c_long, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint32,
    ]
    _play.PlayM4_InputData.restype = ctypes.c_bool
    _play.PlayM4_SetDecCallBack.argtypes = [ctypes.c_long, DECCALLBACK]
    _play.PlayM4_SetDecCallBack.restype = ctypes.c_bool
    _play.PlayM4_Play.argtypes = [ctypes.c_long, ctypes.c_void_p]
    _play.PlayM4_Play.restype = ctypes.c_bool
    _play.PlayM4_Stop.argtypes = [ctypes.c_long]
    _play.PlayM4_Stop.restype = ctypes.c_bool
    _play.PlayM4_CloseStream.argtypes = [ctypes.c_long]
    _play.PlayM4_CloseStream.restype = ctypes.c_bool
    _play.PlayM4_GetLastError.argtypes = [ctypes.c_long]
    _play.PlayM4_GetLastError.restype = ctypes.c_uint32


def _ensure_loaded():
    global _hpr, _hcore, _sdk, _play
    with _sdk_call_lock:
        if _sdk is not None:
            return
        _add_dll_directory(_SDK_DIR)
        _add_dll_directory(_SDK_DIR / "HCNetSDKCom")
        _hpr = _load_dll("hpr.dll")
        _hcore = _load_dll("HCCore.dll")
        _play = _load_dll("PlayCtrl.dll")
        _sdk = _load_dll("HCNetSDK.dll")
        _configure_signatures()


def init() -> bool:
    global _initialized
    with _sdk_call_lock:
        _ensure_loaded()
        if _initialized:
            return True
        assert _sdk is not None
        ok = _sdk.NET_DVR_Init()
        if ok:
            _sdk.NET_DVR_SetConnectTime(5000, 3)
            _initialized = True
            logger.info("HCNetSDK initialized")
        return ok


def get_last_error() -> int:
    _ensure_loaded()
    assert _sdk is not None
    return int(_sdk.NET_DVR_GetLastError())


class CameraLoginError(ConnectionError):
    """Keep authentication metadata so callers do not extend camera lockouts."""

    def __init__(self, ip: str, port: int, code: int, lock_seconds: int = 0):
        self.code = code
        self.lock_seconds = lock_seconds
        super().__init__(f"Login failed for {ip}:{port}: error code {code}")


def login(ip: str, port: int, username: str, password: str) -> tuple[int, NET_DVR_DEVICEINFO_V30]:
    with _sdk_call_lock:
        if not init():
            raise RuntimeError(f"HCNetSDK init failed: error code {get_last_error()}")
        assert _sdk is not None
        login_info = NET_DVR_USER_LOGIN_INFO()
        login_info.sDeviceAddress = ip.encode()
        login_info.wPort = port
        login_info.sUserName = username.encode()
        login_info.sPassword = password.encode()
        login_info.bUseAsynLogin = 0

        device_info = NET_DVR_DEVICEINFO_V40()
        user_id = _sdk.NET_DVR_Login_V40(ctypes.byref(login_info), ctypes.byref(device_info))
        if user_id < 0:
            err = get_last_error()
            raise CameraLoginError(ip, port, err, int(device_info.dwSurplusLockTime))
        logger.info(f"Logged in to {ip}:{port} user_id={user_id}")
        # Наружу отдаём прежнюю V30-часть: всем, кто читает результат, нужна она.
        return int(user_id), device_info.struDeviceV30


def logout(user_id: int):
    if not _sdk or user_id < 0:
        return
    _sdk.NET_DVR_Logout(user_id)
    logger.info(f"Logged out user_id={user_id}")


class HikvisionSDKFrameReader:
    """Логинится на камеру и отдаёт декодированные BGR-кадры (OpenCV)."""

    def __init__(self, ip: str, port: int, username: str, password: str,
                 channel: int, buffer_size: int = 2 * 1024 * 1024, stream_type: int = 0):
        self.ip, self.port = ip, port
        self.username, self.password = username, password
        self.channel = channel
        self.buffer_size = buffer_size
        self.stream_type = stream_type  # 0 — main stream, 1 — substream (см. camera_worker.py)

        self.user_id = -1
        self.real_handle = -1
        self.play_port = ctypes.c_long(-1)
        self.stream_open = False
        self.running = False
        self.last_error = ""

        self._frames: queue.Queue[tuple[bytes, int, int]] = queue.Queue(maxsize=2)
        self._stream_packets: queue.Queue[tuple[int, bytes] | None] = queue.Queue(maxsize=128)
        self._play_buffers = deque(maxlen=256)
        self._pump_thread: threading.Thread | None = None
        self._real_cb = REALDATACALLBACK(self._on_real_data)
        self._dec_cb = DECCALLBACK(self._on_decoded_frame)

    def start(self):
        with _sdk_call_lock:
            _ensure_loaded()
            assert _sdk is not None and _play is not None
            if not _play.PlayM4_GetPort(ctypes.byref(self.play_port)):
                raise RuntimeError("PlayM4_GetPort failed")

            self.user_id, _ = login(self.ip, self.port, self.username, self.password)
            self.running = True
            self._pump_thread = threading.Thread(target=self._pump_stream_data, daemon=True)
            self._pump_thread.start()

            preview_info = NET_DVR_PREVIEWINFO()
            preview_info.lChannel = self.channel
            preview_info.dwStreamType = self.stream_type
            preview_info.dwLinkMode = 0
            preview_info.hPlayWnd = None
            preview_info.bBlocked = 0
            preview_info.dwDisplayBufNum = 1

            self.real_handle = int(_sdk.NET_DVR_RealPlay_V40(
                self.user_id, ctypes.byref(preview_info), self._real_cb, None,
            ))
            if self.real_handle < 0:
                self.running = False
                raise RuntimeError(f"RealPlay failed: error {get_last_error()}")
        logger.info(f"SDK preview started ip={self.ip} channel={self.channel} "
                    f"stream_type={self.stream_type}")

    def read_raw(self, timeout: float = 2.0) -> tuple[bytes, int, int] | None:
        """Кадр как есть — YV12-байты из декодера, без цветового преобразования.

        cvtColor на 3200x1800 стоит 7.2 ms-core, и делать его внутри read() значит
        платить за кадры, которые вызывающий код потом всё равно выбросит по частоте.
        Кто именно и когда разворачивает YV12 в BGR — решает вызывающий."""
        if not self.running:
            return None
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def read(self, timeout: float = 2.0) -> np.ndarray | None:
        if not self.running:
            return None
        try:
            raw, width, height = self._frames.get(timeout=timeout)
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((int(height * 3 / 2), width))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YV12)
        except queue.Empty:
            return None
        except Exception as e:
            self.last_error = f"Frame conversion failed: {e}"
            logger.error(f"SDK frame conversion error [{self.ip}]: {e}")
            return None

    def stop(self):
        self.running = False
        try:
            self._stream_packets.put_nowait(None)
        except queue.Full:
            pass
        if _sdk and self.real_handle >= 0:
            with _sdk_call_lock:
                _sdk.NET_DVR_StopRealPlay(self.real_handle)
            self.real_handle = -1
        if self._pump_thread and self._pump_thread.is_alive():
            self._pump_thread.join(timeout=2)
        self._pump_thread = None
        if _play and self.play_port.value >= 0:
            try:
                with _sdk_call_lock:
                    _play.PlayM4_Stop(self.play_port.value)
                    _play.PlayM4_CloseStream(self.play_port.value)
                    _play.PlayM4_FreePort(self.play_port.value)
            finally:
                self.play_port = ctypes.c_long(-1)
                self.stream_open = False
        if self.user_id >= 0:
            logout(self.user_id)
            self.user_id = -1

    def _on_real_data(self, handle, data_type, buffer, size, user):
        if not self.running or not buffer or size <= 0:
            return
        try:
            packet = (int(data_type), ctypes.string_at(buffer, int(size)))
            try:
                self._stream_packets.put_nowait(packet)
            except queue.Full:
                if data_type != NET_DVR_SYSHEAD:
                    try:
                        self._stream_packets.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._stream_packets.put_nowait(packet)
                    except queue.Full:
                        pass
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"SDK real data callback error [{self.ip}]: {e}")

    def _pump_stream_data(self):
        while self.running:
            try:
                packet = self._stream_packets.get(timeout=1)
            except queue.Empty:
                continue
            if packet is None:
                break
            data_type, data = packet
            try:
                if data_type == NET_DVR_SYSHEAD:
                    self._open_play_stream(data)
                elif data_type == NET_DVR_STREAMDATA and self.stream_open:
                    self._input_stream_data(data)
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"SDK stream pump error [{self.ip}]: {e}")

    def _open_play_stream(self, data: bytes):
        assert _play is not None
        if self.stream_open:
            return
        port = self.play_port.value
        buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        self._play_buffers.append(buf)

        if not _play.PlayM4_SetStreamOpenMode(port, STREAME_REALTIME):
            raise RuntimeError(f"PlayM4_SetStreamOpenMode failed: {self._play_error()}")
        if not _play.PlayM4_OpenStream(port, buf, len(data), self.buffer_size):
            raise RuntimeError(f"PlayM4_OpenStream failed: {self._play_error()}")
        if not _play.PlayM4_SetDecCallBack(port, self._dec_cb):
            raise RuntimeError(f"PlayM4_SetDecCallBack failed: {self._play_error()}")
        if not _play.PlayM4_Play(port, None):
            raise RuntimeError(f"PlayM4_Play failed: {self._play_error()}")
        self.stream_open = True

    def _input_stream_data(self, data: bytes):
        assert _play is not None
        buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        self._play_buffers.append(buf)
        if not _play.PlayM4_InputData(self.play_port.value, buf, len(data)):
            self.last_error = f"PlayM4_InputData failed: {self._play_error()}"

    def _on_decoded_frame(self, port, buffer, size, frame_info, reserved1, reserved2):
        if not self.running or not buffer or size <= 0 or not frame_info:
            return
        info = frame_info.contents
        if info.nType != T_YV12 or info.nWidth <= 0 or info.nHeight <= 0:
            return
        expected = int(info.nWidth * info.nHeight * 3 / 2)
        if size < expected:
            return
        try:
            raw = ctypes.string_at(buffer, expected)
            self._put_frame(raw, int(info.nWidth), int(info.nHeight))
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"SDK decode callback error [{self.ip}]: {e}")

    def _put_frame(self, raw: bytes, width: int, height: int):
        item = (raw, width, height)
        try:
            self._frames.put_nowait(item)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(item)
            except queue.Full:
                pass

    def _play_error(self) -> int | None:
        if not _play or self.play_port.value < 0:
            return None
        return int(_play.PlayM4_GetLastError(self.play_port.value))
