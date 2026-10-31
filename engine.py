"""Camera, background capture, gestures, effects and virtual camera, without any UI.

The window drives an Engine from a background thread: it posts commands and
reads one processed frame plus a small status snapshot per iteration.
"""

from __future__ import annotations

import argparse
import os
import platform
import queue
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .effects import IN, OUT, Effect, create_effect
    from .gestures import DetectionWorker, HoldTrigger
    from .settings import MODE_INFO, MODES, Settings
    from .vanish_sound import ModeSounds
except ImportError:
    from effects import IN, OUT, Effect, create_effect
    from gestures import DetectionWorker, HoldTrigger
    from settings import MODE_INFO, MODES, Settings
    from vanish_sound import ModeSounds


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
HAND_MODEL = APP_DIR / "hand_landmarker.task"
FACE_MODEL = APP_DIR / "face_landmarker.task"

MODEL_URLS = {
    SEGMENTER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    GESTURE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
}

CALIBRATION_SECONDS = 4.0
# How long a sign must be held before it fires, per mode (Goku's is a setting).
# Particles fires ~0.5 s after the V sign shows, like the original demo.
SIGN_HOLD_SECONDS = {"normal": 0.4, "particles": 0.4, "naruto": 0.5}
BACKGROUND_SAMPLE_SECONDS = 0.55
DETECTION_INTERVAL = 0.05
# Hand, face and person tracking run on a copy no wider than this.
ANALYSIS_WIDTH = 1280
# Width of the live view the preview shows while you are vanished (never sent out).
LIVE_VIEW_WIDTH = 384
CAMERA_PERMISSION_WAIT = 30.0
IS_MAC = platform.system() == "Darwin"


def log(message: str) -> None:
    print(f"[VanishCam] {message}", flush=True)


# ------------------------------------------------------------------- devices

@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str


def camera_devices() -> list[CameraDevice]:
    """Webcams with their names, numbered the way OpenCV opens them."""
    if IS_MAC:
        try:
            import AVFoundation

            devices = (list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(
                AVFoundation.AVMediaTypeVideo))
                + list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(
                    AVFoundation.AVMediaTypeMuxed)))
            # OpenCV's AVFoundation backend orders devices by unique ID.
            devices.sort(key=lambda device: str(device.uniqueID()))
            return [CameraDevice(index, str(device.localizedName()))
                    for index, device in enumerate(devices)]
        except Exception:
            pass
    elif platform.system() == "Linux":
        found = []
        for node in sorted(Path("/sys/class/video4linux").glob("video*")):
            try:
                index = int(node.name[5:])
                name = (node / "name").read_text().strip()
            except (OSError, ValueError):
                continue
            found.append(CameraDevice(index, name))
        if found:
            return found
    elif platform.system() == "Windows":
        try:
            from PySide6.QtMultimedia import QMediaDevices

            names = [device.description() for device in QMediaDevices.videoInputs()]
            if names:
                return [CameraDevice(index, name) for index, name in enumerate(names)]
        except Exception:
            pass
    return [CameraDevice(index, f"Camera {index}") for index in probe_cameras()]


def probe_cameras(limit: int = 5) -> list[int]:
    working = []
    for index in range(limit):
        capture = _capture(index)
        ok = capture.isOpened() and capture.read()[0]
        capture.release()
        if ok:
            working.append(index)
    return working


def _capture(index: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
    elif IS_MAC:
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def open_camera(index: int, width: int, height: int, fps: float,
                wait_for_permission: bool = True) -> cv2.VideoCapture:
    capture = _capture(index)
    deadline = time.perf_counter() + (CAMERA_PERMISSION_WAIT if IS_MAC and wait_for_permission else 0)
    # On macOS the first attempt fails while the camera permission dialog is open.
    while not capture.isOpened() and time.perf_counter() < deadline:
        capture.release()
        time.sleep(1.0)
        capture = _capture(index)
    if not capture.isOpened():
        capture.release()
        if IS_MAC:
            raise RuntimeError(
                "The camera could not be opened. Allow camera access for the app you started "
                "VanishCam from in System Settings > Privacy & Security > Camera, then restart "
                "that app. If another app is using the camera, close it or pick another camera."
            )
        raise RuntimeError("The camera is not available. Close other apps using it or pick "
                           "another camera.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def open_virtual_camera(args: argparse.Namespace, settings: Settings, width: int, height: int):
    if not settings.virtual_camera:
        return None
    explicit = args.virtual_camera is True or args.virtual_backend or args.virtual_device
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = f"Virtual camera not available: {error}"
        if explicit:
            raise RuntimeError(message) from error
        log(message)
        return None
    log(f"Virtual camera on: {camera.device} ({width}x{height}, up to {args.fps:g} FPS).")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


def download_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if destination.exists() and destination.stat().st_size > 100_000:
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        log(f"Downloading {destination.name} (first run only)...")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "VanishCam/2.0"})
            with urllib.request.urlopen(request, timeout=60) as response, \
                    temporary.open("wb") as output:
                while chunk := response.read(1024 * 256):
                    output.write(chunk)
            os.replace(temporary, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the MediaPipe models. Check your internet connection "
                "and try again."
            ) from error


def create_segmenter() -> vision.ImageSegmenter:
    return vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    ))


def person_mask(segmenter, rgb: np.ndarray, timestamp_ms: int,
                size: tuple[int, int] | None = None) -> np.ndarray:
    """Soft person mask, scaled to `size` (width, height) or to the input image."""
    width, height = size or (rgb.shape[1], rgb.shape[0])
    result = segmenter.segment_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    # Soften the edge at 960 px wide, where it is cheap, then scale to the frame.
    small = (960, max(1, round(height * 960 / width))) if width > 960 else (width, height)
    soft = cv2.GaussianBlur(cv2.resize(raw, small, interpolation=cv2.INTER_LINEAR),
                            (0, 0), sigmaX=2.2, sigmaY=2.2)
    if small != (width, height):
        soft = cv2.resize(soft, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip((soft - 0.12) / 0.68, 0.0, 1.0)


def model_paths() -> dict[str, str]:
    return {"gesture": str(GESTURE_MODEL), "hand": str(HAND_MODEL), "face": str(FACE_MODEL)}


# -------------------------------------------------------------------- engine

@dataclass
class State:
    # setup -> calibrating -> ready <-> vanishing -> vanished -> appearing -> ready
    phase: str = "setup"
    started: float = 0.0
    background: np.ndarray | None = None
    samples: list = field(default_factory=list)
    mask: np.ndarray | None = None
    mask_at: float = float("-inf")
    vanished_frame: np.ndarray | None = None


@dataclass
class Status:
    """What the window needs to draw its overlays and controls for one frame."""
    mode: str
    phase: str
    countdown: float | None
    hold: float
    camera: int
    virtual_device: str | None
    virtual_error: str
    debug: str
    toasts: list[str]
    error: str
    waiting: str
    sign: bool = False                 # The latest detection saw the trigger sign.
    blocked: bool = False              # The sign is still up from last time: lower it first.
    live: np.ndarray | None = None     # Small raw camera view while vanished, preview only.


class Engine:
    def __init__(self, args: argparse.Namespace, settings: Settings):
        self.args = args
        self.settings = settings
        self.commands: queue.SimpleQueue = queue.SimpleQueue()
        self.state = State()
        self.effects: dict[str, Effect] = {}
        self.effect: Effect | None = None
        # The mode whose effect is loaded. The window shares `settings` and
        # updates settings.mode itself, so that field cannot tell us.
        self.mode: str | None = None
        self.sounds = ModeSounds(settings.sound, settings.volume)
        self.trigger = HoldTrigger()
        self.clock_start = time.perf_counter()
        self.capture = None
        self.segmenter = None
        self.worker = None
        self.virtual_camera = None
        self.virtual_error = ""
        self.virtual_size: tuple[int, int] | None = None
        self.last_segment_ms = -1
        self.last_result = 0
        self.last_submit = float("-inf")
        self.last_debug_print = 0.0
        self.debug = ""
        self.toasts: list[str] = []
        self.error = ""
        self.waiting = ""
        self.pending_toggle = False
        self.sign = False
        self.live: np.ndarray | None = None
        self._analysis: tuple[int, np.ndarray] | None = None
        self._frame_id = 0

    # -- lifecycle

    def open(self) -> None:
        self.waiting = "Loading models..."
        download_models()
        self.segmenter = create_segmenter()
        self.worker = DetectionWorker(model_paths())
        self.set_mode(self.settings.mode, announce=False)
        self.open_capture()

    def open_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.waiting = "Starting the camera... If macOS asks for camera access, click Allow."
        try:
            self.capture = open_camera(self.settings.camera, self.settings.width,
                                       self.settings.height, self.args.fps)
            self.error = ""
        except RuntimeError as error:
            self.error = str(error)
        finally:
            self.waiting = ""
        self.state = State()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.close_virtual()
        if self.worker is not None:
            self.worker.close()
        if self.segmenter is not None:
            self.segmenter.close()
        self.sounds.close()

    def post(self, name: str, *values) -> None:
        """Queue a command from the UI thread; it runs before the next frame."""
        self.commands.put((name, values))

    def run_commands(self) -> None:
        while True:
            try:
                name, values = self.commands.get_nowait()
            except queue.Empty:
                return
            getattr(self, f"command_{name}")(*values)

    # -- commands (run on the engine thread)

    def command_mode(self, mode: str) -> None:
        if mode != self.mode:
            self.set_mode(mode)

    def command_calibrate(self) -> None:
        if self.capture is not None:
            self.state = State(phase="calibrating", started=time.perf_counter())
            log(f"Step out of the frame for {CALIBRATION_SECONDS:g} seconds.")

    def command_toggle_vanish(self) -> None:
        self.pending_toggle = True

    def command_camera(self, index: int) -> None:
        self.settings.camera = index
        self.open_capture()

    def command_resolution(self, width: int, height: int) -> None:
        self.settings.width, self.settings.height = width, height
        self.open_capture()

    def command_retry(self) -> None:
        self.open_capture()

    def command_mirror(self, enabled: bool) -> None:
        if enabled == self.settings.mirror:
            return
        self.settings.mirror = enabled
        # Flip the stored plates so the scene stays aligned without recapturing.
        for name in ("background", "vanished_frame"):
            image = getattr(self.state, name)
            if image is not None:
                setattr(self.state, name, cv2.flip(image, 1))

    def command_virtual(self, enabled: bool) -> None:
        self.settings.virtual_camera = enabled
        self.close_virtual()

    def command_sound(self, enabled: bool) -> None:
        self.settings.sound = self.sounds.enabled = enabled

    def command_volume(self, volume: float) -> None:
        self.settings.volume = volume
        self.sounds.set_volume(volume)

    def command_goku_hold(self, seconds: float) -> None:
        self.settings.goku_hold_seconds = seconds
        if self.mode == "goku":
            self.trigger.hold_seconds = seconds

    # -- helpers

    def set_mode(self, mode: str, announce: bool = True) -> None:
        self.settings.mode = self.mode = mode
        if mode not in self.effects:
            self.effects[mode] = create_effect(mode)
        self.effect = self.effects[mode]
        hold = (self.settings.goku_hold_seconds if mode == "goku"
                else SIGN_HOLD_SECONDS.get(mode, 0.0))
        self.trigger = HoldTrigger(hold)
        self.trigger.require_release()
        self.sounds.preload(mode)
        if self.state.phase in ("vanishing", "vanished", "appearing"):
            self.state.phase = "ready"
        if announce:
            self.toasts.append(MODE_INFO[mode].title)
        log(f"Effect: {MODE_INFO[mode].title}. Sign: {MODE_INFO[mode].gesture}.")

    def close_virtual(self) -> None:
        if self.virtual_camera is not None:
            self.virtual_camera.close()
        self.virtual_camera = None
        self.virtual_size = None
        self.virtual_error = ""

    def timestamp_ms(self) -> int:
        return int((time.perf_counter() - self.clock_start) * 1000)

    def read_frame(self) -> np.ndarray:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("The camera stopped sending images.")
        self._frame_id += 1
        if self.settings.mirror:
            frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        # Some cameras ignore the requested size; keep processing cost bounded.
        if width > self.settings.width:
            scale = self.settings.width / width
            frame = cv2.resize(frame, (self.settings.width, int(round(height * scale))),
                               interpolation=cv2.INTER_AREA)
        return frame

    def analysis_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Downscaled RGB copy for the trackers, computed once per frame."""
        if self._analysis is not None and self._analysis[0] == self._frame_id:
            return self._analysis[1]
        height, width = frame.shape[:2]
        if width > ANALYSIS_WIDTH:
            frame = cv2.resize(frame, (ANALYSIS_WIDTH, int(round(height * ANALYSIS_WIDTH / width))),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._analysis = (self._frame_id, rgb)
        return rgb

    def update_mask(self, frame: np.ndarray) -> np.ndarray:
        timestamp = max(self.timestamp_ms(), self.last_segment_ms + 1)
        self.last_segment_ms = timestamp
        height, width = frame.shape[:2]
        self.state.mask = person_mask(self.segmenter, self.analysis_rgb(frame), timestamp,
                                      (width, height))
        self.state.mask_at = time.perf_counter()
        return self.state.mask

    def begin_effect(self, direction: str, frame: np.ndarray) -> None:
        state = self.state
        mask = self.update_mask(frame)
        self.effect.begin(direction, frame, state.background, mask)
        state.phase = "vanishing" if direction == OUT else "appearing"
        state.started = time.perf_counter()
        if direction == OUT:
            self.sounds.play_vanish(self.mode)
        else:
            self.sounds.play_return(self.mode)

    def poll_gesture(self, frame: np.ndarray, now: float) -> bool:
        """Feed the detector and return True when the trigger sign fires."""
        if self.worker.error is not None:
            raise RuntimeError(f"Gesture detection failed: {self.worker.error}")
        if now - self.last_submit >= DETECTION_INTERVAL:
            if self.worker.submit(self.mode, self.analysis_rgb(frame),
                                  self.timestamp_ms(), now):
                self.last_submit = now
        latest = self.worker.latest()
        if latest is None or latest[0] == self.last_result:
            return False
        sequence, mode, captured_at, detection = latest
        self.last_result = sequence
        if mode != self.mode:
            return False
        self.sign = detection.active
        self.debug = detection.info if self.args.gesture_debug else ""
        if self.args.gesture_debug and now - self.last_debug_print >= 1.0:
            log(f"Gesture: {detection.info}")
            self.last_debug_print = now
        return self.trigger.update(detection.active, captured_at)

    def step(self, frame: np.ndarray) -> np.ndarray:
        state, now = self.state, time.perf_counter()
        toggle, self.pending_toggle = self.pending_toggle, False

        if state.phase == "setup":
            # Runs the detector meanwhile, so its models are loaded once you are set up.
            self.poll_gesture(frame, now)
            return frame

        if state.phase == "calibrating":
            self.poll_gesture(frame, now)
            elapsed = now - state.started
            if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                state.samples.append(frame.copy())
            if elapsed >= CALIBRATION_SECONDS:
                stack = np.stack(state.samples[-18:] or [frame], axis=0)
                state.background = np.median(stack, axis=0).astype(np.uint8)
                state.samples.clear()
                state.phase = "ready"
                self.trigger.require_release()
                self.toasts.append("Background captured. Step back in!")
                log("Background ready. Step back in.")
            return frame

        if state.background is None or state.background.shape != frame.shape:
            self.state = State()
            return frame

        if state.phase in ("ready", "vanished"):
            fired = self.poll_gesture(frame, now)
            if toggle or fired:
                if state.phase == "ready":
                    self.begin_effect(OUT, frame)
                elif toggle or self.settings.gesture_reappear:
                    self.begin_effect(IN, frame)

        if state.phase in ("vanishing", "appearing"):
            age = now - state.started
            if (self.effect.needs_mask(age) and state.mask is not None
                    and now - state.mask_at >= self.effect.mask_interval):
                self.update_mask(frame)
            if age >= self.effect.seconds:
                finished_out = state.phase == "vanishing"
                state.phase = "vanished" if finished_out else "ready"
                state.vanished_frame = self.effect.vanished(state.background) if finished_out else None
                # Whoever is still holding the sign must let go before it fires again.
                self.trigger.require_release()
            else:
                return self.effect.render(frame, state.background, state.mask, age)

        if state.phase == "vanished":
            if state.vanished_frame is None:
                state.vanished_frame = self.effect.vanished(state.background)
            return state.vanished_frame
        return frame

    def send_virtual(self, output: np.ndarray) -> None:
        height, width = output.shape[:2]
        if self.settings.virtual_camera and self.virtual_size != (width, height):
            self.close_virtual()
            self.virtual_size = (width, height)
            try:
                self.virtual_camera = open_virtual_camera(self.args, self.settings, width, height)
            except RuntimeError as error:
                self.virtual_error = str(error)
            if self.virtual_camera is None and not self.virtual_error:
                self.virtual_error = "missing"
        if self.virtual_camera is not None:
            # Call apps mirror your self-view themselves: send the true image.
            clean = cv2.flip(output, 1) if self.settings.mirror else output
            send_virtual_frame(self.virtual_camera, clean)

    def next_frame(self) -> tuple[np.ndarray | None, Status]:
        """Process one camera frame. Returns (output or None, status)."""
        self.run_commands()
        output = None
        if self.capture is not None and not self.error:
            try:
                frame = self.read_frame()
                output = self.step(frame)
                self.send_virtual(output)
                # While you are invisible, the preview still shows you where your hand is.
                self.live = (self._live_view(frame) if self.state.phase == "vanished"
                             else None)
            except RuntimeError as error:
                self.error = str(error)
                log(self.error)
        else:
            time.sleep(0.05)
        return output, self.snapshot()

    @staticmethod
    def _live_view(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        size = (LIVE_VIEW_WIDTH, max(1, round(height * LIVE_VIEW_WIDTH / width)))
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    def snapshot(self) -> Status:
        state, now = self.state, time.perf_counter()
        toasts, self.toasts = self.toasts, []
        return Status(
            mode=self.mode or self.settings.mode,
            phase=state.phase,
            countdown=(CALIBRATION_SECONDS - (now - state.started)
                       if state.phase == "calibrating" else None),
            hold=self.trigger.progress(now) if state.phase in ("ready", "vanished") else 0.0,
            camera=self.settings.camera,
            virtual_device=getattr(self.virtual_camera, "device", None),
            virtual_error=self.virtual_error,
            debug=self.debug,
            toasts=toasts,
            error=self.error,
            waiting=self.waiting,
            sign=self.sign and state.phase in ("ready", "vanished"),
            blocked=(state.phase in ("ready", "vanished") and self.trigger.blocked(now)),
            live=self.live,
        )


__all__ = ["Engine", "State", "Status", "CameraDevice", "camera_devices", "open_camera",
           "open_virtual_camera", "send_virtual_frame", "download_models", "MODES",
           "CALIBRATION_SECONDS", "IN", "OUT"]
