"""Hand-sign recognition for every mode, plus hold/release trigger timing."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, replace

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


GESTURE_RETRY_MS = 140
FACE_MEMORY_SECONDS = 1.5


@dataclass
class Detection:
    active: bool
    hands: int = 0
    info: str = ""


# ---------------------------------------------------------------- finger shapes

def _finger_metrics(points: np.ndarray, straight: float = 0.82, reach: float = 1.10):
    """Straightness, length and extension for index, middle, ring, pinky."""
    straightness, lengths, extended, folded = [], [], [], []
    for base in (5, 9, 13, 17):
        finger = points[base:base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= straight and tip_distance > joint_distance * reach)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)
    if min(lengths[:2]) > 0:
        # Seen from the side, folded fingers can look straight but very short.
        reference = min(lengths[:2])
        for index, base in ((2, 13), (3, 17)):
            reach = float(np.linalg.norm(points[base + 3] - points[base]))
            folded[index] = folded[index] or reach < reference * 0.65
    return straightness, lengths, extended, folded


def victory_landmark_score(points: np.ndarray) -> float:
    """Index and middle extended and spread apart, ring and pinky folded."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0
    straightness, lengths, extended, folded = _finger_metrics(points)
    if min(lengths[:2]) < 1e-6:
        return 0.0
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # The fingertips must be apart; the thumb may stay hidden.
    if float(np.linalg.norm(points[8] - points[12])) < min(lengths[:2]) * 0.20:
        return 0.0
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def two_finger_salute_score(points: np.ndarray) -> float:
    """Index and middle extended and held together, ring and pinky folded."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0
    # Fingers pressed to the forehead flex a little, so allow slightly bent ones.
    straightness, lengths, extended, folded = _finger_metrics(points, straight=0.74, reach=1.05)
    if min(lengths[:2]) < 1e-6:
        return 0.0
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    finger_length = (lengths[0] + lengths[1]) * 0.5
    if float(np.linalg.norm(points[8] - points[12])) > finger_length * 0.42:
        return 0.0
    if float(np.linalg.norm(points[7] - points[11])) > finger_length * 0.40:
        return 0.0
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def _hand_arrays(result, width: int, height: int) -> list[tuple[np.ndarray | None, np.ndarray]]:
    """(world points or None, image points in pixels) for every detected hand."""
    hands = []
    world = getattr(result, "hand_world_landmarks", []) or []
    for index, landmarks in enumerate(result.hand_landmarks):
        image_points = np.array([(p.x * width, p.y * height, 0.0) for p in landmarks])
        world_points = None
        if index < len(world):
            world_points = np.array([(p.x, p.y, p.z) for p in world[index]])
        hands.append((world_points, image_points))
    return hands


# --------------------------------------------------------------------- V sign

def classify_gesture_result(result, image: mp.Image) -> tuple[str, float]:
    # Landmarks come back even when the classifier misses "Victory"
    # (for example with the back of the hand facing the camera).
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # Always check the visible geometry too: estimated depth can be wrong
    # when the palm and folded fingers are hidden.
    for landmarks in result.hand_landmarks:
        points = np.array([(point.x * image.width, point.y * image.height, 0.0)
                           for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    for categories in result.gestures:
        for category in categories:
            if category.category_name == "Victory" and category.score >= 0.62:
                return "Victory", float(category.score)
    if result.hand_landmarks or result.hand_world_landmarks:
        return "Hand_Detected", 0.0
    return "No_Hand", 0.0


def _rotated_image(rgb: np.ndarray, rotation: int) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(np.rot90(rgb, rotation)))


class MultiViewGestureRecognizer:
    """Retries detection on rotated crops without disturbing video tracking."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_rotation: int | None = None
        self.preferred_crop: str | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0

    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(image, timestamp_ms)
        best = classify_gesture_result(result, image)
        if best[0] == "Victory":
            self.preferred_rotation = None
            self.preferred_crop = None
            return best
        if timestamp_ms < self.next_retry_ms:
            return best

        # IMAGE mode forces a fresh detection; the VIDEO model keeps tracking the
        # original frame, so timestamps never mix between rotations.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # A sideways palm is often too small in the full frame. The two lower
        # crops overlap and cover both hands, mirrored or not.
        crops = {
            "full": rgb,
            "lower_left": rgb[int(height * 0.28):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.28):, int(width * 0.22):],
        }
        views = [("full", rotation) for rotation in (0, 1, 3, 2)]
        views += [(crop, rotation) for crop in ("lower_left", "lower_right")
                  for rotation in (0, 1, 3)]
        preferred = (self.preferred_crop, self.preferred_rotation)
        selected = []
        if preferred in views:
            selected.append(preferred)
        # Spread the expensive IMAGE attempts over successive video frames.
        while len(selected) < 3:
            view = views[self.retry_cursor % len(views)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = _rotated_image(crops[crop], rotation)
            candidate = classify_gesture_result(self.alternate.recognize(rotated), rotated)
            if candidate[0] == "Victory":
                self.preferred_rotation = rotation
                self.preferred_crop = crop
                self.next_retry_ms = timestamp_ms + GESTURE_RETRY_MS
                return candidate
            if candidate[0] == "Hand_Detected":
                best = candidate
        self.preferred_rotation = None
        self.preferred_crop = None
        self.next_retry_ms = timestamp_ms + GESTURE_RETRY_MS
        return best

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


def recognize_gesture(recognizer, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
    if isinstance(recognizer, MultiViewGestureRecognizer):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)


# --------------------------------------------------------------- joined hands

def joined_hands_from_landmarks(hands, width: int, height: int) -> bool:
    """Recognize two nearby, similarly oriented open hands in image space."""
    if len(hands) < 2:
        return False
    points = [np.array([(p.x * width, p.y * height) for p in hand])
              for hand in hands[:2] if len(hand) == 21]
    if len(points) != 2 or not all(np.isfinite(hand).all() for hand in points):
        return False
    palms = [hand[[0, 5, 9, 13, 17]].mean(axis=0) for hand in points]
    scale = [max(float(np.linalg.norm(hand[5] - hand[17])),
                 float(np.linalg.norm(hand[0] - hand[9]))) for hand in points]
    if min(scale) < 8.0 or max(scale) > min(scale) * 2.2:
        return False
    size = sum(scale) * 0.5
    directions = [hand[9] - hand[0] for hand in points]
    alignment = float(np.dot(*directions) /
                      max(1e-6, np.linalg.norm(directions[0]) * np.linalg.norm(directions[1])))
    fingertips = [hand[[8, 12, 16, 20]].mean(axis=0) for hand in points]
    return (
        alignment > 0.25
        and np.linalg.norm(palms[0] - palms[1]) < size * 1.55
        and np.linalg.norm(fingertips[0] - fingertips[1]) < size * 1.9
    )


class MultiViewHandLandmarker:
    """Tracks two hands in video and retries difficult views with the hand model."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_view: tuple[str, int] | None = None
        self.next_retry_ms = 0
        self.last_hand_count = 0
        self.retry_cursor = 0

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        if joined_hands_from_landmarks(result.hand_landmarks, image.width, image.height):
            self.preferred_view = None
            return True
        if timestamp_ms < self.next_retry_ms:
            return False

        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        views = {
            "full": rgb,
            "center": rgb[int(height * 0.18):, int(width * 0.12):int(width * 0.88)],
            "lower_left": rgb[int(height * 0.22):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.22):, int(width * 0.22):],
        }
        attempts = [("full", rotation) for rotation in (1, 3, 2)]
        attempts += [("center", rotation) for rotation in (0, 1, 3)]
        attempts += [("lower_left", 0), ("lower_right", 0)]
        selected = []
        if self.preferred_view in attempts:
            selected.append(self.preferred_view)
        while len(selected) < 2:
            view = attempts[self.retry_cursor % len(attempts)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = _rotated_image(views[crop], rotation)
            result = self.alternate.detect(rotated)
            self.last_hand_count = max(self.last_hand_count, len(result.hand_landmarks))
            if joined_hands_from_landmarks(result.hand_landmarks, rotated.width, rotated.height):
                self.preferred_view = (crop, rotation)
                self.next_retry_ms = timestamp_ms + 90
                return True
        self.preferred_view = None
        self.next_retry_ms = timestamp_ms + 80
        return False

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


# ------------------------------------------------------- two fingers, forehead

@dataclass
class ForeheadZone:
    left: float
    right: float
    top: float
    bottom: float
    center: tuple[float, float]
    face_width: float

    def contains(self, point: np.ndarray) -> bool:
        return self.left <= point[0] <= self.right and self.top <= point[1] <= self.bottom


def forehead_zone(face_landmarks, width: int, height: int) -> ForeheadZone | None:
    """Area from the eyebrows up past the hairline, wide enough for the temples."""
    if len(face_landmarks) < 455:
        return None
    point = lambda index: np.array((face_landmarks[index].x * width, face_landmarks[index].y * height))
    left_edge, right_edge = point(234), point(454)
    face_width = float(abs(right_edge[0] - left_edge[0]))
    if face_width < 20:
        return None
    hairline, brow = point(10), point(9)
    eye_y = float((point(33)[1] + point(263)[1]) * 0.5)
    return ForeheadZone(
        left=min(left_edge[0], right_edge[0]) - face_width * 0.18,
        right=max(left_edge[0], right_edge[0]) + face_width * 0.18,
        top=float(hairline[1] - face_width * 0.45),
        bottom=eye_y + face_width * 0.04,
        center=(float((hairline[0] + brow[0]) * 0.5), float((hairline[1] + brow[1]) * 0.5)),
        face_width=face_width,
    )


def salute_at_forehead(world_points: np.ndarray | None, image_points: np.ndarray,
                       zone: ForeheadZone | None) -> tuple[bool, bool]:
    """(fingers in the right shape, fingertips touching the forehead zone)."""
    shape = bool(two_finger_salute_score(image_points))
    if not shape and world_points is not None:
        shape = bool(two_finger_salute_score(world_points))
    if not shape or zone is None:
        return shape, False
    tips = image_points[[8, 12], :2].mean(axis=0)
    pads = image_points[[7, 11], :2].mean(axis=0)
    touching = zone.contains(tips) or zone.contains(pads)
    # The hand must sit beside or on the face, not far in front of the body.
    close = np.linalg.norm(tips - np.array(zone.center)) < zone.face_width * 1.1
    return shape, bool(touching and close)


class ForeheadSaluteDetector:
    """Hand landmarks plus face landmarks, both in VIDEO tracking mode."""

    def __init__(self, hands, face):
        self.hands = hands
        self.face = face
        self.zone: ForeheadZone | None = None
        self.zone_seen_at = float("-inf")

    def detect(self, image: mp.Image, timestamp_ms: int) -> Detection:
        now = time.perf_counter()
        faces = self.face.detect_for_video(image, timestamp_ms).face_landmarks
        if faces:
            zone = forehead_zone(faces[0], image.width, image.height)
            if zone is not None:
                self.zone, self.zone_seen_at = zone, now
        # The hand often hides part of the face; remember where it was.
        zone = self.zone if now - self.zone_seen_at <= FACE_MEMORY_SECONDS else None
        result = self.hands.detect_for_video(image, timestamp_ms)
        hands = _hand_arrays(result, image.width, image.height)
        shape_seen = False
        for world_points, image_points in hands:
            shape, touching = salute_at_forehead(world_points, image_points, zone)
            shape_seen = shape_seen or shape
            if touching:
                return Detection(True, len(hands), "two fingers on forehead")
        info = (f"face={'yes' if zone else 'no'} hands={len(hands)} "
                f"two_fingers={'yes' if shape_seen else 'no'}")
        return Detection(False, len(hands), info)

    def close(self) -> None:
        try:
            self.hands.close()
        finally:
            self.face.close()


# -------------------------------------------------------------- task factory

def _hand_options(model_path: str, hands: int = 2) -> vision.HandLandmarkerOptions:
    return vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=hands,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.45,
    )


class ModeDetector:
    """One detector with a uniform detect() for each mode's trigger sign."""

    def __init__(self, mode: str, models: dict[str, str]):
        self.mode = mode
        if mode in ("normal", "particles"):
            classifier = mp.tasks.components.processors.ClassifierOptions(
                score_threshold=0.55, category_allowlist=["Victory"],
            )
            options = vision.GestureRecognizerOptions(
                base_options=python.BaseOptions(model_asset_path=models["gesture"]),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.5,
                canned_gesture_classifier_options=classifier,
            )
            self.impl = _pair(vision.GestureRecognizer, options, MultiViewGestureRecognizer)
        elif mode == "naruto":
            self.impl = _pair(vision.HandLandmarker, _hand_options(models["hand"]),
                              MultiViewHandLandmarker)
        elif mode == "goku":
            hands = vision.HandLandmarker.create_from_options(_hand_options(models["hand"]))
            try:
                face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=models["face"]),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.4,
                    min_face_presence_confidence=0.4,
                    min_tracking_confidence=0.4,
                ))
            except Exception:
                hands.close()
                raise
            self.impl = ForeheadSaluteDetector(hands, face)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def detect(self, image: mp.Image, timestamp_ms: int) -> Detection:
        if isinstance(self.impl, MultiViewGestureRecognizer):
            name, score = self.impl.recognize_gesture(image, timestamp_ms)
            view = self.impl.preferred_crop or "full"
            return Detection(name == "Victory" and score >= 0.62, int(name != "No_Hand"),
                             f"{name} score={score:.2f} view={view}")
        if isinstance(self.impl, MultiViewHandLandmarker):
            joined = self.impl.recognize_joined_hands(image, timestamp_ms)
            return Detection(joined, self.impl.last_hand_count,
                             f"hands={self.impl.last_hand_count} joined={'yes' if joined else 'no'}")
        return self.impl.detect(image, timestamp_ms)

    def close(self) -> None:
        self.impl.close()


def _pair(task_class, options, wrapper):
    primary = task_class.create_from_options(options)
    try:
        alternate = task_class.create_from_options(
            replace(options, running_mode=vision.RunningMode.IMAGE))
    except Exception:
        primary.close()
        raise
    return wrapper(primary, alternate)


# ------------------------------------------------------------------- timing

class HoldTrigger:
    """Fires once the sign has been held long enough, then waits for a release.

    Short dropouts (a missed detection or two) do not restart the hold.
    """

    def __init__(self, hold_seconds: float = 0.0, grace: float = 0.25, release: float = 0.3):
        self.hold_seconds = hold_seconds
        self.grace = grace
        self.release = release
        self.started: float | None = None
        self.last_active = float("-inf")
        self.armed = True

    def update(self, active: bool, now: float) -> bool:
        if active:
            self.last_active = now
            if not self.armed:
                return False
            if self.started is None:
                self.started = now
            if now - self.started >= self.hold_seconds:
                self.started = None
                self.armed = False
                return True
            return False
        if self.started is not None and now - self.last_active > self.grace:
            self.started = None
        if not self.armed and now - self.last_active >= self.release:
            self.armed = True
        return False

    def progress(self, now: float) -> float:
        if self.started is None or self.hold_seconds <= 0:
            return 0.0
        return float(np.clip((now - self.started) / self.hold_seconds, 0.0, 1.0))

    def require_release(self) -> None:
        self.started = None
        self.armed = False

    def blocked(self, now: float) -> bool:
        """True while the sign is still up after firing, so it cannot fire again yet."""
        return not self.armed and now - self.last_active < self.release


# ------------------------------------------------------------ background work

class DetectionWorker:
    """Runs the mode detector on its own thread so slow frames never stall video.

    The newest submitted frame always replaces an unprocessed older one.
    """

    def __init__(self, models: dict[str, str], factory=ModeDetector):
        self.models = models
        self.factory = factory
        self.detectors: dict[str, ModeDetector] = {}
        self.jobs: queue.Queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.result: tuple[int, str, float, Detection] | None = None
        self.sequence = 0
        self.error: Exception | None = None
        self.busy = False
        self.thread = threading.Thread(target=self._run, name="gesture-detection", daemon=True)
        self.thread.start()

    def submit(self, mode: str, rgb: np.ndarray, timestamp_ms: int, captured_at: float) -> bool:
        if self.busy or self.error is not None:
            return False
        # Mark busy before queueing: the worker clears it when the job is done.
        self.busy = True
        try:
            self.jobs.put_nowait((mode, rgb, timestamp_ms, captured_at))
        except queue.Full:
            self.busy = False
            return False
        return True

    def latest(self) -> tuple[int, str, float, Detection] | None:
        with self.lock:
            return self.result

    def _run(self) -> None:
        last_timestamp: dict[str, int] = {}
        while True:
            job = self.jobs.get()
            if job is None:
                break
            mode, rgb, timestamp_ms, captured_at = job
            try:
                detector = self.detectors.get(mode)
                if detector is None:
                    detector = self.detectors[mode] = self.factory(mode, self.models)
                # VIDEO tasks require strictly increasing timestamps.
                timestamp_ms = max(timestamp_ms, last_timestamp.get(mode, -1) + 1)
                last_timestamp[mode] = timestamp_ms
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection = detector.detect(image, timestamp_ms)
                with self.lock:
                    self.sequence += 1
                    self.result = (self.sequence, mode, captured_at, detection)
            except Exception as error:  # Surface model errors on the main thread.
                self.error = error
            finally:
                self.busy = False
        for detector in self.detectors.values():
            detector.close()

    def close(self) -> None:
        # Wait for any running job, then let the worker close its own models.
        while True:
            try:
                self.jobs.put(None, timeout=0.5)
                break
            except queue.Full:
                if not self.thread.is_alive():
                    break
        self.thread.join(timeout=3)
