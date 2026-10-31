"""Vanish and return animations for every mode.

Each effect is driven by the app with the live camera frame, the clean
background plate and the latest person mask, so the person keeps moving
naturally while an effect plays.
"""

from __future__ import annotations


import cv2
import numpy as np

try:
    from . import naruto_effect as naruto
except ImportError:
    import naruto_effect as naruto


OUT, IN = "out", "in"


def smoothstep(edge0: float, edge1: float, values):
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def ease_in_out_cubic(value: float) -> float:
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def alpha_composite(person: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3 = alpha[:, :, None]
    return (
        person.astype(np.float32) * alpha_3
        + background.astype(np.float32) * (1.0 - alpha_3)
    ).astype(np.uint8)


def mask_box(mask: np.ndarray, threshold: float = 0.3, margin: int = 0):
    """Bounding box (x0, y0, x1, y1) of the person, or None if nobody is there."""
    points = cv2.findNonZero((mask > threshold).astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return (max(0, x - margin), max(0, y - margin),
            min(mask.shape[1], x + width + margin), min(mask.shape[0], y + height + margin))


class Effect:
    """Base class. `begin` starts a vanish (OUT) or a return (IN)."""

    out_seconds = 1.0
    in_seconds = 1.0
    mask_interval = 0.05  # Seconds between person-mask refreshes while playing.

    def __init__(self):
        self.direction = OUT

    @property
    def seconds(self) -> float:
        return self.out_seconds if self.direction == OUT else self.in_seconds

    def begin(self, direction: str, frame: np.ndarray, background: np.ndarray,
              mask: np.ndarray) -> None:
        self.direction = direction

    def needs_mask(self, age: float) -> bool:
        return True

    def render(self, frame: np.ndarray, background: np.ndarray, mask: np.ndarray,
               age: float) -> np.ndarray:
        raise NotImplementedError

    def vanished(self, background: np.ndarray) -> np.ndarray:
        """What the camera shows while you are gone."""
        return background.copy()


# ----------------------------------------------------------------- dissolve

def make_dissolve_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7391)
    y, x = np.mgrid[0:height, 0:width]
    y_normal = y.astype(np.float32) / max(height - 1, 1)
    x_normal = x.astype(np.float32) / max(width - 1, 1)
    low_noise = rng.random((max(2, height // 10), max(2, width // 10)), dtype=np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_noise = rng.random((height, width), dtype=np.float32)
    field = (
        (1.0 - y_normal) * 0.72
        + low_noise * 0.17
        + fine_noise * 0.08
        + np.sin(x_normal * 18.0) * 0.025
    )
    return np.clip(field, 0.0, 1.0), fine_noise


def apply_dissolve(frame: np.ndarray, background: np.ndarray, mask: np.ndarray,
                   progress: float, field: np.ndarray, sparkle_noise: np.ndarray) -> np.ndarray:
    if progress <= 0.002:
        return alpha_composite(frame, background, mask)
    if progress >= 0.999:
        return background.copy()
    box = mask_box(mask, 0.01, 18)
    if box is None:
        return background.copy()
    left, top, right, bottom = box
    area = np.s_[top:bottom, left:right]
    mask = mask[area]
    field = field[area]
    sparkle_noise = sparkle_noise[area]
    eased = ease_in_out_cubic(progress)
    survival = smoothstep(eased - 0.055, eased + 0.055, field)
    output = alpha_composite(frame[area], background[area], mask * survival)
    edge = ((np.abs(field - eased) < 0.018) & (mask > 0.48)
            & (sparkle_noise > 0.54)).astype(np.float32)
    glow = cv2.GaussianBlur(edge, (0, 0), sigmaX=5.0, sigmaY=5.0)
    glow = np.clip(glow * 5.0 + edge, 0.0, 1.0)[:, :, None]
    cyan = np.full_like(output, (255, 228, 92), dtype=np.uint8)
    result = np.clip(
        output.astype(np.float32) * (1.0 - glow * 0.8) + cyan.astype(np.float32) * glow * 0.8,
        0, 255,
    ).astype(np.uint8)
    canvas = background.copy()
    canvas[area] = result
    return canvas


class DissolveEffect(Effect):
    """The original effect: you fade away from the top with a cyan edge."""

    out_seconds = 3.6
    in_seconds = 1.8

    def __init__(self):
        super().__init__()
        self.fields: tuple[np.ndarray, np.ndarray] | None = None

    def begin(self, direction, frame, background, mask):
        super().begin(direction, frame, background, mask)
        height, width = frame.shape[:2]
        if self.fields is None or self.fields[0].shape != (height, width):
            self.fields = make_dissolve_fields(height, width)

    def render(self, frame, background, mask, age):
        progress = min(1.0, age / self.seconds)
        if self.direction == IN:
            progress = 1.0 - progress
        return apply_dissolve(frame, background, mask, progress, *self.fields)


# ---------------------------------------------------------------- particles

PARTICLE_TINTS = np.array(((90, 205, 255), (255, 222, 120), (230, 130, 255), (150, 255, 190)),
                          dtype=np.float32)


class ParticlesEffect(Effect):
    """Your silhouette crumbles into colored fragments carried off by the wind."""

    out_seconds = 2.1
    in_seconds = 1.7
    spread = 1.0  # Seconds between the first and the last fragment leaving.

    def __init__(self):
        super().__init__()
        self.delay_field: np.ndarray | None = None
        self.particles: dict[str, np.ndarray] | None = None
        self.size = 2

    def begin(self, direction, frame, background, mask):
        super().begin(direction, frame, background, mask)
        height, width = frame.shape[:2]
        rng = np.random.default_rng(4242)
        box = mask_box(mask, 0.4) or (width // 3, height // 4, 2 * width // 3, height)
        x0, y0, x1, y1 = box
        box_width, box_height = max(1, x1 - x0), max(1, y1 - y0)

        # Fragments peel away from the downwind (right) side and from the top.
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        x_normal = np.clip((x - x0) / box_width, 0, 1)
        y_normal = np.clip((y - y0) / box_height, 0, 1)
        noise = rng.random((max(2, height // 18), max(2, width // 18)), dtype=np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        field = 0.58 * (1.0 - x_normal) + 0.17 * y_normal + 0.25 * noise
        self.delay_field = np.clip(field, 0, 1) * self.spread

        self.size = max(3, round(height / 190))
        while True:
            grid = mask[y0:y1:self.size, x0:x1:self.size] > 0.5
            if grid.sum() <= 24000 or self.size >= 8:
                break
            self.size += 1
        gy, gx = np.nonzero(grid)
        px = (x0 + gx * self.size).astype(np.float32)
        py = (y0 + gy * self.size).astype(np.float32)
        count = px.size
        centers_y = np.clip(py.astype(int) + self.size // 2, 0, height - 1)
        centers_x = np.clip(px.astype(int) + self.size // 2, 0, width - 1)
        self.particles = {
            "x": px, "y": py,
            "color": frame[centers_y, centers_x].astype(np.float32),
            "tint": PARTICLE_TINTS[rng.integers(0, len(PARTICLE_TINTS), count)],
            "delay": self.delay_field[centers_y, centers_x],
            "vx": rng.uniform(0.04, 0.14, count).astype(np.float32) * width,
            "ax": rng.uniform(0.25, 0.65, count).astype(np.float32) * width,
            "vy": rng.uniform(-0.14, 0.03, count).astype(np.float32) * height,
            "wobble": rng.uniform(0.004, 0.02, count).astype(np.float32) * height,
            "freq": rng.uniform(2.5, 7.0, count).astype(np.float32) * 2 * np.pi,
            "phase": rng.uniform(0, 2 * np.pi, count).astype(np.float32),
            "life": rng.uniform(0.7, 1.1, count).astype(np.float32),
        }

    def render(self, frame, background, mask, age):
        if self.direction == OUT:
            clock = age
        else:
            # Run the vanish backwards, compressed into the shorter return time.
            clock = max(0.0, self.in_seconds - age) * self.out_seconds / self.in_seconds
        canvas = background.copy()
        box = mask_box(mask, 0.01, 4)
        if box is not None:
            # Blend only around the person: full-frame float math costs too much.
            x0, y0, x1, y1 = box
            area = np.s_[y0:y1, x0:x1]
            delay = self.delay_field[area]
            alpha = mask[area] * smoothstep(clock - 0.04, clock + 0.04, delay)
            body = (frame[area].astype(np.float32) * alpha[:, :, None]
                    + background[area].astype(np.float32) * (1 - alpha[:, :, None]))
            if clock > 0.01:
                front = mask[area] * np.exp(-((delay - clock) / 0.035) ** 2)
                body += front[:, :, None] * np.array((150, 220, 255), np.float32) * 0.55
            canvas[area] = np.clip(body, 0, 255).astype(np.uint8)

        p = self.particles
        if p is not None and p["x"].size:
            tau = clock - p["delay"]
            active = (tau > 0) & (tau < p["life"])
            if active.any():
                t = tau[active]
                xs = p["x"][active] + p["vx"][active] * t + 0.5 * p["ax"][active] * t * t
                ys = (p["y"][active] + p["vy"][active] * t
                      + p["wobble"][active] * np.sin(p["freq"][active] * t + p["phase"][active])
                      * np.minimum(1.0, t * 4))
                fade = (1.0 - t / p["life"][active]) ** 0.8
                mix = np.minimum(1.0, t / 0.35)[:, None] * 0.5
                colors = p["color"][active] * (1 - mix) + p["tint"][active] * mix
                height, width = canvas.shape[:2]
                xi, yi = xs.astype(np.int32), ys.astype(np.int32)
                inside = (xi >= 0) & (yi >= 0) & (xi < width - self.size) & (yi < height - self.size)
                xi, yi, fade, colors = xi[inside], yi[inside], fade[inside, None], colors[inside]
                ink = colors * fade
                keep = 1 - fade
                for dy in range(self.size):
                    for dx in range(self.size):
                        target = canvas[yi + dy, xi + dx].astype(np.float32)
                        canvas[yi + dy, xi + dx] = np.clip(ink + target * keep, 0, 255)
        return canvas


# --------------------------------------------------------------------- goku

def _ramp(value: float, start: float, end: float) -> float:
    return float(smoothstep(start, end, value))


class GokuEffect(Effect):
    """Comic speed lines run up your silhouette, you squeeze into a beam and blink out.

    The return plays the same beats in reverse: lines rain down, a beam opens
    into your silhouette and the flash fades.
    """

    out_seconds = 1.05
    in_seconds = 0.9
    line_count = 54

    def __init__(self):
        super().__init__()
        self.box = None
        self.silhouette: np.ndarray | None = None
        self.lines: np.ndarray | None = None

    def begin(self, direction, frame, background, mask):
        super().begin(direction, frame, background, mask)
        height, width = frame.shape[:2]
        box = mask_box(mask, 0.35)
        if box is None:
            box = (int(width * 0.35), int(height * 0.15), int(width * 0.65), height)
        x0, y0, x1, y1 = box
        pad = int((x1 - x0) * 0.12)
        self.box = (max(0, x0 - pad), y0, min(width, x1 + pad), y1)
        self.center_x = (x0 + x1) // 2
        kernel = np.ones((max(3, height // 40),) * 2, np.uint8)
        self.silhouette = cv2.dilate((mask > 0.35).astype(np.float32), kernel)

        rng = np.random.default_rng(7 if direction == OUT else 8)
        columns = (mask[y0:y1, x0:x1] > 0.35).sum(axis=0).astype(np.float64)
        weights = columns + columns.max() * 0.08 + 1e-6
        xs = x0 + rng.choice(x1 - x0, self.line_count, p=weights / weights.sum())
        box_height = max(1, y1 - y0)
        self.lines = np.stack([
            xs + rng.normal(0, (x1 - x0) * 0.02, self.line_count),          # x
            rng.uniform(0, box_height * 1.6, self.line_count),               # phase
            rng.uniform(0.18, 0.7, self.line_count) * box_height,            # length
            rng.uniform(1.6, 3.6, self.line_count) * box_height,             # speed
            rng.uniform(0.0016, 0.006, self.line_count) * height + 0.8,      # half width
            rng.uniform(0.55, 1.0, self.line_count),                         # brightness
        ], axis=1)

    def _timeline(self, age: float) -> dict[str, float]:
        if self.direction == OUT:
            return {
                "lines": _ramp(age, 0.0, 0.12) * (1 - _ramp(age, 0.58, 1.05)),
                "width": 1 - _ramp(age, 0.28, 0.50),
                "white": 0.55 * _ramp(age, 0.04, 0.28) + 0.45 * _ramp(age, 0.28, 0.48),
                "flicker": 1.0 if 0.12 < age < 0.28 else 0.0,
                "beam": _ramp(age, 0.42, 0.50) * (1 - _ramp(age, 0.52, 0.72)),
                "stretch": _ramp(age, 0.28, 0.50),
            }
        return {
            "lines": _ramp(age, 0.0, 0.10) * (1 - _ramp(age, 0.48, 0.9)),
            "width": _ramp(age, 0.22, 0.44),
            "white": 1 - _ramp(age, 0.30, 0.65),
            "flicker": 1.0 if 0.44 < age < 0.56 else 0.0,
            "beam": _ramp(age, 0.08, 0.18) * (1 - _ramp(age, 0.24, 0.40)),
            "stretch": 1 - _ramp(age, 0.22, 0.44),
        }

    def render(self, frame, background, mask, age):
        beats = self._timeline(age)
        output = background.copy()
        height, width = frame.shape[:2]
        visible = beats["width"]
        if beats["flicker"] and int(age * 30) % 2:
            visible *= 0.55

        if beats["width"] > 0.01:
            box = mask_box(mask, 0.05)
            if box is not None:
                x0, y0, x1, y1 = box
                crop = frame[y0:y1, x0:x1].astype(np.float32)
                alpha = mask[y0:y1, x0:x1]
                white = beats["white"]
                crop = crop * (1 - white) + np.array((255, 250, 238), np.float32) * white
                squeeze = max(0.02, beats["width"])
                if squeeze < 0.999 or beats["stretch"] > 0.001:
                    new_width = max(1, int((x1 - x0) * squeeze))
                    new_height = max(1, int((y1 - y0) * (1 + 0.12 * beats["stretch"])))
                    crop = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    alpha = cv2.resize(alpha, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    # Squeeze toward the body's center line, feet stay on the floor.
                    center = (x0 + x1) // 2
                    x0, x1 = center - new_width // 2, center - new_width // 2 + new_width
                    y0, y1 = y1 - new_height, y1
                _paste(output, crop, alpha * visible, x0, y0)

        self._draw_lines(output, beats, age)
        return output

    def _draw_lines(self, output: np.ndarray, beats: dict[str, float], age: float) -> None:
        if beats["lines"] <= 0.003 and beats["beam"] <= 0.003:
            return
        x0, y0, x1, y1 = self.box
        height, width = output.shape[:2]
        left, right = max(0, x0 - 20), min(width, x1 + 20)
        top, bottom = max(0, y0 - 40), min(height, y1 + 10)
        if left >= right or top >= bottom:
            return
        layer = np.zeros((bottom - top, right - left), np.float32)
        box_height = max(1, y1 - y0)
        # Lines rise when you leave and rain down when you arrive.
        sign = -1 if self.direction == OUT else 1
        for x, phase, length, speed, half, bright in self.lines:
            travel = (phase + speed * age) % (box_height + length)
            head = (y1 - travel) if sign < 0 else (y0 - length + travel)
            tail = head + length
            cx = x - left
            spindle = np.array((
                (cx, head - top), (cx + half, (head + tail) * 0.5 - top),
                (cx, tail - top), (cx - half, (head + tail) * 0.5 - top),
            ), np.int32)
            cv2.fillConvexPoly(layer, spindle, float(bright), cv2.LINE_AA)
        # Lines hug the silhouette, with a little spill for the comic feel.
        layer *= 0.3 + 0.7 * self.silhouette[top:bottom, left:right]
        layer *= beats["lines"]
        if beats["beam"] > 0.003:
            beam_half = max(1, int((x1 - x0) * 0.03 * beats["beam"]))
            cx = self.center_x - left
            cv2.rectangle(layer, (cx - beam_half, y0 - top), (cx + beam_half, y1 - top),
                          float(beats["beam"] * 1.4), -1)
        glow = cv2.GaussianBlur(layer, (0, 0), sigmaX=max(2.0, height * 0.007))
        light = (layer * 0.95 + glow * 0.9)[:, :, None]
        tint = np.array((255, 246, 225), np.float32)  # White with a cool edge (BGR).
        region = output[top:bottom, left:right].astype(np.float32)
        output[top:bottom, left:right] = np.clip(region + light * tint, 0, 255).astype(np.uint8)


def _paste(output: np.ndarray, image: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    height, width = output.shape[:2]
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + image.shape[1]), min(height, y + image.shape[0])
    if left >= right or top >= bottom:
        return
    part = image[top - y:bottom - y, left - x:right - x]
    a = np.clip(alpha[top - y:bottom - y, left - x:right - x], 0, 1)[:, :, None]
    region = output[top:bottom, left:right].astype(np.float32)
    output[top:bottom, left:right] = (part * a + region * (1 - a)).astype(np.uint8)


# ------------------------------------------------------------------- naruto

class NarutoEffect(Effect):
    """A puff of smoke swaps you for a wooden log that drops to the floor."""

    out_seconds = naruto.EFFECT_SECONDS
    in_seconds = naruto.RETURN_SECONDS

    def __init__(self):
        super().__init__()
        self.sprite = naruto.load_log()
        self.cache: dict[int, np.ndarray] = {}
        self.anchor = None
        self.landed = None

    def begin(self, direction, frame, background, mask):
        super().begin(direction, frame, background, mask)
        if direction == OUT:
            self.anchor = naruto.substitution_anchor(mask)
            self.landed = naruto.landed_log_anchor(self.anchor, frame.shape[0])
        else:
            self.anchor = naruto.substitution_anchor(mask)

    def needs_mask(self, age):
        # The vanish freezes the person after the first instant; the return tracks you.
        return self.direction == IN or age < 0.13

    def render(self, frame, background, mask, age):
        age = max(0.04, age)
        if self.direction == OUT:
            return naruto.render_substitution(frame, background, mask, self.sprite,
                                              self.anchor, age, self.cache)
        return naruto.render_return(frame, background, mask, self.sprite, self.anchor,
                                    self.landed, age, self.cache)

    def vanished(self, background):
        output = background.copy()
        if self.landed is not None:
            naruto.overlay_log(output, self.sprite, self.landed, cache=self.cache)
        return output


EFFECTS = {
    "normal": DissolveEffect,
    "particles": ParticlesEffect,
    "naruto": NarutoEffect,
    "goku": GokuEffect,
}


def create_effect(mode: str) -> Effect:
    return EFFECTS[mode]()
