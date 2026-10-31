"""Smoke cloud for the Naruto mode.

A puff of smoke bursts out of your silhouette, you are swapped for the empty
room (or back) while it is thickest, then it rolls outward, drifts up and
breaks into wisps. Everything is drawn in code, the way stylized smoke is
usually painted: many round billows, each lit from the upper left with a
cool grey underside, layered back to front so the lumps stay distinct. Each
billow erodes from its noisy edge as it ages, which is what leaves wisps.
"""

from __future__ import annotations

import cv2
import numpy as np


SECONDS = 1.5         # Matches the "poof" sound, which fades out by ~1.6 s.
CUT = 0.16            # The swap happens here, hidden by the thickest smoke.
WORK_WIDTH = 512      # The cloud is drawn at this width, then scaled up.

LIGHT = np.array((248, 248, 250), np.float32)    # BGR: lit side of a billow.
SHADOW = np.array((168, 158, 152), np.float32)   # Cool grey underside.
LIGHT_DIRECTION = np.array((-0.45, -0.7, 0.55), np.float32)  # From the upper left, in front.


def _smoothstep(edge0: float, edge1: float, values):
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _ease(edge0: float, edge1: float, value: float) -> float:
    """Scalar smoothstep, without numpy's per-call overhead (called ~100x a frame)."""
    value = min(1.0, max(0.0, (value - edge0) / max(edge1 - edge0, 1e-6)))
    return value * value * (3.0 - 2.0 * value)


def _fractal(rng: np.random.Generator, size: int, octaves: int = 4, base: int = 3) -> np.ndarray:
    total = np.zeros((size, size), np.float32)
    weight = 0.0
    for octave in range(octaves):
        cells = base * 2 ** octave
        layer = rng.random((cells, cells), dtype=np.float32)
        total += cv2.resize(layer, (size, size), interpolation=cv2.INTER_CUBIC) / 2 ** octave
        weight += 1 / 2 ** octave
    return total / weight


def _lit(height: np.ndarray, step: float) -> np.ndarray:
    """Shade 0..1 of a height map lit from LIGHT_DIRECTION."""
    light = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)
    gy, gx = np.gradient(height, step)
    lit = (-gx * light[0] - gy * light[1] + light[2]) / np.sqrt(gx * gx + gy * gy + 1)
    return np.clip(0.28 + 0.72 * np.clip(lit, 0, 1), 0, 1).astype(np.float32)


def make_billows(rng: np.random.Generator, count: int = 8, size: int = 128):
    """Pre-lit billow sprites: (density 0..1, shade 0..1) pairs.

    Each is a cauliflower-like lump: a bumpy outline, a cloudy inside, and a
    surface of small bumps lit from the upper left, with a grey underside.
    """
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j].astype(np.float32)
    radius = np.sqrt(x * x + y * y)
    step = 2.0 / size
    billows = []
    for _ in range(count):
        edge = 0.68 + 0.32 * _fractal(rng, size, 3, base=4)          # Bumpy outline.
        r = np.clip(radius / edge, 0, 1)
        inside = 0.55 + 0.45 * _fractal(rng, size, 5, base=4)        # Cloudy texture.
        density = cv2.GaussianBlur(np.clip(1 - r ** 3, 0, 1) * inside, (0, 0), size / 100)
        # Height of a soft dome plus small bumps; its slope gives the lighting.
        bumps = cv2.GaussianBlur(_fractal(rng, size, 3, base=5), (0, 0), size / 28)
        height = np.sqrt(np.clip(1 - r * r, 0, 1)) + 0.22 * bumps
        shade = _lit(height, step)
        billows.append((density / max(density.max(), 1e-6), shade.astype(np.float32)))
    return billows


_BILLOWS: list | None = None


def _shared_billows():
    """Billow textures are the same for every burst: make them once."""
    global _BILLOWS
    if _BILLOWS is None:
        _BILLOWS = make_billows(np.random.default_rng(2024), count=12)
    return _BILLOWS


class SmokeCloud:
    """One burst of smoke, laid out around a person mask."""

    def __init__(self, mask: np.ndarray, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self.height, self.width = mask.shape
        self.scale = min(1.0, WORK_WIDTH / self.width)
        work = (max(1, round(self.width * self.scale)), max(1, round(self.height * self.scale)))
        self.work_w, self.work_h = work
        small = cv2.resize(mask.astype(np.float32), work, interpolation=cv2.INTER_AREA)
        self.billows = _shared_billows()

        points = np.argwhere(small > 0.5)
        if len(points) < 10:  # Nobody found: burst in the middle of the frame.
            yy, xx = np.mgrid[int(work[1] * 0.25):work[1], int(work[0] * 0.38):int(work[0] * 0.62)]
            points = np.stack([yy.ravel(), xx.ravel()], 1)
            small = np.zeros_like(small)
            small[points[:, 0], points[:, 1]] = 1.0
        top, left = points.min(0)
        bottom, right = points.max(0)
        box_h, box_w = max(8, bottom - top), max(8, right - left)
        center = np.array([(left + right) / 2, (top + bottom) / 2], np.float32)
        size = float(np.sqrt(box_w * box_h))
        self.size = size

        # Billows born all over you, plus a few rolling out sideways from the base.
        body_count, base_count = 84, 14
        body = points[rng.choice(len(points), body_count)][:, ::-1].astype(np.float32)
        base_y = min(bottom, work[1] - 1) - rng.uniform(0, 0.12, base_count) * box_h
        base_x = center[0] + rng.uniform(-0.6, 0.6, base_count) * box_w
        start = np.vstack([body, np.stack([base_x, base_y], 1)])
        count = len(start)
        away = start - center
        away /= np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-3)
        speed = rng.uniform(0.5, 1.5, count) * size
        speed[-base_count:] *= 1.4
        away[-base_count:, 1] *= 0.25  # Base billows roll along the floor.
        self.start = start
        self.velocity = away * speed[:, None] + np.array([0, -0.15 * size], np.float32)
        self.rise = rng.uniform(0.2, 0.45, count) * size       # Upward drift, per s².
        # A mix of big and small billows (log-normal sizes) looks far less regular.
        self.radius1 = np.clip(rng.lognormal(np.log(0.14), 0.4, count), 0.06, 0.3) * size
        self.radius0 = self.radius1 * rng.uniform(0.4, 0.6, count)
        self.born = np.where(rng.random(count) < 0.7, 0.0, rng.uniform(0, 0.08, count))
        self.life = rng.uniform(0.9, 1.45, count)              # When each billow is gone.
        self.kind = rng.integers(0, len(self.billows), count)
        self.flip = rng.random(count) < 0.5                    # Mirror half: more variety.
        # Painted back to front; billows near the edge of the burst sit behind.
        depth = np.linalg.norm(start - center, axis=1) / max(size, 1) + rng.uniform(0, 0.5, count)
        self.order = np.argsort(-depth)

        # A dense core over you guarantees nothing of you shows at the cut. It is
        # blurred well past your outline so the cloud never takes your shape.
        grow = max(3, int(size * 0.1)) | 1
        core = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
        self.core = cv2.GaussianBlur(core, (0, 0), size * 0.05)
        self.core_shade = _lit(self.core * 6.0 / max(size, 1), 1.0)

    def layer(self, t: float) -> tuple[np.ndarray, np.ndarray] | None:
        """(opacity, shade 0..1) at the working size, or None once it has cleared."""
        if t >= SECONDS:
            return None
        opacity = np.zeros((self.work_h, self.work_w), np.float32)
        paint = np.zeros((self.work_h, self.work_w), np.float32)    # Shade, 0..1.

        # The core: a soft, evenly lit mass behind the billows while you are swapped.
        core = _ease(0.0, 0.05, t) * (1 - _ease(0.18, 0.32, t))
        if core > 0.01:
            alpha = np.clip(self.core * 1.6, 0, 1) * core
            opacity[:] = alpha
            paint[:] = self.core_shade

        tau = t - self.born
        for index in self.order:
            age = float(tau[index])
            life = float(self.life[index])
            if age < 0 or age >= life:
                continue
            # Fast burst outward that slows down (drag), then a lazy rise.
            travel = (1 - np.exp(-4.0 * age)) / 4.0
            x, y = self.start[index] + self.velocity[index] * travel
            y -= self.rise[index] * age * age
            grow = 1 - np.exp(-age / 0.08)
            radius = (self.radius0[index] + (self.radius1[index] - self.radius0[index]) * grow
                      + 0.16 * self.size * age)
            # Each billow eats in from its noisy edge as it ages, leaving wisps.
            erode = 0.04 + 0.92 * _ease(0.3 * life, life, age) ** 0.9
            appear = _ease(0.0, 0.025, age)
            self._paint(opacity, paint, index, x, y, radius, erode, appear)
        if opacity.max() < 0.004:
            return None
        return opacity, paint

    def _paint(self, opacity, paint, index, x, y, radius, erode, appear) -> None:
        diameter = max(3, int(radius * 2))
        left, top = int(x - radius), int(y - radius)
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(self.work_w, left + diameter), min(self.work_h, top + diameter)
        if x0 >= x1 or y0 >= y1:
            return
        density, shade = self.billows[self.kind[index]]
        if self.flip[index]:
            density, shade = density[:, ::-1], shade[:, ::-1]
        crop = np.s_[y0 - top:y1 - top, x0 - left:x1 - left]
        density = cv2.resize(density, (diameter, diameter), interpolation=cv2.INTER_LINEAR)[crop]
        shade = cv2.resize(shade, (diameter, diameter), interpolation=cv2.INTER_LINEAR)[crop]
        alpha = _smoothstep(erode, erode + 0.24, density) * appear
        area = np.s_[y0:y1, x0:x1]
        # "Over": this billow covers what was painted behind it.
        paint[area] = paint[area] * (1 - alpha) + shade * alpha
        opacity[area] = opacity[area] * (1 - alpha) + alpha

    def composite(self, base: np.ndarray, t: float) -> np.ndarray:
        result = self.layer(t)
        if result is None:
            return base.copy()
        opacity, shade = result
        rows, cols = np.nonzero(opacity > 0.004)
        pad = 2
        y0, y1 = max(0, rows.min() - pad), min(self.work_h, rows.max() + pad + 1)
        x0, x1 = max(0, cols.min() - pad), min(self.work_w, cols.max() + pad + 1)
        # Same region at full size; only it is scaled up and blended.
        fx0, fy0 = int(x0 / self.scale), int(y0 / self.scale)
        fx1 = min(self.width, int(np.ceil(x1 / self.scale)))
        fy1 = min(self.height, int(np.ceil(y1 / self.scale)))
        size = (fx1 - fx0, fy1 - fy0)
        alpha = np.clip(cv2.resize(opacity[y0:y1, x0:x1], size, interpolation=cv2.INTER_CUBIC), 0, 1)
        # Color only the smoke's own box, at the working size.
        color = SHADOW + (LIGHT - SHADOW) * shade[y0:y1, x0:x1, None]
        flash = 0.55 * (1 - _ease(0.0, 0.14, t))   # The burst starts nearly white.
        color = np.clip(color + (255 - color) * flash, 0, 255).astype(np.uint8)
        paint = cv2.resize(color, size, interpolation=cv2.INTER_CUBIC)
        output = base.copy()
        region = output[fy0:fy1, fx0:fx1]
        output[fy0:fy1, fx0:fx1] = cv2.blendLinear(paint, region, alpha, 1 - alpha)
        return output
