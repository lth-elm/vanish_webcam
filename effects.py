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

def _noise(rng: np.random.Generator, height: int, width: int, cell: float,
           octaves: int = 4) -> np.ndarray:
    """Fractal value noise in 0..1 whose largest features are about `cell` px."""
    total = np.zeros((height, width), np.float32)
    weight = 0.0
    for octave in range(octaves):
        size = cell / 2 ** octave
        grid = rng.random((max(2, int(height / size) + 2), max(2, int(width / size) + 2)),
                          dtype=np.float32)
        total += cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC) / 2 ** octave
        weight += 1 / 2 ** octave
    total /= weight
    low, high = np.percentile(total, (1, 99))
    return np.clip((total - low) / max(high - low, 1e-6), 0, 1)


class DissolveEffect(Effect):
    """You burn away from the head down along an organic, glowing cyan edge.

    The edge is a noise-shaped front with a white-hot core and a soft bloom; a
    faint charge creeps just ahead of it, and thousands of glowing motes shed
    from it, flicker and float up. The return plays it backwards, faster:
    the motes gather and you re-form from the feet up.
    """

    out_seconds = 2.1
    in_seconds = 1.4
    sweep = (0.05, 1.5)    # The front starts at your head and leaves your feet.
    rim = 0.02             # Width of the glowing edge, in field units.
    glow = np.array((235, 205, 80), np.float32)     # BGR: the mode's cyan (#50CDEB).
    hot = np.array((255, 250, 240), np.float32)     # The white-hot core of the edge.
    motes_count = 3600

    def __init__(self):
        super().__init__()
        self.field: np.ndarray | None = None
        self.box = None
        self.range = (0.0, 1.0)
        self.motes: dict[str, np.ndarray] | None = None
        # The first run loads a few lazy modules; do it now, not on your first vanish.
        still = np.zeros((48, 64, 3), np.uint8)
        blob = np.zeros((48, 64), np.float32)
        blob[12:, 24:40] = 1.0
        self.begin(OUT, still, still, blob)
        self.render(still, still, blob, 1.0)

    def begin(self, direction, frame, background, mask):
        super().begin(direction, frame, background, mask)
        height, width = frame.shape[:2]
        rng = np.random.default_rng()
        box = mask_box(mask, 0.3) or (width // 3, height // 5, 2 * width // 3, height)
        x0, y0, x1, y1 = box
        # Work in your box, widened for the bloom and the motes floating up.
        pad_x, pad_top = int((x1 - x0) * 0.15) + 8, int(height * 0.2)
        self.box = (max(0, x0 - pad_x), max(0, y0 - pad_top), min(width, x1 + pad_x), y1)
        bx0, by0, bx1, by1 = self.box
        box_h = max(1, y1 - y0)
        rows = (np.arange(by0, by1, dtype=np.float32) - y0) / box_h
        # Made at half size (it is smooth) and scaled up: a quarter of the cost.
        noise = _noise(rng, max(2, (by1 - by0) // 2), max(2, (bx1 - bx0) // 2), cell=box_h * 0.11)
        noise = cv2.resize(noise, (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_LINEAR)
        # 0 at the top of your head, 1 at your feet, with a ragged burning edge.
        self.field = np.clip(0.7 * rows[:, None] + 0.3 * noise, -0.2, 1.2).astype(np.float32)

        # The edge sweeps exactly across your own range of the field, so no time
        # is lost before it reaches your head or after it leaves your feet.
        points = np.argwhere(mask[by0:by1, bx0:bx1] > 0.5)
        if len(points):
            values = self.field[points[:, 0], points[:, 1]]
            low, high = np.percentile(values, (0.5, 99.5))
        else:
            low, high = 0.0, 1.0
        self.range = (float(low) - 2 * self.rim, float(high) + 2 * self.rim)

        # Motes: points on you, each set free when the edge passes it.
        if len(points):
            pick = points[rng.choice(len(points), min(self.motes_count, len(points)))]
            level = self.field[pick[:, 0], pick[:, 1]]
            count = len(pick)
            self.motes = {
                "x": (pick[:, 1] + bx0).astype(np.float32),
                "y": (pick[:, 0] + by0).astype(np.float32),
                "born": self._time_at(level),
                "life": rng.uniform(0.35, 0.75, count).astype(np.float32),
                "vx": (rng.normal(0, 0.05, count) * box_h).astype(np.float32),
                "vy": (-rng.uniform(0.08, 0.3, count) * box_h).astype(np.float32),
                "sway": rng.uniform(0.004, 0.015, count).astype(np.float32) * box_h,
                "freq": rng.uniform(3, 8, count).astype(np.float32),
                "phase": rng.uniform(0, 2 * np.pi, count).astype(np.float32),
                "bright": rng.uniform(0.5, 1.0, count).astype(np.float32),
            }
        else:
            self.motes = None
        self.dot = max(1, round(height / 720))   # Mote size, at half resolution.

    def _level(self, clock: float) -> float:
        """Where the edge is (field units) at a point of the vanish timeline."""
        start, end = self.sweep
        share = min(1.0, max(0.0, (clock - start) / (end - start)))
        eased = 0.5 * share + 0.5 * share * share * (3 - 2 * share)   # Gentle in and out.
        low, high = self.range
        return low + eased * (high - low)

    def _time_at(self, level: np.ndarray) -> np.ndarray:
        """Inverse of _level, for the motes' birth times."""
        low, high = self.range
        eased = np.clip((level - low) / max(high - low, 1e-6), 0, 1)
        # Invert the easing with a few Newton steps; plenty for timing.
        guess = eased.copy()
        for _ in range(5):
            value = 0.5 * guess + 0.5 * guess * guess * (3 - 2 * guess) - eased
            slope = 0.5 + 3 * guess * (1 - guess)
            guess = np.clip(guess - value / slope, 0, 1)
        start, end = self.sweep
        return (start + guess * (end - start)).astype(np.float32)

    def render(self, frame, background, mask, age):
        if self.direction == OUT:
            clock = age
        else:
            # The vanish backwards, from while its motes are still in the air,
            # so the return starts with them gathering rather than with nothing.
            clock = max(0.0, 1 - age / self.in_seconds) * (self.sweep[1] + 0.25)
        level = self._level(clock)
        output = background.copy()
        x0, y0, x1, y1 = self.box
        area = np.s_[y0:y1, x0:x1]
        ahead = self.field - level                # > 0: still there; < 0: burnt away.
        keep = (np.clip(ahead * 125 + 0.5, 0, 1) * mask[area]).astype(np.float32)
        output[area] = cv2.blendLinear(frame[area], background[area], keep, 1 - keep)

        # All the light is soft, so it is worked out at half size and added on top.
        half = (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2))
        field = cv2.resize(self.field, half, interpolation=cv2.INTER_AREA)
        person = cv2.resize(mask[area], half, interpolation=cv2.INTER_AREA)
        ahead = field - level
        still = np.clip(ahead * 125 + 0.5, 0, 1) * person
        # A cyan charge creeps over you just ahead of the edge.
        charge = np.exp(-np.maximum(ahead, 0) / 0.09) * still
        light = self._motes(clock, (half[1], half[0]), (x0, y0))
        if 0.0 < clock < self.sweep[1] + 0.1:
            light += np.exp(-(ahead / self.rim) ** 2) * np.minimum(1, person * 1.5)
        if light.max() < 0.003 and charge.max() < 0.003:
            return output
        quarter = (max(1, half[0] // 2), max(1, half[1] // 2))
        bloom = cv2.resize(cv2.GaussianBlur(cv2.resize(light, quarter, interpolation=cv2.INTER_AREA),
                                            (0, 0), max(1.0, (y1 - y0) * 0.004)),
                           half, interpolation=cv2.INTER_LINEAR)
        core = np.clip(light, 0, 1)
        glow = (0.4 * charge + 0.75 * core + 0.75 * np.clip(bloom * 1.6, 0, 1))[:, :, None]
        add = self.glow * glow + (self.hot - self.glow) * (0.55 * core ** 3)[:, :, None]
        add = cv2.resize(np.clip(add, 0, 255).astype(np.uint8), (x1 - x0, y1 - y0),
                         interpolation=cv2.INTER_LINEAR)
        output[area] = cv2.add(output[area], add)
        return output

    def _motes(self, clock: float, shape, origin) -> np.ndarray:
        """Glowing motes as a light map at half the size of the work box."""
        light = np.zeros(shape, np.float32)
        p = self.motes
        if p is None:
            return light
        tau = clock - p["born"]
        live = (tau > 0) & (tau < p["life"])
        if not live.any():
            return light
        t = tau[live]
        x = (p["x"][live] + p["vx"][live] * t
             + p["sway"][live] * np.sin(p["freq"][live] * t * 2 * np.pi + p["phase"][live]))
        y = p["y"][live] + p["vy"][live] * t * (0.6 + 0.8 * t)   # Speeds up as it rises.
        fade = (1 - t / p["life"][live]) ** 1.5
        flicker = 0.7 + 0.3 * np.sin(p["phase"][live] * 7 + t * 40)
        value = 2.2 * p["bright"][live] * fade * flicker
        height, width = shape
        xi = ((x - origin[0]) / 2).astype(np.int32)
        yi = ((y - origin[1]) / 2).astype(np.int32)
        inside = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
        index = yi[inside] * width + xi[inside]
        light = np.bincount(index, value[inside], height * width).astype(np.float32)
        light = light.reshape(height, width)
        if self.dot > 1:
            light = cv2.dilate(light, np.ones((self.dot, self.dot), np.uint8))
        return light


# ---------------------------------------------------------------- particles

class ParticlesEffect(Effect):
    """A ragged front sweeps up from your feet to your head. Everything below it
    breaks into small cyan blocks that drift up and fade; above it you stay live.

    Measured on the original author's demo (30 fps): the front starts slowly at
    your feet, speeds up, reaches the top of the head after ~0.64 s, and the last
    blocks are gone ~0.95 s after the sign.
    Light parts of you turn bright cyan, dark parts (hair, beard) stay dark navy.
    The return plays the vanish backwards: you rebuild from the head down.
    """

    sweep = 0.64          # Seconds for the front to cross you.
    ease = 1.8            # > 1: the front starts slowly and speeds up, like the original.
    margin = 0.06         # The front starts this far below you and ends this far above.
    out_seconds = 0.95
    in_seconds = 0.95
    cyan = np.array((250, 222, 50), np.float32)   # BGR, from the demo (its camera washes it out a bit).
    navy = np.array((78, 52, 28), np.float32)
    keep_share = 0.52     # Share of the grid cells under you that become blocks.

    def __init__(self):
        super().__init__()
        self.field: np.ndarray | None = None
        self.particles: dict[str, np.ndarray] | None = None
        self.size = 6
        # The first run loads a few lazy modules; do it now, not on your first vanish.
        still = np.zeros((48, 64, 3), np.uint8)
        blob = np.zeros((48, 64), np.float32)
        blob[12:, 24:40] = 1.0
        self.begin(OUT, still, still, blob)
        self.render(still, still, blob, 0.3)

    def begin(self, direction, frame, background, mask):
        super().begin(direction, frame, background, mask)
        height, width = frame.shape[:2]
        rng = np.random.default_rng()
        box = mask_box(mask, 0.35) or (width // 3, height // 4, 2 * width // 3, height)
        x0, y0, x1, y1 = box

        # 0 at your feet, 1 at the top of your head, with a ragged edge.
        rows = (y1 - np.arange(height, dtype=np.float32)) / max(1, y1 - y0)
        coarse = rng.random((max(2, height // 24), max(2, width // 24)), dtype=np.float32)
        fine = rng.random((max(2, height // 9), max(2, width // 9)), dtype=np.float32)
        noise = (0.1 * cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
                 + 0.05 * cv2.resize(fine, (width, height), interpolation=cv2.INTER_LINEAR))
        self.field = rows[:, None] + noise - 0.075

        self.size = size = max(3, round(height * 0.019))
        grid = mask[y0:y1:size, x0:x1:size] > 0.5
        grid &= rng.random(grid.shape) < self.keep_share
        gy, gx = np.nonzero(grid)
        px = x0 + gx * size + rng.uniform(-0.3, 0.3, gx.size) * size
        py = y0 + gy * size + rng.uniform(-0.3, 0.3, gy.size) * size
        cx = np.clip(px + size / 2, 0, width - 1).astype(np.int32)
        cy = np.clip(py + size / 2, 0, height - 1).astype(np.int32)
        bgr = frame[cy, cx].astype(np.float32)
        luma = (bgr @ np.array((0.114, 0.587, 0.299), np.float32)) / 255
        # Blocks are bright cyan; a share of those from dark parts of you (hair,
        # beard) stay dark, like on the original.
        count = px.size
        dark = (luma < 0.28) & (rng.random(count) < 0.45)
        shade = np.where(dark, 0.08, 1.0).astype(np.float32)[:, None]
        self.particles = {
            "x": px.astype(np.float32), "y": py.astype(np.float32),
            "color": self.navy + (self.cyan - self.navy) * shade,
            "born": self._front_time(self.field[cy, cx]),
            "life": rng.uniform(0.1, 0.24, count).astype(np.float32),
            "vx": (rng.normal(0, 0.035, count) * height).astype(np.float32),
            "vy": (-rng.uniform(0.02, 0.14, count) * height).astype(np.float32),
            "lift": (-rng.uniform(0.0, 0.25, count) * height).astype(np.float32),
        }

    def _front_time(self, level):
        """When the front reaches a field level."""
        share = np.clip((level + self.margin) / (1 + 2 * self.margin), 0, 1)
        return share ** (1 / self.ease) * self.sweep

    def _front_level(self, clock: float) -> float:
        share = min(1.0, max(0.0, clock / self.sweep)) ** self.ease
        return share * (1 + 2 * self.margin) - self.margin

    def render(self, frame, background, mask, age):
        # The return is the vanish played backwards.
        clock = age if self.direction == OUT else max(0.0, self.seconds - age)
        output = background.copy()
        level = self._front_level(clock)
        box = mask_box(mask, 0.02, 2)
        if box is not None and level < 1.2:
            x0, y0, x1, y1 = box
            keep = smoothstep(level - 0.008, level + 0.008, self.field[y0:y1, x0:x1])
            weight = (mask[y0:y1, x0:x1] * keep).astype(np.float32)
            output[y0:y1, x0:x1] = cv2.blendLinear(frame[y0:y1, x0:x1], background[y0:y1, x0:x1],
                                                   weight, 1 - weight)
        self._draw_blocks(output, clock)
        return output

    def _draw_blocks(self, output: np.ndarray, clock: float) -> None:
        p = self.particles
        if p is None or not p["x"].size:
            return
        tau = clock - p["born"]
        active = (tau > 0) & (tau < p["life"])
        if not active.any():
            return
        t = tau[active]
        size = self.size
        height, width = output.shape[:2]
        xs = (p["x"][active] + p["vx"][active] * t).astype(np.int32)
        ys = (p["y"][active] + p["vy"][active] * t + 0.5 * p["lift"][active] * t * t).astype(np.int32)
        alpha = 1 - (t / p["life"][active]) ** 2.5  # Solid, then gone quickly.
        colors = p["color"][active]
        inside = (xs > -size) & (ys > -size) & (xs < width) & (ys < height)
        if not inside.any():
            return
        xs, ys, alpha, colors = xs[inside], ys[inside], alpha[inside], colors[inside]

        # Splat every block into the region it covers in one pass.
        left, top = max(0, int(xs.min())), max(0, int(ys.min()))
        right, bottom = min(width, int(xs.max()) + size), min(height, int(ys.max()) + size)
        region_w, region_h = right - left, bottom - top
        dy, dx = np.divmod(np.arange(size * size), size)
        px = xs[:, None] + dx
        py = ys[:, None] + dy
        valid = (px >= left) & (px < right) & (py >= top) & (py < bottom)
        index = ((py - top) * region_w + (px - left))[valid]
        cells = region_w * region_h

        def splat(values):
            return np.bincount(index, np.broadcast_to(values[:, None], px.shape)[valid], cells)

        total = splat(alpha).reshape(region_h, region_w, 1)
        paint = np.stack([splat(alpha * colors[:, c]) for c in range(3)], axis=1)
        paint = paint.reshape(region_h, region_w, 3) / np.maximum(total, 1e-6)
        cover = np.minimum(1.0, total)
        area = output[top:bottom, left:right].astype(np.float32)
        lit = area * (1 - cover) + paint * cover
        # A soft halo, as the blocks glow on the original. Blurred at a quarter
        # of the size: it is soft anyway, and a full-size blur costs ~30 ms at 1080p.
        light = paint * cover
        small = cv2.resize(light, (max(1, region_w // 4), max(1, region_h // 4)),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(0.8, size * 0.9 / 4))
        glow = cv2.resize(small, (region_w, region_h), interpolation=cv2.INTER_LINEAR)
        output[top:bottom, left:right] = np.clip(lit + 0.4 * glow, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------- goku

def _ramp(value: float, start: float, end: float) -> float:
    return float(np.clip((value - start) / max(end - start, 1e-6), 0.0, 1.0))


class _GokuBeats:
    """When each layer of the teleport shows, in seconds from the trigger."""

    def __init__(self, seconds, body, outline, born, grow, die):
        self.seconds = seconds
        self.body = body          # (start, end): your opacity goes 1 -> 0 (out) or 0 -> 1 (in).
        self.outline = outline    # (on, full, fade, off)
        self.born = born          # Most streaks are there on the first frame, the rest by this.
        self.grow = grow          # Time for a streak to reach its full length.
        self.die = die            # (first, last): streaks blink out one by one, most near the end.


# Measured frame by frame on the original author's demo (30 fps): streaks drop
# over you, two frames later you blink out through one see-through frame, a
# jittery outline of you stays while the streaks thin out, and everything is
# gone after nine frames. The return flashes the outline and the streaks for
# one frame, pops you in, and the streaks thin out over the next six.
GOKU_OUT = _GokuBeats(0.30, body=(0.03, 0.075), outline=(0.03, 0.05, 0.265, 0.30),
                      born=0.03, grow=0.03, die=(0.17, 0.265))
GOKU_IN = _GokuBeats(0.25, body=(0.015, 0.03), outline=(-1.0, 0.0, 0.02, 0.035),
                     born=0.03, grow=0.03, die=(0.16, 0.245))


class GokuEffect(Effect):
    """Instant Transmission: streaks fall over you, you blink out and a
    jittery white outline of your silhouette lingers for a beat.

    The return plays the same beats the other way: the outline and the
    streaks show where you will be, then you pop back in.
    """

    mask_interval = 1.0
    line_density = 34  # Streaks across the full frame width (the original shows ~23 white, ~9 dark).
    dark_share = 0.28

    def __init__(self):
        super().__init__()
        self.beats = GOKU_OUT
        self.mask: np.ndarray | None = None
        self.box = None
        self.lines: np.ndarray | None = None
        self.dark: np.ndarray | None = None
        self.outline: list[np.ndarray] = []
        self.outline_width = 2
        # The first run loads a few lazy modules; do it now, not on your first teleport.
        still = np.zeros((48, 64, 3), np.uint8)
        blob = np.zeros((48, 64), np.float32)
        blob[12:, 24:40] = 1.0
        self.begin(OUT, still, still, blob)
        self.render(still, still, blob, 0.05)

    @property
    def out_seconds(self) -> float:
        return GOKU_OUT.seconds

    @property
    def in_seconds(self) -> float:
        return GOKU_IN.seconds

    def begin(self, direction, frame, background, mask):
        super().begin(direction, frame, background, mask)
        self.beats = GOKU_OUT if direction == OUT else GOKU_IN
        self.mask = mask.astype(np.float32)
        self.box = mask_box(mask, 0.02, 2)
        rng = np.random.default_rng()
        height, width = frame.shape[:2]
        box = mask_box(mask, 0.35) or (int(width * 0.3), int(height * 0.1),
                                       int(width * 0.7), height)
        self.lines, self.dark = self._make_lines(rng, box, width, height)
        self.outline = _jittered_outline(mask, rng)
        self.outline_width = max(1, round(height / 320))

    def _make_lines(self, rng, box, width, height):
        x0, y0, x1, y1 = box
        margin = width * 0.1
        left, right = max(0.0, x0 - margin), min(width - 1.0, x1 + margin)
        count = max(8, round(self.line_density * (right - left) / width))
        # One streak per slot, nudged inside it: even coverage, no tidy grid.
        slots = np.linspace(left, right, count + 1)
        xs = slots[:-1] + rng.uniform(0.1, 0.9, count) * (slots[1] - slots[0])
        top = rng.uniform(y0 - 0.3 * height, y0 + 0.45 * (y1 - y0), count).clip(0, height * 0.7)
        length = rng.uniform(0.22, 0.88, count) * height
        scale = height / 720
        dark = rng.random(count) < self.dark_share
        thickness = np.where(dark, 1.0, rng.choice((1.0, 1.5, 1.5, 2.0), count)) * scale
        thickness = np.maximum(1, np.round(thickness))
        beats = self.beats
        born = np.where(rng.random(count) < 0.75, 0.0, rng.uniform(0, beats.born, count))
        first, last = beats.die
        die = first + (last - first) * np.sqrt(rng.random(count))
        lines = np.stack([xs, top, np.minimum(height - 1, top + length), thickness, born, die], 1)
        return lines, dark

    def needs_mask(self, age):
        return False  # Everything is drawn from the mask taken at the trigger.

    def render(self, frame, background, mask, age):
        beats = self.beats
        body = _ramp(age, *beats.body)
        if self.direction == OUT:
            body = 1.0 - body
        if body >= 0.999:
            output = frame.copy()
        elif body <= 0.001:
            output = background.copy()
        else:
            # Half-gone, you read as a pale ghost rather than a dim copy.
            output = _blend_person(frame, background, self.mask, self.box, body,
                                   pale=0.5 * (1 - body))

        streaks = self._draw_streaks(output, age)
        if streaks is not None:
            output = cv2.addWeighted(streaks, 0.78, output, 0.22, 0)
        on, full, fade, off = beats.outline
        outline = 0.8 * _ramp(age, on, full) * (1.0 - _ramp(age, fade, off))
        if outline > 0.01 and self.outline:
            overlay = output.copy()
            cv2.polylines(overlay, self.outline, False, (240, 244, 246),
                          self.outline_width, cv2.LINE_AA, shift=_SHIFT)
            output = cv2.addWeighted(overlay, outline, output, 1.0 - outline, 0)
        return output

    def _draw_streaks(self, output: np.ndarray, age: float) -> np.ndarray | None:
        lines = self.lines
        if lines is None or not lines.size:
            return None
        alive = (age >= lines[:, 4]) & (age < lines[:, 5])
        if not alive.any():
            return None
        overlay = output.copy()
        grow = np.clip((age - lines[:, 4]) / self.beats.grow, 0.6, 1.0)
        for (x, top, bottom, thickness, _, _), dark, reach in zip(
                lines[alive], self.dark[alive], grow[alive]):
            end = top + (bottom - top) * reach
            color = (40, 40, 42) if dark else (250, 250, 250)
            cv2.line(overlay, (int(x * _ONE), int(top * _ONE)), (int(x * _ONE), int(end * _ONE)),
                     color, int(thickness), cv2.LINE_AA, shift=_SHIFT)
        return overlay


_SHIFT = 2           # Sub-pixel bits for cv2 drawing.
_ONE = 1 << _SHIFT


def _blend_person(frame, background, mask, box, alpha, pale=0.0):
    output = background.copy()
    if box is not None:
        x0, y0, x1, y1 = box
        person = frame[y0:y1, x0:x1]
        if pale > 0:
            person = cv2.addWeighted(person, 1 - pale, np.full_like(person, 255), pale, 0)
        weight = mask[y0:y1, x0:x1] * np.float32(alpha)
        output[y0:y1, x0:x1] = cv2.blendLinear(person, background[y0:y1, x0:x1],
                                               weight, 1 - weight)
    return output


def _jittered_outline(mask: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """The silhouette's edges (holes included) as wobbly open polylines.

    Edges that run along the frame border are left out, like in the original.
    """
    height, width = mask.shape
    scale = min(1.0, 640 / max(width, 1))  # Trace at <= 640 px wide: cheaper and smoother.
    small = cv2.resize(mask, (max(1, round(width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.2)
    contours, _ = cv2.findContours((small > 0.5).astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    min_area = small.size * 0.0015
    step = max(2.0, height / 240)       # Distance between wobble points, full-size pixels.
    wobble = height / 420               # How far the line strays from the true edge.
    border = 2.5 / scale
    polylines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        points = contour[:, 0, :].astype(np.float64) / scale
        points = _resample(points, step)
        if len(points) < 4:
            continue
        # Push each point along the edge normal by a little smoothed noise.
        tangent = np.roll(points, -1, 0) - np.roll(points, 1, 0)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        noise = rng.normal(0, 1, len(points))
        noise = 0.7 * noise + 0.15 * (np.roll(noise, 1) + np.roll(noise, -1))
        points = points + normal * (noise * wobble)[:, None]
        on_edge = ((points[:, 0] < border) | (points[:, 1] < border)
                   | (points[:, 0] > width - 1 - border) | (points[:, 1] > height - 1 - border))
        for run in _runs(~on_edge, closed=True):
            if len(run) >= 3:
                polylines.append(np.round(points[run] * _ONE).astype(np.int32))
    return polylines


def _resample(points: np.ndarray, step: float) -> np.ndarray:
    """Evenly spaced points along a closed contour."""
    closed = np.vstack([points, points[:1]])
    lengths = np.hypot(*np.diff(closed, axis=0).T)
    distance = np.concatenate([[0.0], np.cumsum(lengths)])
    total = distance[-1]
    if total < step * 3:
        return points
    samples = np.arange(0.0, total, step)
    return np.stack([np.interp(samples, distance, closed[:, 0]),
                     np.interp(samples, distance, closed[:, 1])], 1)


def _runs(keep: np.ndarray, closed: bool) -> list[np.ndarray]:
    """Index runs where `keep` is True; on a closed loop the last run wraps to the first."""
    if keep.all():
        indices = np.arange(len(keep))
        return [np.append(indices, 0)] if closed else [indices]
    start = int(np.argmin(keep))  # Start on a gap so no run is cut in two.
    order = np.roll(np.arange(len(keep)), -start)
    runs, current = [], []
    for index in order:
        if keep[index]:
            current.append(index)
        elif current:
            runs.append(np.array(current))
            current = []
    if current:
        runs.append(np.array(current))
    return runs


# ------------------------------------------------------------------- naruto

class NarutoEffect(Effect):
    """A puff of smoke bursts out of you; when it is thickest you are swapped
    for the empty room, then it rolls out, rises and breaks into wisps.

    The return uses the same cloud and swaps the room for you instead.
    """

    out_seconds = naruto.SECONDS
    in_seconds = naruto.SECONDS

    def __init__(self):
        super().__init__()
        self.cloud: naruto.SmokeCloud | None = None
        # The first run loads a few lazy modules; do it now, not on your first vanish.
        still = np.zeros((48, 64, 3), np.uint8)
        blob = np.zeros((48, 64), np.float32)
        blob[12:, 24:40] = 1.0
        self.begin(OUT, still, still, blob)
        self.render(still, still, blob, 0.2)

    def begin(self, direction, frame, background, mask):
        super().begin(direction, frame, background, mask)
        self.cloud = naruto.SmokeCloud(mask)

    def needs_mask(self, age):
        return False  # The cloud is laid out from the mask taken at the trigger.

    def render(self, frame, background, mask, age):
        before, after = (frame, background) if self.direction == OUT else (background, frame)
        return self.cloud.composite(before if age < naruto.CUT else after, age)


EFFECTS = {
    "normal": DissolveEffect,
    "particles": ParticlesEffect,
    "naruto": NarutoEffect,
    "goku": GokuEffect,
}


def create_effect(mode: str) -> Effect:
    return EFFECTS[mode]()
