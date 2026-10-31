"""Synthesize the original effect sounds in assets/ (no recorded or sampled audio).

Run from the repository root:  python tools/make_sounds.py
The output is deterministic, so regenerating gives identical files.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


RATE = 44100
ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _time(seconds: float) -> np.ndarray:
    return np.arange(int(RATE * seconds), dtype=np.float64) / RATE


def _sweep(start_hz: float, end_hz: float, seconds: float) -> np.ndarray:
    """Exponential frequency sweep, returned as phase-continuous sine."""
    t = _time(seconds)
    ratio = end_hz / start_hz
    phase = 2 * np.pi * start_hz * seconds * (ratio ** (t / seconds) - 1) / np.log(ratio)
    return np.sin(phase)


def _envelope(length: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(length) / RATE
    rise = np.clip(t / max(attack, 1e-4), 0, 1)
    return rise * np.exp(-t / max(decay, 1e-4))


def _bandpassed_noise(rng: np.random.Generator, seconds: float,
                      low_hz: np.ndarray | float, high_hz: np.ndarray | float) -> np.ndarray:
    """White noise filtered in short FFT blocks so the band can move over time."""
    length = int(RATE * seconds)
    block = 2048
    hop = block // 2
    window = np.hanning(block)
    noise = rng.standard_normal(length + block)
    output = np.zeros(length + block)
    freqs = np.fft.rfftfreq(block, 1 / RATE)
    low = np.broadcast_to(low_hz, (length,))
    high = np.broadcast_to(high_hz, (length,))
    for start in range(0, length, hop):
        center = min(start + hop, length - 1)
        spectrum = np.fft.rfft(noise[start:start + block] * window)
        band = (freqs >= low[center]) & (freqs <= high[center])
        spectrum[~band] = 0
        output[start:start + block] += np.fft.irfft(spectrum, block) * window
    return output[:length] / max(1e-9, np.abs(output).max())


def _bell(seconds: float, partials: list[tuple[float, float]], decay: float) -> np.ndarray:
    t = _time(seconds)
    tone = sum(gain * np.sin(2 * np.pi * hz * t * (1 + 0.003 * np.sin(2 * np.pi * 6 * t)))
               for hz, gain in partials)
    return tone * np.exp(-t / decay)


def teleport(reverse: bool) -> np.ndarray:
    """A bright 'zip' with a metallic ring: out climbs, in falls."""
    rng = np.random.default_rng(81 if reverse else 80)
    seconds = 0.75
    length = int(RATE * seconds)
    mix = np.zeros(length)

    zip_seconds = 0.16
    zip_tone = (_sweep(2600, 380, zip_seconds) if reverse else _sweep(380, 2600, zip_seconds))
    zip_tone = np.tanh(2.2 * zip_tone)  # A little edge, like a comic sound effect.
    zip_env = np.sin(np.linspace(0, np.pi, zip_tone.size)) ** 0.6
    zip_at = int(RATE * (0.05 if reverse else 0.0))
    mix[zip_at:zip_at + zip_tone.size] += 0.55 * zip_tone * zip_env

    whoosh = _bandpassed_noise(rng, 0.32, 900, np.linspace(2500, 9000, int(RATE * 0.32)))
    whoosh *= np.sin(np.linspace(0, np.pi, whoosh.size)) ** 1.5
    mix[:whoosh.size] += 0.35 * whoosh

    ring = _bell(0.6, [(1760, 1.0), (2637, 0.55), (4186, 0.3), (5274, 0.12)], decay=0.16)
    ring_at = int(RATE * (0.0 if reverse else 0.1))
    end = min(length, ring_at + ring.size)
    mix[ring_at:end] += 0.28 * ring[:end - ring_at] * _envelope(end - ring_at, 0.004, 10)

    # A soft low thump grounds the moment you blink out (or land).
    thump = np.sin(2 * np.pi * 95 * _time(0.18)) * _envelope(int(RATE * 0.18), 0.003, 0.05)
    thump_at = int(RATE * (0.17 if reverse else 0.12))
    mix[thump_at:thump_at + thump.size] += 0.45 * thump

    stereo = np.stack([mix, mix], axis=1)
    return _fade(stereo, 0.004, 0.06)


def wind() -> np.ndarray:
    """Wind gust that sweeps from left to right, matching the particle drift."""
    rng = np.random.default_rng(12)
    seconds = 2.3
    t = _time(seconds)
    sweep = 400 + 1600 * np.sin(np.pi * np.clip(t / seconds, 0, 1)) ** 2
    gust = _bandpassed_noise(rng, seconds, sweep * 0.5, sweep * 2.2)
    swell = np.sin(np.pi * np.clip(t / seconds, 0, 1)) ** 1.6
    flutter = 0.82 + 0.18 * np.sin(2 * np.pi * 5.3 * t + 1.2 * np.sin(2 * np.pi * 0.7 * t))
    sparkle = _bandpassed_noise(rng, seconds, 5000, 11000)
    sparkle *= swell
    mono = 0.7 * gust * swell * flutter + 0.08 * sparkle
    pan = np.clip(t / seconds, 0, 1)
    stereo = np.stack([mono * np.cos(pan * np.pi / 2), mono * np.sin(pan * np.pi / 2)], axis=1)
    return _fade(stereo * 1.25, 0.05, 0.3)


def shimmer() -> np.ndarray:
    """Rising glassy shimmer for the classic dissolve."""
    rng = np.random.default_rng(33)
    seconds = 3.6
    length = int(RATE * seconds)
    mix = np.zeros((length, 2))
    notes = [1318.5, 1568.0, 1975.5, 2349.3, 2637.0, 3136.0]
    for index in range(46):
        at = int(RATE * (seconds * 0.85) * (index / 46) ** 0.8 + rng.integers(0, 2000))
        hz = notes[int(rng.integers(0, len(notes)))] * (2 if rng.random() < 0.3 else 1)
        ping = _bell(0.7, [(hz, 1.0), (hz * 2.01, 0.2)], decay=0.18)
        end = min(length, at + ping.size)
        pan = rng.random()
        gain = 0.09 * (0.6 + 0.4 * rng.random())
        mix[at:end, 0] += gain * ping[:end - at] * (1 - pan)
        mix[at:end, 1] += gain * ping[:end - at] * pan
    air = _bandpassed_noise(rng, seconds, 3000, 9000)
    mix += 0.04 * (air * np.linspace(0, 1, length) ** 2)[:, None]
    return _fade(mix, 0.02, 0.4)


def _fade(samples: np.ndarray, fade_in: float, fade_out: float) -> np.ndarray:
    samples = samples.copy()
    rise = int(RATE * fade_in)
    fall = int(RATE * fade_out)
    samples[:rise] *= np.linspace(0, 1, rise)[:, None]
    samples[-fall:] *= np.linspace(1, 0, fall)[:, None]
    return samples


def write_wav(path: Path, samples: np.ndarray, peak: float = 0.85) -> None:
    samples = samples / max(1e-9, np.abs(samples).max()) * peak
    data = (samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(RATE)
        output.writeframes(data.tobytes())


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    write_wav(ASSETS / "teleport_out.wav", teleport(reverse=False))
    write_wav(ASSETS / "teleport_in.wav", teleport(reverse=True))
    write_wav(ASSETS / "particles_wind.wav", wind(), peak=0.7)
    write_wav(ASSETS / "dissolve_shimmer.wav", shimmer(), peak=0.6)
    print(f"Sounds written to {ASSETS}")


if __name__ == "__main__":
    main()
