"""Synthesize the wind and shimmer sounds in assets/ (no recorded or sampled audio).

The Goku teleport sound is not made here: see assets/README.md.

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


def wind() -> np.ndarray:
    """Particles: a whoosh that climbs with the front sweeping up your body, and a
    crackle of tiny grains for the blocks peeling off. Timed to the effect:
    the front eases in over 0.64 s and the last blocks are gone by 0.95 s."""
    rng = np.random.default_rng(12)
    seconds = 1.1
    t = _time(seconds)
    front = np.clip(t / 0.64, 0, 1) ** 1.8          # Same easing as the visual front.
    centre = 500 * (6000 / 500) ** front
    whoosh = _bandpassed_noise(rng, seconds, centre * 0.55, centre * 1.6)
    swell = np.clip(t / 0.05, 0, 1) * (0.55 + 0.45 * front) * np.exp(-np.clip(t - 0.66, 0, None) / 0.12)

    # Grains: short clicks of high noise, densest where the front moves fastest.
    crackle = np.zeros_like(t)
    length = int(RATE * 0.01)
    rate = 60 + 260 * front                          # Grains per second.
    at = rng.exponential(1 / rate[0])
    while at < 0.85:
        start = int(at * RATE)
        grain = np.diff(rng.standard_normal(length + 1)) * np.hanning(length)  # Bright click.
        crackle[start:start + length] += grain * rng.uniform(0.3, 1.0)
        at += rng.exponential(1 / rate[start])
    crackle *= np.exp(-np.clip(t - 0.7, 0, None) / 0.1)

    mono = 0.75 * whoosh * swell + 0.25 * crackle
    pan = 0.5 + 0.25 * np.sin(2 * np.pi * 1.3 * t)  # A little movement across the stereo field.
    stereo = np.stack([mono * np.cos(pan * np.pi / 2), mono * np.sin(pan * np.pi / 2)], axis=1)
    return _fade(stereo, 0.01, 0.15)


def shimmer() -> np.ndarray:
    """Classic dissolve: a soft electric crackle that follows the glowing edge down
    (0.05 to 1.5 s, easing in and out), over glassy chimes for the motes, then a
    bright tail as the last motes drift away by 2.1 s."""
    rng = np.random.default_rng(33)
    seconds = 2.3
    length = int(RATE * seconds)
    t = _time(seconds)
    mix = np.zeros((length, 2))
    # Glassy chimes, densest while the edge sweeps, falling in pitch as it moves down.
    notes = [1318.5, 1568.0, 1975.5, 2349.3, 2637.0, 3136.0]
    for index in range(40):
        at_s = 0.05 + 1.5 * (index / 40) + rng.uniform(0, 0.05)
        hz = notes[int(rng.integers(0, len(notes)))] * (2 if rng.random() < 0.3 else 1)
        hz *= 1.0 - 0.18 * at_s / 1.6
        ping = _bell(0.6, [(hz, 1.0), (hz * 2.01, 0.2)], decay=0.14)
        at = int(RATE * at_s)
        end = min(length, at + ping.size)
        pan = rng.random()
        gain = 0.08 * (0.6 + 0.4 * rng.random())
        mix[at:end, 0] += gain * ping[:end - at] * (1 - pan)
        mix[at:end, 1] += gain * ping[:end - at] * pan
    # Electric crackle: bursts of bright noise, gated at random, along the sweep.
    sweep = np.clip((t - 0.05) / 1.45, 0, 1)
    swell = np.sin(np.pi * sweep) ** 0.7 * (t < 1.55)
    gate = (rng.random(int(seconds * 300)) < 0.45).astype(float)
    gate = np.repeat(gate, int(np.ceil(length / gate.size)))[:length]
    gate = np.convolve(gate, np.hanning(64) / 32, mode="same")
    crackle = _bandpassed_noise(rng, seconds, 2500, 9000) * gate * swell
    hum = np.sin(2 * np.pi * 110 * t + 0.6 * np.sin(2 * np.pi * 220 * t)) * swell
    air = _bandpassed_noise(rng, seconds, 5000, 11000) * np.exp(-np.clip(t - 1.5, 0, None) / 0.3)
    air *= np.clip(t / 0.8, 0, 1)
    mono = 0.14 * crackle + 0.05 * hum + 0.05 * air
    mix += mono[:, None]
    return _fade(mix, 0.01, 0.3)


def gather(vanish: np.ndarray) -> np.ndarray:
    """Classic return: the vanish sound backwards, squeezed to the 1.4 s return,
    so the motes' chimes swell up into you as you re-form."""
    start = vanish[:int(RATE * 1.75)][::-1]
    length = int(RATE * 1.4)
    position = np.linspace(0, len(start) - 1, length)
    squeezed = np.stack([np.interp(position, np.arange(len(start)), start[:, channel])
                         for channel in range(2)], axis=1)
    return _fade(squeezed, 0.15, 0.04)


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
    write_wav(ASSETS / "particles_wind.wav", wind(), peak=0.7)
    dissolve = shimmer()
    write_wav(ASSETS / "dissolve_shimmer.wav", dissolve, peak=0.6)
    write_wav(ASSETS / "dissolve_gather.wav", gather(dissolve), peak=0.6)
    print(f"Sounds written to {ASSETS}")


if __name__ == "__main__":
    main()
