"""Preload effect sounds and play them without blocking video frames."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SOUND_PATH = ASSETS_DIR / "vanish.MP3"

# Sound played when each mode makes you vanish, and when it brings you back.
MODE_SOUNDS = {
    "normal": ("dissolve_shimmer.wav", None),
    "particles": ("particles_wind.wav", None),
    "naruto": ("vanish.MP3", "vanish.MP3"),
    "goku": ("teleport_out.wav", "teleport_in.wav"),
}


class VanishSound:
    """One decoded sound. Uses pygame when available, else afplay or ffplay."""

    def __init__(self, path: Path = SOUND_PATH, volume: float = 1.0):
        if not path.is_file():
            raise RuntimeError(f"Sound file not found: {path}")
        self.path = path
        self.volume = volume
        self.process = None
        self.player = None
        self.owns_mixer = False

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        try:
            from pygame import mixer
        except ImportError:
            self.player = _command_player()
            if self.player is None:
                raise RuntimeError("To play sounds, install pygame (pip install pygame) or ffplay.")
            return

        self.mixer = mixer
        self.owns_mixer = mixer.get_init() is None
        try:
            if self.owns_mixer:
                mixer.init(frequency=44100, size=-16, channels=2, buffer=256)
            self.sound = mixer.Sound(str(path))  # Decode once, before opening the webcam.
            self.sound.set_volume(volume)
        except Exception as error:
            if self.owns_mixer:
                mixer.quit()
            raise RuntimeError(f"Could not open sound {path.name}: {error}") from error

    def set_volume(self, volume: float) -> None:
        self.volume = volume
        if self.player is None:
            self.sound.set_volume(volume)

    def play(self) -> None:
        if self.player is not None:
            self._stop_process()
            command = [*self.player, str(self.path)]
            if self.player[0].endswith("afplay"):
                command[1:1] = ["-v", f"{self.volume:.2f}"]
            try:
                self.process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as error:
                print(f"[VanishCam] Could not play {self.path.name}: {error}")
            return
        self.sound.stop()
        self.sound.play()

    def _stop_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def close(self) -> None:
        if self.player is not None:
            self._stop_process()
            return
        self.sound.stop()
        if self.owns_mixer:
            self.mixer.quit()
            self.owns_mixer = False


def _command_player() -> list[str] | None:
    if platform.system() == "Darwin" and shutil.which("afplay"):
        return [shutil.which("afplay")]
    ffplay = shutil.which("ffplay")
    if ffplay:
        return [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", "-i"]
    return None


class ModeSounds:
    """Lazily loads the vanish/return sounds for whichever mode is active."""

    def __init__(self, enabled: bool = True, volume: float = 0.8):
        self.enabled = enabled
        self.volume = volume
        self.cache: dict[str, VanishSound] = {}

    def _get(self, name: str | None) -> VanishSound | None:
        if not self.enabled or name is None:
            return None
        if name not in self.cache:
            try:
                self.cache[name] = VanishSound(ASSETS_DIR / name, self.volume)
            except RuntimeError as error:
                print(f"[VanishCam] {error}")
                self.cache[name] = None
        return self.cache[name]

    def set_volume(self, volume: float) -> None:
        self.volume = volume
        for sound in self.cache.values():
            if sound is not None:
                sound.set_volume(volume)

    def preload(self, mode: str) -> None:
        for name in MODE_SOUNDS.get(mode, (None, None)):
            self._get(name)

    def play_vanish(self, mode: str) -> None:
        sound = self._get(MODE_SOUNDS.get(mode, (None, None))[0])
        if sound is not None:
            sound.play()

    def play_return(self, mode: str) -> None:
        sound = self._get(MODE_SOUNDS.get(mode, (None, None))[1])
        if sound is not None:
            sound.play()

    def close(self) -> None:
        # Close the mixer owner last so the other sounds can still stop cleanly.
        sounds = [sound for sound in self.cache.values() if sound is not None]
        for sound in sorted(sounds, key=lambda item: item.owns_mixer):
            sound.close()
        self.cache.clear()
