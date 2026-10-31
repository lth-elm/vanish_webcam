"""Persistent user settings, stored as JSON in the per-user config folder."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass, fields
from pathlib import Path


MODES = ("normal", "particles", "naruto", "goku")
RESOLUTIONS = ((1920, 1080), (1280, 720), (960, 540))


@dataclass(frozen=True)
class ModeInfo:
    title: str
    gesture: str
    description: str
    hint: str        # Shown over the preview while waiting for the sign.
    accent: str      # Hex RGB.


MODE_INFO = {
    "normal": ModeInfo("Classic Dissolve", "V sign",
                       "Fade away with a glowing cyan edge.",
                       "Show a V sign to vanish", "#50CDEB"),
    "particles": ModeInfo("Particles", "V sign",
                          "Blow away in colored fragments.",
                          "Show a V sign to turn into particles", "#FFBE5A"),
    "naruto": ModeInfo("Naruto Substitution", "Palms pressed together",
                       "Smoke, and a log takes your place.",
                       "Press your palms together to vanish", "#F09650"),
    "goku": ModeInfo("Goku Teleport", "Two fingers on your forehead, 1 s",
                     "Speed lines, a flash, and you're gone.",
                     "Hold two fingers on your forehead to teleport", "#FFD23C"),
}


def config_dir() -> Path:
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "VanishCam"


SETTINGS_PATH = config_dir() / "settings.json"


@dataclass
class Settings:
    mode: str = "goku"
    camera: int = 0
    width: int = 1920
    height: int = 1080
    mirror: bool = True
    # True: send to the virtual camera when one is installed; False: preview only.
    virtual_camera: bool = True
    sound: bool = True
    volume: float = 0.8
    # Hints drawn over the preview only, never sent to the virtual camera.
    hud: bool = True
    # Repeating the sign while vanished brings you back.
    gesture_reappear: bool = True
    goku_hold_seconds: float = 1.0
    panel_visible: bool = True

    def validate(self) -> "Settings":
        if self.mode not in MODES:
            self.mode = Settings.mode
        self.camera = max(0, int(self.camera))
        if (self.width, self.height) not in RESOLUTIONS:
            self.width, self.height = Settings.width, Settings.height
        self.volume = min(1.0, max(0.0, float(self.volume)))
        self.goku_hold_seconds = min(2.0, max(0.5, float(self.goku_hold_seconds)))
        return self


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    if not isinstance(data, dict):
        return Settings()
    known = {field.name for field in fields(Settings)}
    defaults = Settings()
    values = {}
    for name, value in data.items():
        if name not in known:
            continue  # Keys from older versions.
        default = getattr(defaults, name)
        if type(value) is type(default):
            values[name] = value
        elif isinstance(default, float) and type(value) is int:
            values[name] = float(value)
    return Settings(**values).validate()


def save_settings(settings: Settings, path: Path = SETTINGS_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except OSError as error:
        print(f"[VanishCam] Could not save settings to {path}: {error}")
