"""PyInstaller one-file build for the Windows webcam app."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs


root = Path(SPECPATH).resolve()
icon_file = root / "assets" / "vanish_icon.ico"
resources = [
    (root / "hand_landmarker.task", "."),
    (root / "models" / "gesture_recognizer.task", "models"),
    (root / "models" / "selfie_segmenter_landscape.tflite", "models"),
    (root / "face_landmarker.task", "."),
    (root / "assets" / "vanish.MP3", "assets"),
    (root / "assets" / "teleport.wav", "assets"),
    (root / "assets" / "particles_wind.wav", "assets"),
    (root / "assets" / "dissolve_shimmer.wav", "assets"),
    (root / "assets" / "dissolve_gather.wav", "assets"),
]
missing = [str(path) for path, _ in resources if not path.is_file()]
if not icon_file.is_file():
    missing.append(str(icon_file))
if missing:
    raise SystemExit("Files needed for the build are missing:\n" + "\n".join(missing))

a = Analysis(
    [str(root / "vanish_cam.py")],
    pathex=[str(root)],
    binaries=collect_dynamic_libs("mediapipe"),
    datas=[(str(path), destination) for path, destination in resources],
    hiddenimports=["pygame.mixer", "mediapipe.tasks.c", "PySide6.QtMultimedia"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VanishCam",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(icon_file),
)
