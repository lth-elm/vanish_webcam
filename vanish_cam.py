"""VanishCam: make yourself vanish from your webcam with a hand sign."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np

try:
    from .effects import IN, OUT, create_effect
    from .engine import (
        FACE_MODEL, GESTURE_MODEL, HAND_MODEL, IS_MAC, SEGMENTER_MODEL, Engine, camera_devices,
        create_segmenter, download_models, log, model_paths,
    )
    from .gestures import ModeDetector
    from .settings import MODES, Settings, load_settings, save_settings
    from .vanish_sound import ASSETS_DIR, MODE_SOUNDS, VanishSound
except ImportError:
    from effects import IN, OUT, create_effect
    from engine import (
        FACE_MODEL, GESTURE_MODEL, HAND_MODEL, IS_MAC, SEGMENTER_MODEL, Engine, camera_devices,
        create_segmenter, download_models, log, model_paths,
    )
    from gestures import ModeDetector
    from settings import MODES, Settings, load_settings, save_settings
    from vanish_sound import ASSETS_DIR, MODE_SOUNDS, VanishSound


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("FPS must be between 1 and 120.")
    return fps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vanish from your webcam with a hand sign. Settings you change in the "
                    "window are remembered; these options override them for one run.",
    )
    parser.add_argument("--mode", choices=MODES, help="Start with this effect selected")
    parser.add_argument("--camera", type=int, help="Webcam number (see --list-cameras)")
    parser.add_argument("--width", type=int, help="Video width, e.g. 1920")
    parser.add_argument("--height", type=int, help="Video height, e.g. 1080")
    parser.add_argument("--fps", type=positive_fps, default=30.0,
                        help="Maximum camera and virtual camera FPS (default: 30)")
    parser.add_argument("--fullscreen", action="store_true", help="Start full screen")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the preview")
    parser.add_argument("--no-sound", action="store_true", help="Mute effect sounds")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Require the virtual camera (--no-virtual-camera: preview window only)",
    )
    parser.add_argument("--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
                        help="Virtual camera driver to use")
    parser.add_argument("--virtual-device", help="Name of the virtual camera device")
    parser.add_argument("--list-cameras", action="store_true", help="List webcams and exit")
    parser.add_argument("--reset-settings", action="store_true",
                        help="Forget saved settings and start from the defaults")
    parser.add_argument("--gesture-debug", action="store_true",
                        help="Show what the hand tracker sees (preview and terminal)")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> dict[str, object]:
    """Apply one-run command-line options; returns {field: value} that was overridden."""
    overrides: dict[str, object] = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.camera is not None:
        overrides["camera"] = args.camera
    if args.width and args.height:
        overrides["width"], overrides["height"] = args.width, args.height
    if args.no_mirror:
        overrides["mirror"] = False
    if args.no_sound:
        overrides["sound"] = False
    if args.virtual_camera is not None:
        overrides["virtual_camera"] = args.virtual_camera
    for name, value in overrides.items():
        setattr(settings, name, value)
    return overrides


def list_cameras() -> int:
    devices = camera_devices()
    if not devices:
        print("No camera found.")
    for device in devices:
        print(f"  --camera {device.index}: {device.name}")
    return 0


def self_test() -> int:
    """Check models, assets and audio without opening a webcam."""
    download_models()
    required = [SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, FACE_MODEL]
    required += [ASSETS_DIR / name for pair in MODE_SOUNDS.values() for name in pair if name]
    missing = sorted({path.name for path in required if not path.is_file()})
    if missing:
        raise RuntimeError(f"Missing files: {', '.join(missing)}")
    create_segmenter().close()
    for mode in MODES:
        ModeDetector(mode, model_paths()).close()
        effect = create_effect(mode)
        frame = np.zeros((90, 160, 3), np.uint8)
        mask = np.zeros((90, 160), np.float32)
        mask[30:90, 60:100] = 1
        for direction in (OUT, IN):
            effect.begin(direction, frame, frame, mask)
            effect.render(frame, frame, mask, effect.seconds / 2)
    for name in {name for pair in MODE_SOUNDS.values() for name in pair if name}:
        VanishSound(ASSETS_DIR / name).close()
    import PySide6.QtWidgets  # noqa: F401  The window toolkit must be installed too.
    log("Self-test passed: models, effects, assets, audio and the window toolkit all work.")
    return 0


def name_mac_app() -> None:
    """Show "VanishCam" instead of "Python" in the macOS menu bar and Dock."""
    try:
        from Foundation import NSBundle

        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = "VanishCam"
    except Exception:
        pass


def run_window(args: argparse.Namespace, settings: Settings) -> int:
    if IS_MAC:
        name_mac_app()
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")
    # OpenCV's Linux wheels point Qt at their own plugins, which breaks PySide6.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    from PySide6.QtWidgets import QApplication

    try:
        from .window import MainWindow
    except ImportError:
        from window import MainWindow

    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("VanishCam")
    application.setApplicationDisplayName("VanishCam")
    application.setStyle("Fusion")
    window = MainWindow(Engine(args, settings))
    window.video.show_hud = settings.hud
    if args.fullscreen:
        window.showFullScreen()
    else:
        window.show()
    window.raise_()
    window.activateWindow()
    return application.exec()


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.list_cameras:
        return list_cameras()
    settings = Settings() if args.reset_settings else load_settings()
    remembered = replace(settings)
    overrides = apply_overrides(settings, args)
    try:
        return run_window(args, settings)
    finally:
        # Keep changes made in the window. A one-run command-line override is
        # forgotten unless it was changed again in the window.
        final = replace(settings)
        for name, value in overrides.items():
            if getattr(final, name) == value:
                setattr(final, name, getattr(remembered, name))
        save_settings(final)


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[VanishCam] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
