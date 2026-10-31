import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import effects
import engine as app
import vanish_cam
from gestures import Detection
from settings import Settings, load_settings, save_settings


def arguments(**overrides):
    values = dict(mode=None, camera=None, width=None, height=None, fps=30.0,
                  fullscreen=False, no_mirror=False, no_sound=True, virtual_camera=None,
                  virtual_backend=None, virtual_device=None, list_cameras=False,
                  reset_settings=False, gesture_debug=False, self_test=False)
    values.update(overrides)
    return argparse.Namespace(**values)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeWorker:
    def __init__(self):
        self.error = None
        self.active = False
        self.sequence = 0
        self.result = None

    def submit(self, mode, rgb, timestamp_ms, captured_at):
        self.sequence += 1
        self.result = (self.sequence, mode, captured_at, Detection(self.active))
        return True

    def latest(self):
        return self.result

    def close(self):
        pass


class FakeSegmenter:
    def __init__(self, mask):
        self.mask = mask

    def segment_for_video(self, image, timestamp_ms):
        return SimpleNamespace(confidence_masks=[SimpleNamespace(numpy_view=lambda: self.mask)])

    def close(self):
        pass


def scene(height=90, width=160):
    background = np.zeros((height, width, 3), np.uint8)
    background[:, :, 1] = np.linspace(30, 140, width, dtype=np.uint8)
    frame = background.copy()
    mask = np.zeros((height, width), np.float32)
    mask[20:, 60:100] = 1.0
    frame[mask > 0] = (40, 60, 220)
    return frame, background, mask


def make_engine(mode="normal", **settings):
    frame, background, mask = scene()
    with patch("builtins.print"):
        instance = app.Engine(arguments(), Settings(mode=mode, sound=False, **settings))
        instance.segmenter = FakeSegmenter(mask)
        instance.worker = FakeWorker()
        instance.set_mode(mode, announce=False)
    return instance, frame, background


class VirtualCameraTests(unittest.TestCase):
    def test_bgr_format_and_actual_frame_dimensions(self):
        camera = Mock(device="OBS Virtual Camera")
        module = SimpleNamespace(Camera=Mock(return_value=camera),
                                 PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module), patch("builtins.print"):
            self.assertIs(app.open_virtual_camera(arguments(), Settings(), 640, 480), camera)
        module.Camera.assert_called_once_with(width=640, height=480, fps=30.0, fmt="BGR",
                                              backend=None, device=None)

    def test_missing_driver_falls_back_unless_explicitly_required(self):
        module = SimpleNamespace(Camera=Mock(side_effect=RuntimeError("driver missing")),
                                 PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module), patch("builtins.print"):
            self.assertIsNone(app.open_virtual_camera(arguments(), Settings(), 640, 480))
            for options in (dict(virtual_camera=True), dict(virtual_backend="obs")):
                with self.assertRaisesRegex(RuntimeError, "driver missing"):
                    app.open_virtual_camera(arguments(**options), Settings(), 640, 480)

    def test_disabled_virtual_camera_is_never_opened(self):
        module = SimpleNamespace(Camera=Mock())
        with patch.dict("sys.modules", pyvirtualcam=module):
            settings = Settings(virtual_camera=False)
            self.assertIsNone(app.open_virtual_camera(arguments(), settings, 640, 480))
        module.Camera.assert_not_called()

    def test_send_preserves_colors_and_resizes(self):
        camera = Mock(width=40, height=24)
        frame = np.zeros((24, 40, 3), np.uint8)
        frame[:] = (10, 70, 200)  # BGR: red and blue must not swap.
        app.send_virtual_frame(camera, frame[:, ::-1])
        sent = camera.send.call_args.args[0]
        self.assertTrue(sent.flags.c_contiguous)
        np.testing.assert_array_equal(sent, frame)
        app.send_virtual_frame(camera, np.full((48, 80, 3), 42, np.uint8))
        self.assertEqual(camera.send.call_args.args[0].shape, (24, 40, 3))


class EngineFlowTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("builtins.print")
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_until(self, instance, clock, frame, seconds, step=0.05):
        output = None
        end = clock.now + seconds
        while clock.now < end:
            clock.now += step
            instance._frame_id += 1
            output = instance.step(frame)
        return output

    def test_waits_for_the_user_before_the_countdown(self):
        clock = Clock()
        with patch("time.perf_counter", clock):
            instance, frame, background = make_engine()
            instance.capture = Mock()
            self.run_until(instance, clock, frame, 10)
            self.assertEqual(instance.state.phase, "setup")
            instance.post("calibrate")
            instance.run_commands()
            self.assertEqual(instance.state.phase, "calibrating")

    def test_full_cycle_with_gesture_vanish_and_gesture_return(self):
        clock = Clock()
        with patch("time.perf_counter", clock), patch.object(app, "DETECTION_INTERVAL", 0):
            instance, frame, background = make_engine()
            instance.capture = Mock()
            instance.post("calibrate")
            instance.run_commands()
            self.run_until(instance, clock, background, app.CALIBRATION_SECONDS + 0.1)
            self.assertEqual(instance.state.phase, "ready")
            np.testing.assert_array_equal(instance.state.background, background)
            self.assertIn("Background captured. Step back in!", instance.snapshot().toasts)

            self.run_until(instance, clock, frame, 0.6)  # No sign yet.
            self.assertEqual(instance.state.phase, "ready")
            instance.worker.active = True
            self.run_until(instance, clock, frame, 0.3)   # Shorter than the V-sign hold.
            self.assertEqual(instance.state.phase, "ready")
            self.run_until(instance, clock, frame, 0.15)
            self.assertEqual(instance.state.phase, "vanishing")

            output = self.run_until(instance, clock, frame, instance.effect.seconds + 0.1)
            self.assertEqual(instance.state.phase, "vanished")
            np.testing.assert_array_equal(output, background)

            # Still holding the V: nothing happens until it is released, and the
            # preview says so.
            self.run_until(instance, clock, frame, 0.5)
            self.assertEqual(instance.state.phase, "vanished")
            self.assertTrue(instance.snapshot().blocked)
            instance.worker.active = False
            self.run_until(instance, clock, frame, 0.5)
            instance.worker.active = True
            self.run_until(instance, clock, frame, 0.45)
            self.assertEqual(instance.state.phase, "appearing")
            output = self.run_until(instance, clock, frame, instance.effect.seconds + 0.1)
            self.assertEqual(instance.state.phase, "ready")
            np.testing.assert_array_equal(output, frame)

    def test_goku_needs_the_configured_hold(self):
        clock = Clock()
        with patch("time.perf_counter", clock), patch.object(app, "DETECTION_INTERVAL", 0):
            instance, frame, background = make_engine("goku")
            instance.state = app.State(phase="ready", started=clock.now, background=background)
            self.run_until(instance, clock, frame, 0.5)
            instance.worker.active = True
            self.run_until(instance, clock, frame, 0.6)
            self.assertEqual(instance.state.phase, "ready")
            self.run_until(instance, clock, frame, 0.3)
            self.assertEqual(instance.state.phase, "vanishing")
            instance.post("goku_hold", 1.5)
            instance.run_commands()
            self.assertEqual(instance.trigger.hold_seconds, 1.5)

    def test_particles_needs_a_short_hold(self):
        clock = Clock()
        with patch("time.perf_counter", clock), patch.object(app, "DETECTION_INTERVAL", 0):
            instance, frame, background = make_engine("particles")
            instance.state = app.State(phase="ready", started=clock.now, background=background)
            self.run_until(instance, clock, frame, 0.5)
            instance.worker.active = True
            self.run_until(instance, clock, frame, 0.3)
            self.assertEqual(instance.state.phase, "ready")
            self.run_until(instance, clock, frame, 0.2)
            self.assertEqual(instance.state.phase, "vanishing")

    def test_live_view_only_while_vanished_and_never_sent_out(self):
        frame, background, mask = scene()
        instance, _, _ = make_engine()
        capture = Mock()
        capture.read.return_value = (True, frame)
        instance.capture = capture
        camera = Mock(width=160, height=90, device="OBS Virtual Camera")
        instance.state = app.State(phase="vanished", background=background[:, ::-1].copy())
        with patch.object(app, "open_virtual_camera", return_value=camera):
            output, status = instance.next_frame()
        self.assertIsNotNone(status.live)
        self.assertEqual(status.live.shape[1], app.LIVE_VIEW_WIDTH)
        # The call only ever gets the empty room.
        np.testing.assert_array_equal(camera.send.call_args.args[0], background)
        instance.state.phase = "ready"
        with patch.object(app, "open_virtual_camera", return_value=camera):
            _, status = instance.next_frame()
        self.assertIsNone(status.live)

    def test_naruto_needs_palms_held_half_a_second(self):
        instance, _, _ = make_engine("naruto")
        self.assertEqual(instance.trigger.hold_seconds, 0.5)

    def test_v_key_toggles_without_a_sign(self):
        instance, frame, background = make_engine()
        instance.state = app.State(phase="ready", background=background)
        instance.post("toggle_vanish")
        instance.run_commands()
        instance.step(frame)
        self.assertEqual(instance.state.phase, "vanishing")

    def test_switching_mode_brings_you_back(self):
        instance, frame, background = make_engine()
        instance.state = app.State(phase="vanished", background=background)
        instance.post("mode", "goku")
        instance.run_commands()
        self.assertEqual(instance.settings.mode, "goku")
        self.assertEqual(instance.state.phase, "ready")
        self.assertEqual(instance.trigger.hold_seconds, 0.8)
        self.assertIsInstance(instance.effect, effects.GokuEffect)

    def test_mode_switch_from_the_window_loads_the_new_effect(self):
        # The window shares `settings` with the engine and updates the mode
        # before the engine reads its command; the effect must still change.
        instance, frame, background = make_engine("goku")
        for mode, kind in (("normal", effects.DissolveEffect),
                           ("particles", effects.ParticlesEffect),
                           ("naruto", effects.NarutoEffect)):
            instance.settings.mode = mode
            instance.post("mode", mode)
            instance.run_commands()
            self.assertIsInstance(instance.effect, kind)
            self.assertEqual(instance.snapshot().mode, mode)

    def test_mirror_toggle_flips_the_saved_background(self):
        instance, frame, background = make_engine()
        instance.state = app.State(phase="ready", background=background)
        instance.post("mirror", False)
        instance.run_commands()
        np.testing.assert_array_equal(instance.state.background, background[:, ::-1])

    def test_next_frame_sends_an_unmirrored_frame_to_the_virtual_camera(self):
        frame, background, mask = scene()
        instance, _, _ = make_engine()
        capture = Mock()
        capture.read.return_value = (True, frame)
        instance.capture = capture
        camera = Mock(width=160, height=90, device="OBS Virtual Camera")
        with patch.object(app, "open_virtual_camera", return_value=camera):
            output, status = instance.next_frame()
        np.testing.assert_array_equal(output, frame[:, ::-1])  # Mirrored preview.
        np.testing.assert_array_equal(camera.send.call_args.args[0], frame)
        self.assertEqual(status.virtual_device, "OBS Virtual Camera")
        self.assertEqual(status.phase, "setup")

    def test_camera_failure_is_reported_and_retry_reopens(self):
        instance, frame, _ = make_engine()
        with patch.object(app, "open_camera", side_effect=RuntimeError("no access")):
            instance.open_capture()
        self.assertEqual(instance.next_frame()[1].error, "no access")
        capture = Mock()
        capture.read.return_value = (True, frame)
        with patch.object(app, "open_camera", return_value=capture), \
                patch.object(app, "open_virtual_camera", return_value=None):
            instance.post("retry")
            output, status = instance.next_frame()
        self.assertEqual(status.error, "")
        self.assertIsNotNone(output)

    def test_large_frames_are_analysed_at_a_smaller_size(self):
        instance, _, _ = make_engine()
        big = np.zeros((1080, 1920, 3), np.uint8)
        self.assertEqual(instance.analysis_rgb(big).shape, (720, 1280, 3))


class WindowTests(unittest.TestCase):
    def test_controls_post_commands_to_the_engine(self):
        from PySide6.QtWidgets import QApplication
        import window

        application = QApplication.instance() or QApplication([])
        engine = Mock()
        engine.settings = Settings(mode="goku")
        engine.snapshot.side_effect = RuntimeError("not used")
        with patch.object(window.EngineThread, "start"), \
                patch.object(window, "camera_devices",
                             return_value=[app.CameraDevice(0, "FaceTime HD Camera"),
                                           app.CameraDevice(1, "iPhone Camera")]):
            main = window.MainWindow(engine)
            self.assertEqual(main.camera_combo.itemText(1), "iPhone Camera")
            main.select_mode("particles")
            main.start_calibration()
            main.next_camera()
            main.sound_switch.toggle()
        posted = [call.args for call in engine.post.call_args_list]
        self.assertIn(("mode", "particles"), posted)
        self.assertIn(("calibrate",), posted)
        self.assertIn(("camera", 1), posted)
        self.assertIn(("sound", False), posted)
        self.assertTrue(main.mode_cards["particles"].isChecked())
        main.thread.requestInterruption()
        application.processEvents()


class SettingsTests(unittest.TestCase):
    def test_round_trip_and_bad_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            self.assertEqual(load_settings(path), Settings())
            self.assertEqual((Settings().width, Settings().height), (1920, 1080))
            save_settings(Settings(mode="particles", sound=False, width=1280, height=720), path)
            loaded = load_settings(path)
            self.assertEqual((loaded.mode, loaded.sound, loaded.width), ("particles", False, 1280))
            path.write_text(json.dumps({"mode": "nope", "volume": 7, "width": 5, "height": 5,
                                        "unknown": 1, "sound": "yes"}))
            loaded = load_settings(path)
            self.assertEqual(loaded.mode, Settings.mode)
            self.assertEqual(loaded.volume, 1.0)
            self.assertEqual((loaded.width, loaded.height), (Settings.width, Settings.height))
            self.assertTrue(loaded.sound)
            path.write_text("{broken")
            self.assertEqual(load_settings(path), Settings())

    def test_command_line_overrides_are_not_saved(self):
        settings = Settings(camera=0, mirror=True)
        overrides = vanish_cam.apply_overrides(settings, arguments(camera=2, no_mirror=True))
        self.assertEqual(overrides, {"camera": 2, "mirror": False, "sound": False})
        self.assertEqual((settings.camera, settings.mirror), (2, False))


if __name__ == "__main__":
    unittest.main()
