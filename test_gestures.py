import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import mediapipe as mp
import numpy as np

from gestures import (
    DetectionWorker, Detection, ForeheadSaluteDetector, HoldTrigger, MultiViewGestureRecognizer,
    MultiViewHandLandmarker, forehead_zone, joined_hands_from_landmarks, recognize_gesture,
    salute_at_forehead, two_finger_salute_score, victory_landmark_score,
)


def hand_points(extended=(True, True, False, False), bases=(-0.45, 0.0, 0.4, 0.75),
                spreads=(-0.35, 0.25, 0.0, 0.0)):
    points = np.zeros((21, 3))
    for base, x, opened, spread in zip((5, 9, 13, 17), bases, extended, spreads):
        if opened:
            points[base:base + 4] = [
                (x, 1.0, 0.0), (x + spread * 0.45, 1.7, 0.0),
                (x + spread * 0.75, 2.15, 0.0), (x + spread, 2.5, 0.0),
            ]
        else:
            points[base:base + 4] = [
                (x, 1.0, 0.0), (x, 1.5, 0.2),
                (x, 1.3, 0.5), (x, 0.95, 0.3),
            ]
    return points


def salute_points():
    """Index and middle straight and side by side, ring and pinky folded."""
    return hand_points(bases=(-0.12, 0.12, 0.4, 0.75), spreads=(0.0, 0.0, 0.0, 0.0))


def landmarks(points):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]


class VictoryGestureTests(unittest.TestCase):
    def test_victory_with_back_palm_rotation_mirror_and_scale(self):
        points = hand_points()
        for angle in (0.0, 0.7, 1.57, 2.6, np.pi):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            tilt = np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
            for mirror in (1, -1):
                for scale in (0.02, 1.0, 100.0):
                    with self.subTest(angle=angle, mirror=mirror, scale=scale):
                        transformed = points @ rotation @ tilt
                        transformed[:, 0] *= mirror
                        transformed = transformed * scale + (0.5, 0.3, 0.1)
                        self.assertGreater(victory_landmark_score(transformed), 0.62)

    def test_other_gestures_do_not_trigger(self):
        for fingers in (
            (False, False, False, False),  # Fist.
            (True, True, True, True),  # Open palm.
            (True, False, False, False),  # Index only.
            (True, False, False, True),  # Horns.
            (True, True, True, False),  # Three fingers.
        ):
            with self.subTest(fingers=fingers):
                self.assertEqual(victory_landmark_score(hand_points(fingers)), 0.0)

    def test_fingers_held_together_are_not_a_v(self):
        self.assertEqual(victory_landmark_score(salute_points()), 0.0)

    def test_invalid_or_degenerate_landmarks(self):
        for points in (np.zeros((21, 3)), np.zeros((20, 3)), np.full((21, 3), np.nan)):
            self.assertEqual(victory_landmark_score(points), 0.0)

    def test_landmarks_trigger_without_classifier_category(self):
        result = SimpleNamespace(
            hand_world_landmarks=[landmarks(hand_points())], hand_landmarks=[], gestures=[])
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        name, score = recognize_gesture(recognizer, None, 1000)
        self.assertEqual(name, "Victory")
        self.assertGreater(score, 0.62)

    def test_normalized_landmark_fallback_handles_video_aspect_ratio(self):
        image = SimpleNamespace(width=960, height=540)
        points = hand_points() * 40 + (480, 200, 0)
        result = SimpleNamespace(hand_world_landmarks=[],
                                 hand_landmarks=[landmarks(points / (960, 540, 960))], gestures=[])
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, image, 1000)[0], "Victory")

    def test_existing_classifier_still_works(self):
        result = SimpleNamespace(
            hand_world_landmarks=[], hand_landmarks=[],
            gestures=[[SimpleNamespace(category_name="Victory", score=0.9)]])
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, None, 1000), ("Victory", 0.9))

    def test_occluded_folded_fingers_can_appear_short_and_straight(self):
        points = hand_points()
        for base in (13, 17):
            points[base:base + 4] = [(0.4, 1.0, 0.0), (0.4, 1.1, 0.0),
                                     (0.4, 1.2, 0.0), (0.4, 1.3, 0.0)]
        self.assertGreater(victory_landmark_score(points), 0.62)

    def test_hidden_fingers_with_coincident_landmarks(self):
        points = hand_points()
        points[13:17] = points[13]
        points[17:21] = points[17]
        self.assertGreater(victory_landmark_score(points), 0.62)

    def test_visible_geometry_is_used_even_with_inaccurate_world_landmarks(self):
        image = SimpleNamespace(width=960, height=540)
        points = hand_points() * 40 + (480, 200, 0)
        world = hand_points()
        world[8] = world[6]
        self.assertEqual(victory_landmark_score(world), 0.0)
        result = SimpleNamespace(hand_world_landmarks=[landmarks(world)],
                                 hand_landmarks=[landmarks(points / (960, 540, 960))], gestures=[])
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, image, 1000)[0], "Victory")

    def test_diagnostics_distinguish_missing_hand_and_rejected_gesture(self):
        for hands, expected in (([], "No_Hand"),
                                ([landmarks(hand_points((False,) * 4))], "Hand_Detected")):
            result = SimpleNamespace(hand_world_landmarks=hands, hand_landmarks=[], gestures=[])
            recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
            self.assertEqual(recognize_gesture(recognizer, None, 1000)[0], expected)


class MultiViewTests(unittest.TestCase):
    def setUp(self):
        self.empty = SimpleNamespace(hand_world_landmarks=[], hand_landmarks=[], gestures=[])
        self.victory = SimpleNamespace(
            hand_world_landmarks=[], hand_landmarks=[],
            gestures=[[SimpleNamespace(category_name="Victory", score=0.9)]])
        self.image = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=np.zeros((32, 48, 3), dtype=np.uint8))

    def detector(self, recognize):
        return MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.empty),
            SimpleNamespace(recognize=recognize))

    def test_rotated_detection_and_preferred_view_are_used(self):
        calls = []

        def recognize(image):
            calls.append((image.width, image.height))
            return self.victory if image.width == 32 else self.empty

        detector = self.detector(recognize)
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "Victory")
        self.assertEqual(calls, [(48, 32), (32, 48)])
        self.assertEqual(detector.preferred_rotation, 1)
        calls.clear()
        self.assertEqual(recognize_gesture(detector, self.image, 1140)[0], "Victory")
        self.assertEqual(calls, [(32, 48)])

    def test_failed_search_spreads_three_views_per_frame_and_is_throttled(self):
        calls = []
        detector = self.detector(lambda image: calls.append(image) or self.empty)
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "No_Hand")
        self.assertEqual(len(calls), 3)
        recognize_gesture(detector, self.image, 1100)  # Within the retry interval.
        self.assertEqual(len(calls), 3)
        recognize_gesture(detector, self.image, 1140)
        self.assertEqual(len(calls), 6)
        detector.alternate.recognize = lambda image: self.victory
        self.assertEqual(recognize_gesture(detector, self.image, 1420)[0], "Victory")
        detector.alternate.recognize = lambda image: self.empty
        self.assertEqual(recognize_gesture(detector, self.image, 1560)[0], "No_Hand")

    def test_primary_victory_avoids_extra_inference(self):
        calls = []
        detector = MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.victory),
            SimpleNamespace(recognize=lambda image: calls.append(image) or self.empty))
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "Victory")
        self.assertEqual(calls, [])

    def test_lower_hand_crop_recovers_a_small_sideways_v(self):
        def recognize(image):
            return self.victory if (image.width, image.height) == (37, 24) else self.empty

        detector = self.detector(recognize)
        # The crop is reached within a few retries spread over successive frames.
        found = [recognize_gesture(detector, self.image, 1000 + step * 150)[0]
                 for step in range(4)]
        self.assertIn("Victory", found)
        self.assertEqual(detector.preferred_crop, "lower_left")


class JoinedHandsTests(unittest.TestCase):
    @staticmethod
    def hand(center_x, upward=True):
        points = [SimpleNamespace(x=center_x, y=0.53) for _ in range(21)]
        points[0] = SimpleNamespace(x=center_x, y=0.70)
        for index, offset in ((5, -0.045), (9, -0.015), (13, 0.015), (17, 0.045)):
            points[index] = SimpleNamespace(x=center_x + offset, y=0.54 if upward else 0.84)
        for index, offset in ((8, -0.045), (12, -0.015), (16, 0.015), (20, 0.045)):
            points[index] = SimpleNamespace(x=center_x + offset, y=0.34 if upward else 1.04)
        return points

    def test_two_nearby_hands_recognized_but_one_or_separated_hands_are_not(self):
        left, right = self.hand(0.45), self.hand(0.55)
        self.assertTrue(joined_hands_from_landmarks([left, right], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left, self.hand(0.83)], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left, self.hand(0.55, upward=False)], 640, 480))

    def test_rotated_joined_hands_still_recognized(self):
        hands = [self.hand(0.45), self.hand(0.55)]
        rotated = [[SimpleNamespace(x=point.y, y=1 - point.x) for point in hand] for hand in hands]
        self.assertTrue(joined_hands_from_landmarks(rotated, 480, 640))

    def test_recognizer_uses_two_hands_in_the_current_frame(self):
        primary, alternate = Mock(), Mock()
        primary.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[self.hand(0.45), self.hand(0.55)])
        recognizer = MultiViewHandLandmarker(primary, alternate)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), np.uint8))
        self.assertTrue(recognizer.recognize_joined_hands(image, 100))
        self.assertEqual(recognizer.last_hand_count, 2)
        alternate.detect.assert_not_called()

    def test_hand_model_recovers_joined_hands_in_a_cropped_view(self):
        primary, alternate = Mock(), Mock()
        primary.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[self.hand(0.45)])
        alternate.detect.side_effect = [
            SimpleNamespace(hand_landmarks=[]),
            SimpleNamespace(hand_landmarks=[self.hand(0.45), self.hand(0.55)]),
        ]
        recognizer = MultiViewHandLandmarker(primary, alternate)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), np.uint8))
        self.assertTrue(recognizer.recognize_joined_hands(image, 100))
        self.assertEqual(recognizer.preferred_view, ("full", 3))


def face(width=640, height=480, center=(320, 260), size=160):
    """Face landmarks with just the points the forehead zone uses."""
    cx, cy = center
    points = [SimpleNamespace(x=cx / width, y=cy / height, z=0.0) for _ in range(478)]
    place = {234: (cx - size / 2, cy), 454: (cx + size / 2, cy), 10: (cx, cy - size * 0.75),
             9: (cx, cy - size * 0.35), 33: (cx - size * 0.2, cy - size * 0.2),
             263: (cx + size * 0.2, cy - size * 0.2)}
    for index, (x, y) in place.items():
        points[index] = SimpleNamespace(x=x / width, y=y / height, z=0.0)
    return points


def hand_at(points, tip_xy, scale=30.0):
    """Place hand points (finger length ~1.5 units) so the fingertips land on tip_xy."""
    placed = points * scale
    tips = placed[[8, 12], :2].mean(axis=0)
    placed[:, :2] += np.array(tip_xy) - tips
    return placed


class GokuPoseTests(unittest.TestCase):
    def test_salute_shape_needs_two_fingers_together(self):
        self.assertGreater(two_finger_salute_score(salute_points()), 0.6)
        self.assertEqual(two_finger_salute_score(hand_points()), 0.0)  # A V is spread apart.
        for fingers in ((False,) * 4, (True,) * 4, (True, False, False, False)):
            with self.subTest(fingers=fingers):
                points = hand_points(fingers, bases=(-0.12, 0.12, 0.4, 0.75),
                                     spreads=(0, 0, 0, 0))
                self.assertEqual(two_finger_salute_score(points), 0.0)

    def test_salute_is_rotation_invariant(self):
        for angle in (0.0, 0.5, -0.5, 1.2):
            c, s = np.cos(angle), np.sin(angle)
            rotated = salute_points() @ np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            self.assertGreater(two_finger_salute_score(rotated), 0.6)

    def test_fingers_must_touch_the_forehead(self):
        zone = forehead_zone(face(), 640, 480)
        self.assertIsNotNone(zone)
        on_forehead = hand_at(salute_points(), (320, 170))
        on_temple = hand_at(salute_points(), (395, 215))
        on_chin = hand_at(salute_points(), (320, 330))
        far_away = hand_at(salute_points(), (560, 120))
        self.assertEqual(salute_at_forehead(None, on_forehead, zone), (True, True))
        self.assertEqual(salute_at_forehead(None, on_temple, zone), (True, True))
        self.assertEqual(salute_at_forehead(None, on_chin, zone), (True, False))
        self.assertEqual(salute_at_forehead(None, far_away, zone), (True, False))
        # No face found yet: the shape is seen but cannot count as a teleport.
        self.assertEqual(salute_at_forehead(None, on_forehead, None), (True, False))

    def test_detector_remembers_the_face_when_the_hand_covers_it(self):
        image = SimpleNamespace(width=640, height=480)
        hand = landmarks(hand_at(salute_points(), (320, 170)) / (640, 480, 640))
        face_task = Mock()
        face_task.detect_for_video.side_effect = [
            SimpleNamespace(face_landmarks=[face()]),
            SimpleNamespace(face_landmarks=[]),  # The fingers now hide the face.
        ]
        hand_task = Mock()
        hand_task.detect_for_video.side_effect = [
            SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[]),
            SimpleNamespace(hand_landmarks=[hand], hand_world_landmarks=[]),
        ]
        detector = ForeheadSaluteDetector(hand_task, face_task)
        self.assertFalse(detector.detect(image, 10).active)
        self.assertTrue(detector.detect(image, 60).active)


class HoldTriggerTests(unittest.TestCase):
    def test_hold_must_last_and_brief_dropouts_are_forgiven(self):
        trigger = HoldTrigger(hold_seconds=1.0, grace=0.25)
        self.assertFalse(trigger.update(True, 0.0))
        self.assertFalse(trigger.update(False, 0.2))  # One missed detection.
        self.assertFalse(trigger.update(True, 0.5))
        self.assertAlmostEqual(trigger.progress(0.5), 0.5)
        self.assertTrue(trigger.update(True, 1.0))

    def test_breaking_the_pose_restarts_the_countdown(self):
        trigger = HoldTrigger(hold_seconds=1.0, grace=0.25)
        trigger.update(True, 0.0)
        trigger.update(False, 0.7)
        self.assertFalse(trigger.update(True, 1.0))
        self.assertFalse(trigger.update(True, 1.9))
        self.assertTrue(trigger.update(True, 2.0))

    def test_must_release_before_firing_again(self):
        trigger = HoldTrigger(hold_seconds=0.0, release=0.4)
        self.assertTrue(trigger.update(True, 0.0))
        self.assertFalse(trigger.update(True, 0.5))  # Still holding.
        self.assertFalse(trigger.update(False, 0.6))
        self.assertFalse(trigger.update(True, 0.7))  # Released too briefly.
        self.assertFalse(trigger.update(False, 1.2))
        self.assertTrue(trigger.update(True, 1.3))

    def test_reports_when_a_held_sign_blocks_firing_again(self):
        trigger = HoldTrigger(hold_seconds=0.0, release=0.3)
        self.assertFalse(trigger.blocked(0.0))
        self.assertTrue(trigger.update(True, 0.0))
        trigger.update(True, 0.5)
        self.assertTrue(trigger.blocked(0.55))      # Still up: lower it first.
        trigger.update(False, 0.9)
        self.assertFalse(trigger.blocked(0.9))      # Lowered long enough: armed again.


class DetectionWorkerTests(unittest.TestCase):
    def test_runs_detector_off_the_main_thread_and_closes_it(self):
        detector = Mock()
        detector.detect.return_value = Detection(True, 1, "ok")
        worker = DetectionWorker({}, factory=lambda mode, models: detector)
        rgb = np.zeros((8, 8, 3), np.uint8)
        self.assertTrue(worker.submit("goku", rgb, 5, 1.0))
        for _ in range(200):
            if worker.latest() is not None:
                break
            import time
            time.sleep(0.005)
        sequence, mode, captured_at, detection = worker.latest()
        self.assertEqual((mode, captured_at, detection.active), ("goku", 1.0, True))
        worker.close()
        detector.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
