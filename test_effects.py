import unittest

import numpy as np

from effects import IN, OUT, EFFECTS, create_effect
import naruto_effect as naruto


def scene(height=180, width=320):
    background = np.zeros((height, width, 3), np.uint8)
    background[:, :, 1] = np.linspace(40, 120, width, dtype=np.uint8)
    frame = background.copy()
    mask = np.zeros((height, width), np.float32)
    mask[40:, 130:190] = 1.0
    frame[mask > 0] = (40, 60, 220)
    return frame, background, mask


class EveryEffectTests(unittest.TestCase):
    def test_vanish_ends_on_the_background_and_return_ends_on_you(self):
        frame, background, mask = scene()
        person = mask > 0
        for mode in EFFECTS:
            with self.subTest(mode=mode):
                effect = create_effect(mode)
                effect.begin(OUT, frame, background, mask)
                start = effect.render(frame, background, mask, 0.0)
                np.testing.assert_allclose(start[person].mean(axis=0), (40, 60, 220), atol=40)
                middle = effect.render(frame, background, mask, effect.seconds * 0.5)
                self.assertTrue(np.any(middle != frame))
                end = effect.render(frame, background, mask, effect.seconds)
                gone = effect.vanished(background)
                self.assertLess(np.abs(end.astype(int) - background).mean(), 3)
                np.testing.assert_array_equal(gone, background)
                self.assertGreater(np.abs(end[person].astype(int) - frame[person]).mean(), 60)

                effect.begin(IN, frame, background, mask)
                back = effect.render(frame, background, mask, effect.seconds)
                self.assertLess(np.abs(back[person].astype(int) - frame[person]).mean(), 25)

    def test_effects_keep_the_frame_shape_and_type(self):
        frame, background, mask = scene(97, 211)  # Odd sizes catch off-by-one crops.
        for mode in EFFECTS:
            effect = create_effect(mode)
            for direction in (OUT, IN):
                effect.begin(direction, frame, background, mask)
                for age in np.linspace(0, effect.seconds, 9):
                    output = effect.render(frame, background, mask, float(age))
                    self.assertEqual(output.shape, frame.shape)
                    self.assertEqual(output.dtype, np.uint8)

    def test_effects_survive_an_empty_mask(self):
        frame, background, _ = scene()
        empty = np.zeros(frame.shape[:2], np.float32)
        for mode in EFFECTS:
            effect = create_effect(mode)
            effect.begin(OUT, frame, background, empty)
            effect.render(frame, background, empty, effect.seconds * 0.4)

    def test_dissolve_burns_from_the_head_down_with_a_cyan_edge(self):
        frame, background, mask = scene()
        effect = create_effect("normal")
        effect.begin(OUT, frame, background, mask)
        middle = effect.render(frame, background, mask, 0.75)
        legs, head = np.s_[150:170, 140:180], np.s_[45:60, 140:180]
        self.assertGreater(np.abs(middle[head].astype(int) - frame[head]).mean(), 60)
        self.assertLess(np.abs(middle[legs].astype(int) - frame[legs]).mean(), 60)
        # The edge glows: bright pixels, blue above red (BGR), somewhere on you.
        glowing = (middle[:, :, 0] > 200) & (middle[:, :, 0] > middle[:, :, 2] + 40)
        self.assertGreater(glowing.sum(), 20)

    def test_particles_sweep_up_from_your_feet(self):
        frame, background, mask = scene()
        effect = create_effect("particles")
        effect.begin(OUT, frame, background, mask)
        self.assertAlmostEqual(effect.seconds, 0.95)
        middle = effect.render(frame, background, mask, 0.45)
        legs, head = np.s_[150:170, 140:180], np.s_[45:60, 140:180]
        # Below the front you are gone, above it you are still there.
        self.assertGreater(np.abs(middle[legs].astype(int) - frame[legs]).mean(), 60)
        self.assertLess(np.abs(middle[head].astype(int) - frame[head]).mean(), 10)
        # The blocks around the front are cyan (BGR: blue and green high, red low).
        cyan = (middle[:, :, 0] > 200) & (middle[:, :, 1] > 170) & (middle[:, :, 2] < 120)
        self.assertGreater(cyan.sum(), 30)

    def test_particles_return_rebuilds_you_from_the_head_down(self):
        frame, background, mask = scene()
        effect = create_effect("particles")
        effect.begin(IN, frame, background, mask)
        early = effect.render(frame, background, mask, 0.6)
        legs, head = np.s_[150:170, 140:180], np.s_[45:60, 140:180]
        self.assertLess(np.abs(early[head].astype(int) - frame[head]).mean(), 10)
        self.assertGreater(np.abs(early[legs].astype(int) - frame[legs]).mean(), 60)

    def test_goku_follows_the_original_beats(self):
        frame, background, mask = scene()
        person = mask > 0
        edge = np.zeros_like(person)
        edge[38:42, 128:192] = True  # Just around the top of the head.
        effect = create_effect("goku")
        effect.begin(OUT, frame, background, mask)
        self.assertAlmostEqual(effect.seconds, 0.30)
        # Streaks fall over you while you are still there...
        start = effect.render(frame, background, mask, 0.0)
        self.assertGreater(np.abs(start.astype(int) - frame).max(), 100)
        # ...three frames in you are gone, only your outline and the streaks remain.
        gone = effect.render(frame, background, mask, 0.1)
        self.assertGreater(np.abs(gone[person].astype(int) - frame[person]).mean(), 60)
        self.assertGreater(gone[edge].max(), 200)
        # Just before the end the streaks are gone and only the outline is left.
        late = effect.render(frame, background, mask, 0.27)
        changed = np.abs(late.astype(int) - background).sum(axis=2) > 60
        near_edge = np.zeros_like(person)
        near_edge[35:, 124:196] = True
        self.assertTrue(changed[near_edge].any())
        self.assertFalse(changed[~near_edge].any())

    def test_goku_return_shows_the_outline_then_you(self):
        frame, background, mask = scene()
        person = mask > 0
        effect = create_effect("goku")
        effect.begin(IN, frame, background, mask)
        first = effect.render(frame, background, mask, 0.0)
        self.assertGreater(np.abs(first[person].astype(int) - frame[person]).mean(), 60)
        self.assertGreater(np.abs(first.astype(int) - background).max(), 100)
        popped = effect.render(frame, background, mask, 0.05)
        self.assertLess(np.abs(popped[person].astype(int) - frame[person]).mean(), 40)

    def test_goku_outline_skips_the_frame_border(self):
        frame, background, mask = scene()
        effect = create_effect("goku")
        effect.begin(OUT, frame, background, mask)
        points = np.concatenate(effect.outline) / 4
        self.assertGreater(len(points), 20)
        self.assertLess(points[:, 1].max(), frame.shape[0] - 2)  # No line along the bottom edge.


class NarutoTests(unittest.TestCase):
    def test_smoke_hides_you_completely_at_the_swap(self):
        frame, background, mask = scene()
        person = mask > 0
        effect = create_effect("naruto")
        for direction in (OUT, IN):
            effect.begin(direction, frame, background, mask)
            for age in (naruto.CUT - 0.01, naruto.CUT + 0.01):
                hidden = effect.render(frame, background, mask, age)
                # Pale smoke everywhere you are, neither you (red) nor the room (green).
                self.assertGreater(hidden[person].min(), 120)

    def test_smoke_clears_by_the_end(self):
        frame, background, mask = scene()
        effect = create_effect("naruto")
        effect.begin(OUT, frame, background, mask)
        self.assertIsNone(effect.cloud.layer(naruto.SECONDS))
        late = effect.render(frame, background, mask, naruto.SECONDS - 0.01)
        self.assertLess(np.abs(late.astype(int) - background).mean(), 3)

    def test_burst_still_works_with_nobody_in_view(self):
        frame, background, _ = scene()
        effect = create_effect("naruto")
        effect.begin(OUT, frame, background, np.zeros(frame.shape[:2], np.float32))
        middle = effect.render(frame, background, background[:, :, 0] * 0.0, 0.3)
        self.assertTrue(np.any(middle != frame))


if __name__ == "__main__":
    unittest.main()
