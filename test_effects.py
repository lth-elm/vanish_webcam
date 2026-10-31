import unittest

import numpy as np

from effects import IN, OUT, EFFECTS, create_effect
from naruto_effect import (
    EFFECT_SECONDS, falling_log_anchor, landed_log_anchor, load_log, overlay_log,
    render_substitution, substitution_anchor,
)


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
                if mode != "naruto":  # Naruto's smoke bursts on the very first frame.
                    np.testing.assert_allclose(start[person].mean(axis=0), (40, 60, 220), atol=40)
                middle = effect.render(frame, background, mask, effect.seconds * 0.5)
                self.assertTrue(np.any(middle != frame))
                end = effect.render(frame, background, mask, effect.seconds)
                gone = effect.vanished(background)
                if mode != "naruto":  # Naruto leaves a log where you stood.
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

    def test_particles_fly_downwind(self):
        frame, background, mask = scene()
        effect = create_effect("particles")
        effect.begin(OUT, frame, background, mask)
        output = effect.render(frame, background, mask, 0.9)
        changed = np.abs(output.astype(int) - background).sum(axis=2) > 30
        changed[mask > 0] = False
        columns = np.nonzero(changed.any(axis=0))[0]
        self.assertGreater(columns.size, 0)
        self.assertGreater(columns.mean(), 190)  # To the right of the person.

    def test_goku_lines_rise_then_linger_after_you_are_gone(self):
        frame, background, mask = scene()
        effect = create_effect("goku")
        effect.begin(OUT, frame, background, mask)
        person = mask > 0
        linger = effect.render(frame, background, mask, 0.7)
        # You are gone but bright streaks remain over where you stood.
        self.assertGreater(linger[person].max(), 200)
        self.assertGreater(np.abs(linger.astype(int) - background)[person].mean(), 3)


class NarutoTests(unittest.TestCase):
    def test_generated_asset_has_real_transparency(self):
        sprite = load_log()
        self.assertEqual(sprite.shape[2], 4)
        self.assertEqual(int(sprite[:, :, 3].min()), 0)
        self.assertEqual(int(sprite[:, :, 3].max()), 255)

    def test_smoke_animation_leaves_log_behind(self):
        frame = np.full((160, 240, 3), 130, np.uint8)
        background = np.full_like(frame, 35)
        mask = np.ones(frame.shape[:2], np.float32)
        sprite = load_log()
        anchor = (120, 85, 100)
        middle = render_substitution(frame, background, mask, sprite, anchor, 0.55)
        final = render_substitution(frame, background, mask, sprite, anchor, EFFECT_SECONDS)
        self.assertTrue(np.any(middle != frame))
        self.assertTrue(np.any(middle != background))
        self.assertTrue(np.any(final != background))
        np.testing.assert_array_equal(final[0, 0], background[0, 0])

    def test_log_anchor_tracks_person_region(self):
        mask = np.zeros((160, 240), np.float32)
        mask[20:130, 140:210] = 1.0
        x, y, size = substitution_anchor(mask)
        self.assertGreater(x, 140)
        self.assertGreater(size, 35)
        self.assertTrue(20 < y < 130)

    def test_log_falls_then_stays_on_floor(self):
        anchor = (120, 80, 90)
        ys = [falling_log_anchor(anchor, 240, age)[1] for age in (0.1, 0.3, 0.5, 0.7)]
        self.assertEqual(ys[0], anchor[1])
        self.assertGreater(ys[2], ys[1])
        self.assertGreater(ys[3], ys[2])
        floor = landed_log_anchor(anchor, 240)
        background = np.full((240, 240, 3), 35, np.uint8)
        output = render_substitution(background, background, np.zeros((240, 240), np.float32),
                                     load_log(), anchor, EFFECT_SECONDS)
        expected = background.copy()
        overlay_log(expected, load_log(), floor)
        np.testing.assert_array_equal(output, expected)

    def test_log_stays_while_vanished_and_goes_when_you_return(self):
        frame, background, mask = scene(240, 320)
        effect = create_effect("naruto")
        effect.begin(OUT, frame, background, mask)
        self.assertTrue(np.any(effect.vanished(background) != background))
        effect.begin(IN, frame, background, mask)
        back = effect.render(frame, background, mask, effect.seconds)
        outside = mask == 0
        self.assertLess(np.abs(back[outside].astype(int) - background[outside]).mean(), 2)


if __name__ == "__main__":
    unittest.main()
