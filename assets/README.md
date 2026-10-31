# Smoke

The Naruto smoke cloud is drawn in code by `naruto_effect.py` (billow textures,
lighting and motion are all procedural). A
[CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet) was
reviewed but is not included: procedural smoke scales and follows the detected
person.

# Teleport sound

`teleport.wav` is the Instant Transmission sound effect from *Dragon Ball Z*, the
same one heard in the original author's Goku demo. It was supplied by the project
owner as an MP3 and prepared with Python and ffmpeg: the 0.82 s of leading silence
was cut (keeping 3 ms before the onset, so it plays the moment the effect starts),
the clip was kept to 1.4 s with a 0.12 s fade-out, and it was raised 6 dB to peak at
-3 dBFS. Vanish and return both play it. It is not original to this project.
