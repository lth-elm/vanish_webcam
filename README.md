# VanishCam: vanish from your webcam with a hand sign

Make a hand sign and you disappear from your own webcam, in front of a clean
background. Make it again and you come back. It works in the preview window and,
through a virtual camera, in Zoom, Meet, Teams, Discord or OBS.

Everything runs on your computer. Video is never recorded or sent anywhere.

## The four effects

| Key | Effect | Sign to vanish (and to come back) | What happens |
| --- | --- | --- | --- |
| 1 | **Classic Dissolve** | V sign (index and middle fingers apart) | You fade away from the top with a glowing cyan edge. |
| 2 | **Particles** | V sign | Your silhouette crumbles into colored fragments that the wind carries off to the right. |
| 3 | **Naruto Substitution** | Palms pressed together, fingers side by side | A puff of smoke and a wooden log drops where you stood. |
| 4 | **Goku Teleport** | Index and middle fingers held together on your forehead for 1 second | Comic speed lines run up your silhouette, you flash and squeeze into a beam, and blink out with a teleport sound. Repeat the sign to teleport back in. |

After an effect plays, drop your hand before signing again. A progress bar in the
preview shows the Goku hold.

## Start it

**macOS:** double-click `VanishCam.command`, or in Terminal:

```bash
./VanishCam.command
```

**Windows:** double-click `VanishCam.bat`. **Linux:** `bash VanishCam.command`.

The first run creates a Python environment in `.venv`, installs the components and
downloads two MediaPipe models from Google (about 10 MB), so it needs internet once. It
installs again on its own after an update changes `requirements.txt`.
You need Python 3.10 or newer, and on macOS an Apple silicon Mac (MediaPipe has no
Intel Mac build).

The VanishCam window opens with your camera on the left and a settings panel on the
right. Then:

1. On macOS, allow camera access when asked.
2. Pick an effect in the panel (click a card, or press `1` to `4`).
3. Press **Space** (or **Capture background**), then **step out of the frame** for the
   4-second countdown while it learns what is behind you.
4. Step back in and make the sign shown at the bottom of the preview.

## Use it in a video call

VanishCam sends its output to a virtual camera, which you pick as your camera in the
call app.

- **macOS and Windows:** install [OBS Studio](https://obsproject.com/) 30 or newer. On
  macOS, open OBS once, click **Start Virtual Camera**, approve the system extension,
  then click **Stop Virtual Camera** and quit OBS. VanishCam now finds
  **OBS Virtual Camera** on its own.
- In Zoom, Meet, Teams, etc., choose **OBS Virtual Camera** as the camera.
- Only one program can feed the OBS virtual camera, so don't start it in OBS while
  VanishCam is running.

The virtual camera gets the clean, un-mirrored image. Call apps mirror your self-view
themselves, so other people see you (and any text behind you) the right way round. The
hints, countdown and key help are drawn only in the local preview window.

To use VanishCam inside an OBS scene instead, run it with `--no-virtual-camera` and add
a **Window Capture** of the **VanishCam** window (press `H` to hide the hints and `Tab`
to hide the panel first).
[Unity Capture](https://github.com/schellingb/UnityCapture) also works on Windows:
`--virtual-backend unitycapture`.

## Settings and keys

Everything is in the panel on the right, and each setting shows its key:

| Key | Action |
| --- | --- |
| `1` to `4` | Switch effect |
| `Space` | Capture the background (starts the countdown) |
| `V` | Vanish, or come back, without the sign |
| `C` | Next camera (or pick one by name in the Camera list) |
| `M` | Mirror the preview on/off |
| `O` | Virtual camera on/off |
| `S` | Sound on/off (volume slider in the panel) |
| `G` | Same sign brings you back, on/off |
| `H` | Hide or show the hints on the preview |
| `Tab` | Hide or show the panel |
| `F` | Full screen |
| `Q`, or `Esc` outside full screen | Quit |

The panel also has the resolution (1080p by default, or 720p and 540p for slower
computers) and how long the Goku sign must be held. Everything you change is
remembered in:

- macOS: `~/Library/Application Support/VanishCam/settings.json`
- Windows: `%APPDATA%\VanishCam\settings.json`
- Linux: `~/.config/VanishCam/settings.json`

Command-line options apply to one run and are not saved:

```bash
./VanishCam.command --mode goku --camera 1 --width 1920 --height 1080
```

| Option | Effect |
| --- | --- |
| `--mode normal\|particles\|naruto\|goku` | Start with this effect selected |
| `--camera N` | Use webcam number N (see `--list-cameras`) |
| `--width W --height H` | Video size: 1920 x 1080, 1280 x 720 or 960 x 540 |
| `--no-mirror`, `--no-sound`, `--fullscreen` | What they say |
| `--virtual-camera` / `--no-virtual-camera` | Require the virtual camera, or never use it |
| `--virtual-backend`, `--virtual-device`, `--fps` | Virtual camera driver, device and max FPS |
| `--list-cameras` | List webcams and exit |
| `--gesture-debug` | Show what the hand tracker sees, in the preview and the terminal |
| `--reset-settings` | Start from the default settings |

## Troubleshooting

- **"Camera could not be opened" on macOS:** open System Settings > Privacy & Security
  > Camera, allow the app you launched VanishCam from (Terminal, iTerm, VS Code...), and
  restart that app. Close other apps that use the camera.
- **Wrong camera (for example your iPhone via Continuity Camera):** pick the right one
  in the panel's Camera list, by name. The list refreshes each time you open it.
- **Choppy video:** choose 720p in the panel.
- **The sign is not recognized:** keep your hand fully in the frame and well lit, and
  run with `--gesture-debug`. For Goku, keep your face visible when you start, and hold
  the two fingers together and straight against your forehead. For Naruto, keep both
  hands in view.
- **Ghosting or bits of you left behind:** press `Space`, step fully out of frame, and
  let it capture the background again. Avoid moving the camera or changing the lighting
  after that.
- **No virtual camera:** see "Use it in a video call". The app keeps working with the
  preview window only.

## Development

```bash
.venv/bin/python -m unittest discover -p "test_*.py"
.venv/bin/python vanish_cam.py --self-test
```

| File | What it does |
| --- | --- |
| `vanish_cam.py` | Entry point: command-line options, settings, self-test |
| `window.py` | The window: preview with hints, settings panel, keyboard shortcuts |
| `engine.py` | Camera, background capture, gestures, effects and virtual camera, run on a background thread |
| `gestures.py` | Sign recognition for every effect, hold/release timing, detection thread |
| `effects.py` | Vanish and return animations for every effect |
| `naruto_effect.py` | Smoke and log drawing for the Naruto effect |
| `settings.py` | Saved settings |
| `vanish_sound.py` | Effect sounds |
| `tools/make_sounds.py` | Synthesizes the `.wav` sounds in `assets/` |
| `BUILD_WINDOWS.bat`, `VanishCam.spec` | Build a standalone Windows `VanishCam.exe` |

Video processing and gesture tracking each run on their own thread, so neither the
window nor the video stalls while the other works.

## Assets

- The teleport, wind and shimmer sounds are synthesized from scratch by
  `tools/make_sounds.py`; no recorded or sampled audio is used.
- The [wooden log](assets/substitution_log.png) is an original illustration (see
  [assets/README.md](assets/README.md)). Smoke, particles and speed lines are drawn in
  code. No anime frames or assets are included.
- Hand, face and person tracking use Google's [MediaPipe](https://ai.google.dev/edge/mediapipe)
  models.

## Credits

Based on [vanish_webcam](https://github.com/Nicovic17/vanish_webcam) by Nicovic17,
who created the original dissolve and Naruto effects. You can support the
original author through [PayPal](https://paypal.me/nicovic17) or
[Revolut](https://revolut.me/nicoauriemma).
