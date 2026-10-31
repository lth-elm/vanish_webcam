#!/bin/bash
# Double-click in Finder (macOS) or run "bash VanishCam.command" (Linux) to start VanishCam.
# Any options are passed through, e.g.:  ./VanishCam.command --mode goku
cd "$(dirname "$0")" || exit 1

pause_on_error() {
  echo
  read -r -p "VanishCam stopped with an error (see above). Press Enter to close."
}

if [ ! -x .venv/bin/python ]; then
  PYTHON=""
  # Prefer 3.12, the version VanishCam is tested with; any Python 3.10+ works.
  for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
  done
  if [ -z "$PYTHON" ]; then
    echo "Python 3 is required. Install it from https://www.python.org/downloads/"
    pause_on_error; exit 1
  fi
  echo "[VanishCam] Creating the Python environment (first run only)..."
  "$PYTHON" -m venv .venv || { pause_on_error; exit 1; }
fi

# Install on the first run, and again whenever requirements.txt changes.
if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "[VanishCam] Installing components (needs internet, only after an update)..."
  .venv/bin/python -m pip install --quiet --upgrade pip &&
    .venv/bin/python -m pip install --quiet -r requirements.txt &&
    cp requirements.txt .venv/requirements.installed || { pause_on_error; exit 1; }
fi

.venv/bin/python vanish_cam.py "$@" || { pause_on_error; exit 1; }
