@echo off
setlocal
rem Double-click to start VanishCam on Windows. Options are passed through,
rem for example:  VanishCam.bat --mode goku
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [VanishCam] Creating the Python environment ^(first run only^)...
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 goto :error
)

rem Install on the first run, and again whenever requirements.txt changes.
fc /b requirements.txt .venv\requirements.installed >nul 2>nul
if errorlevel 1 (
  echo [VanishCam] Installing components ^(needs internet, only after an update^)...
  "%VENV_PY%" -m pip install --quiet --upgrade pip
  if errorlevel 1 goto :error
  "%VENV_PY%" -m pip install --quiet -r requirements.txt
  if errorlevel 1 goto :error
  copy /y requirements.txt .venv\requirements.installed >nul
)

"%VENV_PY%" vanish_cam.py %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo VanishCam could not start. See the messages above, or README.md for common fixes.
pause
exit /b 1
