@echo off
rem ---------------------------------------------------------------------------
rem  Builds a stand-alone Windows app: dist\RoadSpeed\RoadSpeed.exe
rem  The result runs on any Windows PC without Python. Copy the whole
rem  dist\RoadSpeed folder anywhere (or zip it to share it).
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
title Building RoadSpeed.exe

set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python 3.9+ is needed to build the app: https://www.python.org/downloads/
  pause & exit /b 1
)

if not exist ".build-venv\Scripts\python.exe" %PY% -m venv .build-venv || goto fail
".build-venv\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check -q
".build-venv\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller --disable-pip-version-check || goto fail

".build-venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed ^
  --name RoadSpeed --icon assets\icon.ico ^
  --add-data "assets;assets" ^
  --collect-all imageio_ffmpeg ^
  gui.py || goto fail

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d=[Environment]::GetFolderPath('Desktop'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut(\"$d\Road Speed Grid.lnk\"); $s.TargetPath='%~dp0dist\RoadSpeed\RoadSpeed.exe'; $s.WorkingDirectory='%~dp0dist\RoadSpeed'; $s.Save()" >nul 2>nul

echo.
echo  Built: %~dp0dist\RoadSpeed\RoadSpeed.exe
echo  A "Road Speed Grid" shortcut is on your Desktop.
start "" "%~dp0dist\RoadSpeed"
pause
exit /b 0

:fail
echo.
echo  Build failed - see the messages above.
pause
exit /b 1
