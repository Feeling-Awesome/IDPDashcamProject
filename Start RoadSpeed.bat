@echo off
rem ---------------------------------------------------------------------------
rem  Road Speed Grid - double-click to start.
rem  First run: installs what the app needs into a private folder (.venv) and
rem  puts a "Road Speed Grid" icon on your Desktop. After that it just opens.
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" goto run

title Road Speed Grid - first-time setup
echo.
echo  Setting up Road Speed Grid (first run only, needs internet, 1-3 minutes)...
echo.
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul && set "PY=python"
if not defined PY goto nopython

%PY% -c "import tkinter" >nul 2>nul || goto notk
%PY% -m venv .venv || goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt --disable-pip-version-check || goto fail

rem Desktop shortcut with the app icon
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d=[Environment]::GetFolderPath('Desktop'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut(\"$d\Road Speed Grid.lnk\"); $s.TargetPath='%~dp0.venv\Scripts\pythonw.exe'; $s.Arguments='\"%~dp0gui.py\"'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%~dp0assets\icon.ico'; $s.Save()" >nul 2>nul
echo.
echo  Done. A "Road Speed Grid" icon was added to your Desktop.
echo.

:run
start "" ".venv\Scripts\pythonw.exe" "%~dp0gui.py" %*
exit /b 0

:nopython
echo  Python is not installed (or is older than 3.9).
echo  1. Install it from https://www.python.org/downloads/
echo     - tick "Add python.exe to PATH" on the first screen of the installer
echo  2. Then double-click "Start RoadSpeed.bat" again.
start "" https://www.python.org/downloads/
pause
exit /b 1

:notk
echo  Your Python has no Tk (the window toolkit). Please reinstall Python from
echo  https://www.python.org/downloads/ and keep "tcl/tk and IDLE" ticked.
pause
exit /b 1

:fail
echo.
echo  Setup failed - see the messages above. Delete the ".venv" folder and try again.
if exist ".venv" rmdir /s /q ".venv"
pause
exit /b 1
