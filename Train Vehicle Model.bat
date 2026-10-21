@echo off
rem ---------------------------------------------------------------------------
rem  Road Speed Grid - train the vehicle size/orientation model (NVIDIA GPU).
rem
rem  Double-click. It will:
rem    1. set up a private Python folder (.train-venv) with PyTorch for your GPU
rem    2. download the KITTI dataset (12 GB, resumes if interrupted) into
rem       %USERPROFILE%\RoadSpeed-training  (or the folder you drop on this file)
rem    3. train for ~20-40 minutes and save speedgrid\models\vehicle3d.onnx
rem  Run it again any time: finished steps are skipped.
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
title Road Speed Grid - training the vehicle model
set "DATA=%USERPROFILE%\RoadSpeed-training"
if not "%~1"=="" set "DATA=%~1"

echo.
echo  Step 1/3: checking the graphics card and Python
where nvidia-smi >nul 2>nul && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
where nvidia-smi >nul 2>nul || echo  WARNING: nvidia-smi not found - is the NVIDIA driver installed? Training would be very slow on the CPU.

if exist ".train-venv\Scripts\python.exe" goto haveenv
set "PY="
py -3 -c "import sys; sys.exit(0 if (3, 9) <= sys.version_info < (3, 14) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if (3, 9) <= sys.version_info < (3, 14) else 1)" >nul 2>nul && set "PY=python"
if not defined PY goto nopython
%PY% -m venv .train-venv || goto fail
".train-venv\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check -q
echo  Installing PyTorch with CUDA (about 2.5 GB, one time only)...
".train-venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --disable-pip-version-check || goto fail
".train-venv\Scripts\python.exe" -m pip install numpy opencv-python onnx --disable-pip-version-check || goto fail

:haveenv
".train-venv\Scripts\python.exe" -c "import torch; ok=torch.cuda.is_available(); print('  PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND (will use CPU - slow)')"

echo.
echo  Step 2/3: KITTI dataset in "%DATA%" (needs ~13 GB free)
".train-venv\Scripts\python.exe" training\prepare_kitti.py --data "%DATA%" || goto fail

echo.
echo  Step 3/3: training
".train-venv\Scripts\python.exe" training\train_vehicle3d.py --data "%DATA%" || goto fail

echo.
echo  ===========================================================================
echo   Done. The model is in speedgrid\models\vehicle3d.onnx - the app uses it
echo   automatically next time you analyse a video. Send vehicle3d.json to Claude
echo   if you want the accuracy checked.
echo  ===========================================================================
pause
exit /b 0

:nopython
echo  Python 3.9 - 3.13 is needed (PyTorch has no build for newer ones yet).
echo  Install Python 3.12 from https://www.python.org/downloads/ and tick
echo  "Add python.exe to PATH", then run this file again.
pause
exit /b 1

:fail
echo.
echo  Something went wrong - see the messages above. Copy them to Claude.
pause
exit /b 1
