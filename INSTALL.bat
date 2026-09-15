@echo off
title AI-Powered 3D Model Creator - Installer
color 0B

echo =======================================================================
echo    AI-Powered 3D Model Creator (RX 6600, among others) - Installer
echo =======================================================================
echo.
echo  This script will automatically set up your environment and dependencies.
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH!
    echo Please install Python 3.10 or 3.11 from python.org and check "Add to PATH".
    echo.
    pause
    exit /b 1
)

echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/4] Detecting GPU and installing PyTorch...
echo.
echo Please select your hardware configuration:
echo   1. AMD Radeon GPU (RX 6600, RX 6700, RX 7600, etc.) or Intel Arc [RECOMMENDED]
echo   2. NVIDIA GeForce GPU (RTX 3060, 4060, etc.)
echo   3. CPU Only
echo.
set /p GPU_CHOICE="Enter your choice (1, 2, or 3) [Default 1]: "
if "%GPU_CHOICE%"=="" set GPU_CHOICE=1

if "%GPU_CHOICE%"=="1" (
    echo Installing PyTorch CPU core with DirectML for AMD/Intel...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install torch-directml
) else if "%GPU_CHOICE%"=="2" (
    echo Installing PyTorch with CUDA 12.1 for NVIDIA...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installing PyTorch CPU...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo [3/4] Installing project requirements (diffusers, transformers, rembg, pymeshlab, gradio)...
pip install -r requirements.txt

echo.
echo [4/4] Verifying installation...
python -c "import torch, trimesh, gradio, PIL; print('[OK] Core libraries loaded successfully!')"

echo.
echo =======================================================================
echo   Installation Complete!
echo   To start the Web Studio, double-click: START_WEB_UI.bat
echo   To generate via drag-and-drop, use:     CLI_GENERATE.bat
echo =======================================================================
echo.
pause
