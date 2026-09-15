@echo off
title AI-Powered 3D Model Creator - CLI Generator
color 0A
echo.
echo  =============================================================
echo    AI-Powered 3D Model Creator (RX 6600, among others)
echo    CLI Image-to-3D Generator
echo  =============================================================
echo.
echo   USAGE: Drag and drop a PNG/JPG image onto this .bat file
echo   Or run from console:
echo     CLI_GENERATE.bat path\to\image.png [ultra/rapida/media/alta]
echo.

if "%~1"=="" (
    echo  [ERROR] No image selected!
    echo  Please drag and drop an image onto this file to start.
    echo.
    pause
    exit /b 1
)

set IMAGEN=%~1
set CALIDAD=%~2
if "%CALIDAD%"=="" set CALIDAD=ultra

echo  Selected Image: %IMAGEN%
echo  Quality Preset: %CALIDAD%
echo.

cd /d "%~dp0"
python app/cli.py "%IMAGEN%" --calidad %CALIDAD%

echo.
echo  Generation finished! Press any key to exit...
pause >nul
