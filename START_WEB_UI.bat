@echo off
title AI-Powered 3D Model Creator - Web Studio
color 0B
echo.
echo  =============================================================
echo    AI-Powered 3D Model Creator (RX 6600, among others)
echo    Universal AI Image-to-3D Studio - Open Source
echo  =============================================================
echo.
echo   Starting local Web Studio...
echo   Opening in your default browser at:
echo   http://localhost:7860/?__theme=dark
echo.

cd /d "%~dp0"
python app/web_ui.py

pause
