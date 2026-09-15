@echo off
title TripoSR - Generador 3D desde Imagen (AMD Radeon RX 6600)
cd /d "%~dp0"
echo ============================================================
echo   Iniciando Servidor Web TripoSR (Image-to-3D)
echo   Optimizado para AMD Radeon RX 6600 (8GB VRAM) en Windows
echo ============================================================
echo.
echo Abriendo interfaz grafica en tu navegador...
echo Abre: http://127.0.0.1:7860
echo.
python gradio_app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Ocurrio un error al ejecutar la interfaz.
    pause
)
