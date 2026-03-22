@echo off
title Instalador Qwen Voice Clone
color 0A

echo =======================================================
echo      INSTALADOR AUTOMATICO - QWEN VOICE CLONE
echo =======================================================
echo.

echo [1/4] Comprobando instalacion de Python...
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo ERROR: Python no esta instalado o no esta agregado al PATH del sistema.
    echo Por favor instala Python 3.10 o superior desde https://www.python.org/downloads/
    echo y asegurate de marcar la casilla "Add Python to PATH" durante la instalacion.
    echo.
    pause
    exit /b
)
echo Python detectado correctamente.
echo.

echo [2/4] Creando entorno virtual (venv)...
if not exist "venv\Scripts\activate.bat" (
    py -3.11 -m venv venv
    if %errorlevel% neq 0 (
        color 0C
        echo ERROR: No se pudo crear el entorno virtual.
        pause
        exit /b
    )
    echo Entorno virtual creado exitosamente.
) else (
    echo El entorno virtual ya existe, se omitira la creacion.
)

echo.
echo [3/4] Activando entorno virtual y actualizando pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
echo Entorno activado.
echo.

echo [4/4] Seleccion de dependencias...
echo =======================================================
echo Tienes una tarjeta grafica dedicada NVIDIA?
echo (Esto instalara PyTorch con soporte CUDA para mayor velocidad)
echo =======================================================
echo 1. SI (Tengo NVIDIA de la serie GTX/RTX)
echo 2. NO (No tengo NVIDIA / Prefiero usar el procesador CPU)
echo.
set /p choice="Escribe 1 o 2 y presiona Enter: "

if "%choice%"=="1" (
    echo.
    echo Instalando PyTorch con soporte CUDA para NVIDIA...
    echo Esto puede tardar varios minutos y descargar varios Gigabytes...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo.
    echo Instalando PyTorch en modo CPU...
)

echo.
echo Instalando dependencias principales del proyecto (TTS y transcripcion)...
pip install qwen-tts fastapi uvicorn[standard] python-multipart soundfile numpy aiofiles openai-whisper

echo.
color 0B
echo =======================================================
echo              INSTALACION COMPLETADA
echo =======================================================
echo Todo ha sido instalado y configurado correctamente.
echo.
echo Para abrir el programa, haz doble clic en el archivo:
echo -^> start.bat
echo.
pause
