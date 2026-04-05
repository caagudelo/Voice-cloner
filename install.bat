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

echo [4/4] Detectando Hardware Grafico Automaticamente...
echo =======================================================
echo Identificando la arquitectura de tu PC para instalar la version optima...

echo import subprocess, sys > detect_gpu.py
echo try: >> detect_gpu.py
echo     has_nvidia = False >> detect_gpu.py
echo     gpu_info = "" >> detect_gpu.py
echo     try: >> detect_gpu.py
echo         gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).upper() >> detect_gpu.py
echo         has_nvidia = True >> detect_gpu.py
echo     except: >> detect_gpu.py
echo         pass >> detect_gpu.py
echo     if not has_nvidia: >> detect_gpu.py
echo         print('--- No se detecto tarjeta NVIDIA (o faltan drivers). Instalando version CPU ---') >> detect_gpu.py
echo         subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']) >> detect_gpu.py
echo     elif '5070' in gpu_info or '5080' in gpu_info or '5090' in gpu_info: >> detect_gpu.py
echo         print('--- NVIDIA Serie 5000 detectada. Instalando version Avanzada (CUDA 12.8 Nightly)... ---') >> detect_gpu.py
echo         subprocess.run([sys.executable, '-m', 'pip', 'install', '--pre', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/nightly/cu128']) >> detect_gpu.py
echo     else: >> detect_gpu.py
echo         print('--- NVIDIA Estandar detectada. Instalando version Estable (CUDA 12.4)... ---') >> detect_gpu.py
echo         subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu124']) >> detect_gpu.py
echo except Exception as e: >> detect_gpu.py
echo     print('Error en autodeteccion, usando modo seguro (CPU):', e) >> detect_gpu.py
echo     subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']) >> detect_gpu.py

python detect_gpu.py
del detect_gpu.py

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
