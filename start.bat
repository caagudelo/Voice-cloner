@echo off
echo ================================================
echo   Qwen Voice Clone - Iniciando Servidor
echo ================================================
echo.

REM Verificar si existe entorno virtual
if exist "venv\Scripts\activate.bat" (
    echo Activando entorno virtual...
    call venv\Scripts\activate.bat
) else (
    echo NOTA: No se encontro entorno virtual.
    echo Para crear uno ejecuta: py -3.11 -m venv venv
    echo.
)

REM Iniciar servidor
echo Iniciando servidor en http://localhost:8000
echo Presiona Ctrl+C para detener el servidor
echo.
python app.py

pause
