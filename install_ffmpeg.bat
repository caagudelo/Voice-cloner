@echo off
title Instalador de FFmpeg
color 0A

echo =======================================================
echo      DESCARGA E INSTALACION DE FFMPEG (REQUERIDO)
echo =======================================================
echo.

echo Comprobando FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% equ 0 (
    echo FFmpeg ya esta instalado en el sistema.
    pause
    exit /b
)

echo FFmpeg no encontrado. Iniciando descarga automatica...
echo Esto puede tomar un momento dependiendo de tu conexion.
echo.

set "FFMPEG_ZIP=%TEMP%\ffmpeg.zip"
set "FFMPEG_DIR=C:\ffmpeg"

REM Descargar build estatico de gyan.dev
powershell -Command "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile '%FFMPEG_ZIP%'"

if not exist "%FFMPEG_ZIP%" (
    color 0C
    echo ERROR: Fallo la descarga de FFmpeg.
    pause
    exit /b
)

echo.
echo Extrayendo archivos...
if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"

REM Usar powershell para extraer el zip temporalmente
powershell -Command "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%TEMP%\ffmpeg_ext' -Force"

REM Mover el contenido de la carpeta interna a C:\ffmpeg
for /d %%I in ("%TEMP%\ffmpeg_ext\ffmpeg-*") do (
    xcopy /E /Y "%%I\*" "%FFMPEG_DIR%\" >nul
)

REM Limpiar temporales
del "%FFMPEG_ZIP%"
rmdir /S /Q "%TEMP%\ffmpeg_ext"

echo.
echo Agregando FFmpeg al PATH del sistema...
powershell -Command "$userPath = [Environment]::GetEnvironmentVariable('Path', 'User'); if ($userPath -notmatch [regex]::Escape('C:\ffmpeg\bin')) { $newPath = $userPath + ';C:\ffmpeg\bin'; [Environment]::SetEnvironmentVariable('Path', $newPath, 'User'); }"

echo.
color 0B
echo =======================================================
echo     FFMPEG INSTALADO CORRECTAMENTE EN C:\ffmpeg
echo =======================================================
echo.
echo IMPORTANTE: Para que los cambios surtan efecto en tu 
echo servidor de Voz, debes CERRAR LA VENTANA NEGRA ACTUAL 
echo Y VOLVER A ABRIRLA usando start.bat
echo.
pause
