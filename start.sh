#!/bin/bash
echo "================================================"
echo "  Qwen Voice Clone - Iniciando Servidor"
echo "================================================"
echo

# Activar entorno virtual si existe
if [ -f "venv/bin/activate" ]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate
else
    echo "NOTA: No se encontro entorno virtual."
    echo "Para crear uno ejecuta: python -m venv venv"
    echo
fi

# Iniciar servidor
echo "Iniciando servidor en http://localhost:8000"
echo "Presiona Ctrl+C para detener el servidor"
echo
python app.py
