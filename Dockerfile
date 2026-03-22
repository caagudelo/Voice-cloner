# Usar la imagen oficial de Python 3.11 slim como base
FROM python:3.11-slim

# Evitar que Python escriba archivos .pyc y forzar salida sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar FFmpeg y dependencias del sistema requeridas
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl tzdata && \
    rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos
COPY requirements.txt .

# Instalar las dependencias de Python (incluyendo PyTorch con soporte CUDA)
# PyTorch con CUDA 12.1 según está especificado como ideal para el proyecto
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir openai-whisper && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copiar el código fuente
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
