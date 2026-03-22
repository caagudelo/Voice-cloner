# Qwen Voice Clone | Rocka Dev 🎙️✨

Una potente aplicación web para la **clonación de voz** de alta calidad basada en el motor **Qwen3-TTS** y el sistema de transcripción **Whisper Engine**. Diseñada con una interfaz moderna y optimizada para el manejo eficiente de recursos de hardware.

![Screenshots](https://img.shields.io/badge/Status-Online-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-blueviolet)

---

## 🌟 Características Principales

- **🧬 Clonación de Voz de Precisión**: Utiliza una huella de voz de solo 10 segundos para generar narraciones completas con el mismo timbre y tono.
- **✨ Transcripción Automática**: Integración con **Whisper Large** para transcribir automáticamente tus archivos de referencia.
- **✂️ Editor de Audio Integrado**: Recorta y selecciona el fragmento exacto de tu audio de referencia directamente desde el navegador con visualización de forma de onda (Wavesurfer.js).
- **🌍 Soporte Multi-idioma**: Genera voces en Español, Inglés, Chino, Japonés, Coreano, Francés, Alemán y más.
- **🚀 Optimización de VRAM**: Sistema de limpieza inteligente de memoria GPU y carga dinámica (Lazy Loading) de modelos para GPUs con 8GB o menos.
- **📥 Múltiples Formatos**: Descarga los resultados en alta fidelidad (WAV) o formatos comprimidos (MP3).

---

## 📋 Requisitos del Sistema

### Hardware Recomendado
- **GPU**: NVIDIA RTX Series (Recomendado 8GB+ VRAM).
- **CPU**: Modo fallback disponible si no se detecta GPU (mucho más lento).
- **RAM**: 16GB mínimo para una experiencia fluida.

### Software
- **Python**: 3.11.x
- **FFmpeg**: Necesario para la conversión a MP3.
- **Controladores CUDA**: Versión 12.1 o superior.

---

## 🚀 Instalación y Configuración

### 1. Requisitos Previos (Windows)
1. **Python 3.11**: Descarga e instala [Python 3.11.x](https://www.python.org/downloads/). 
   **⚠️ IMPORTANTE:** Asegúrate de marcar la casilla **"Add Python to PATH"** durante la instalación.
2. (Recomendado) Tarjeta gráfica NVIDIA (Serie GTX/RTX) para aceleración por hardware vía CUDA.

### 2. Instalación Automática (Recomendado para Windows)
El proyecto incluye scripts interactivos que configuran todo el entorno por ti:

1. **Instalar dependencias de Python** (`install.bat`):
   - Haz doble clic en el archivo `install.bat` desde el explorador de Windows.
   - El script comprobará que tengas Python instalado en el PATH, creará un entorno virtual aislado (`venv`) y actualizará pip.
   - Por último, te preguntará si tienes una tarjeta gráfica dedicada NVIDIA. Responde con `1` (Sí) o `2` (No/CPU). Dependiendo de tu respuesta, descargará automáticamente la versión de PyTorch más adecuada y el resto de las aplicaciones (FastAPI, Qwen, Whisper, etc). 

2. **Instalar FFmpeg - Requerido para MP3** (`install_ffmpeg.bat`):
   - Haz doble clic en el archivo `install_ffmpeg.bat`.
   - Este script revisa si ya existe una instalación existente de FFmpeg en tu computadora.
   - Si no lo encuentra, lo descargará automáticamente, lo extraerá en la carpeta `C:\ffmpeg` y añadirá los binarios al `PATH` de tu sistema.
   - **Nota vital:** Una vez que FFmpeg se instala, deberás cerrar cualquier consola que tuvieras abierta para que la computadora detecte los cambios del registro.

### 3. Instalación Manual (Avanzado, Linux o macOS)
Si prefieres hacerlo de manera tradicional por consola:

```bash
# 1. Crear y activar el entorno virtual
python3.11 -m venv venv

# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# 2. Instalar dependencias base
pip install -r requirements.txt
pip install openai-whisper

# 3. Instalar PyTorch
# Con soporte CUDA (para GPUs NVIDIA en Windows/Linux):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Solo para CPU (Mac o sistemas sin GPU Nvidia):
pip install torch torchvision torchaudio

# 4. (Opcional) Instalar dependencias adicionales para optimizar VRAM de video
# pip install flash-attn --no-build-isolation
```

### 4. Ejecución con Docker (Recomendado para servidores)
Si deseas desplegar la aplicación sin afectar tu sistema local, puedes usar Docker y Docker Compose. El proyecto ya incluye soporte completo para passthrough de GPU con NVIDIA.

#### Requisitos para Docker GPU:
- [Docker y Docker Compose](https://docs.docker.com/get-docker/) instalados.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) instalado en el sistema anfitrión.

#### Pasos para ejecutar:
```bash
# 1. Construir la imagen y levantar el contenedor en segundo plano (dettached mode)
docker-compose up -d --build

# 2. Ver registros en vivo (Opcional)
docker-compose logs -f
```
El contenedor de base expondrá el puerto `8000` en tu máquina local y montará automáticamente las carpetas locales (`modelo/`, `uploads/`, `outputs/`) como volúmenes para evitar descargas manuales o pérdida de datos.

---

## 💻 Uso de la Aplicación

1. **Iniciar el Servidor**:
   Ejecuta el archivo `start.bat`. El servidor estará disponible en `http://localhost:8000`.

2. **Cargar Referencia**:
   Sube un audio de referencia (WAV, MP3, OGG). Se recomienda un fragmento limpio de 5 a 10 segundos.

3. **Editar y Transcribir**:
   - Usa el editor visual para seleccionar el mejor fragmento.
   - Haz clic en **✨ Transcribir** para que la IA detecte el texto dicho o escríbelo manualmente.

4. **Generar**:
   Escribe el texto que quieres que la voz clonada diga, selecciona el idioma y haz clic en **Generar Clonación de Voz ⚡**.

---

## 🎯 Estrategias para Capturar Acentos Regionales

Capturar un acento específico (como el colombiano "paisa", "costeño", etc.) requiere optimizar la prosodia (ritmo y entonación). Sigue estas estrategias:

1. **Selección "Melódica"**: Las regiones de entre 5 y 10 segundos funcionan mejor. Elige un fragmento donde la persona sea muy **expresiva**. La IA clona la voz basándose en la curva de entonación detectada.
2. **Transcripción Fonética**: En el cuadro de "Transcripción del Audio", escribe **exactamente** lo que se escucha. Si la persona dice "verdá" en lugar de "verdad", escríbelo así. Esto ayuda a la IA a mapear sonidos regionales específicos a las palabras.
3. **Consistencia de Estilo**: Para mantener el acento en el resultado final, redacta el **Texto Objetivo** usando el mismo lenguaje y modismos de la persona. Si le pides una frase formal a un audio de referencia coloquial, el acento tenderá a neutralizarse.
4. **Calidad de Referencia**: Asegúrate de que el audio de referencia no tenga música de fondo ni eco. El ruido "ensucia" la firma acústica del acento.
5. **Ajuste de Temperatura (Factor de Creatividad)**: El sistema usa por defecto una temperatura de 0.75 para mayor estabilidad. Si el acento suena muy neutral o robótico, subir la temperatura a **1.0** (en el archivo `app.py`) permite que la IA sea más expresiva y respete mejor las subidas y bajadas de tono regionales, aunque esto puede generar pequeñas inconsistencias en la dicción.

---

## 🛠️ Tecnologías Utilizadas

- **Backend**: FastAPI (Python 3.11)
- **Frontend**: HTML5, CSS3 Moderno, JavaScript
- **Motores IA**: 
  - Qwen3-TTS (Síntesis de voz)
  - OpenAI Whisper Large (Transcripción)
- **Procesamiento de Audio**: Soundfile, FFmpeg
- **Visualización**: Wavesurfer.js (Herramientas de edición de ondas)

---

## 🤝 Créditos

Diseñado y desarrollado por **Camilo Andres Agudelo**. 
🌐 **Web:** [cagudelo.com](https://cagudelo.com)

Potenciado por tecnologías de código abierto de vanguardia en el campo de la Inteligencia Artificial.

---
> [!NOTE]
> La primera vez que inicies la aplicación, el sistema descargará el modelo Qwen3-TTS (~2-4GB). Se recomienda una conexión a internet estable.
