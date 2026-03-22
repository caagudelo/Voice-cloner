import os
import uuid
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import gc
import whisper
import subprocess
import shutil

# Configuracion CUDA para GPUs con memoria limitada (8GB)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

app = FastAPI(title="Qwen Voice Clone", description="Aplicacion para clonar voz usando Qwen3-TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorios
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "modelo"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Variables globales
model = None
current_device = None

SUPPORTED_LANGUAGES = [
    "Spanish", "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Italian"
]

# Ya no mantenemos un modelo global de Whisper en memoria para ahorrar VRAM.
# El modelo se cargará y descargará bajo demanda.

def get_gpu_info():
    """Obtiene informacion detallada de la GPU"""
    if not torch.cuda.is_available():
        return None

    gpu_info = {
        "name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
    }

    # Memoria disponible actual
    if torch.cuda.is_available():
        gpu_info["free_memory_gb"] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        gpu_info["used_memory_gb"] = torch.cuda.memory_allocated(0) / (1024**3)

    return gpu_info


def clear_gpu_memory():
    """Limpia la memoria de la GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def trim_audio_file(file_path: str, start_time: float, end_time: float) -> str:
    """
    Recorta un archivo de audio usando soundfile (sin depender de ffmpeg para el recorte inicial).
    Si los tiempos son invalidos, no hace nada. Solo recorta y reescribe si end_time > start_time > 0.
    """
    if start_time is None or end_time is None:
        return file_path
        
    start_time = float(start_time)
    end_time = float(end_time)
    
    if start_time < 0 or start_time >= end_time:
        return file_path

    try:
        data, samplerate = sf.read(file_path)
        
        # Convertir tiempo en segundos a frames/muestras
        start_frame = int(start_time * samplerate)
        end_frame = int(end_time * samplerate)
        
        # Limites del archivo original
        start_frame = max(0, start_frame)
        end_frame = min(len(data), end_frame)
        
        # Recortar usando slices nativos de numpy
        cropped_data = data[start_frame:end_frame]
        
        # Crear un nuevo archivo cropped y devolver la ruta original simulada o reemplazarlo
        temp_crop_path = file_path + ".crop.wav"
        sf.write(temp_crop_path, cropped_data, samplerate)
        
        # Reemplazar el original por el recortado
        os.replace(temp_crop_path, file_path)
        return file_path
    except Exception as e:
        print(f"Error al recortar audio: {e}")
        return file_path


def get_model(force_cpu=False):
    """
    Carga el modelo de forma lazy.

    Args:
        force_cpu: Si es True, fuerza el uso de CPU aunque haya GPU disponible
    """
    global model, current_device

    if model is None:
        from qwen_tts import Qwen3TTSModel

        # Determinar dispositivo
        use_gpu = torch.cuda.is_available() and not force_cpu

        if use_gpu:
            gpu_info = get_gpu_info()
            print("=" * 50)
            print(f"GPU detectada: {gpu_info['name']}")
            print(f"VRAM total: {gpu_info['total_memory_gb']:.1f} GB")
            print(f"Compute capability: {gpu_info['compute_capability']}")
            print("=" * 50)

            # Limpiar memoria antes de cargar
            clear_gpu_memory()

            device = "cuda:0"
            # float16 es mas compatible que bfloat16 en RTX series
            dtype = torch.float16

            model_kwargs = {
                "device_map": device,
                "torch_dtype": dtype,
            }

            # Intentar usar Flash Attention 2 (mas eficiente en memoria)
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Flash Attention 2: Habilitado")
            except ImportError:
                print("Flash Attention 2: No disponible (opcional)")
                # Usar SDPA como alternativa eficiente
                model_kwargs["attn_implementation"] = "sdpa"
                print("Usando SDPA (Scaled Dot Product Attention)")
        else:
            print("=" * 50)
            print("Usando CPU (sin GPU detectada o modo forzado)")
            print("NOTA: La generacion sera mas lenta en CPU")
            print("=" * 50)

            device = "cpu"
            dtype = torch.float32

            model_kwargs = {
                "device_map": device,
                "torch_dtype": dtype,
            }

        print(f"\nCargando modelo Qwen3-TTS desde: {MODEL_PATH}")
        print("Esto puede tomar 1-2 minutos la primera vez...")

        try:
            model = Qwen3TTSModel.from_pretrained(
                str(MODEL_PATH),
                **model_kwargs
            )
            current_device = device

            if use_gpu:
                gpu_info = get_gpu_info()
                print(f"\nModelo cargado en GPU")
                print(f"VRAM utilizada: {gpu_info['used_memory_gb']:.2f} GB")
                print(f"VRAM libre: {gpu_info['free_memory_gb']:.2f} GB")
            else:
                print(f"\nModelo cargado en CPU")

        except torch.cuda.OutOfMemoryError:
            print("\nERROR: Memoria GPU insuficiente")
            print("Intentando cargar en CPU...")
            clear_gpu_memory()

            # Fallback a CPU
            model = Qwen3TTSModel.from_pretrained(
                str(MODEL_PATH),
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            current_device = "cpu"
            print("Modelo cargado en CPU (fallback)")

    return model


@app.get("/", response_class=HTMLResponse)
async def home():
    """Sirve la pagina principal"""
    html_path = BASE_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/languages")
async def get_languages():
    """Retorna los idiomas soportados"""
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form(None, description="Idioma del audio (codigo ISO, ej: 'es', 'en')"),
    start_time: float = Form(None, description="Inicio del recorte (segundos)"),
    end_time: float = Form(None, description="Fin del recorte (segundos)"),
):
    """Transcribe un archivo de audio usando Whisper large (carga bajo demanda) y permite recorte de región exacta"""
    temp_audio_path = UPLOAD_DIR / f"temp_transcribe_{uuid.uuid4()}{Path(audio.filename).suffix or '.wav'}"
    
    try:
        with open(temp_audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
            
        # 1. RECORTAR EL AUDIO A LA REGIÓN SELECCIONADA POR EL USUARIO EN LA WEB
        if start_time is not None and end_time is not None:
            trim_audio_file(str(temp_audio_path), start_time, end_time)
            
        print("\nIniciando proceso de transcripcion...")
        # Limpiar cualquier residuo de memoria en la GPU
        clear_gpu_memory()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Cargando modelo Whisper (large)... Esto requerira aprox. 3-4GB VRAM.")
        w_model = whisper.load_model("large", device=device)
        
        if language:
            print(f"Transcribiendo audio: {audio.filename} (Forzando idioma: {language})...")
            result = w_model.transcribe(str(temp_audio_path), language=language)
        else:
            print(f"Transcribiendo audio: {audio.filename} (Auto-detectando idioma)...")
            result = w_model.transcribe(str(temp_audio_path))
            
        texto_transcrito = result["text"].strip()
        print(f"Transcripcion completada: '{texto_transcrito[:50]}...'")
        
        # ELIMINAR EL MODELO PARA LIBERAR LA MEMORIA VRAM INMEDIATAMENTE
        del w_model
        clear_gpu_memory()
        print("Modelo Whisper 'large' ha sido eliminado de la GPU y memoria liberada para Qwen.")
        
        return {
            "success": True, 
            "text": texto_transcrito
        }
    except torch.cuda.OutOfMemoryError as e:
        # En caso de faltante de memoria, intentar limpiar
        clear_gpu_memory()
        print("ERROR: Memoria VRAM agotada. Quizas Qwen esta ocupando demasiada y Whisper no cabe.")
        raise HTTPException(status_code=500, detail="Memoria GPU agotada (Out Of Memory). Intenta de nuevo.")
    except Exception as e:
        print(f"Error en transcripcion: {e}")
        try:
            del w_model
        except NameError:
            pass
        clear_gpu_memory()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_audio_path.exists():
            temp_audio_path.unlink()



@app.get("/api/status")
async def get_status():
    """Verifica el estado del servidor con info detallada"""
    status = {
        "status": "online",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "current_device": current_device or ("cuda" if torch.cuda.is_available() else "cpu"),
    }

    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        status["gpu"] = {
            "name": gpu_info["name"],
            "total_memory_gb": round(gpu_info["total_memory_gb"], 2),
            "used_memory_gb": round(gpu_info.get("used_memory_gb", 0), 2),
            "free_memory_gb": round(gpu_info.get("free_memory_gb", gpu_info["total_memory_gb"]), 2),
        }

    return status


@app.post("/api/clone")
async def clone_voice(
    ref_audio: UploadFile = File(..., description="Audio de referencia (WAV, MP3, OGG)"),
    ref_text: str = Form(..., description="Transcripcion original"),
    target_text: str = Form(..., description="Texto a sintetizar"),
    language: str = Form("Spanish", description="Idioma"),
    start_time: float = Form(None, description="Inicio del recorte (segundos)"),
    end_time: float = Form(None, description="Fin del recorte (segundos)"),
):
    """
    Clona una voz a partir de un audio de referencia.
    - Soporta recorte exacto en base a coordenadas enviadas por el navegador web (max 10s)
    """

    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Idioma no soportado. Usa uno de: {SUPPORTED_LANGUAGES}"
        )

    # Guardar archivo de audio temporalmente
    audio_id = str(uuid.uuid4())
    audio_ext = Path(ref_audio.filename).suffix or ".wav"
    temp_audio_path = UPLOAD_DIR / f"{audio_id}{audio_ext}"

    try:
        # Guardar audio subido
        with open(temp_audio_path, "wb") as f:
            content = await ref_audio.read()
            f.write(content)

        # RECORTAR LA PISTA ANTES DE ENVIAR A QWEN
        if start_time is not None and end_time is not None:
            trim_audio_file(str(temp_audio_path), start_time, end_time)

        # Cargar modelo
        tts_model = get_model()

        print(f"\nGenerando voz clonada...")
        print(f"Texto: '{target_text[:80]}{'...' if len(target_text) > 80 else ''}'")
        print(f"Idioma: {language}")

        # Generar con manejo de memoria y parametros optimizados de calidad
        # - temperature: Más bajo (0.7-0.8) hace la voz mas estable/limpia. Más alto (1.0+) la hace mas caótica/expresiva.
        # - top_p: Controla la aleatoriedad, 0.85 es buen balance.
        # - repetition_penalty: Ayuda a evitar que el modelo se trabe tartamudeando o con ruidos metálicos.
        try:
            with torch.inference_mode():
                wavs, sr = tts_model.generate_voice_clone(
                    text=target_text,
                    language=language,
                    ref_audio=str(temp_audio_path),
                    ref_text=ref_text,
                    temperature=1,
                    top_p=0.85,
                    repetition_penalty=1.1
                )
        except torch.cuda.OutOfMemoryError:
            print("Memoria GPU insuficiente durante generacion, limpiando cache...")
            clear_gpu_memory()

            # Reintentar con texto mas corto si es muy largo (manteniendo los parametros)
            with torch.inference_mode():
                wavs, sr = tts_model.generate_voice_clone(
                    text=target_text,
                    language=language,
                    ref_audio=str(temp_audio_path),
                    ref_text=ref_text,
                    temperature=0.75,
                    top_p=0.85,
                    repetition_penalty=1.1
                )

        # Guardar resultado en WAV (original)
        output_filename_wav = f"cloned_{audio_id}.wav"
        output_path_wav = OUTPUT_DIR / output_filename_wav
        sf.write(str(output_path_wav), wavs[0], sr)
        
        # Convertir a MP3 usando FFmpeg
        output_filename_mp3 = f"cloned_{audio_id}.mp3"
        output_path_mp3 = OUTPUT_DIR / output_filename_mp3
        
        has_ffmpeg = shutil.which("ffmpeg") is not None
        mp3_generated = False
        
        if has_ffmpeg:
            try:
                print(f"Convirtiendo a MP3...")
                # Ejecutar ffmpeg silenciosamente (-y para sobreescribir si existe)
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(output_path_wav),
                    "-codec:a", "libmp3lame", "-qscale:a", "2", 
                    str(output_path_mp3)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                mp3_generated = True
                print(f"MP3 generado exitosamente")
            except Exception as e:
                print(f"Advertencia: No se pudo convertir a MP3: {e}")

        # Limpiar memoria despues de generar
        clear_gpu_memory()

        print(f"Audio(s) generado(s) exitosamente")

        if torch.cuda.is_available():
            gpu_info = get_gpu_info()
            print(f"VRAM libre despues de generar: {gpu_info['free_memory_gb']:.2f} GB")

        response_data = {
            "success": True,
            "message": "Voz clonada exitosamente",
            "audio_url": f"/api/audio/{output_filename_wav}",
            "filename": output_filename_wav,
            "has_mp3": mp3_generated
        }
        
        if mp3_generated:
            response_data["audio_url_mp3"] = f"/api/audio/{output_filename_mp3}"
            response_data["filename_mp3"] = output_filename_mp3
            
        return response_data

    except torch.cuda.OutOfMemoryError:
        clear_gpu_memory()
        raise HTTPException(
            status_code=500,
            detail="Memoria GPU insuficiente. Intenta con un texto mas corto o reinicia el servidor."
        )
    except Exception as e:
        clear_gpu_memory()
        raise HTTPException(status_code=500, detail=f"Error al procesar: {str(e)}")

    finally:
        # Limpiar archivo temporal
        if temp_audio_path.exists():
            temp_audio_path.unlink()


@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    """Descarga un archivo de audio generado"""
    audio_path = OUTPUT_DIR / filename

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio no encontrado")

    # Detectar el tipo de medio basado en la extension
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"

    return FileResponse(
        path=str(audio_path),
        media_type=media_type,
        filename=filename
    )


@app.post("/api/load-model")
async def load_model_endpoint():
    """Pre-carga el modelo en memoria"""
    try:
        get_model()
        return {
            "success": True,
            "message": "Modelo cargado correctamente",
            "device": current_device
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar modelo: {str(e)}")


@app.post("/api/clear-memory")
async def clear_memory():
    """Limpia la cache de memoria GPU"""
    clear_gpu_memory()

    result = {"success": True, "message": "Memoria limpiada"}

    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        result["free_memory_gb"] = round(gpu_info["free_memory_gb"], 2)

    return result


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("       Qwen Voice Clone - Servidor Web")
    print("=" * 60)
    print()

    # Info del sistema
    print("CONFIGURACION DEL SISTEMA:")
    print("-" * 40)

    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        print(f"  GPU: {gpu_info['name']}")
        print(f"  VRAM: {gpu_info['total_memory_gb']:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  PyTorch: {torch.__version__}")
    else:
        print("  GPU: No detectada")
        print("  Modo: CPU")
        print(f"  PyTorch: {torch.__version__}")

    print()
    print(f"  Modelo: {MODEL_PATH}")
    print()
    print("-" * 40)
    print("  Servidor: http://localhost:8000")
    print("  API Docs: http://localhost:8000/docs")
    print("-" * 40)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
