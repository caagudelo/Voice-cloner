import subprocess, sys > detect_gpu.py 
try: >> detect_gpu.py 
    has_nvidia = False >> detect_gpu.py 
    gpu_info = "" >> detect_gpu.py 
    try: >> detect_gpu.py 
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).upper() >> detect_gpu.py 
        has_nvidia = True >> detect_gpu.py 
    except: >> detect_gpu.py 
        pass >> detect_gpu.py 
    if not has_nvidia: >> detect_gpu.py 
        print('--- No se detecto tarjeta NVIDIA (o faltan drivers). Instalando version CPU ---') >> detect_gpu.py 
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']) >> detect_gpu.py 
    elif '5070' in gpu_info or '5080' in gpu_info or '5090' in gpu_info: >> detect_gpu.py 
        print('--- NVIDIA Serie 5000 detectada. Instalando version Avanzada (CUDA 12.8 Nightly)... ---') >> detect_gpu.py 
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--pre', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/nightly/cu128']) >> detect_gpu.py 
    else: >> detect_gpu.py 
        print('--- NVIDIA Estandar detectada. Instalando version Estable (CUDA 12.4)... ---') >> detect_gpu.py 
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu124']) >> detect_gpu.py 
except Exception as e: >> detect_gpu.py 
    print('Error en autodeteccion, usando modo seguro (CPU):', e) >> detect_gpu.py 
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']) >> detect_gpu.py 
