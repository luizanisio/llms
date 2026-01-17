import platform
import subprocess
import sys
import os
import re
from pathlib import Path

# ===== FUNÇÕES AUXILIARES DE SEGURANÇA =====

def safe_read_file(filepath, encoding='utf-8'):
    """Leitura segura de arquivos do sistema."""
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except (FileNotFoundError, PermissionError, IOError):
        return None

def safe_subprocess_run(cmd, shell=False, timeout=5):
    """Execução segura de comandos do sistema.
    
    Args:
        cmd: Comando a ser executado (lista ou string)
        shell: Se True, executa via shell (use com cuidado!)
        timeout: Timeout em segundos
    
    Returns:
        Output do comando ou None em caso de erro
    """
    try:
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()
        
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False  # Não levanta exceção em caso de erro
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return None

# ===== FUNÇÕES DE COLETA DE INFORMAÇÕES DO HARDWARE =====

def get_cpu_info():
    """Obtém informações detalhadas da CPU."""
    try:
        cpuinfo = safe_read_file("/proc/cpuinfo")
        if cpuinfo:
            for line in cpuinfo.split('\n'):
                if "model name" in line:
                    return line.split(":")[1].strip()
        return platform.processor()
    except:
        return "CPU Genérica"

def get_cpu_cores():
    """Obtém número de núcleos da CPU."""
    try:
        return os.cpu_count()
    except:
        return "?"

def get_cpu_frequency():
    """Obtém frequência da CPU (base e máxima)."""
    try:
        cpuinfo = safe_read_file("/proc/cpuinfo")
        if cpuinfo:
            for line in cpuinfo.split('\n'):
                if "cpu MHz" in line:
                    freq_mhz = float(line.split(":")[1].strip())
                    return f"{freq_mhz / 1000:.2f} GHz"
        
        # Alternativa: tentar lscpu
        output = safe_subprocess_run(['lscpu'])
        if output:
            for line in output.split('\n'):
                if 'CPU max MHz' in line or 'CPU MHz' in line:
                    freq = re.search(r'(\d+\.?\d*)', line.split(':')[1])
                    if freq:
                        return f"{float(freq.group(1)) / 1000:.2f} GHz"
    except:
        pass
    return "Desconhecida"

def get_cpu_architecture():
    """Obtém arquitetura da CPU."""
    try:
        return platform.machine()
    except:
        return "Desconhecida"

def get_cpu_cache():
    """Obtém informações de cache da CPU."""
    try:
        output = safe_subprocess_run(['lscpu'])
        if output:
            cache_info = {}
            for line in output.split('\n'):
                if 'L1d cache' in line:
                    cache_info['L1d'] = line.split(':')[1].strip()
                elif 'L1i cache' in line:
                    cache_info['L1i'] = line.split(':')[1].strip()
                elif 'L2 cache' in line:
                    cache_info['L2'] = line.split(':')[1].strip()
                elif 'L3 cache' in line:
                    cache_info['L3'] = line.split(':')[1].strip()
            if cache_info:
                return cache_info
    except:
        pass
    return None

def get_ram_info():
    """Obtém informação total de RAM."""
    try:
        # SEGURANÇA: Usa lista em vez de shell=True
        output = safe_subprocess_run(['free', '-h'])
        if output:
            lines = output.split('\n')
            parts = lines[1].split()
            return parts[1]
    except:
        pass
    
    # Fallback: lê /proc/meminfo
    try:
        meminfo = safe_read_file("/proc/meminfo")
        if meminfo:
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    gb = kb / (1024 ** 2)
                    return f"{gb:.1f}G"
    except:
        pass
    
    return "RAM desconhecida"

def get_ram_details():
    """Obtém detalhes da memória RAM (tipo, frequência)."""
    try:
        # Tenta usar dmidecode (requer root normalmente)
        output = safe_subprocess_run(['sudo', '-n', 'dmidecode', '-t', 'memory'], timeout=10)
        if output:
            details = {}
            lines = output.split('\n')
            for i, line in enumerate(lines):
                if 'Type:' in line and 'Type Detail' not in line:
                    mem_type = line.split(':')[1].strip()
                    if mem_type not in ['Unknown', 'Other']:
                        details['type'] = mem_type
                if 'Speed:' in line and 'Configured' not in line:
                    speed = line.split(':')[1].strip()
                    if 'MHz' in speed:
                        details['speed'] = speed
                        break
            return details if details else None
    except:
        pass
    return None

def get_gpu_info():
    """Obtém informações básicas da GPU NVIDIA."""
    try:
        # SEGURANÇA: Usa lista em vez de shell=True
        cmd = [
            'nvidia-smi',
            '--query-gpu=name,memory.total,driver_version',
            '--format=csv,noheader'
        ]
        output = safe_subprocess_run(cmd)
        
        if output:
            gpus = output.split('\n')
            count = len(gpus)
            
            if count > 0:
                # Pega info da primeira GPU (assume-se homogeneidade se houver mais de uma)
                first_gpu = gpus[0].split(',')
                return {
                    "count": count,
                    "name": first_gpu[0].strip(),
                    "memory": first_gpu[1].strip(),
                    "driver": first_gpu[2].strip()
                }
    except:
        pass
    return None

def get_gpu_details():
    """Obtém informações detalhadas da GPU (compute capability, CUDA cores, etc)."""
    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=name,compute_cap,clocks.max.sm,clocks.max.memory',
            '--format=csv,noheader'
        ]
        output = safe_subprocess_run(cmd)
        
        if output:
            parts = output.split('\n')[0].split(',')
            details = {
                'name': parts[0].strip(),
                'compute_capability': parts[1].strip() if len(parts) > 1 else 'N/A',
                'clock_sm': parts[2].strip() if len(parts) > 2 else 'N/A',
                'clock_memory': parts[3].strip() if len(parts) > 3 else 'N/A'
            }
            return details
    except:
        pass
    return None

def get_os_info():
    """Obtém informações do sistema operacional."""
    try:
        # Tenta ler /etc/os-release que é padrão na maioria das distros modernas
        os_release = safe_read_file("/etc/os-release")
        if os_release:
            for line in os_release.split('\n'):
                if line.startswith("PRETTY_NAME="):
                    return line.split("=")[1].strip().strip('"')
    except:
        pass
    return f"{platform.system()} {platform.release()}"

def get_kernel_version():
    """Obtém versão do kernel Linux."""
    try:
        return platform.release()
    except:
        return "Desconhecida"

def get_storage_info():
    """Obtém informações de armazenamento."""
    try:
        output = safe_subprocess_run(['df', '-h', '/'])
        if output:
            lines = output.split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                return {
                    'total': parts[1],
                    'used': parts[2],
                    'available': parts[3],
                    'percent': parts[4]
                }
        
        # Alternativa: tenta lsblk
        output = safe_subprocess_run(['lsblk', '-d', '-o', 'NAME,SIZE,TYPE'])
        if output:
            return {'raw': output}
    except:
        pass
    return None

def get_storage_type():
    """Detecta tipo de armazenamento principal (SSD/NVMe/HDD)."""
    try:
        # Verifica rotação (0 = SSD/NVMe, >0 = HDD)
        devices = Path('/sys/block').glob('sd*')
        for device in devices:
            rotational_file = device / 'queue' / 'rotational'
            if rotational_file.exists():
                content = safe_read_file(str(rotational_file))
                if content:
                    if content.strip() == '0':
                        # Verifica se é NVMe
                        if 'nvme' in str(device):
                            return 'NVMe SSD'
                        return 'SSD'
                    else:
                        return 'HDD'
        
        # Verifica dispositivos NVMe
        nvme_devices = list(Path('/sys/block').glob('nvme*'))
        if nvme_devices:
            return 'NVMe SSD'
    except:
        pass
    return 'Desconhecido'

# ===== FUNÇÕES DE COLETA DE INFORMAÇÕES DE SOFTWARE =====

def get_python_version():
    """Obtém versão do Python."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def get_installed_packages():
    """Obtém versões de pacotes Python relevantes para ML."""
    packages = {}
    try:
        import importlib.metadata as importlib_metadata
    except ImportError:
        import importlib_metadata
    
    # Lista de pacotes importantes para documentar
    important_packages = [
        'torch', 'tensorflow', 'transformers', 'accelerate', 
        'bitsandbytes', 'unsloth', 'peft', 'trl', 'datasets',
        'numpy', 'pandas', 'scikit-learn'
    ]
    
    for package in important_packages:
        try:
            version = importlib_metadata.version(package)
            packages[package] = version
        except:
            pass
    
    return packages

def get_cuda_version():
    """Obtém versão do CUDA Toolkit instalada no sistema."""
    try:
        # Tenta nvcc --version
        output = safe_subprocess_run(['nvcc', '--version'])
        if output:
            # Procura por "release X.X"
            match = re.search(r'release (\d+\.\d+)', output)
            if match:
                return match.group(1)
    except:
        pass
    
    # Fallback: verifica arquivo de versão
    try:
        cuda_version_file = safe_read_file("/usr/local/cuda/version.txt")
        if cuda_version_file:
            match = re.search(r'CUDA Version (\d+\.\d+)', cuda_version_file)
            if match:
                return match.group(1)
    except:
        pass
    
    return "Não detectada"

def get_cudnn_version():
    """Obtém versão do cuDNN."""
    try:
        # Tenta ler header do cuDNN
        cudnn_header = safe_read_file("/usr/include/cudnn_version.h")
        if not cudnn_header:
            cudnn_header = safe_read_file("/usr/local/cuda/include/cudnn_version.h")
        
        if cudnn_header:
            major = re.search(r'#define CUDNN_MAJOR\s+(\d+)', cudnn_header)
            minor = re.search(r'#define CUDNN_MINOR\s+(\d+)', cudnn_header)
            patch = re.search(r'#define CUDNN_PATCHLEVEL\s+(\d+)', cudnn_header)
            
            if major and minor and patch:
                return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"
    except:
        pass
    
    return "Não detectada"

def generate_latex_snippet():
    """Gera snippet LaTeX completo para documentação do ambiente computacional."""
    print("Coletando informações do sistema...\n")
    
    # Coleta todas as informações
    cpu_model = get_cpu_info()
    cpu_cores = get_cpu_cores()
    cpu_freq = get_cpu_frequency()
    cpu_arch = get_cpu_architecture()
    cpu_cache = get_cpu_cache()
    
    ram_total = get_ram_info()
    ram_details = get_ram_details()
    
    os_name = get_os_info()
    kernel = get_kernel_version()
    
    storage_type = get_storage_type()
    storage_info = get_storage_info()
    
    gpu_info = get_gpu_info()
    gpu_details = get_gpu_details()
    
    python_ver = get_python_version()
    packages = get_installed_packages()
    cuda_toolkit = get_cuda_version()
    cudnn = get_cudnn_version()
    
    print("-" * 70)
    print(">>> RELATÓRIO COMPLETO DO AMBIENTE COMPUTACIONAL <<<")
    print("-" * 70)
    print("\n=== HARDWARE ===")
    print(f"CPU: {cpu_model}")
    print(f"  - Núcleos: {cpu_cores}")
    print(f"  - Frequência: {cpu_freq}")
    print(f"  - Arquitetura: {cpu_arch}")
    if cpu_cache:
        print(f"  - Cache: L2={cpu_cache.get('L2', 'N/A')}, L3={cpu_cache.get('L3', 'N/A')}")
    
    print(f"\nRAM: {ram_total}")
    if ram_details:
        print(f"  - Tipo: {ram_details.get('type', 'N/A')}")
        print(f"  - Frequência: {ram_details.get('speed', 'N/A')}")
    
    print(f"\nArmazenamento: {storage_type}")
    if storage_info and 'total' in storage_info:
        print(f"  - Total: {storage_info['total']}")
        print(f"  - Disponível: {storage_info['available']}")
    
    if gpu_info:
        print(f"\nGPU: {gpu_info['count']}x {gpu_info['name']}")
        print(f"  - VRAM: {gpu_info['memory']}")
        print(f"  - Driver: {gpu_info['driver']}")
        if gpu_details:
            print(f"  - Compute Capability: {gpu_details.get('compute_capability', 'N/A')}")
            print(f"  - Clock SM: {gpu_details.get('clock_sm', 'N/A')}")
    
    print(f"\n=== SOFTWARE ===")
    print(f"SO: {os_name}")
    print(f"Kernel: {kernel}")
    print(f"Python: {python_ver}")
    print(f"CUDA Toolkit: {cuda_toolkit}")
    print(f"cuDNN: {cudnn}")
    
    print(f"\nBibliotecas Python:")
    for pkg, ver in sorted(packages.items()):
        print(f"  - {pkg}: {ver}")
    
    print("\n" + "-" * 70)
    print(">>> TEXTO PARA QUALIFICAÇÃO (LaTeX) <<<")
    print("-" * 70 + "\n")
    
    # Gera parágrafo introdutório e tabela LaTeX
    latex_intro = f"""% Ambiente Computacional (Gerado automaticamente)
\\subsection{{Infraestrutura Computacional}}

Os experimentos foram conduzidos em um ambiente computacional de alto desempenho, configurado especificamente para o treinamento e avaliação de modelos de linguagem de grande escala. A Tabela~\\ref{{tab:env}} apresenta as especificações detalhadas do hardware e software utilizados."""
    
    # Inicia a tabela
    latex_table = """

\\begin{table}[h]
\\centering
\\caption{Configuração do ambiente de execução}
\\label{tab:env}
\\begin{tabular}{@{}ll@{}}
\\toprule
"""
    
    # Adiciona informações de CPU
    latex_table += f"CPU & {cpu_model} \\\\\n"
    if cpu_freq != "Desconhecida":
        # Formata frequência para LaTeX
        freq_formatted = cpu_freq.replace(" GHz", "\\,GHz")
        latex_table += f"Frequência base & {freq_formatted} \\\\\n"
    
    if cpu_cores:
        latex_table += f"Núcleos/Threads & {cpu_cores} \\\\\n"
    
    # Adiciona informações de RAM
    if ram_total:
        # Formata corretamente para LaTeX
        if 'Gi' in ram_total:
            ram_formatted = ram_total.replace('Gi', '\\,GB')
        elif 'G' in ram_total and 'GB' not in ram_total:
            ram_formatted = ram_total.replace('G', '\\,GB')
        else:
            ram_formatted = ram_total
        
        ram_line = f"Memória RAM & {ram_formatted}"
        
        if ram_details:
            if 'type' in ram_details:
                ram_line += f" {ram_details['type']}"
            if 'speed' in ram_details:
                speed_formatted = ram_details['speed'].replace(" MHz", "\\,MHz")
                ram_line += f" ({speed_formatted})"
        
        latex_table += ram_line + " \\\\\n"
    
    # Adiciona informações de GPU
    if gpu_info:
        gpu_line = f"GPU & "
        if gpu_info['count'] > 1:
            gpu_line += f"{gpu_info['count']}x "
        gpu_line += gpu_info['name']
        
        if gpu_info['memory']:
            # Converte MiB para GB se necessário
            mem_str = gpu_info['memory']
            if 'MiB' in mem_str:
                mib_value = float(mem_str.replace(' MiB', '').replace('MiB', ''))
                gb_value = mib_value / 1024
                mem_formatted = f"{gb_value:.0f}\\,GB"
            else:
                mem_formatted = mem_str.replace(" GiB", "\\,GB").replace(" GB", "\\,GB")
            gpu_line += f" ({mem_formatted} VRAM)"
        
        latex_table += gpu_line + " \\\\\n"
        
        # Adiciona Compute Capability se disponível
        if gpu_details and gpu_details.get('compute_capability') != 'N/A':
            latex_table += f"Compute Capability & {gpu_details['compute_capability']} \\\\\n"
    else:
        latex_table += "GPU & Não utilizada nos testes \\\\\n"
    
    # Adiciona informações de armazenamento
    storage_line = f"Armazenamento & {storage_type}"
    if storage_info and 'total' in storage_info:
        total_formatted = storage_info['total'].replace("T", "\\,TB").replace("G", "\\,GB")
        used_formatted = storage_info['used'].replace("T", "\\,TB").replace("G", "\\,GB")
        storage_line += f" {total_formatted} ({used_formatted} em uso)"
    latex_table += storage_line + " \\\\\n"
    
    # Adiciona informações de SO
    latex_table += f"Sistema Operacional & {os_name} \\\\\n"
    latex_table += f"Kernel & {kernel} \\\\\n"
    
    # Adiciona versão do Python
    latex_table += f"Python & {python_ver} \\\\\n"
    
    # Adiciona informações de CUDA se houver GPU
    if gpu_info:
        latex_table += f"Driver NVIDIA & {gpu_info['driver']} \\\\\n"
        
        if cuda_toolkit != "Não detectada":
            latex_table += f"CUDA Toolkit & {cuda_toolkit} \\\\\n"
        
        if cudnn != "Não detectada":
            latex_table += f"cuDNN & {cudnn} \\\\\n"
    
    # Adiciona bibliotecas principais
    libs_to_show = []
    if 'torch' in packages:
        libs_to_show.append(f"PyTorch {packages['torch']}")
    if 'transformers' in packages:
        libs_to_show.append(f"Transformers {packages['transformers']}")
    if 'unsloth' in packages:
        libs_to_show.append(f"Unsloth {packages['unsloth']}")
    if 'accelerate' in packages:
        libs_to_show.append(f"Accelerate {packages['accelerate']}")
    
    if libs_to_show:
        latex_table += f"Bibliotecas principais & {', '.join(libs_to_show)} \\\\\n"
    
    # Fecha a tabela
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    latex_text = latex_intro + latex_table
    
    print(latex_text)
    print("\n" + "-" * 70)
    
    return {
        'cpu': {'model': cpu_model, 'cores': cpu_cores, 'freq': cpu_freq, 'arch': cpu_arch, 'cache': cpu_cache},
        'ram': {'total': ram_total, 'details': ram_details},
        'storage': {'type': storage_type, 'info': storage_info},
        'gpu': gpu_info,
        'gpu_details': gpu_details,
        'os': {'name': os_name, 'kernel': kernel},
        'software': {'python': python_ver, 'packages': packages, 'cuda': cuda_toolkit, 'cudnn': cudnn},
        'latex': latex_text
    }

if __name__ == "__main__":
    generate_latex_snippet()
