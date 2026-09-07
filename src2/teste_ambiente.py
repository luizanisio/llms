#!/usr/bin/env python3
"""
Autor: Luiz Anísio
Verifica dependências, versões e recursos do ambiente de treinamento/avaliação.
Exibe dicas de instalação para pacotes ausentes.

Uso:
    python teste_ambiente.py
    conda run -n treina python src/teste_ambiente.py
"""

import sys
import importlib.metadata
import importlib.util

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OK   = "✅"
_WARN = "⚠️ "
_ERR  = "❌"
_OPT  = "💡"

_total = 0
_ok = 0
_ausentes_obrigatorios = []
_ausentes_opcionais = []


def _versao(pacote: str) -> str | None:
    try:
        return importlib.metadata.version(pacote)
    except importlib.metadata.PackageNotFoundError:
        return None


def _importavel(modulo: str) -> bool:
    return importlib.util.find_spec(modulo) is not None


def _linha(label: str, versao: str | None, obrigatorio: bool = True,
           dica: str = "", extra: str = "") -> None:
    global _total, _ok
    _total += 1
    if versao:
        _ok += 1
        sufixo = f"  {extra}" if extra else ""
        print(f"   {_OK} {label:<28} {versao}{sufixo}")
    else:
        tag = _ERR if obrigatorio else _OPT
        print(f"   {tag} {label:<28} não instalado{'  ← obrigatório' if obrigatorio else '  ← opcional'}")
        if dica:
            print(f"      → {dica}")
        if obrigatorio:
            _ausentes_obrigatorios.append(label)
        else:
            _ausentes_opcionais.append(label)


def _secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print("   " + "─" * 60)


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("  VERIFICAÇÃO DO AMBIENTE DE TREINAMENTO / AVALIAÇÃO")
print("=" * 65)
print(f"  Python {sys.version.split()[0]}  |  {sys.executable}")

# ---------------------------------------------------------------------------
# CUDA / GPU
# ---------------------------------------------------------------------------

_secao("🖥️  CUDA e GPU")

try:
    import torch
    _torch_ver = torch.__version__
    cuda_ok = torch.cuda.is_available()
    print(f"   {_OK if cuda_ok else _ERR} CUDA disponível:           {'Sim' if cuda_ok else 'Não'}")
    if cuda_ok:
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            cc = f"{props.major}.{props.minor}"
            print(f"   {_OK} GPU[{i}] {props.name:<24} {mem_gb:.1f} GB  compute {cc}")
    print(f"   {_OK} torch.version.cuda:        {torch.version.cuda}")
    # SDPA backends
    flash_sdp  = cuda_ok and torch.backends.cuda.flash_sdp_enabled()
    mem_sdp    = cuda_ok and torch.backends.cuda.mem_efficient_sdp_enabled()
    math_sdp   = cuda_ok and torch.backends.cuda.math_sdp_enabled()
    print(f"   {'✅' if flash_sdp else '❌'} SDPA flash_sdp:            {'ativo' if flash_sdp else 'inativo'}")
    print(f"   {'✅' if mem_sdp  else '❌'} SDPA mem_efficient_sdp:    {'ativo' if mem_sdp  else 'inativo'}")
    print(f"   {'✅' if math_sdp else '❌'} SDPA math_sdp:             {'ativo' if math_sdp else 'inativo'}")
except ImportError:
    print(f"   {_ERR} torch não instalado — impossível verificar CUDA/GPU")
    flash_sdp = False

# ---------------------------------------------------------------------------
# Flash Attention
# ---------------------------------------------------------------------------

_secao("⚡  Flash Attention (economia de VRAM O(n) vs O(n²))")

# SDPA nativo — não conta como pacote ausente, é parte do PyTorch
if flash_sdp:
    print(f"   {_OK} {'SDPA nativo (PyTorch >= 2.0)':<28} ativo")
    print(f"      → Economia de VRAM equivalente ao pacote flash-attn para LoRA fine-tuning.")
    print(f"        Ative no YAML: flash_attention_2: true")
else:
    print(f"   {_WARN} {'SDPA nativo (PyTorch >= 2.0)':<28} inativo")
    print(f"      → Atualize o PyTorch: pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cu128")

# Pacote pip flash-attn (opcional)
_flash_attn_ver = _versao("flash-attn")
if _flash_attn_ver:
    _linha("flash-attn (pacote pip)", _flash_attn_ver, obrigatorio=False)
elif flash_sdp:
    # SDPA nativo cobre o caso de uso — não conta como ausente problemático
    _total += 1
    _ok += 1
    _ausentes_opcionais  # não adiciona
    print(f"   {_OPT} {'flash-attn (pacote pip)':<28} não instalado  ← opcional, já atendido pelo PyTorch SDPA")
    print(f"      → Instale apenas para: sliding window attention / FA-3 no H100")
    print(f"        TORCH_CUDA_ARCH_LIST=\"8.6\" MAX_JOBS=1 pip install flash-attn --no-build-isolation")
else:
    _linha(
        "flash-attn (pacote pip)",
        None,
        obrigatorio=False,
        dica=(
            'TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=1 '
            "pip install flash-attn --no-build-isolation\n"
            "        ⏱️  ~60-90 min  |  SDPA nativo inativo — este pacote é a alternativa"
        ),
    )

# ---------------------------------------------------------------------------
# Treinamento — pacotes obrigatórios
# ---------------------------------------------------------------------------

_secao("🤖  Treinamento (obrigatórios)")

_linha("torch",               _versao("torch"))
_linha("transformers",        _versao("transformers"))
_linha("accelerate",          _versao("accelerate"))
_linha("bitsandbytes",        _versao("bitsandbytes"))
_linha("trl",                 _versao("trl"))
_linha("peft",                _versao("peft"))
_linha("datasets",            _versao("datasets"))

# ---------------------------------------------------------------------------
# Otimizadores de memória
# ---------------------------------------------------------------------------

_secao("🔥  Otimizadores de VRAM (recomendados)")

_linha(
    "liger-kernel",
    _versao("liger-kernel"),
    obrigatorio=False,
    dica="pip install liger-kernel   # fused CE + RoPE + RMSNorm — reduz ~40% VRAM pico",
)
_linha(
    "triton",
    _versao("triton"),
    obrigatorio=False,
    dica="pip install triton   # necessário pelo liger-kernel",
)

# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------

_secao("🚀  Inferência")

_linha(
    "vllm",
    _versao("vllm"),
    obrigatorio=False,
    dica="TMPDIR=/var/tmp pip install vllm   # inferência de alta performance",
)
_linha("sentence-transformers", _versao("sentence-transformers"))

# Unsloth (opcional — só para export GGUF / predição legada)
_linha(
    "unsloth",
    _versao("unsloth"),
    obrigatorio=False,
    dica="pip install -r requirements-unsloth.txt   # apenas para export GGUF / predição legada",
)

# ---------------------------------------------------------------------------
# Avaliação e comparação
# ---------------------------------------------------------------------------

_secao("📊  Avaliação / Comparação")

_linha("bert-score",           _versao("bert-score"))
_linha("rouge-score",          _versao("rouge-score"))
_linha("python-Levenshtein",   _versao("python-Levenshtein"))
_linha("fuzzywuzzy",           _versao("fuzzywuzzy"))
_linha("tiktoken",             _versao("tiktoken"))

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

_secao("🛠️   Utilitários")

_linha("numpy",        _versao("numpy"))
_linha("pandas",       _versao("pandas"))
_linha("matplotlib",   _versao("matplotlib"))
_linha("seaborn",      _versao("seaborn"))
_linha("tqdm",         _versao("tqdm"))
_linha("PyYAML",       _versao("PyYAML"))
_linha("regex",        _versao("regex"))
_linha("openpyxl",     _versao("openpyxl"))
_linha("xlsxwriter",   _versao("xlsxwriter"))
_linha("tensorboardX", _versao("tensorboardX"))
_linha("ipywidgets",   _versao("ipywidgets"))
_linha("cryptography", _versao("cryptography"))
_linha("python-dotenv",_versao("python-dotenv"))

# ---------------------------------------------------------------------------
# APIs
# ---------------------------------------------------------------------------

_secao("🌐  APIs")

_linha(
    "openai",
    _versao("openai"),
    obrigatorio=False,
    dica="pip install openai",
)
_linha(
    "openrouter",
    _versao("openrouter"),
    obrigatorio=False,
    dica="pip install openrouter",
)

# ---------------------------------------------------------------------------
# Resumo final
# ---------------------------------------------------------------------------

print()
print("=" * 65)

_ausentes = len(_ausentes_obrigatorios) + len(_ausentes_opcionais)
_icone = _OK if not _ausentes_obrigatorios else _ERR

print(f"  {_icone}  {_ok}/{_total} pacotes instalados", end="")
if _ausentes_opcionais:
    print(f"  |  {len(_ausentes_opcionais)} opcional(is) ausente(s)", end="")
if _ausentes_obrigatorios:
    print(f"  |  {len(_ausentes_obrigatorios)} OBRIGATÓRIO(S) AUSENTE(S)", end="")
print()

if _ausentes_obrigatorios:
    print(f"\n  {_ERR}  Obrigatórios ausentes:")
    for p in _ausentes_obrigatorios:
        print(f"       • {p}")
    print(f"\n     Instale via: TMPDIR=/var/tmp pip install -r requirements.txt")

if _ausentes_opcionais:
    print(f"\n  {_OPT}  Opcionais ausentes (ver dicas acima):")
    for p in _ausentes_opcionais:
        print(f"       • {p}")

if not _ausentes_obrigatorios and not _ausentes_opcionais:
    print(f"\n  Ambiente completo — tudo pronto para treinar! 🎉")
elif not _ausentes_obrigatorios:
    print(f"\n  Ambiente de treinamento OK. Opcionais ausentes não bloqueiam o treino.")

print("=" * 65)
print()
