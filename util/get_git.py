"""
 Utilitário criado por Luiz Anísio 05/2025 – v0.1.1
 --------------------------------------------------
 Exemplo de uso no Jupyter/Colab (estando em qualquer subpasta):
 >>> from util.get_git import sync_git_util
 >>> Util = sync_git_util(dest_root="../")        # copia repo para ../src
 >>> Util.flatten_listas([1, 2, [3, 4]])
"""

# ─── 1) CONFIGURAÇÕES GERAIS ─────────────────────────────────────────
REPO_URL   = "https://github.com/luizanisio/llms.git"   # repositório público
SUBDIR     = "src"          # subpasta que nos interessa dentro do repo
DEFAULT_DIR_NAME = "src"    # nome da pasta para onde o SUBDIR será copiado

# ─── 2) IMPORTS ──────────────────────────────────────────────────────
import subprocess, shutil, sys, importlib
from pathlib import Path

# ─── 3) FUNÇÃO DE SINCRONIZAÇÃO ─────────────────────────────────────
def _sync_git_subdir(repo_url: str, subdir: str, local_path: Path):
    """
    Garante que `local_path` receba o conteúdo de `subdir` do repositório.
    Faz clone ou pull em ~/.cache/git_repos e copia apenas o trecho desejado.
    """
    cache_root = Path.home() / ".cache" / "git_repos"
    repo_name  = Path(repo_url).stem
    repo_path  = cache_root / repo_name
    cache_root.mkdir(parents=True, exist_ok=True)

    if repo_path.exists():
        subprocess.run(["git", "-C", str(repo_path), "pull", "--quiet"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_path)], check=True)

    if local_path.exists():
        shutil.rmtree(local_path)
    shutil.copytree(repo_path / subdir, local_path)

    # Coloca a pasta no sys.path para que o import funcione
    abs_local = str(local_path.resolve())
    if abs_local not in sys.path:
        sys.path.insert(0, abs_local)

# ─── 4) IMPORTAÇÃO / RELOAD ─────────────────────────────────────────
def _import_or_reload(module_name: str):
    """Importa ou faz reload do módulo indicado."""
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)

# ─── 5) FUNÇÃO-FACILITADORA ─────────────────────────────────────────
def sync_git_util(dest_root: str = ".", *, repo_url: str = REPO_URL) -> "Util":
    """
    ↓↓↓ Principais parâmetros ↓↓↓
    dest_root (str) – Onde a pasta `src` será criada.  
                      "."   → ./src (padrão, no diretório atual)  
                      "../" → ../src  
                      "/tmp"→ /tmp/src  etc.
    repo_url  (str) – Caso deseje apontar para outro fork / URL.
    """
    dest_root_path = Path(dest_root).expanduser().resolve()
    local_dir      = dest_root_path / DEFAULT_DIR_NAME

    _sync_git_subdir(repo_url, SUBDIR, local_dir)

    # importa o modulo src.util
    util_module = _import_or_reload("src.util")
    print(f"[OK] src.util carregado de {local_dir}")
    util_module.Util.verifica_versao()
    return util_module.Util
