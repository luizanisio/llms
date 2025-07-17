# -*- coding: utf-8 -*-

"""
 Utilitário criado por Luiz Anísio 05/2025 – v0.1.1
    - 07/2025 – v0.2.0 - JsonAnalise
 --------------------------------------------------
 Exemplo de uso no Jupyter/Colab (estando em qualquer subpasta):
 >>> import get_git
 >>> Util = get_git.sync_git_util()
 >>> JsonAnalise = get_git.sync_git_json_analise()
 >>> Util.flatten_listas([1, 2, [3, 4]])
 >>> JsonAnalise.teste_compara(3)

 Para escolher a pasta:
 >>> Util = get_git.sync_git_util(dest_root="../")  # copia repo para ../src
"""

# ─── 1) CONFIGURAÇÕES GERAIS ─────────────────────────────────────────
REPO_URL   = "https://github.com/luizanisio/llms.git"   # repositório público
SUBDIR     = "src"          # subpasta que nos interessa dentro do repo
DEFAULT_DIR_NAME = "src"    # nome da pasta para onde o SUBDIR será copiado
TEMPO_AB = dict() # guarda o tempo de atualização entre origem e destino 
                  # evita carregar mais de uma vez em 20s
TEMPO_AB_IGNORA = 20 # 20s por padrão ignora recarga                  

# ─── 2) IMPORTS ──────────────────────────────────────────────────────
import subprocess, shutil, sys, importlib
from pathlib import Path
from time import time
import os

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

def _atualiza(dest_root: str = ".", *, repo_url: str = REPO_URL):
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
    tempo = TEMPO_AB.get(f'{dest_root}__{repo_url}', 0)
    tempo = time()-tempo
    if (not os.path.isdir(local_dir)) or (tempo > TEMPO_AB_IGNORA):
      _sync_git_subdir(repo_url, SUBDIR, local_dir)
      #print(f"[OK] recarga da pasta {local_dir} realizada") 
      TEMPO_AB[f'{dest_root}__{repo_url}'] = time()
      return f'{local_dir} (git)'
    else:
      #print(f"[OK] recarga da pasta {local_dir} ignorada pois foi atualizada em {tempo:.1f}s") 
      return f'{local_dir} (cache)'

# ─── 5) FUNÇÔES IMPORTAÇÃO ─────────────────────────────────────────
def sync_git_util(dest_root: str = ".", *, repo_url: str = REPO_URL) -> "Util":
    """ Util = get_git.sync_git_util()
    """
    local_dir = _atualiza(dest_root=dest_root, repo_url=repo_url)

    # importa o modulo src.util
    util_module = _import_or_reload("src.util")
    print(f"[OK] Util carregado de {local_dir}")
    util_module.Util.verifica_versao()
    return util_module.Util

def sync_git_json_analise(dest_root: str = ".", *, repo_url: str = REPO_URL) -> "JsonAnalise":
    """ JsonAnalise = get_git.sync_git_json_analise()
    """
    local_dir = _atualiza(dest_root=dest_root, repo_url=repo_url)

    # importa o modulo src.util
    util_module = _import_or_reload("src.util_json")
    print(f"[OK] JsonAnalise carregado de {local_dir}")
    util_module.JsonAnalise.verifica_versao()
    return util_module.JsonAnalise

        
        
