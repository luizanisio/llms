# LLMs
Pacotes em desenvolvimento para estudos com LLMs

## Em construção 
- Desenvolvimento de pacotes para predição, treinamento e avaliação de LLMs

## Notebook
- Para carregar as classes no colab ou jupyter
```python
#@title Importando classes do git
# ███ 1) CONFIGURAÇÕES ────────────────────────────────────────────────
REPO_URL = "https://github.com/luizanisio/llms.git"   # meu repositório público
SUBDIR    = "src"          # subpasta do repositório
LOCAL_DIR = "./src"        # pasta local

# ███ 2) FUNÇÃO DE SINCRONIZAÇÃO ─────────────────────────────────────
import subprocess, shutil, sys, importlib
from pathlib import Path

def sync_git_subdir(repo_url: str, subdir: str, local_dir: str):
    """
    Baixa (ou dá pull) no repo e mantém em cache em ~/.cache/git_repos.
    Depois copia apenas o subdir desejado para o notebook,
    substituindo a versão antiga se houver.
    """
    cache_root = Path.home() / ".cache" / "git_repos"
    repo_name  = Path(repo_url).stem           # ex.: llms
    repo_path  = cache_root / repo_name

    cache_root.mkdir(parents=True, exist_ok=True)

    if repo_path.exists():
        # repositório já clonado → apenas atualiza
        subprocess.run(["git", "-C", str(repo_path), "pull", "--quiet"], check=True)
    else:
        # clone raso (depth 1) para economizar tempo/banda
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True
        )

    # Remove versão antiga do subdiretório local (se houver) e copia a nova
    local_path = Path(local_dir)
    if local_path.exists():
        shutil.rmtree(local_path)
    shutil.copytree(repo_path / subdir, local_path)

    # Garante que o Python encontre o pacote
    if str(local_path.resolve()) not in sys.path:
        sys.path.insert(0, str(local_path.resolve()))

# ███ 3) IMPORTAÇÃO / RELOAD ─────────────────────────────────────────
def import_or_reload(module_name: str):
    """
    Importa o módulo se for a primeira vez;
    caso já exista em sys.modules, faz reload para pegar alterações.
    """
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)

# ███ 4) EXECUÇÃO ────────────────────────────────────────────────────
sync_git_subdir(REPO_URL, SUBDIR, LOCAL_DIR)

util = import_or_reload("src.util")   # exemplo de uso
print("src.util carregado ↺")

Util = util.Util
# Se quiser checar versão/função:
Util.verifica_versao()

```
