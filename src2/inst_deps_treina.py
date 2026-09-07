# Autor: Luiz Anísio
# Fonte: https://github.com/luizanisio/llms/tree/main/src

import os, shlex, subprocess, sys
from time import sleep

def _is_colab() -> bool:
    """Detecta se o código está rodando no Google Colab."""
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False

def _limpar():
    try:
      from IPython.display import clear_output        
      clear_output(wait=True)
    except:
      print('Não foi possível limpar o output :( \n')

def _pip(cmd: str, msg_ok: str = "✅ Instalação concluída"):
    """Roda pip, mostra progresso e limpa a saída se tudo der certo."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "pip", *shlex.split(cmd)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    # stream do log
    for line in proc.stdout:
        print(line, end="")          # mantém a barra de progresso visível
    proc.wait()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    _limpar()

    print('\n', msg_ok)

def inst_dependencias():
    arq1 = "/content/pip_unsloth_ok.txt"
    arq2 = "/content/pip_transformers_ok.txt"

    if not os.path.isfile(arq1):
        _unsloth = "unsloth[colab-new]==2025.8.9"
        print(f"Instalando {_unsloth} ...")
        _pip(f"install {_unsloth}", '✅ Unsloth instalado _o/')
        import unsloth
        print("Unsloth OK _o/")
        with open(arq1, "w") as f:
            f.write("Unsloth instalado _o/")
    else:
        print("Unsloth já ok _o/")

    if not os.path.isfile(arq2):
        _transformers = "transformers>=4.53.0,<4.54.0"
        print(f"Instalando {_transformers} ...")
        _pip(f"install --upgrade --force-reinstall --no-cache-dir {_transformers}", '✅ Transformers instalado _o/')
        import transformers
        print("Transformers OK _o/")
        with open(arq2, "w") as f:
            f.write("Transformers instalado _o/")
    else:
        print("Transformers já ok _o/")

def testar_dependencias(verificar_unsloth=False):
    arq1 = "pip_unsloth_ok.txt"
    arq2 = "pip_transformers_ok.txt"
    if _is_colab():
       _unsloth = "unsloth[colab-new]==2025.8.9"
    else:
       _unsloth = "unsloth==2025.8.9"
    #_transformers = "transformers>=4.53.0,<4.54.0" if verificar_unsloth else "transformers>=4.55.0,<4.56.0"
    _transformers = "transformers>=4.55.3,<4.56.0"

    if os.path.isfile(arq1) and os.path.isfile(arq2):
       print('✅ Instalação já realizada nesse ambiente _o/')
       return

    try:
        print("Verificando dependências ....")
        if verificar_unsloth:
           import unsloth
           from unsloth import FastModel
           from unsloth.chat_templates import get_chat_template
        import torch, transformers
        if verificar_unsloth:
           print("imports unsloth e transformers ok ___o/")
        else:
           print("imports transformers ok ___o/")
    except ImportError as e:
        print(f"\n\nOCORREU UM ERRO DE IMPORT: {e}")
        if verificar_unsloth:
           print("Tentando instalar transformers e unsloth")
        else:
           print("Tentando instalar transformers")

        if not os.path.isfile(arq1) and verificar_unsloth:
          print(f"Instalando {_unsloth} ...")
          _pip(f"install {_unsloth}", '✅ Unsloth instalado _o/')
          import unsloth
          print("Unsloth OK _o/")
          with open(arq1, "w") as f:
              f.write("Unsloth instalado _o/")

        if not os.path.isfile(arq2):
          print(f"Instalando {_transformers} ...")
          _pip(f"install --upgrade --force-reinstall --no-cache-dir {_transformers}", '✅ Transformers instalado _o/')
          import transformers
          print("Transformers OK _o/")
          with open(arq2, "w") as f:
              f.write("Transformers instalado _o/")

        if _is_colab():
            _limpar()
            print("🔁 Novos pacotes instalados. Reiniciando a sessão do Colab...")
            print("⚠️ Rode novamente essa célula!!!!!", flush=True)
            sleep(1)
            print('...')
            os.kill(os.getpid(), 9)        

def testar_dependencias_analise()           :
    arq1 = "pip_analise_ok.txt"
    _analise = "python-Levenshtein rouge-score"
    if os.path.isfile(arq1):
       print('✅ Instalação já realizada nesse ambiente _o/')
       return
    try:
        print("Verificando dependências ....")
        import Levenshtein
        from rouge_score import rouge_scorer
        print("imports Levenshtein e rouge_score ok ___o/")
    except ImportError as e:
        print(f"\n\nOCORREU UM ERRO DE IMPORT: {e}")
        print("Tentando instalar Levenshtein e rouge_score")
        print(f"Instalando {_analise} ...")

        _pip(f"install {_analise}", '✅ Levenshtein e Rouge instalados _o/')
        import Levenshtein
        from rouge_score import rouge_scorer
        print(f"{_analise} OK _o/")
        with open(arq1, "w") as f:
            f.write(f"{_analise} instalados _o/")
