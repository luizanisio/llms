import argparse
import shutil
import os
import sys
try:
  from huggingface_hub import snapshot_download
except ImportError:
  print("Erro: O pacote 'huggingface_hub' não foi instalado.")
  print("Instale com: pip install huggingface_hub")
  sys.exit(1)  

def zip_reduzido(nome_pasta, nome_arquivo):
    print(f'Compactando {nome_pasta}...', flush=True)
    shutil.make_archive(nome_pasta, 'zip', nome_arquivo)
    print(f'Compactado! {nome_arquivo}', flush=True)

if __name__ == '__main__':
    desc = '''Baixar modelo do Hugging Face e opcionalmente zipar.

Exemplos de uso:
  python baixar.py --modelo Qwen/Qwen2.5-7B-Instruct
  python baixar.py --modelo Qwen/Qwen2.5-1.5B-Instruct
  python baixar.py --modelo Qwen/Qwen2.5-0.5B-Instruct
  python baixar.py --modelo Qwen/Qwen3.6-35B-A3B'''

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--pasta', default='.', help='Local onde será baixado o modelo (padrão: pasta atual)')
    parser.add_argument('--modelo', required=True, help='Path do modelo no hugging face')
    parser.add_argument('--zip', action='store_true', help='Zipar o modelo e remover a pasta ao final')

    import sys
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    repo_id = args.modelo
    drive = args.pasta
    model_name = repo_id.split('/')[-1]

    local_dir = os.path.join(drive, model_name)
    os.makedirs(local_dir, exist_ok=True)

    print(f'Baixando modelo {model_name} para a pasta {local_dir}')
    if args.zip:
        arquivo_zip = os.path.join(drive, model_name)
        print(f' - zipando ao final para {arquivo_zip}.zip e removendo pasta original')

    # Habilita suporte ao hf_transfer (download extremamente rápido em Rust) se o pacote estiver instalado
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        max_workers=16  # Aumenta a quantidade de conexões simultâneas (o padrão é bem menor)
    )

    if args.zip:
        print('Compactando...')
        arquivo_zip = os.path.join(drive, model_name)
        zip_reduzido(arquivo_zip, local_dir)
        print(f'Removendo pasta original {local_dir}...')
        shutil.rmtree(local_dir)

    print('ok')