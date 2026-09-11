#!/usr/bin/env python3
"""
Consolida os gráficos de treinamento de um experimento.

Pode ser executado:
  - Da pasta do experimento (ex: pubmed-experimento/)  → busca em ./treinos/
  - Da pasta treinos/ diretamente                      → busca em ./

Funcionalidades:
  1. Detecta YAMLs de treinamento (contêm chaves 'modelo' e 'treinamento')
  2. Oferece regenerar os gráficos antes de consolidar (via treinar_unsloth.py --graficos)
  3. Copia PNGs das subpastas <pasta>/treinamento/ para graficos/(<protocolo>)<arquivo>.png

Exemplo de renomeação:
  Qwen2.5-1.5B-Instruct(b)/treinamento/uso_hardware_grafico.png
  → graficos/(b)uso_hardware_grafico.png
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def extrair_protocolo(nome_pasta: str) -> str | None:
    """Extrai o texto entre parênteses no nome da pasta. Ex: 'Qwen2.5-1.5B-Instruct(d1)' → 'd1'."""
    match = re.search(r"\(([^)]+)\)$", nome_pasta)
    return match.group(1) if match else None


def detectar_pasta_treinos(cwd: Path) -> Path | None:
    """Detecta a pasta que contém as subpastas de treinamento.

    Retorna o caminho se encontrar subpastas com protocolo (parênteses),
    verificando primeiro o CWD e depois ./treinos/.
    """
    for candidata in [cwd, cwd / "treinos"]:
        if not candidata.is_dir():
            continue
        for item in candidata.iterdir():
            if item.is_dir() and extrair_protocolo(item.name):
                return candidata
    return None


def detectar_pasta_experimento(cwd: Path, pasta_treinos: Path) -> Path:
    """Retorna a pasta do experimento (onde ficam os YAMLs).

    Se estamos na pasta treinos/, o experimento é o pai.
    Se estamos na pasta do experimento, é o próprio CWD.
    """
    if cwd == pasta_treinos:
        return pasta_treinos.parent
    return cwd


def _eh_yaml_treinamento(caminho: Path) -> bool:
    """Verifica se um YAML é de treinamento checando as chaves 'modelo' e 'treinamento'."""
    if yaml is None:
        # Fallback sem PyYAML: busca por linhas de nível raiz
        try:
            texto = caminho.read_text(encoding="utf-8")
            tem_modelo = bool(re.search(r"^modelo:", texto, re.MULTILINE))
            tem_treinamento = bool(re.search(r"^treinamento:", texto, re.MULTILINE))
            return tem_modelo and tem_treinamento
        except Exception:
            return False

    try:
        with open(caminho, encoding="utf-8") as f:
            dados = yaml.safe_load(f)
        if not isinstance(dados, dict):
            return False
        return "modelo" in dados and "treinamento" in dados
    except Exception:
        return False


def listar_yamls_treinamento(pasta_experimento: Path) -> list[Path]:
    """Retorna lista de YAMLs de treinamento encontrados na pasta do experimento."""
    yamls = []
    for arq in sorted(pasta_experimento.glob("*.yaml")):
        if _eh_yaml_treinamento(arq):
            yamls.append(arq)
    return yamls


def regenerar_graficos(yamls: list[Path], pasta_experimento: Path) -> None:
    """Executa treinar_unsloth.py --graficos para cada YAML."""
    script = Path(__file__).resolve().parent / "treinar_unsloth.py"
    if not script.exists():
        print(f"  ⚠️  Script não encontrado: {script}")
        return

    for i, yaml_path in enumerate(yamls, 1):
        nome = yaml_path.name
        print(f"\n  [{i}/{len(yamls)}] {nome}")
        try:
            resultado = subprocess.run(
                [sys.executable, str(script), str(yaml_path), "--graficos"],
                cwd=str(pasta_experimento),
                timeout=300,
            )
            if resultado.returncode != 0:
                print(f"    ⚠️  Saiu com código {resultado.returncode}")
        except subprocess.TimeoutExpired:
            print(f"    ⚠️  Timeout ao processar {nome}")
        except Exception as e:
            print(f"    ❌ Erro: {e}")


def listar_pngs(base: Path) -> list[tuple[Path, str]]:
    """Retorna lista de (caminho_png, novo_nome) para todos os PNGs encontrados."""
    arquivos = []
    for pasta_treino in sorted(base.iterdir()):
        if not pasta_treino.is_dir():
            continue
        protocolo = extrair_protocolo(pasta_treino.name)
        if protocolo is None:
            continue
        pasta_treinamento = pasta_treino / "treinamento"
        if not pasta_treinamento.is_dir():
            continue
        for png in sorted(pasta_treinamento.glob("*.png")):
            novo_nome = f"({protocolo}){png.name}"
            arquivos.append((png, novo_nome))
    return arquivos


def main():
    cwd = Path.cwd()
    base = detectar_pasta_treinos(cwd)

    if base is None:
        print("Erro: nenhuma pasta de treinamento encontrada.")
        print(f"  Diretório atual: {cwd}")
        print("  Execute este script da pasta do experimento ou da pasta treinos/.")
        return

    pasta_experimento = detectar_pasta_experimento(cwd, base)

    # --- Fase 1: Detectar YAMLs e oferecer regeneração ---
    yamls = listar_yamls_treinamento(pasta_experimento)

    if yamls:
        print(f"\n📋 {len(yamls)} YAML(s) de treinamento encontrado(s) em {pasta_experimento.name}/:")
        for y in yamls:
            print(f"   • {y.name}")

        resposta = input("\nRegerar gráficos antes de consolidar? (s/N): ").strip().lower()
        if resposta in ("s", "sim", "y", "yes"):
            print(f"\n📈 Regenerando gráficos...")
            regenerar_graficos(yamls, pasta_experimento)
            print(f"\n✅ Regeneração concluída.")
    else:
        print(f"\nℹ️  Nenhum YAML de treinamento encontrado em {pasta_experimento.name}/.")

    # --- Fase 2: Consolidar PNGs ---
    destino = base / "graficos"
    arquivos = listar_pngs(base)

    print(f"\n📂 Consolidação de gráficos:")
    print(f"   Origem:   {base}")
    print(f"   Destino:  {destino}")
    print(f"   Arquivos: {len(arquivos)} PNGs encontrados")

    if not arquivos:
        print("   Nenhum arquivo para copiar.")
        return

    input("\nPressione Enter para consolidar...")

    # Limpar e recriar a pasta destino
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir()

    for png, novo_nome in arquivos:
        shutil.copy2(png, destino / novo_nome)

    print(f"\n✅ {len(arquivos)} arquivos copiados com sucesso.")


if __name__ == "__main__":
    main()
