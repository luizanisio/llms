#!/usr/bin/env python3
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Utilitário de exportação para GGUF usando Unsloth.

Este script é o ÚNICO lugar que ainda usa Unsloth, exclusivamente para
exportação GGUF. O pipeline de treinamento não depende mais dele.

Uso:
    python exportar_gguf.py <modelo_dir> [--quantization q4_k_m|q8_0|f16]

Exemplos:
    python exportar_gguf.py ./modelos/meu_modelo_merged --quantization q4_k_m
    python exportar_gguf.py ./modelos/meu_modelo_lora --quantization q8_0 --merge
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️ Aviso: Unsloth não está instalado.")
    print("   Para exportar para GGUF, instale: pip install unsloth")


def exportar_gguf(
    model_dir: str,
    output_dir: str = None,
    quantization: str = "q4_k_m",
    merge_first: bool = False,
    max_seq_length: int = 4096,
):
    """Exporta modelo para formato GGUF usando Unsloth.

    Args:
        model_dir: Diretório do modelo (pode ser LoRA ou merged)
        output_dir: Diretório de saída (padrão: model_dir + "_gguf_<quant>")
        quantization: Método de quantização (q4_k_m, q8_0, f16, etc.)
        merge_first: Se True, faz merge de adaptadores LoRA antes de exportar
        max_seq_length: Tamanho máximo de sequência
    """
    if not UNSLOTH_AVAILABLE:
        print("❌ Erro: Unsloth é necessário para exportação GGUF.")
        print("   Instale com: pip install unsloth")
        sys.exit(1)

    model_dir = os.path.abspath(model_dir)

    if not os.path.exists(model_dir):
        print(f"❌ Erro: Diretório não encontrado: {model_dir}")
        sys.exit(1)

    # Verifica se é modelo LoRA
    is_lora = os.path.exists(os.path.join(model_dir, "adapter_config.json"))

    if is_lora and not merge_first:
        print("⚠️ Aviso: Este parece ser um modelo LoRA (adaptadores).")
        print("   Para exportar GGUF, você precisa de um modelo completo (merged).")
        print("   Use --merge para mesclar automaticamente antes de exportar.")
        sys.exit(1)

    # Define diretório de saída
    if output_dir is None:
        output_dir = f"{model_dir}_gguf_{quantization}"

    output_dir = os.path.abspath(output_dir)

    print("="*80)
    print("📦 EXPORTAÇÃO PARA GGUF (usando Unsloth)")
    print("="*80)
    print(f"  Modelo:       {model_dir}")
    print(f"  Saída:        {output_dir}")
    print(f"  Quantização:  {quantization}")
    print(f"  É LoRA:       {is_lora}")
    print(f"  Merge first:  {merge_first}")
    print("="*80 + "\n")

    # Carrega modelo
    print("🔄 Carregando modelo...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,  # Para exportação, carregar em precisão completa
        )
        print("✅ Modelo carregado com sucesso!\n")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        sys.exit(1)

    # Merge se necessário
    if merge_first and is_lora:
        print("🔄 Mesclando adaptadores LoRA ao modelo base...")
        try:
            # Unsloth faz merge automaticamente ao salvar GGUF se for modelo PEFT
            print("   (O Unsloth fará merge automaticamente durante exportação)")
        except Exception as e:
            print(f"❌ Erro ao mesclar: {e}")
            sys.exit(1)

    # Exporta para GGUF
    print(f"\n💾 Exportando para GGUF ({quantization})...")
    print("   ⚠️ Isso pode demorar e consumir memória significativa.")
    print("   ⏳ Aguarde...\n")

    try:
        model.save_pretrained_gguf(
            output_dir,
            tokenizer,
            quantization_method=quantization,
        )
        print(f"\n✅ Exportação concluída com sucesso!")
        print(f"📂 Modelo GGUF salvo em: {output_dir}")

        # Lista arquivos gerados
        if os.path.exists(output_dir):
            gguf_files = [f for f in os.listdir(output_dir) if f.endswith('.gguf')]
            if gguf_files:
                print(f"\n📄 Arquivos gerados:")
                for f in gguf_files:
                    file_path = os.path.join(output_dir, f)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   - {f} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"\n❌ Erro durante exportação: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print("🎉 Exportação GGUF finalizada!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Exporta modelos para formato GGUF usando Unsloth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Exportar modelo merged para GGUF Q4_K_M (padrão)
  python exportar_gguf.py ./modelos/meu_modelo_merged

  # Exportar com quantização Q8_0 (maior qualidade)
  python exportar_gguf.py ./modelos/meu_modelo_merged --quantization q8_0

  # Exportar modelo LoRA (faz merge primeiro)
  python exportar_gguf.py ./modelos/meu_modelo_lora --merge --quantization q4_k_m

Métodos de quantização suportados:
  - q4_k_m   : 4-bit, balanceado (recomendado, ~3.5GB para 7B)
  - q8_0     : 8-bit, alta qualidade (~7GB para 7B)
  - f16      : 16-bit, máxima qualidade (~14GB para 7B)
  - q4_0     : 4-bit, mais compacto (~3.2GB para 7B)
  - q5_k_m   : 5-bit, balanceado (~4.5GB para 7B)

Nota: Este é o ÚNICO script que ainda usa Unsloth. O pipeline de treinamento
não depende mais dele.
        """
    )

    parser.add_argument(
        "model_dir",
        help="Diretório do modelo a exportar (pode ser LoRA ou merged)"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_dir",
        default=None,
        help="Diretório de saída (padrão: <model_dir>_gguf_<quantization>)"
    )

    parser.add_argument(
        "-q", "--quantization",
        default="q4_k_m",
        choices=["q4_k_m", "q8_0", "f16", "q4_0", "q5_k_m", "q5_0", "q6_k"],
        help="Método de quantização GGUF (padrão: q4_k_m)"
    )

    parser.add_argument(
        "--merge",
        action="store_true",
        help="Se modelo for LoRA, mescla adaptadores antes de exportar"
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Tamanho máximo de sequência (padrão: 4096)"
    )

    args = parser.parse_args()

    exportar_gguf(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        quantization=args.quantization,
        merge_first=args.merge,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
