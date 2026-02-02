#!/usr/bin/env python3
"""
Autor: Luiz Anisio 02/2026

Módulo de dicas para o pacote treinar_unsloth.py
Contém constantes e funções para injeção de dicas em arquivos YAML de configuração.

Funções:
    - injetar_dicas_yaml: Injeta comentários de dicas no conteúdo YAML.
"""

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Dicas e Comentários para os Templates YAML
# ---------------------------------------------------------------------------

DICAS_YAML = {
    # Seções Principais
    "formatos": """#| =========================================================================
#| FORMATOS
#| =========================================================================
#| Configuração dos formatos de entrada e saída de dados.""",
    
    "dataset": """#| =========================================================================
#| DATASET (PARQUET)
#| =========================================================================
#| Configuração para carregamento de dados via arquivos parquet.""",
    
    "pastas": """#| =========================================================================
#| DATASET (PASTAS)
#| =========================================================================
#| Configuração para carregamento de dados via estrutura de pastas (arquivos txt/json).""",
    
    "modelo": """#| =========================================================================
#| MODELO
#| =========================================================================
#| Configurações do modelo base e diretório de saída.""",
    
    "treinamento": """#| =========================================================================
#| TREINAMENTO
#| =========================================================================
#| Configurações de hiperparâmetros. Defaults do Unsloth são otimizados.
#| Guia: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide""",
    
    "lora": """#| =========================================================================
#| LORA (Low-Rank Adaptation)
#| =========================================================================
#| Configurações do adaptador LoRA. QLoRA (4-bit) economiza 75% VRAM vs LoRA (16-bit).""",
    
    "misc": """#| =========================================================================
#| MISCELÂNEA
#| =========================================================================
#| Configurações diversas (logs, variáveis de ambiente).""",

    # Dicas Específicas - Treinamento
    "treinamento/eval_steps": "#| Frequência de avaliação (ex: '15%' das steps ou número inteiro).",
    "treinamento/batch_size": """#| Amostras por GPU. Menor gasta menos VRAM.
#| Para evitar OOM, use 1 ou 2 e aumente grad_batch_size.
#| [7B] Com 4-bit (RTX 3060): use 1.
#| [H100] Pode usar 8-16.""",
    "treinamento/grad_batch_size": """#| Acumulação de gradiente. Aumenta tempo de treino mas economiza VRAM.
#| Batch Efetivo = batch_size * grad_batch_size * num_gpus (Recomendado: 16).
#| [7B] Aumente para 20-40 se batch_size=1.""",
    "treinamento/num_train_epochs": """#| Passadas no dataset. Recomendado: 1-3 épocas.
#| Mais que 3 pode causar overfitting (memorização) em instruct datasets.
#| [H100] Mais poder = mais épocas viáveis; pode experimentar 5-10.""",
    "treinamento/max_seq_length": """#| Contexto máximo. Qwen/Llama suportam long context, mas 2048-4096 é comum.
#| Valor maior consome mais VRAM.
#| [7B] Suporta até 128k, mas 4096-8192 é prático.""",
    "treinamento/learning_rate": """#| Taxa de aprendizado. QLoRA/LoRA: start 2e-4. RL (DPO): 5e-6.
#| Se overfitting/loss < 0.2: Reduzir. Se underfitting: Aumentar.
#| [7B] Modelos maiores são mais sensíveis; force 1e-4.""",
    "treinamento/nbits": """#| Quantização: 4 (QLoRA) economiza muita memória. 16 (LoRA) para máxima precisão.
#| [H100] Pode usar 8 ou 16 (Full Precision) para máxima qualidade.""",
    "treinamento/train_on_responses_only": "#| True: treina apenas nas respostas (recomendado para chat). Aumenta acurácia.",
    "treinamento/warmup_steps": """#| Passos de aquecimento. Recomendado: 5-10% do total de steps.
#| [7B] Aumentar para 10-20 steps em modelos maiores.""",
    "treinamento/weight_decay": "#| Decaimento de peso (regularização). Padrão 0.01.",
    "treinamento/optim": """#| Otimizador. Padrão Unsloth: 'adamw_8bit'.
#| Outras opções: 'adamw_torch', 'paged_adamw_8bit'.""",
    "treinamento/lr_scheduler_type": """#| Scheduler. 'linear': decaimento constante (estável).
#| 'cosine': decaimento suave (melhor convergência final em >1 época).""",
    
    # Dicas Específicas - Modelo
    "modelo/base_model_name": """#| Slug HuggingFace ou caminho local.
#| [7B] Ex: Qwen/Qwen2.5-7B-Instruct (requer ~24GB VRAM em 4-bit)""",
    "modelo/saida": "#| Diretório de saída para o adaptador LoRA.",

    # Dicas Específicas - LoRA
    "lora/r": """#| Rank: 8, 16, 32, 64, 128. Recomendado: 16 ou 32.
#| Maior rank = mais capacidade, mas usa mais VRAM e pode overfitar.
#| [7B/H100] Pode aumentar para 32, 64 ou 128.""",
    "lora/alpha": """#| Alpha: Escala do ajuste. Recomendado: igual a r ou 2*r.
#| Ex: r=16 -> alpha=32.""",
    "lora/target_modules": "#| Módulos alvo. Recomendado: todos os lineares (q,k,v,o,gate,up,down) para melhor qualidade.",
    "lora/dropout": "#| Regularização. Unsloth otimizado para 0, mas use 0.1 se notar overfitting.",
}


def injetar_dicas_yaml(yaml_content: str, dicas: Dict[str, str]) -> str:
    """
    Injeta comentários/dicas no conteúdo YAML gerado.
    Substitui comentários iniciados com '#|' e preserva outros.
    
    Args:
        yaml_content: String contendo o YAML gerado.
        dicas: Dicionário mapeando chaves ou caminhos (ex: 'treinamento/batch_size') para comentários.
               Os comentários devem iniciar com '#|'.
        
    Returns:
        String do YAML com os comentários inseridos/atualizados.
    """
    lines = yaml_content.split('\n')
    new_lines = []
    stack = [] # [(indent, key)]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        
        # Se for linha vazia, mantém
        if not stripped:
            new_lines.append(line)
            i += 1
            continue
            
        # Se for comentário
        if stripped.startswith('#'):
            # Se for comentário de dica (#|), ignoramos pois será reinjetado se houver dica
            if stripped.startswith('#|'):
                i += 1
                continue
            else:
                # Comentário normal do usuário, mantém
                new_lines.append(line)
                i += 1
                continue

        indent = len(line) - len(stripped)
        
        # Ajusta a pilha baseada na indentação atual
        while stack and stack[-1][0] >= indent:
            stack.pop()
            
        # Tenta identificar uma chave YAML na linha
        if ':' in stripped:
            key_part = stripped.split(':', 1)[0].strip()
            # Remove aspas se houver
            key_part = key_part.strip("'\"")
            
            # Reconstrói o caminho atual (ex: treinamento/batch_size)
            curr_path = "/".join([item[1] for item in stack] + [key_part])
            
            # Busca dica pelo caminho completo ou apenas pela chave
            dica = dicas.get(curr_path) or dicas.get(key_part)
            
            if dica:
                # Prepara indentação do comentário
                indent_str = " " * indent
                dica_lines = dica.strip().split('\n')
                for d_line in dica_lines:
                    # Garante que começa com #| e tem espaço
                    clean_d_line = d_line.strip()
                    if not clean_d_line.startswith('#|'):
                         clean_d_line = '#| ' + clean_d_line.lstrip('#').lstrip()
                    new_lines.append(f"{indent_str}{clean_d_line}")
            
            stack.append((indent, key_part))
            
        new_lines.append(line)
        i += 1
        
    return '\n'.join(new_lines)
