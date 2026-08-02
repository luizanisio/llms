#!/usr/bin/env python3
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Módulo dedicado a inspecionar os datasets do currículo sem iniciar o treinamento.
Gera um relatório markdown contendo as quantidades de linhas, filtros aplicados
e origens dos dados (dry-run).
"""

import os
import copy
import json
from typing import Optional

# Importação de módulos do framework
import util  # garante o PYTHONPATH correto
from treinar_unsloth_util import YamlTreinamento
from treinar_unsloth_dataset import DatasetTreinamento
from treinar_unsloth_logging import get_logger, log_separador
from util_print import print_cores

logger = get_logger(__name__)

def gerar_relatorio_datasets(yaml_path: str, print_console: bool = True) -> Optional[str]:
    """
    Simula o carregamento dos datasets para contabilizar as instâncias.
    Gera relatorio_datasets.md na pasta de saída do modelo.
    """
    if print_console:
        logger.info("\n")
        log_separador(caractere="=", largura=80)
        logger.info("<azul>📊 GERANDO RELATÓRIO DE DATASETS (DRY-RUN)</azul>")
        log_separador(caractere="=", largura=80)

    try:
        # Carregamos com validar_caminhos=False se apenas quisermos simular, 
        # mas como os dados serão lidos, eles devem existir.
        yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    except Exception as e:
        logger.error(f"❌ Erro ao ler {yaml_path}: {e}")
        return None

    # Pasta de destino
    output_dir = yaml_config.treinamento_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    arquivo_saida = os.path.join(output_dir, "relatorio_datasets.md")
    
    # Extrai pipeline
    etapas = yaml_config.curriculum_treino
    if not etapas:
        logger.warning("Nenhuma etapa de treinamento configurada.")
        return None
        
    dataset_manager = yaml_config.dataset_manager
    global_filtro_divisao = copy.deepcopy(yaml_config.curriculum_config.divisao.dataset_filtro)
    
    relatorio_linhas = []
    relatorio_linhas.append(f"# Relatório de Datasets do Treinamento")
    relatorio_linhas.append(f"**Configuração**: `{yaml_path}`\n")
    relatorio_linhas.append(f"**Modelo de Saída**: `{yaml_config.modelo.saida}`\n")
    
    # Tabela principal de etapas
    relatorio_linhas.append("## Etapas do Curriculum\n")
    relatorio_linhas.append("| Etapa (Alias) | Arquivo CSV | Filtro Ativo | Max Seq Length | Filtra Excesso | Treino (Qtd) | Validação (Qtd) | Teste (Qtd) | Ignorados/Erros (Qtd) |")
    relatorio_linhas.append("|---|---|---|---|---|---|---|---|---|")

    total_etapas = len(etapas)
    contagens_treino = []  # instâncias de treino por etapa (usado pela seção do modo FUSÃO)

    for idx, etapa in enumerate(etapas):
        alias = etapa.alias or f"Etapa {idx}"
        
        # Simula o arquivo e filtro da etapa
        if etapa.arquivo:
            yaml_config.curriculum_config.divisao.arquivo = etapa.arquivo
            
        filtro_atual = etapa.dataset_filtro if etapa.dataset_filtro is not None else global_filtro_divisao
        yaml_config.curriculum_config.divisao.dataset_filtro = filtro_atual
        
        # Força limpeza do cache do CSV para forçar releitura com a configuração da etapa
        dataset_manager._dados_divisao = None
        
        # Carrega dados
        try:
            msg_treino = dataset_manager.carregar_mensagens_de_pastas(alvo="treino")
            qtd_treino = len(msg_treino)
        except Exception as e:
            logger.warning(f"Erro no treino da etapa {alias}: {e}")
            qtd_treino = "ERRO"
            
        try:
            msg_val = dataset_manager.carregar_mensagens_de_pastas(alvo="validacao")
            qtd_val = len(msg_val)
        except Exception as e:
            logger.warning(f"Erro na validacao da etapa {alias}: {e}")
            qtd_val = "ERRO"
            
        try:
            msg_teste = dataset_manager.carregar_mensagens_de_pastas(alvo="teste")
            qtd_teste = len(msg_teste)
        except Exception as e:
            logger.warning(f"Erro no teste da etapa {alias}: {e}")
            qtd_teste = "ERRO"
            
        # Calcular ignorados (arquivos da divisão que não viraram instâncias válidas)
        try:
            df_divisao = dataset_manager.carregar_ou_criar_divisao()
            qtd_total_divisao = len(df_divisao)
            
            soma_carregados = (qtd_treino if isinstance(qtd_treino, int) else 0) + \
                              (qtd_val if isinstance(qtd_val, int) else 0) + \
                              (qtd_teste if isinstance(qtd_teste, int) else 0)
            
            qtd_ignorados = qtd_total_divisao - soma_carregados
            qtd_ignorados = max(0, qtd_ignorados)
        except Exception as e:
            logger.warning(f"Erro ao calcular ignorados da etapa {alias}: {e}")
            qtd_ignorados = "ERRO"
            
        # Formatação
        arquivo_nome = os.path.basename(etapa.arquivo) if etapa.arquivo else "N/A"
        str_filtro = json.dumps(filtro_atual, ensure_ascii=False) if filtro_atual else "(Sem filtro)"
        
        msl_val = etapa.max_seq_length if (hasattr(etapa, "max_seq_length") and etapa.max_seq_length > 0) else yaml_config.treinamento.max_seq_length
        str_msl = str(msl_val) if msl_val > 0 else "Auto"
        filtrar_msl = "Sim" if getattr(yaml_config.treinamento, 'filtrar_max_seq_length', False) else "Não"
        
        relatorio_linhas.append(f"| {alias} | {arquivo_nome} | `{str_filtro}` | {str_msl} | {filtrar_msl} | {qtd_treino} | {qtd_val} | {qtd_teste} | {qtd_ignorados} |")

        contagens_treino.append(qtd_treino if isinstance(qtd_treino, int) else 0)

        if print_console:
            print_cores(f"   <verde>✓</verde> Etapa '{alias}': Treino={qtd_treino}, Validação={qtd_val}, Teste={qtd_teste}, Ignorados={qtd_ignorados}", color_auto=False)

    # --- Modo FUSÃO: lista os spans do stream concatenado (dry-run) ---
    fusao = getattr(yaml_config.curriculum_config, "fusao", None)
    if fusao and fusao.ativo:
        from treinar_unsloth_fusao import corte_da_etapa
        import torch as _torch
        n_gpus = max(_torch.cuda.device_count(), 1) if _torch.cuda.is_available() else 1
        batch_efetivo = (yaml_config.treinamento.batch_size *
                         yaml_config.treinamento.grad_batch_size * n_gpus)

        relatorio_linhas.append("\n## Modo FUSÃO (execução contínua)\n")
        relatorio_linhas.append(f"- **Tipo (regime único do run)**: `{fusao.tipo}`")
        relatorio_linhas.append(f"- **learning_rate (pico do cosine)**: {fusao.learning_rate}")
        relatorio_linhas.append(f"- **warmup_grupo_steps**: {fusao.warmup_grupo_steps}")
        relatorio_linhas.append(f"- **seed_shuffle**: {fusao.seed_shuffle}")
        relatorio_linhas.append(f"- **Batch efetivo**: {batch_efetivo} "
                                f"(batch {yaml_config.treinamento.batch_size} × grad "
                                f"{yaml_config.treinamento.grad_batch_size} × {n_gpus} GPU(s))\n")
        relatorio_linhas.append("| Span | Alias | Instâncias | pace_epochs | Stream [início, fim) | step_início (~) | unfreeze_corte |")
        relatorio_linhas.append("|---|---|---|---|---|---|---|")
        inst_acum = 0
        for idx, (etapa, qtd) in enumerate(zip(etapas, contagens_treino)):
            inicio = inst_acum
            inst_acum += qtd * max(1, etapa.pace_epochs)
            corte = corte_da_etapa(etapa)
            corte_str = (f"{etapa.unfreeze_layers_pct:g}%" if etapa.unfreeze_layers_pct >= 0
                         else (str(etapa.unfreeze_layers_from) if etapa.unfreeze_layers_from >= 0 else "—"))
            relatorio_linhas.append(
                f"| {idx} | {etapa.alias} | {qtd} | {etapa.pace_epochs} | "
                f"[{inicio}, {inst_acum}) | {inicio // batch_efetivo} | {corte_str} |"
            )
        tem_gating = any(corte_da_etapa(e) >= 0 for e in etapas)
        relatorio_linhas.append(f"\n- **Total do stream (Σ instâncias × pace)**: {inst_acum}")
        relatorio_linhas.append(f"- **Optimizer steps (~)**: {inst_acum // batch_efetivo}")
        relatorio_linhas.append(f"- **Gating de LR por grupos**: {'sim (cortes declarados)' if tem_gating else 'não (sem unfreeze — otimizador padrão)'}")

        if print_console:
            print_cores(f"   <azul>🧬 Modo FUSÃO ({fusao.tipo}): {len(etapas)} span(s), "
                        f"{inst_acum} instâncias no stream, ~{inst_acum // batch_efetivo} steps, "
                        f"gating={'sim' if tem_gating else 'não'}</azul>", color_auto=False)

    # Simula Avaliação Global
    relatorio_linhas.append("\n## Avaliação Global\n")
    if total_etapas > 1 and yaml_config.treinamento.eval_global:
        # Usa a mesma função que o treino real para decidir se eval global é necessário
        check = dataset_manager.verificar_eval_global_necessario(yaml_config.curriculum)
        
        if check["necessario"]:
            status_str = "✅ **ATIVO**"
        else:
            status_str = "❌ **DESATIVADO**"
        
        relatorio_linhas.append(f"- **Status**: {status_str}")
        relatorio_linhas.append(f"- **Motivo**: {check['motivo']}")
        
        # Detalha validação por etapa
        if check["ids_por_etapa"]:
            relatorio_linhas.append(f"- **IDs de validação globais (união)**: {len(check['ids_global'])}")
            relatorio_linhas.append("")
            relatorio_linhas.append("| Etapa | Validação (IDs) |")
            relatorio_linhas.append("|---|---|")
            for alias, ids in check["ids_por_etapa"].items():
                relatorio_linhas.append(f"| {alias} | {len(ids)} |")
            relatorio_linhas.append(f"| **Global (união)** | **{len(check['ids_global'])}** |")
        
        # Carrega contagem real da divisão unificada
        try:
            divisao_unificada = dataset_manager.carregar_divisao_completa(yaml_config.curriculum)
            yaml_config.curriculum_config.divisao.dataset_filtro = global_filtro_divisao
            msg_global = dataset_manager.carregar_mensagens_de_pastas(alvo="validacao", divisao=divisao_unificada)
            relatorio_linhas.append(f"\n- **Instâncias carregadas (validação unificada)**: {len(msg_global)}")
        except Exception as e:
            relatorio_linhas.append(f"\n- Erro ao computar validação unificada: {str(e)}")
        
        if print_console:
            emoji = "✅" if check["necessario"] else "❌"
            print_cores(f"   {emoji} Eval Global: {check['motivo']}", color_auto=False)
            if check["ids_por_etapa"]:
                for alias, ids in check["ids_por_etapa"].items():
                    print_cores(f"      → {alias}: {len(ids)} IDs de validação", color_auto=False)
                print_cores(f"      → Global (união): {len(check['ids_global'])} IDs", color_auto=False)
    else:
        if total_etapas <= 1:
            motivo = "apenas 1 etapa no currículo"
        else:
            motivo = "desativado via YAML (treinamento.eval_global: false)"
        relatorio_linhas.append(f"- *Avaliação global não aplicável: {motivo}.*")
        if print_console:
            print_cores(f"   <cinza>ℹ Eval Global: {motivo}</cinza>", color_auto=False)
            
    # Restaura configuração para não alterar estado do objeto (caso continue)
    yaml_config.curriculum_config.divisao.dataset_filtro = global_filtro_divisao
    dataset_manager._dados_divisao = None

    # Grava Relatório
    conteudo = "\n".join(relatorio_linhas) + "\n"
    try:
        with open(arquivo_saida, "w", encoding="utf-8") as f:
            f.write(conteudo)
        if print_console:
            log_separador(caractere="-", largura=80)
            logger.info(f"<verde>✅ Relatório de datasets gerado com sucesso em: {arquivo_saida}</verde>")
    except Exception as e:
        logger.error(f"❌ Erro ao salvar relatório em {arquivo_saida}: {e}")
        
    return arquivo_saida

if __name__ == "__main__":
    # Teste rápido se chamado diretamente
    import sys
    if len(sys.argv) > 1:
        cfg = sys.argv[1]
        gerar_relatorio_datasets(cfg)
    else:
        print("Uso: treinar_unsloth_datasets_relatorio.py <arquivo.yaml>")
