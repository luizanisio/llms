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
    relatorio_linhas.append("| Etapa (Alias) | Arquivo CSV | Filtro Ativo | Treino (Qtd) | Validação (Qtd) | Teste (Qtd) | Ignorados/Erros (Qtd) |")
    relatorio_linhas.append("|---|---|---|---|---|---|---|")

    total_etapas = len(etapas)
    
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
        
        relatorio_linhas.append(f"| {alias} | {arquivo_nome} | `{str_filtro}` | {qtd_treino} | {qtd_val} | {qtd_teste} | {qtd_ignorados} |")
        
        if print_console:
            print_cores(f"   <verde>✓</verde> Etapa '{alias}': Treino={qtd_treino}, Validação={qtd_val}, Teste={qtd_teste}, Ignorados={qtd_ignorados}", color_auto=False)

    # Simula Avaliação Global
    relatorio_linhas.append("\n## Avaliação Global\n")
    if total_etapas > 1 and yaml_config.treinamento.eval_global:
        # Prepara a divisão completa
        try:
            divisao_unificada = dataset_manager.carregar_divisao_completa(yaml_config.curriculum)
            yaml_config.curriculum_config.divisao.dataset_filtro = global_filtro_divisao
            
            msg_global = dataset_manager.carregar_mensagens_de_pastas(alvo="validacao", divisao=divisao_unificada)
            qtd_global = len(msg_global)
            
            str_filtro_g = json.dumps(global_filtro_divisao, ensure_ascii=False) if global_filtro_divisao else "(Sem filtro)"
            
            relatorio_linhas.append(f"- **Filtro Global Ativo**: `{str_filtro_g}`")
            relatorio_linhas.append(f"- **Total de Instâncias (Validação Unificada)**: {qtd_global}")
            
            if print_console:
                print_cores(f"   <verde>✓</verde> Eval Global Unificado: Validação={qtd_global}", color_auto=False)
        except Exception as e:
            relatorio_linhas.append(f"- Erro ao computar avaliação global: {str(e)}")
            logger.warning(f"Erro no eval global: {e}")
    else:
        relatorio_linhas.append("- *Avaliação global desativada ou não aplicável (apenas 1 etapa no currículo).*")
        if print_console:
            print_cores("   <cinza>ℹ Eval Global: N/A</cinza>", color_auto=False)
            
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
