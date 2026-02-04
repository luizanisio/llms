# -*- coding: utf-8 -*-
"""
Comparação de extrações de espelhos usando múltiplas métricas de similaridade e configuração via YAML.

Autor: Luiz Anísio
Refatorado: 25/01/2026
Descrição:
-----------
Compara dados extraídos por diferentes abordagens e modelos baseados em um arquivo de configuração YAML.
Suporta múltiplas métricas (BERTScore, ROUGE, Levenshtein) configuráveis por campo.
Gera planilhas Excel, CSVs de estatísticas e gráficos comparativos.

Uso:
    python comparar_extracoes.py config_summa.yaml
"""

import os
import sys
import argparse
import yaml
import regex as re
import pandas as pd

# ============================================================================
# PROTEÇÃO E SETUP INICIAL
# ============================================================================
# Adiciona paths de utilitários
sys.path.extend(['./utils', './src', '../../src'])

# Verificação para multiprocessing (BERTScore safe)
_IS_MAIN_PROCESS = __name__ == '__main__' or not hasattr(sys.modules.get('__mp_main__', None), '__file__')

def ler_configuracao(caminho_yaml):
    """Lê e valida o arquivo de configuração YAML."""
    if not os.path.exists(caminho_yaml):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {caminho_yaml}")
    
    with open(caminho_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"📖 Configuração carregada de: {caminho_yaml}")
    return config

def _inicializar_ambiente():
    """Inicializa ambiente (carrega .env e configurações globais)."""
    from util import UtilEnv
    UtilEnv.carregar_env('.env', pastas=['../', './'])
    
    # Recupera configurações de ambiente ou defaults
    max_workers = UtilEnv.get_int('MAX_WORKERS_ANALISE', 10)
    device_bert = UtilEnv.get_str('BERTSCORE_DEVICE', 'auto')
    
    return max_workers, device_bert

# ============================================================================
# FUNÇÕES DE ANÁLISE
# ============================================================================

def processar_analise_estatistica(dados_analise, pasta_saida, config):
    """
    Executa a análise estatística (LLM-as-a-Judge) baseada nos pares definidos no YAML.
    """
    if not config.get('execucao', {}).get('analise_estatistica', False):
        return

    print("\n📊 Iniciando Análise Estatística (LLM-as-a-Judge)...")
    
    try:
        from util_analise_estatistica import AnaliseEstatistica
    except ImportError:
        print("❌ Módulo util_analise_estatistica não encontrado.")
        return

    lista_dados = []
    pk = dados_analise.config.nome_campo_id
    
    rotulo_base = config['modelo_base']['rotulo']
    familia_base = config['modelo_base'].get('familia', 'Base')
    
    # Itera sobre os modelos de comparação definidos no YAML
    # Itera sobre os modelos de comparação definidos no YAML (respeitando flag ativo)
    modelos_ativos = [m for m in config.get('modelos_comparacao', []) if m.get('ativo', True)]
    
    for modelo in modelos_ativos:
        rotulo_agente = modelo['rotulo']
        nome_familia = modelo.get('familia', rotulo_agente)
        
        print(f"   Processando par: {familia_base} ({rotulo_base}) vs {nome_familia} ({rotulo_agente})...")
        
        if rotulo_base not in dados_analise.rotulos or rotulo_agente not in dados_analise.rotulos:
            print(f"      ⚠️  Saltando: Rótulos {rotulo_base} ou {rotulo_agente} não encontrados nos dados.")
            continue

        for item in dados_analise.dados:
            id_peca = item.get(pk)
            if not id_peca: continue
            
            tokens = dados_analise.get_tokens(id_peca)
            evals = dados_analise.get_avaliacao(id_peca)
            
            v1 = evals.get(f'{rotulo_base}_F1')
            v2 = evals.get(f'{rotulo_agente}_F1')
            
            c1 = tokens.get(f'{rotulo_base}_total', 1) or 1
            c2 = tokens.get(f'{rotulo_agente}_total', 1) or 1
            
            if v1 is not None and v2 is not None:
                lista_dados.append({
                    'valor1': v1,
                    'valor2': v2,
                    'custo1': c1,
                    'custo2': c2,
                    'familia': nome_familia
                })
    
    if not lista_dados:
        print("❌ Nenhum dado compatível encontrado para análise estatística.")
        return

    df_stat = pd.DataFrame(lista_dados)
    arquivo_saida = os.path.join(pasta_saida, 'relatorio_analise_estatistica.md')
    
    analise = AnaliseEstatistica(df_stat, config={
        'rotulo1': 'Base',
        'rotulo2': 'Agente',
        'arquivo_saida': arquivo_saida
    })
    analise.processar_analise()
    analise.salvar_relatorio()
    print(f"\n✅ Análise Estatística concluída e salva em: {arquivo_saida}")

def configurar_metricas(config_yaml):
    """Converte a configuração YAML para o formato esperado pelo JsonAnalise."""
    conf_comp = config_yaml.get('configuracao_comparacao', {})
    campos = conf_comp.get('campos', {})
    
    # Estrutura base
    config_final = {
        'nivel_campos': conf_comp.get('nivel_campos', 1),
        'padronizar_simbolos': conf_comp.get('padronizar_simbolos', True),
        'rouge_stemmer': conf_comp.get('rouge_stemmer', True),
        'campos_bertscore': campos.get('bertscore') or [],
        'campos_rouge': campos.get('rouge_l') or [],
        'campos_rouge2': campos.get('rouge_2') or [],
        'campos_rouge1': campos.get('rouge_1') or [],
        'campos_levenshtein': campos.get('levenshtein') or [],
        # Métricas SBERT (Sentence-BERT)
        'campos_sbert': campos.get('sbert') or [],
        'campos_sbert_pequeno': campos.get('sbert_pequeno') or [],
        'campos_sbert_medio': campos.get('sbert_medio') or [],
        'campos_sbert_grande': campos.get('sbert_grande') or []
    }
    
    # Ajuste para teste rápido (desativa BERTScore e SBERT grande)
    if config_yaml.get('execucao', {}).get('teste_rapido', False):
        print("⚠️  Modo TESTE RÁPIDO: Desabilitando BERTScore/SBERT e movendo campos para ROUGE-L.")
        # Move campos BERTScore para ROUGE-L
        campos_removidos = config_final['campos_bertscore']
        config_final['campos_bertscore'] = []
        for c in campos_removidos:
            if c not in config_final['campos_rouge']:
                config_final['campos_rouge'].append(c)
        # Move campos SBERT para ROUGE-L
        for sbert_key in ['campos_sbert', 'campos_sbert_pequeno', 'campos_sbert_medio', 'campos_sbert_grande']:
            campos_sbert = config_final[sbert_key]
            config_final[sbert_key] = []
            for c in campos_sbert:
                if c not in config_final['campos_rouge']:
                    config_final['campos_rouge'].append(c)
                
    return config_final

def extrair_campos_unicos(config_metricas):
    """Extrai lista única de todos os campos envolvidos na comparação, preservando ordem."""
    todos_campos = []
    seen = set()
    
    # Ordem de prioridade na visualização/processamento (inclui SBERT)
    metricas_ordem = ['campos_bertscore', 'campos_rouge', 'campos_rouge2', 'campos_rouge1', 'campos_levenshtein',
                      'campos_sbert', 'campos_sbert_pequeno', 'campos_sbert_medio', 'campos_sbert_grande']
    
    for chave in metricas_ordem:
        campos = config_metricas.get(chave, [])
        for c in campos:
            if c not in seen:
                seen.add(c)
                todos_campos.append(c)
    
    # Remove campos especiais como (global) que não existem no JSON
    return [c for c in todos_campos if not c.startswith('(')]

# ============================================================================
# MAIN
# ============================================================================

def resolver_caminho(caminho_relativo, base_dir):
    """Resolve caminhos relativos baseado no diretório do arquivo de configuração."""
    if os.path.isabs(caminho_relativo):
        return caminho_relativo
    return os.path.normpath(os.path.join(base_dir, caminho_relativo))

def _gerar_grafico_erros(dados_analise, pasta_saida):
    """Gera gráfico de barras empilhadas com status dos documentos por modelo."""
    from util_graficos import UtilGraficos, Cores
    import pandas as pd
    
    rotulos_modelos = dados_analise.rotulos_modelos
    stats = {m: {'Sucesso': 0, 'Erro': 0, 'Inexistente': 0} for m in rotulos_modelos}
    
    for linha in dados_analise.dados:
        for modelo in rotulos_modelos:
            val = linha.get(modelo)
            # Verifica status baseado no valor
            if val is None:
                stats[modelo]['Inexistente'] += 1
            elif isinstance(val, dict) and 'erro' in val:
                erro_msg = str(val['erro'])
                if 'Inexistente' in erro_msg:
                     stats[modelo]['Inexistente'] += 1
                else:
                     stats[modelo]['Erro'] += 1
            else:
                stats[modelo]['Sucesso'] += 1
                
    df_stats = pd.DataFrame(stats).transpose()
    
    # Ordena colunas para mapear na paleta RdYlGn (Red -> Green)
    # 1. Inexistente (Red)
    # 2. Erro (Yellow/Orange)
    # 3. Sucesso (Green)
    cols_ordem = ['Inexistente', 'Erro', 'Sucesso']
    # Garante que as colunas existem no DF
    cols_existentes = [c for c in cols_ordem] # Sempre cria as 3 para consistência de cor?
    
    # Garante que todas as colunas existem
    for c in cols_ordem:
        if c not in df_stats.columns:
            df_stats[c] = 0
            
    df_stats = df_stats[cols_ordem]
    
    arquivo = os.path.join(pasta_saida, 'status_modelos.png')
    UtilGraficos.gerar_grafico_empilhado(
        df_stats, 
        titulo='Status das Extrações por Modelo',
        ylabel='Quantidade de Documentos',
        xlabel='Modelo',
        arquivo_saida=arquivo,
        paleta_cores=Cores.RdYlGn
    )
    print(f"   ✓ Gráfico de status gerado: {os.path.basename(arquivo)}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comparador de Extrações JSON via YAML")
    parser.add_argument('config_file', help="Caminho do arquivo de configuração YAML")
    args = parser.parse_args()

    # 1. Carregar configuração
    caminho_yaml_abs = os.path.abspath(args.config_file)
    base_dir_yaml = os.path.dirname(caminho_yaml_abs)
    config = ler_configuracao(caminho_yaml_abs)
    
    # 2. Inicializar ambiente
    max_workers_env, _ = _inicializar_ambiente()
    max_workers = config['execucao'].get('max_workers', max_workers_env)
    
    # 3. Setup de Pastas e Rótulos (Resolvendo caminhos)
    pasta_saida_raw = config['saida']['pasta']
    pasta_saida = resolver_caminho(pasta_saida_raw, base_dir_yaml)
    
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
        
    modelo_base = config['modelo_base']
    modelos_comp_all = config['modelos_comparacao']
    
    # Filtra apenas modelos ativos (padrão: ativo=True)
    modelos_comp = [m for m in modelos_comp_all if m.get('ativo', True)]
    
    # Log de itens ignorados
    ignorados = [m['rotulo'] for m in modelos_comp_all if not m.get('ativo', True)]
    if ignorados:
        print(f"⚠️  Modelos ignorados explicitamente (ativo=false): {', '.join(ignorados)}")

    # Prepara listas para CargaDadosComparacao
    origem_raw = modelo_base['pasta']
    origem = resolver_caminho(origem_raw, base_dir_yaml)
    
    rotulo_origem = modelo_base.get('rotulo', 'BASE')
    # Read campo_id from YAML config (defined ahead of time for use later)
    rotulo_id = config.get('configuracao_comparacao', {}).get('nome_campo_id', 'id')
    
    pastas_destinos = [resolver_caminho(m['pasta'], base_dir_yaml) for m in modelos_comp]
    rotulos_destinos = [m['rotulo'] for m in modelos_comp]
    
    # Validações básicas
    if not os.path.isdir(origem):
        print(f"❌ Pasta base não encontrada: {origem}")
        sys.exit(1)
    
    # 4. Configuração das Métricas
    config_comparacao = configurar_metricas(config)
    campos_comparacao = extrair_campos_unicos(config_comparacao)
    
    print("\n⚙️  Configuração de Comparação:")
    print(f"   Origem (Base): {origem} [{rotulo_origem}]")
    print(f"   Destinos: {len(pastas_destinos)} pastas")
    print(f"   Campos: {campos_comparacao}")
    print(f"   Saída: {pasta_saida}")
    
    # 5. Carga de Dados
    # Imports tardios para evitar problemas de circularidade ou init desnecessário
    from util_json_carga import CargaDadosComparacao
    from util_json import JsonAnaliseDataFrame
    
    # Regex e Máscaras
    config_masks = config.get('configuracao_comparacao', {}).get('mascaras', {})
    
    def to_regex(val, is_extracao=False):
        if not val: return None
        if val.startswith('^'): return val
        # Se não é regex explícito, assume sufixo e cria regex capture group
        # Escape do sufixo para segurança
        return f"^(.+){re.escape(val)}$"

    mascara_extracao = config_masks.get('extracao', r'^(\d{12})\.\d+\.\d*\.json$')
    # Garante que extracao seja tratada como regex se não especificado (fallback legacy)
    # ou se user passar sufixo, converte.
    mascara_extracao = to_regex(mascara_extracao, is_extracao=True)
    
    mascara_tokens = to_regex(config_masks.get('tokens', '.json'))
    mascara_avaliacao = to_regex(config_masks.get('avaliacao', '.avaliacao.json'))
    mascara_observabilidade = to_regex(config_masks.get('observabilidade', '.obs.json'))

    print(f"\n🔍 Máscaras definidas:")
    print(f"   Extração: {mascara_extracao}")
    print(f"   Tokens:   {mascara_tokens}")
    print(f"   Aval.:    {mascara_avaliacao}")
    print(f"   Obs.:     {mascara_observabilidade}")

    carga = CargaDadosComparacao(
        pasta_origem=origem,
        pastas_destinos=pastas_destinos,
        rotulo_id=rotulo_id,
        rotulo_origem=rotulo_origem,
        rotulos_destinos=rotulos_destinos,
        campos_comparacao=campos_comparacao,
        mascara_extracao=mascara_extracao,
        mascara_tokens=mascara_tokens,
        mascara_avaliacao=mascara_avaliacao,
        mascara_observabilidade=mascara_observabilidade,
        pasta_log_erros=pasta_saida,
        ignorar_erro_extracao=config['execucao'].get('ignorar_erro_extracao', False)
    )
    
    dados_analise = carga.carregar()
    print(dados_analise.resumo())
    
    if not dados_analise.dados:
        print("❌ Nenhum dado encontrado!. Verifique as pastas e os padrões de nome de arquivo.")
        sys.exit(1)

    # 6. Analisador e Processamento
    nome_arquivo_base = config['saida'].get('arquivo_base', 'comparacao_resultados')
    arquivo_excel = os.path.join(pasta_saida, f'{nome_arquivo_base}.xlsx')
    
    # Checa reuso
    regerar = config['saida'].get('regerar_planilha_base', True)
    analisador_instanciado = False
    
    # Define flags de execução
    flag_graficos = config['execucao'].get('gerar_graficos', False)
    flag_llm = config['execucao'].get('llm_as_a_judge', False)
    
    # Lógica Principal de Execução ou Reuso
    if os.path.isfile(arquivo_excel) and not regerar:
        print(f"\n⚠️  Arquivo Excel já existe e 'regerar_planilha_base' é FALSE.")
        print(f"   Pulando re-análise completa. Usando arquivo existente: {arquivo_excel}")
        
        # Se for para apenas gerar gráficos ou LLM judge em cima do existente
        if flag_graficos or flag_llm:
            # mas talvez não precise processar tudo se tiver hooks específicos. 
            # O código original instanciava tudo. Vamos instanciar para garantir consistência.
            print("   Instanciando analisador para operações em arquivo existente...")
            pasta_jsons = os.path.join(pasta_saida, 'jsons')
            analisador = JsonAnaliseDataFrame(
                dados_analise,
                config=config_comparacao,
                pasta_analises=pasta_jsons,  # JSON files in jsons subfolder
                max_workers=max_workers,
                incluir_valores_analise=True,
                gerar_exemplos_md=False,
                gerar_relatorio=False
            )
            analisador_instanciado = True
    else:
        # EXECUÇÃO COMPLETA
        print(f"\n🚀 Iniciando análise completa...")
        pasta_jsons = os.path.join(pasta_saida, 'jsons')
        analisador = JsonAnaliseDataFrame(
            dados_analise, # Passa o objeto JsonAnaliseDados carregado
            config=config_comparacao,
            pasta_analises=pasta_jsons,  # JSON files in jsons subfolder
            pasta_markdown=pasta_saida,  # Markdown files in main folder (if supported)
            max_workers=max_workers,
            incluir_valores_analise=True,
            gerar_exemplos_md=True,
            max_exemplos_md_por_metrica=5,
            gerar_relatorio=True
        )
        analisador_instanciado = True
        
        # Relatório Markdown Overview
        if analisador.relatorio:
            analisador.relatorio.set_overview(
                titulo=f"Comparação {rotulo_origem} vs Modelos",
                descricao="Análise comparativa configurada via YAML.",
                rotulos=analisador.rotulos,
                total_documentos=len(dados_analise.dados),
                campos_comparacao=campos_comparacao
            )
            analisador.relatorio.set_config(config_comparacao, campos_comparacao)
            
        # Gera o DataFrame e Exporta
        # Gera o DataFrame e Exporta
        print("📊 Exportando CSVs e Excel...")
        analisador.to_df() # Gera intenamente
        
        # Exporta CSV para a pasta raiz (usa caminho absoluto para sair da pasta_analises/jsons)
        arquivo_csv = os.path.join(pasta_saida, f'{nome_arquivo_base}.csv')
        analisador.exportar_csv(arquivo_csv)
        
        # Exporta Excel para a pasta raiz (usa caminho absoluto)
        analisador.exportar_excel(
            arquivo_excel, 
            incluir_estatisticas=True, 
            usar_formatacao_avancada=True,
            congelar_paineis=True,
            gerar_graficos=False # Gráficos gerados no passo seguinte se solicitado
        )
        print(f"✅ Análise Base salva em: {arquivo_excel}")

    # 7. Pós-Processamento (Gráficos, LLM Judge, Stats)
    
    if flag_graficos and analisador_instanciado:
        print("\n📈 Gerando/Atualizando Gráficos no Excel...")
        # Gera gráfico de status/erros
        _gerar_grafico_erros(dados_analise, pasta_saida)
        
        if os.path.isfile(arquivo_excel):
            analisador.gerar_graficos_de_excel(arquivo_excel, pasta_saida=pasta_saida)
    
    if flag_llm and analisador_instanciado:
        print("\n⚖️  Executando LLM-as-a-Judge (atualização do Excel)...")
        if os.path.isfile(arquivo_excel):
            analisador.atualizar_avaliacao_llm_no_excel(arquivo_excel, gerar_graficos=True)

    if config['execucao'].get('analise_estatistica', False):
        processar_analise_estatistica(dados_analise, pasta_saida, config)

    # 8. Estatísticas Finais no Console
    if analisador_instanciado:
        print("\n📈 Estatísticas Globais (Resumo):")
        stats = analisador.estatisticas_globais()
        
        if not stats.empty:
            # Prioridade de métricas para determinar melhor modelo (ordem de preferência)
            # 1. Avaliação LLM (se disponível)
            # 2. BERTScore F1
            # 3. ROUGE-L F1
            # 4. ROUGE-2 F1
            # 5. ROUGE-1 F1
            metricas_prioridade = [
                r'\(global\)_llm_.*',       # LLM evaluation
                r'\(global\)_bertscore_F1', # BERTScore
                r'\(global\)_rouge_F1',     # ROUGE-L (rouge padrão = rouge-L)
                r'\(global\)_rougel_F1',    # ROUGE-L alternativo
                r'\(global\)_rouge2_F1',    # ROUGE-2
                r'\(global\)_rouge1_F1',    # ROUGE-1
            ]
            
            metrica_escolhida = None
            for padrao in metricas_prioridade:
                match = stats[stats['metrica'].str.contains(padrao, regex=True, na=False)]
                if not match.empty:
                    metrica_escolhida = match.iloc[0]['metrica']
                    break
            
            if metrica_escolhida:
                f1_topo = stats[stats['metrica'] == metrica_escolhida]
                if not f1_topo.empty:
                    idx_vencedor = f1_topo['mean'].idxmax()
                    vencedor = f1_topo.loc[idx_vencedor]
                    print(f"\n🏆 Melhor Modelo em '{metrica_escolhida}':")
                    print(f"   {vencedor['modelo']} (Média: {vencedor['mean']:.4f})")

    print("\n✅ Processo finalizado com sucesso.")

if __name__ == '__main__':
    main()
