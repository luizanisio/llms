# -*- coding: utf-8 -*-
"""
Comparação de extrações de espelhos usando múltiplas métricas de similaridade e configuração via YAML.

Autor: Luiz Anísio
Refatorado: 25/01/2026
A versão em evolução está em: https://github.com/luizanisio/llms 

Descrição:
-----------
Compara dados extraídos por diferentes abordagens e modelos baseados em um arquivo de configuração YAML.
Suporta múltiplas métricas (BERTScore, ROUGE, Levenshtein) configuráveis por campo.
Gera planilhas Excel, CSVs de estatísticas e gráficos comparativos.

Uso:
    python comparar_extracoes.py --config config_summa.yaml
"""

import os
import sys

class LoggerDuplo:
    """Redireciona a saída padrão (stdout) para o console e para um arquivo de log simultaneamente."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        """Retorna se o terminal subjacente é interativo (necessário para tqdm/transformers)."""
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

    def fileno(self):
        """Retorna o file descriptor do terminal subjacente (necessário para algumas bibliotecas)."""
        return self.terminal.fileno()
import argparse
import yaml
import regex as re
import pandas as pd

# ============================================================================
# PROTEÇÃO E SETUP INICIAL
# ============================================================================
import util  # garante que a pasta src está no sys.path (via _UTIL_SRC_DIR em util.py)

# Verificação para multiprocessing (BERTScore safe)
_IS_MAIN_PROCESS = __name__ == '__main__' or not hasattr(sys.modules.get('__mp_main__', None), '__file__')

def ler_configuracao(caminho_yaml):
    """Lê e valida o arquivo de configuração YAML."""
    if not os.path.exists(caminho_yaml):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {caminho_yaml}")
    
    with open(caminho_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"📖 Configuração carregada de: {caminho_yaml}")
    
    # Validação de rótulos únicos
    rotulo_base = config.get('modelo_base', {}).get('rotulo')
    if rotulo_base:
        rotulos_comp = [m.get('rotulo') for m in config.get('modelos_comparacao', []) if m.get('ativo', True)]
        if rotulo_base in rotulos_comp:
            raise ValueError(f"ERRO DE CONFIGURAÇÃO: O rótulo do modelo base ('{rotulo_base}') é igual a um dos rótulos de comparação. Altere para torná-los únicos (ex: '{rotulo_base}(Base)').")
            
        rotulos_vistos = set()
        for r in rotulos_comp:
            if not r: continue
            if r in rotulos_vistos:
                raise ValueError(f"ERRO DE CONFIGURAÇÃO: O rótulo '{r}' aparece duplicado em 'modelos_comparacao'. Cada rótulo deve ser único.")
            rotulos_vistos.add(r)
            
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

def processar_analise_estatistica(analisador, dados_analise, pasta_saida, config, lang='en'):
    """
    Executa análise estatística (Friedman, Wilcoxon, Nemenyi, Shapiro-Wilk)
    para cada combinação campo×métrica configurada em campos_estatisticas.
    """
    try:
        from util_analise_estatistica import executar_analise_estatistica
    except ImportError:
        print("❌ Módulo util_analise_estatistica não encontrado.")
        return []

    return executar_analise_estatistica(
        analisador=analisador,
        dados_analise=dados_analise,
        config=config,
        pasta_saida=pasta_saida,
        lang=lang
    )

def calcular_divisao_grupos(config):
    """
    Calcula os percentuais de divisão de grupos baseado no yaml e aplica regras
    de distribuição de valores não informados. Retorna tupla (treino, teste, validacao).
    """
    divisao = config.get('execucao', {}).get('divisao')
    if not isinstance(divisao, dict):
        divisao = config.get('execucao-divisao')
        
    if not isinstance(divisao, dict):
        return (0.7, 0.2, 0.1) # treino, teste, validacao

    treino = divisao.get('treino')
    teste = divisao.get('teste')
    validacao = divisao.get('validacao')

    valores = []
    for v in [treino, teste, validacao]:
        if v is not None:
            valores.append(float(v))
        else:
            valores.append(None)
            
    treino, teste, validacao = valores

    usando_porcentagem = any(v > 1.0 for v in [treino, teste, validacao] if v is not None)
    if usando_porcentagem:
        treino = treino / 100.0 if treino is not None else None
        teste = teste / 100.0 if teste is not None else None
        validacao = validacao / 100.0 if validacao is not None else None

    soma_atual = sum(v for v in [treino, teste, validacao] if v is not None)
    num_faltantes = sum(1 for v in [treino, teste, validacao] if v is None)

    if num_faltantes > 0:
        restante = max(0.0, 1.0 - soma_atual)
        parte = restante / num_faltantes
        if treino is None: treino = parte
        if teste is None: teste = parte
        if validacao is None: validacao = parte

    soma_final = round(treino + teste + validacao, 4)
    if soma_final > 0 and soma_final != 1.0:
        treino /= soma_final
        teste /= soma_final
        validacao /= soma_final

    return (treino, teste, validacao)

def configurar_metricas(config_yaml, base_dir="", pasta_modelos_ativa=""):
    """Converte a configuração YAML para o formato esperado pelo JsonAnalise."""
    conf_comp = config_yaml.get('configuracao_comparacao', {})
    campos = conf_comp.get('campos', {})
    
    # Estrutura base
    config_final = {
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
        'campos_sbert_grande': campos.get('sbert_grande') or [],
        # Configuração de modelos personalizados (opcional)
        'modelos_sbert': conf_comp.get('modelos', {}).get('sbert', {}),
        'modelo_bertscore': conf_comp.get('modelos', {}).get('bertscore', None),
        'bertscore_batch_size': conf_comp.get('modelos', {}).get('bertscore_batch_size', None),
        'sbert_batch_size': conf_comp.get('modelos', {}).get('sbert_batch_size', None),
        'campos_virtuais': config_yaml.get('campos_virtuais', {})
    }
    
    # Processamento de Aliases para Modelos
    modelos_conf = conf_comp.get('modelos', {})
    sbert_conf = modelos_conf.get('sbert', {})
    
    defaults_modelos = {
        'sbert_pequeno': ('paraphrase-multilingual-MiniLM-L12-v2', 'MiniLM'),
        'sbert_medio': ('paraphrase-multilingual-mpnet-base-v2', 'MPNet'),
        'sbert_grande': ('intfloat/multilingual-e5-large', 'E5-Large'),
        'bertscore': ('bert-base-multilingual-cased', 'mBERT')
    }
    
    def _obter_alias(dict_fonte, chave_valor, chave_alias, default_val, default_alias):
        v = dict_fonte.get(chave_valor)
        a = dict_fonte.get(chave_alias)
        if not v:
            return (default_val, default_alias)
        if not a:
            import os as _os
            a = _os.path.basename(v.rstrip('/\\'))
        return (v, a)

    v_peq, a_peq = _obter_alias(sbert_conf, 'pequeno', 'pequeno_alias', *defaults_modelos['sbert_pequeno'])
    v_med, a_med = _obter_alias(sbert_conf, 'medio', 'medio_alias', *defaults_modelos['sbert_medio'])
    v_gra, a_gra = _obter_alias(sbert_conf, 'grande', 'grande_alias', *defaults_modelos['sbert_grande'])
    v_bert, a_bert = _obter_alias(modelos_conf, 'bertscore', 'bertscore_alias', *defaults_modelos['bertscore'])

    config_final['modelos_aliases'] = {
        'sbert_pequeno': (a_peq, v_peq),
        'sbert_medio': (a_med, v_med),
        'sbert_grande': (a_gra, v_gra),
        'bertscore': (a_bert, v_bert)
    }
    
    # Resolve modelos locais
    if config_final['modelos_sbert']:
        for k, v in config_final['modelos_sbert'].items():
            if v and not os.path.isabs(v) and not v.startswith(("hf://", "huggingface://")):
                v_res = resolver_caminho(v, base_dir, pasta_modelos_ativa)
                if os.path.exists(v_res):
                    config_final['modelos_sbert'][k] = v_res
                    
    v_bert = config_final['modelo_bertscore']
    if v_bert and not os.path.isabs(v_bert) and not v_bert.startswith(("hf://", "huggingface://")):
        v_res = resolver_caminho(v_bert, base_dir, pasta_modelos_ativa)
        if os.path.exists(v_res):
            config_final['modelo_bertscore'] = v_res
    
    
    # Ajuste para teste rápido (desativa BERTScore e SBERT)
    if config_yaml.get('execucao', {}).get('teste_rapido', False):
        print("⚠️  Modo TESTE RÁPIDO: Desabilitando BERTScore/SBERT e movendo campos para ROUGE-L.")
        
        # Move campos BERTScore para ROUGE-L
        campos_removidos = config_final['campos_bertscore'].copy()
        config_final['campos_bertscore'] = []
        for c in campos_removidos:
            if c not in config_final['campos_rouge']:
                config_final['campos_rouge'].append(c)
                
        # Move campos SBERT para ROUGE-L
        for sbert_key in ['campos_sbert', 'campos_sbert_pequeno', 'campos_sbert_medio', 'campos_sbert_grande']:
            campos_sbert = config_final[sbert_key].copy()
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
# Utilitários  →  centralizados em util_menu_opcoes.py
# ============================================================================

# ============================================================================
# MAIN
# ============================================================================

def resolver_caminho(caminho_relativo, base_dir, pasta_base=""):
    """Resolve caminhos relativos baseado no diretório do arquivo de configuração ou pasta_base."""
    return util.Util.resolver_caminho(caminho_relativo, base_dir, pasta_base)

# As funções de geração de gráficos foram extraídas para comparar_extracoes_graficos.py


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comparador de Extrações JSON via YAML")
    parser.add_argument('--config', dest='config_file', default=None, help="Caminho do arquivo de configuração YAML")
    parser.add_argument('--graficos', action='store_true', help="Apenas atualiza os gráficos a partir de uma comparação já realizada")
    parser.add_argument('--estatisticas', action='store_true', help="Apenas atualiza as estatísticas a partir de uma comparação já realizada")
    parser.add_argument('--planilha', action='store_true', help="Apenas atualiza a formatação da planilha base a partir de uma comparação já realizada")
    args = parser.parse_args()

    qualquer_flag_parcial = args.graficos or args.estatisticas or args.planilha

    # 1. Carregar configuração
    caminho_yaml_abs = ""
    if args.config_file:
        caminho_yaml_abs = os.path.abspath(args.config_file)
    else:
        from util_menu_opcoes import escolher_yaml
        escolha = escolher_yaml(
            pasta='.',
            chave_obrigatoria=['modelo_base', 'modelos_comparacao'],
            titulo="Arquivos de configuração encontrados:",
            padrao_recente=True,
            limite=10,
            opcoes_extras=[
                ("Criar um novo arquivo de configuração", "CRIAR_NOVO"),
                ("Sair sem escolher", None)
            ]
        )
        if escolha is None:
            print("\nSaindo sem escolher...")
            sys.exit(0)
        elif escolha == "CRIAR_NOVO":
            nome = input("\nNome do novo arquivo (ex: config_novo.yaml): ").strip()
            if not nome:
                print("Nome não fornecido. Saindo...")
                sys.exit(0)
            if not nome.endswith(('.yaml', '.yml')):
                nome += '.yaml'
            caminho_yaml_abs = os.path.abspath(nome)
            if os.path.exists(caminho_yaml_abs):
                print(f"⚠️  O arquivo {nome} já existe. Edite-o e execute novamente.")
                sys.exit(0)
            
            # Tentar copiar de um existente para facilitar
            if os.path.exists('config_espelho.yaml'):
                import shutil
                shutil.copy('config_espelho.yaml', caminho_yaml_abs)
                print(f"✅ Arquivo {nome} criado a partir de config_espelho.yaml!")
            else:
                try:
                    from comparar_extracoes_exemplo import YAML_EXEMPLO_PASTA, YAML_EXEMPLO_PARQUET
                    
                    print("\n────────────────────────────────────────────────────────────")
                    print("  Qual formato de template você deseja usar?")
                    print("────────────────────────────────────────────────────────────")
                    print("  [1] Formato com PARQUET (Novo padrão, ideal para grandes volumes)")
                    print("  [2] Formato com PASTA de JSONs (Mais simples, pastas com jsons)")
                    print("────────────────────────────────────────────────────────────")
                    
                    while True:
                        formato = input("  Escolha uma opção [1/2]: ").strip()
                        if formato in ['1', '2']:
                            break
                        print("  ⚠️  Opção inválida. Digite 1 ou 2.")
                    
                    if formato == '2':
                        template_yaml = YAML_EXEMPLO_PASTA.strip() + "\n"
                        msg_sucesso = f"✅ Arquivo {nome} criado usando o template de PASTA!"
                    else:
                        template_yaml = YAML_EXEMPLO_PARQUET.strip() + "\n"
                        msg_sucesso = f"✅ Arquivo {nome} criado usando o template PARQUET!"
                except ImportError:
                    template_yaml = "# Novo arquivo de configuração YAML\n# Não foi possível carregar o exemplo em comparar_extracoes_exemplo.py\n"
                    msg_sucesso = f"✅ Arquivo {nome} criado com template básico!"

                with open(caminho_yaml_abs, 'w', encoding='utf-8') as f:
                    f.write(template_yaml)
                print(msg_sucesso)
                
            print(f"Edite-o e execute o script novamente com: python comparar_extracoes.py {nome}")
            sys.exit(0)
        else:
            caminho_yaml_abs = escolha  # já é caminho absoluto retornado por escolher_yaml

    base_dir_yaml = os.path.dirname(caminho_yaml_abs)
    config = ler_configuracao(caminho_yaml_abs)
    
    misc = config.get("misc", {}) or {}
    
    pastas_base = misc.get("pastas_base", [])
    if isinstance(pastas_base, str):
        pastas_base = [pastas_base]
        
    pasta_base_ativa = ""
    for pb in pastas_base:
        pb_abs = pb if os.path.isabs(pb) else os.path.normpath(os.path.join(base_dir_yaml, pb))
        if os.path.isdir(pb_abs):
            pasta_base_ativa = pb_abs
            break
            
    if not pasta_base_ativa and pastas_base:
        pb = pastas_base[0]
        pasta_base_ativa = pb if os.path.isabs(pb) else os.path.normpath(os.path.join(base_dir_yaml, pb))
        
    pastas_modelos = misc.get("pastas_modelos", misc.get("pasta_modelos", []))
    if isinstance(pastas_modelos, str):
        pastas_modelos = [pastas_modelos] if pastas_modelos else []
        
    pasta_modelos_ativa = pasta_base_ativa
    if pastas_modelos:
        for pm in pastas_modelos:
            pm_abs = pm if os.path.isabs(pm) else os.path.normpath(os.path.join(pasta_base_ativa or base_dir_yaml, pm))
            if os.path.isdir(pm_abs):
                pasta_modelos_ativa = pm_abs
                break

    pastas_modelos_treinados = misc.get("pastas_modelos_treinados", [])
    if isinstance(pastas_modelos_treinados, str):
        pastas_modelos_treinados = [pastas_modelos_treinados] if pastas_modelos_treinados else []
    
    # 2. Inicializar ambiente
    max_workers_env, _ = _inicializar_ambiente()
    max_workers = config['execucao'].get('max_workers', max_workers_env)
    
    # 3. Setup de Pastas e Rótulos (Resolvendo caminhos)
    pasta_saida_raw = config['saida']['pasta']
    pasta_saida = resolver_caminho(pasta_saida_raw, base_dir_yaml, pasta_base_ativa)
    
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
        
    # Configura Logger Duplo para console e arquivo
    arquivo_log_execucao = os.path.join(pasta_saida, 'execucao.log')
    sys.stdout = LoggerDuplo(arquivo_log_execucao)
    print(f"📄 Log de execução iniciado em: {arquivo_log_execucao}")
    # 3.4 Pega as configurações gerais
    modelo_base = config['modelo_base']
    modelos_comp_all = config.get('modelos_comparacao', [])
    
    # Filtra apenas modelos ativos (padrão: ativo=True)
    modelos_comp = [m for m in modelos_comp_all if m.get('ativo', True)]
    
    # Log de itens ignorados
    ignorados = [m['rotulo'] for m in modelos_comp_all if not m.get('ativo', True)]
    if ignorados:
        print(f"⚠️  Modelos ignorados explicitamente (ativo=false): {', '.join(ignorados)}")

    # Configura Logger de Memória
    arquivo_log_memoria = os.path.join(pasta_saida, 'uso_memoria.csv')
    try:
        from util_sysinfo import MemoryLogger
        MemoryLogger.set_log_file(arquivo_log_memoria, tempo_atualizacao=30)
        MemoryLogger.set_nome_etapa("INICIO - Configuração Carregada")
        tem_memory_logger = True
    except ImportError:
        tem_memory_logger = False

    # Prepara listas para CargaDadosComparacao
    rotulo_origem = modelo_base.get('rotulo', 'BASE')
    # Read campo_id from YAML config (defined ahead of time for use later)
    rotulo_id = config.get('configuracao_comparacao', {}).get('nome_campo_id', 'id')
    rotulos_destinos = [m['rotulo'] for m in modelos_comp]
    
    # 3.5 Pré-processamento: Dataset (Parquet/CSV) → Pasta de JSONs
    campos_dataset = config.get('configuracao_comparacao', {}).get('campos_dataset', config.get('configuracao_comparacao', {}).get('campos_parquet', {}))
    pasta_extracao_raw = config.get('saida', {}).get('pasta_extracao', config.get('saida', {}).get('pasta_parquet', ''))
    
    houve_reextracao = False
    
    # Valida obrigatoriedade de pasta_extracao quando há entrada em formato de tabela
    todos_modelos_config = [modelo_base] + modelos_comp
    tem_dataset = any(m.get('arquivo', '').endswith('.parquet') or m.get('arquivo', '').endswith('.csv') for m in todos_modelos_config)
    if tem_dataset and not pasta_extracao_raw:
        print("❌ 'saida.pasta_parquet' ou 'saida.pasta_extracao' é obrigatório quando se usa arquivos .parquet ou .csv como entrada.")
        sys.exit(1)
        
    # Lê filtro de IDs se configurado
    ids_filtro = None
    config_filtro = config.get('configuracao_comparacao', {}).get('filtro', {})
    if config_filtro and isinstance(config_filtro, dict):
        arquivo_filtro = config_filtro.get('arquivo')
        campo_id_filtro = config_filtro.get('campo_id')
        
        if arquivo_filtro and campo_id_filtro:
            arquivo_filtro_abs = resolver_caminho(arquivo_filtro, base_dir_yaml, pasta_base_ativa)
            if os.path.exists(arquivo_filtro_abs):
                from util_pandas import ler_dataset, aplicar_filtro_dataset
                try:
                    df_filtro = ler_dataset(arquivo_filtro_abs)
                    
                    filtro_query = config_filtro.get('dataset_filtro')
                    if filtro_query:
                        df_filtro = aplicar_filtro_dataset(df_filtro, filtro_query)
                        
                    if campo_id_filtro in df_filtro.columns:
                        ids_filtro = set(df_filtro[campo_id_filtro].astype(str).str.strip())
                        print(f"🔍 Filtro carregado: {len(ids_filtro)} IDs de '{arquivo_filtro}' (campo '{campo_id_filtro}')")
                    else:
                        print(f"⚠️  Aviso: Coluna '{campo_id_filtro}' não encontrada no arquivo de filtro '{arquivo_filtro}'.")
                except Exception as e:
                    print(f"⚠️  Aviso: Erro ao ler arquivo de filtro '{arquivo_filtro}': {e}")
            else:
                print(f"⚠️  Aviso: Arquivo de filtro não encontrado: {arquivo_filtro_abs}")
    
    def _resolver_entrada_modelo(modelo_config):
        """
        Resolve a entrada de um modelo: se for .parquet, extrai para pasta.
        Retorna o caminho da PASTA com os JSONs (seja direta ou extraída do parquet).
        """
        rotulo = modelo_config.get('rotulo', 'Desconhecido')
        print(f"\n--- Preparando modelo: {rotulo} ---")
        
        nonlocal houve_reextracao
        arquivo = modelo_config.get('arquivo', '')
        pasta = modelo_config.get('pasta', '')
        
        caminhos_busca = pastas_modelos_treinados + pastas_modelos + pastas_base
        
        if arquivo and (arquivo.endswith('.parquet') or arquivo.endswith('.csv')):
            from comparar_extracoes_util import ExtracaoDataset, resolver_pasta_dataset
            arquivo_abs = resolver_caminho(arquivo, base_dir_yaml, pasta_base_ativa)
            
            if not os.path.exists(arquivo_abs):
                for p_busca in caminhos_busca:
                    p_busca_abs = p_busca if os.path.isabs(p_busca) else os.path.normpath(os.path.join(base_dir_yaml, p_busca))
                    caminho_teste = os.path.join(p_busca_abs, arquivo)
                    if os.path.exists(caminho_teste):
                        arquivo_abs = caminho_teste
                        print(f"🔍 Arquivo dataset resolvido em: {arquivo_abs}")
                        break
            
            pasta_extracao_abs = resolver_caminho(pasta_extracao_raw, base_dir_yaml, pasta_base_ativa)
            pasta_destino = resolver_pasta_dataset(arquivo_abs, pasta_extracao_abs)
            
            campos_modelo = modelo_config.get('campos_parquet', campos_dataset)
            saida_json_config = campos_modelo.get('saida_json', campos_dataset.get('saida_json', True))
            dataset_filtro_config = modelo_config.get('dataset_filtro')
            extrator = ExtracaoDataset(arquivo_abs, pasta_destino, campos_modelo, ids_filtro=ids_filtro, saida_json=saida_json_config, dataset_filtro=dataset_filtro_config)
            erros = extrator.validar_colunas()
            if erros:
                print(f"❌ Erro ao validar dataset '{arquivo_abs}':")
                for e in erros:
                    print(f"   - {e}")
                sys.exit(1)
                
            if not extrator.ja_extraido():
                houve_reextracao = True
                
            return extrator.extrair()
        elif pasta:
            pasta_abs = resolver_caminho(pasta, base_dir_yaml, pasta_base_ativa)
            if os.path.exists(pasta_abs):
                return pasta_abs
                
            for p_busca in caminhos_busca:
                p_busca_abs = p_busca if os.path.isabs(p_busca) else os.path.normpath(os.path.join(base_dir_yaml, p_busca))
                caminho_teste = os.path.join(p_busca_abs, pasta)
                if os.path.exists(caminho_teste):
                    print(f"🔍 Pasta do modelo resolvida em: {caminho_teste}")
                    return caminho_teste
            
            print(f"❌ Pasta do modelo '{modelo_config.get('rotulo', '?')}' ({pasta}) não encontrada diretamente nem nas pastas listadas em 'misc'.")
            sys.exit(1)
        else:
            print(f"❌ Modelo '{modelo_config.get('rotulo', '?')}' deve ter 'arquivo' (.parquet ou .csv) ou 'pasta' definido.")
            sys.exit(1)
    
    origem = _resolver_entrada_modelo(modelo_base)
    pastas_destinos = [_resolver_entrada_modelo(m) for m in modelos_comp]
    
    # Validações básicas
    if not os.path.isdir(origem):
        print(f"❌ Pasta base não encontrada: {origem}")
        sys.exit(1)
        
    for p_dest, r_dest in zip(pastas_destinos, rotulos_destinos):
        if not os.path.isdir(p_dest):
            print(f"❌ Pasta de destino não encontrada para o modelo '{r_dest}': {p_dest}")
            sys.exit(1)
    
    # 4. Configuração das Métricas
    config_comparacao = configurar_metricas(config, base_dir_yaml, pasta_modelos_ativa)
    campos_comparacao = extrair_campos_unicos(config_comparacao)
    
    print("\n⚙️  Configuração de Comparação:")
    print(f"   Origem (Base): {origem} [{rotulo_origem}]")
    print(f"   Destinos: {len(pastas_destinos)} pastas")
    print(f"   Campos: {campos_comparacao}")
    print(f"   Saída: {pasta_saida}")
    
    # Log de modelos personalizados (se configurados)
    modelos_sbert = config_comparacao.get('modelos_sbert', {})
    modelo_bertscore = config_comparacao.get('modelo_bertscore', None)
    if modelos_sbert:
        for tamanho, nome_modelo in modelos_sbert.items():
            print(f"   🔧 SBERT [{tamanho}]: {nome_modelo} (personalizado)")
    if modelo_bertscore:
        print(f"   🔧 BERTScore: {modelo_bertscore} (personalizado)")
    
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

    # Flag avaliar_global: garante que se o filtro de campos do json retornar
    # vazio (por não haver campos folha mapeados explicitamente no yaml), o json
    # original inteiro ainda será passado adiante para o motor de similaridade 
    # se qualquer métrica global (ex: SBERT, BERTScore) ou estrutural estiver configurada.
    avaliar_global = any('(global)' in config.get(k, []) for k in [
        'campos_bertscore', 'campos_rouge', 'campos_rouge1', 'campos_rouge2',
        'campos_levenshtein', 'campos_sbert', 'campos_sbert_pequeno',
        'campos_sbert_medio', 'campos_sbert_grande'
    ])

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
        ignorar_erro_extracao=config['execucao'].get('ignorar_erro_extracao', False),
        ids_filtro=ids_filtro,
        campos_virtuais=config.get('campos_virtuais', {}),
        avaliar_global=avaliar_global
    )
    
    if tem_memory_logger: MemoryLogger.set_nome_etapa("ANTES DE CARREGAR DADOS JSON")
    dados_analise = carga.carregar()
    if tem_memory_logger: MemoryLogger.set_nome_etapa("DEPOIS DE CARREGAR DADOS JSON")
    print(dados_analise.resumo())
    
    if not dados_analise.dados:
        print("❌ Nenhum dado encontrado!. Verifique as pastas e os padrões de nome de arquivo.")
        sys.exit(1)

    # 6. Analisador e Processamento
    nome_arquivo_base = config['saida'].get('arquivo_base', 'comparacao_resultados')
    arquivo_excel = os.path.join(pasta_saida, f'{nome_arquivo_base}.xlsx')
    
    # Checa reuso e valida execução parcial
    if qualquer_flag_parcial:
        if houve_reextracao:
            print("\n❌ ERRO: O arquivo base ou modelo foi atualizado ou reextraído.")
            print("Execute a comparação completa (sem flags parciais) antes de gerar apenas gráficos, estatísticas ou planilhas.")
            sys.exit(1)
        if not os.path.isfile(arquivo_excel):
            print("\n❌ ERRO: Planilha de comparação não encontrada.")
            print(f"Esperado: {arquivo_excel}")
            print("Execute a comparação completa primeiro (sem usar as flags --graficos, --estatisticas ou --planilha).")
            sys.exit(1)

        flag_graficos = args.graficos
        flag_estatisticas = args.estatisticas
        flag_planilha = args.planilha
        flag_llm = False # Se for execução parcial, pulamos LLM por envolver custo de API
        regerar = False
    else:
        # Execução Padrão: faz tudo!
        flag_graficos = True
        flag_estatisticas = True
        flag_planilha = False
        regerar = True
        flag_llm = config.get('execucao', {}).get('llm_as_a_judge', False)

    analisador_instanciado = False
    
    lang_graficos = config.get('saida', {}).get('linguagem_graficos', '').strip().lower()
    if lang_graficos not in ('pt', 'en'):
        lang_graficos = 'en'
    
    # Lógica Principal de Execução ou Reuso
    if os.path.isfile(arquivo_excel) and not regerar:
        print(f"\n⚠️  Execução Parcial Solicitada via CLI.")
        print(f"   Pulando re-análise pesada. Usando arquivo existente: {arquivo_excel}")
        
        if flag_graficos or flag_estatisticas or flag_planilha or flag_llm:
            print("   Instanciando analisador para operações em arquivo existente...")
            pasta_jsons = os.path.join(pasta_saida, 'jsons')
            analisador = JsonAnaliseDataFrame(
                dados_analise,
                config=config_comparacao,
                pasta_analises=pasta_jsons,  # JSON files in jsons subfolder
                max_workers=max_workers,
                incluir_valores_analise=True,
                gerar_exemplos_md=False,
                gerar_relatorio=False,
                lang=lang_graficos
            )
            analisador_instanciado = True
            
            if flag_planilha:
                print("📊 Regerando Excel e formatação...")
                analisador.to_df()
                analisador.exportar_excel(
                    arquivo_excel, 
                    incluir_estatisticas=True, 
                    usar_formatacao_avancada=True,
                    congelar_paineis=True,
                    gerar_graficos=False 
                )
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
            gerar_relatorio=True,
            lang=lang_graficos
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
        print("📊 Exportando CSVs e Excel...")
        if tem_memory_logger: MemoryLogger.set_nome_etapa("ANTES DA GERAÇÃO DO DATAFRAME")
        analisador.to_df() # Gera intenamente
        if tem_memory_logger: MemoryLogger.set_nome_etapa("DEPOIS DA GERAÇÃO DO DATAFRAME")
        
        # Exporta CSV para a pasta raiz (usa caminho absoluto para sair da pasta_analises/jsons)
        arquivo_csv = os.path.join(pasta_saida, f'{nome_arquivo_base}.csv')
        analisador.exportar_csv(arquivo_csv)
        
        # Exporta Excel para a pasta raiz (usa caminho absoluto)
        if tem_memory_logger: MemoryLogger.set_nome_etapa("ANTES DE EXPORTAR EXCEL")
        analisador.exportar_excel(
            arquivo_excel, 
            incluir_estatisticas=True, 
            usar_formatacao_avancada=True,
            congelar_paineis=True,
            gerar_graficos=False # Gráficos gerados no passo seguinte se solicitado
        )
        if tem_memory_logger: MemoryLogger.set_nome_etapa("DEPOIS DE EXPORTAR EXCEL")
        print(f"✅ Análise Base salva em: {arquivo_excel}")

    # 7. Pós-Processamento (Gráficos, LLM Judge, Stats)
    
    if flag_graficos and analisador_instanciado:
        from comparar_extracoes_graficos import CompararExtracoesGraficos
        print("\n📈 Gerando/Atualizando Gráficos no Excel...")
        # Gera gráfico de status/erros
        CompararExtracoesGraficos.gerar_grafico_erros(dados_analise, pasta_saida, lang=lang_graficos)
        
        # gerar_graficos_de_excel faz a limpeza da pasta graficos, então deve rodar primeiro
        if os.path.isfile(arquivo_excel):
            analisador.gerar_graficos_de_excel(arquivo_excel, pasta_saida=pasta_saida)
        
        # Gera gráficos de métricas de treinamento (após limpeza, para não serem apagados)
        CompararExtracoesGraficos.gerar_graficos_treinamento(config, base_dir_yaml, pasta_base_ativa, dados_analise, pasta_saida, lang=lang_graficos)
        
        # Gera gráficos de custo-eficiência (tokens/instâncias vs F1 Score)
        if os.path.isfile(arquivo_excel):
            CompararExtracoesGraficos.gerar_graficos_custo_eficiencia(
                config, base_dir_yaml, pasta_base_ativa, dados_analise, pasta_saida,
                arquivo_excel=arquivo_excel, lang=lang_graficos
            )
    
    if flag_llm and analisador_instanciado:
        print("\n⚖️  Executando LLM-as-a-Judge (atualização do Excel)...")
        if os.path.isfile(arquivo_excel):
            analisador.atualizar_avaliacao_llm_no_excel(arquivo_excel, gerar_graficos=True, pasta_saida=pasta_saida)

    if flag_estatisticas and analisador_instanciado:
        processar_analise_estatistica(analisador, dados_analise, pasta_saida, config, lang=lang_graficos)

    # 8. Divisão dos Dados (Treino/Teste/Validação) (pula se for execução parcial)
    if qualquer_flag_parcial:
        print("\n🏁 Execução parcial finalizada com sucesso!")
        return

    print("\n🗂️  Gerando divisões de dados (Treino/Teste/Validação)...")
    try:
        from util_json_divisoes import UtilJsonDivisoes, contar_chaves_recursivo
        divisao_grupos = calcular_divisao_grupos(config)
        print(f"   Configuração de divisão: Treino={divisao_grupos[0]:.2f}, Teste={divisao_grupos[1]:.2f}, Validação={divisao_grupos[2]:.2f}")
        
        # Constrói mapa de chaves do ground truth para classificação de complexidade
        mapa_chaves = {}
        rotulo_true = dados_analise.rotulo_true
        rotulo_id_key = dados_analise.rotulo_id
        for item in dados_analise.dados:
            id_doc = item.get(rotulo_id_key)
            json_true = item.get(rotulo_true)
            if id_doc is not None and isinstance(json_true, dict):
                mapa_chaves[str(id_doc)] = contar_chaves_recursivo(json_true)
        
        # Constrói mapa de tokens por documento (total e output por modelo)
        # As chaves nos JSONs de análise usam o formato "{rotulo_true}_{rotulo_modelo}"
        # enquanto os tokens usam apenas "{rotulo_modelo}" como prefixo.
        # Remapeamos para que as chaves do mapa usem o formato dos JSONs de análise.
        mapa_tokens = {}
        if dados_analise.tem_tokens:
            for item in dados_analise.dados:
                id_doc = item.get(rotulo_id_key)
                if id_doc is not None:
                    tokens = dados_analise.get_tokens(str(id_doc), por_mil=False)
                    if tokens:
                        doc_tokens = {}
                        for rotulo_modelo in dados_analise.rotulos_modelos:
                            nome_analise = f'{rotulo_true}_{rotulo_modelo}'
                            total = tokens.get(f'{rotulo_modelo}_total')
                            output = tokens.get(f'{rotulo_modelo}_output')
                            if total is not None:
                                doc_tokens[f'{nome_analise}_total'] = total
                            if output is not None:
                                doc_tokens[f'{nome_analise}_output'] = output
                        if doc_tokens:
                            mapa_tokens[str(id_doc)] = doc_tokens

        divisao_config = config.get('execucao', {}).get('divisao')
        if not isinstance(divisao_config, dict):
            divisao_config = config.get('execucao-divisao')
        
        arquivo_referencia = None
        if isinstance(divisao_config, dict):
            arquivo_referencia = divisao_config.get('arquivo_referencia')

        util_divisoes = UtilJsonDivisoes(pasta_analises=pasta_saida, divisao_grupos=divisao_grupos, mapa_chaves=mapa_chaves, mapa_tokens=mapa_tokens, arquivo_referencia=arquivo_referencia)
        util_divisoes.processar()
    except Exception as e:
        print(f"❌ Erro ao gerar divisões: {e}")

    # 9. Estatísticas Finais no Console
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

    if 'tem_memory_logger' in locals() and tem_memory_logger: 
        MemoryLogger.set_nome_etapa("FIM DA EXECUÇÃO")
        MemoryLogger.finalizar()
        
    print("\n✅ Processo finalizado com sucesso.")

if __name__ == '__main__':
    main()
