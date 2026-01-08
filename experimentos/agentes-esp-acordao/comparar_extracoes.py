# -*- coding: utf-8 -*-
"""
Comparação de extrações de espelhos usando múltiplas métricas de similaridade.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/experimentos/agentes-esp-acordao
Data: 14/11/2025

Descrição:
-----------
Compara espelhos extraídos por diferentes abordagens (RAW, base, agentes) e modelos
(GPT-5, Gemma-3 12b/27b) usando BERTScore, ROUGE-L, ROUGE-2 e Levenshtein.
Seleciona métricas apropriadas para cada tipo de campo conforme filosofia documentada.

IMPORTANTE:
-----------
Os imports e configurações pesadas são isolados em funções/blocos condicionais para
evitar que os processos workers do BERTScore reimportem configurações desnecessárias.
Quando multiprocessing usa 'spawn', cada processo filho reimporta o módulo principal.
"""

import os
import sys

# Imports leves que não causam problemas com multiprocessing
import regex as re

# ============================================================================
# PROTEÇÃO CONTRA REIMPORTAÇÃO POR WORKERS DO MULTIPROCESSING
# ============================================================================
# Verificação se este é o processo principal ou um worker do multiprocessing
# Processos workers criados por 'spawn' reimportam o módulo, mas não devem
# executar a inicialização completa do projeto
_IS_MAIN_PROCESS = __name__ == '__main__' or not hasattr(sys.modules.get('__mp_main__', None), '__file__')

def _inicializar_ambiente():
    """
    Inicializa o ambiente do projeto (paths, .env, BERTScore workers).
    Esta função só deve ser chamada no processo principal.
    """
    global MAX_WORKERS_ANALISE, PASTA_ENTRADA_RAIZ
    
    # Adiciona paths de utilitários
    sys.path.extend(['./utils', './src', '../../src'])
    
    # Importa e carrega configurações
    from util import UtilEnv
    UtilEnv.carregar_env('.env', pastas=['../', './'])
    
    # NOTA: BERTScore agora usa implementação simplificada com cache MD5 automático
    # Não é mais necessário configurar workers - a biblioteca bert_score gerencia internamente
    
    # Lê variáveis de ambiente
    # BERTSCORE_DEVICE ainda é utilizado pela nova implementação
    device_bert = UtilEnv.get_str('BERTSCORE_DEVICE', 'auto')
    
    MAX_WORKERS_ANALISE = UtilEnv.get_int('MAX_WORKERS_ANALISE', 10)
    PASTA_ENTRADA_RAIZ = os.getenv('PASTA_ENTRADA_RAIZ') or './saidas/'
    
    # Documenta configurações (f-string apenas para documentação)
    f''' 
      CONSTANTES E CONFIGURAÇÕES DE VARIÁVEIS DE AMBIENTE
      - `{MAX_WORKERS_ANALISE}`: número máximo de workers para análise paralela
      - `{PASTA_ENTRADA_RAIZ}`: pasta raíz de entrada dos espelhos
      - `{device_bert}`: dispositivo para BERTScore (cuda/cpu/auto)
    '''
    
    return MAX_WORKERS_ANALISE, PASTA_ENTRADA_RAIZ

# Valores padrão para quando importado por workers
MAX_WORKERS_ANALISE = 10
PASTA_ENTRADA_RAIZ = './saidas/'
'''
Compara com JsonAnalise os espelhos RAW, base e extrações feitas pelos agentes.

FILOSOFIA DE SELEÇÃO DE MÉTRICAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. BERTScore → Textos longos com nuances semânticas
2. ROUGE-L   → Estruturas/sequências ordenadas
3. ROUGE-2   → Frases médias, precisão de bigramas
4. ROUGE-1   → Termos individuais, palavras-chave (padrão para (estrutura))
5. Levenshtein → Textos curtos exatos (nomes, IDs, valores numéricos)

RAZÕES:   
✨ BENEFÍCIO: Cada campo é analisado pela métrica mais adequada ao seu tipo,
   gerando múltiplas perspectivas onde necessário (ex: teseJuridica tem tanto
   semântica profunda quanto precisão de fraseamento).
'''

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES PADRÃO (reduz duplicação nos cenários)
# ═══════════════════════════════════════════════════════════════════════════════
ROTULO_ID_PADRAO = 'id'
CAMPOS_COMPARACAO_PADRAO = [
    'jurisprudenciaCitada', 'notas', 'informacoesComplementares', 
    'termosAuxiliares', 'teseJuridica', 'tema', 'referenciasLegislativas'
]

# Variáveis globais configuradas pelos cenários
ORIGEM, DESTINOS, D_ROTULOS, CAMPOS_COMPARACAO, PASTA_SAIDA_COMPARACAO, ROTULO_ID, ROTULO_ORIGEM = None, None, None, None, None, None, None
TESTE = False

def ajustar_300_avaliacao_tcc():
    # carrega a lista do arquivo ./saidas/300tcc.txt
    # remove os arquivos de valiação que não estão na lista
    if not os.path.isfile('./saidas/300tcc.txt'):
        print('Arquivo 300tcc.txt não encontrado')
        return
    print('Arquivo 300tcc.txt encontrado, filtrando...')
    with open('./saidas/300tcc.txt', 'r') as f:
        lista_300 = f.read().splitlines()
    q = 0
    for arquivo in os.listdir('./saidas/espelhos_base_gpt5_300/'):
        # o id do arquivo é o nome do arquivo sem o path e sem a extensão .avaliacao.json
        id_peca = os.path.basename(arquivo).replace('.avaliacao.json', '').replace('.json', '')
        if id_peca not in lista_300:
            os.remove('./saidas/espelhos_base_gpt5_300/' + arquivo)
            print(f'Removido: {arquivo}')
            q += 1
    print(f'Mantidos apenas os arquivos de avaliação dos 300 do TCC ({q} removidos)')

def base_raw():
    global ORIGEM, DESTINOS, D_ROTULOS, CAMPOS_COMPARACAO, PASTA_SAIDA_COMPARACAO, ROTULO_ID, ROTULO_ORIGEM
    ORIGEM = 'espelhos_raw/'
    DESTINOS = ['espelhos_base_gpt5/', 'espelhos_agentes_gpt5/', 'espelhos_base_gemma3_12b/', 'espelhos_agentes_gemma3_12b/', 'espelhos_base_gemma3_27b/', 'espelhos_agentes_gemma3_27b/']
    D_ROTULOS = ['base_gpt5','agentes_gpt5','base_gemma3(12)','agentes_gemma3(12)','base_gemma3(27)','agentes_gemma3(27)']
    ROTULO_ID = ROTULO_ID_PADRAO
    ROTULO_ORIGEM = 'RAW'
    CAMPOS_COMPARACAO = CAMPOS_COMPARACAO_PADRAO
    PASTA_SAIDA_COMPARACAO = 'analises_comparacao_raw/'
def base_gpt5():
    global ORIGEM, DESTINOS, D_ROTULOS, CAMPOS_COMPARACAO, PASTA_SAIDA_COMPARACAO, ROTULO_ID, ROTULO_ORIGEM
    ORIGEM = 'espelhos_base_gpt5/'
    DESTINOS = ['espelhos_agentes_gpt5/', 'espelhos_base_gemma3_12b/', 'espelhos_agentes_gemma3_12b/', 'espelhos_base_gemma3_27b/', 'espelhos_agentes_gemma3_27b/']
    D_ROTULOS = ['agentes_gpt5','base_gemma3(12)','agentes_gemma3(12)','base_gemma3(27)','agentes_gemma3(27)']
    ROTULO_ID = ROTULO_ID_PADRAO
    ROTULO_ORIGEM = 'base_gpt5'
    CAMPOS_COMPARACAO = CAMPOS_COMPARACAO_PADRAO
    PASTA_SAIDA_COMPARACAO = 'analises_comparacao_base_gpt5/'
def base_gpt5_300():
    base_gpt5()
    global ORIGEM, PASTA_SAIDA_COMPARACAO, TESTE, DESTINOS, D_ROTULOS
    #DESTINOS = ['espelhos_agentes_gpt5/', 'espelhos_base_gemma3_12b/', 'espelhos_base_gemma3_27b/']
    #D_ROTULOS = ['agentes_gpt5','base_gemma3(12)','base_gemma3(27)']
    ORIGEM = 'espelhos_base_gpt5_300/'
    PASTA_SAIDA_COMPARACAO = 'analises_comparacao_300/'
    TESTE = False # não usa bertscore para teste rápido
    ajustar_300_avaliacao_tcc()
def base_gpt5_p():
    base_gpt5()
    global ORIGEM, PASTA_SAIDA_COMPARACAO, TESTE, DESTINOS, D_ROTULOS
    #DESTINOS = ['espelhos_agentes_gpt5/', 'espelhos_base_gemma3_12b/', 'espelhos_base_gemma3_27b/']
    #D_ROTULOS = ['agentes_gpt5','base_gemma3(12)','base_gemma3(27)']
    ORIGEM = 'espelhos_base_p/'
    PASTA_SAIDA_COMPARACAO = 'analises_comparacao_teste/'
    TESTE = True # não usa bertscore para teste rápido
def base_gpt5_ag():
    base_gpt5_p()
    global ORIGEM
    ORIGEM = 'espelhos_agentes_p/'
def base_gpt5_g():
    base_gpt5_p()
    global ORIGEM, TESTE
    ORIGEM = 'espelhos_base_gpt5/'
    TESTE = True # não usa bertscore para teste rápido
# Função para inicializar cenário padrão - chamada apenas no __main__
# base_gpt5() é o cenário padrão, mas só será executado no processo principal

def _configurar_cenario():
    """Configura cenário e valida pastas. Chamada apenas no processo principal."""
    global ORIGEM, DESTINOS, D_ROTULOS, CAMPOS_COMPARACAO, PASTA_SAIDA_COMPARACAO
    global ROTULO_ID, ROTULO_ORIGEM, TESTE, CONFIG_COMPARACAO
    
    # Seleciona cenário padrão >>> Aqui pode ser alterado para testar cenários menores e mais rápidos como o _p
    base_gpt5_300()
    
    # Valida configuração
    assert len(DESTINOS) == len(D_ROTULOS), 'Número de destinos e rótulos deve ser igual!'
    
    # Ajusta caminhos com PASTA_ENTRADA_RAIZ
    ORIGEM = os.path.join(PASTA_ENTRADA_RAIZ, ORIGEM)
    DESTINOS = [os.path.join(PASTA_ENTRADA_RAIZ, d) for d in DESTINOS]
    PASTA_SAIDA_COMPARACAO = os.path.join(PASTA_ENTRADA_RAIZ, PASTA_SAIDA_COMPARACAO)
    
    print('Pasta raíz:', PASTA_ENTRADA_RAIZ)
    print('Origem:', ORIGEM)
    print('Destinos:', DESTINOS)
    print('Saída:', PASTA_SAIDA_COMPARACAO)
    
    assert os.path.isdir(ORIGEM), f'Pasta de origem "{ORIGEM}" não existe!'
    for d in DESTINOS:
        assert os.path.isdir(d), f'Pasta de destinos "{d}" não existe!'
    
    # Configuração otimizada para nova estrutura JsonAnalise (sem metrica_global)
    CONFIG_COMPARACAO = {
        # Nível de campos (1 = apenas raiz, 2 = raiz + 1 nível aninhado)
        'nivel_campos': 1,
        
        # ═══════════════════════════════════════════════════════════════════════
        # CAMPOS COM MÚLTIPLAS MÉTRICAS (análise multidimensional)
        # ═══════════════════════════════════════════════════════════════════════
        
        # BERTScore: similaridade semântica profunda (textos longos)
        'campos_bertscore': [
            '(global)',                  # Visão geral do documento
            'teseJuridica',              # Teses jurídicas complexas + ROUGE-L (semântica + precisão)
            'notas',                     # Textos descritivos (admite parafraseamento)
            'termosAuxiliares',          # Lista de termos técnicos + ROUGE-2 (semântica + bigramas)
            'informacoesComplementares'  # Informações adicionais (texto livre)
        ],
        
        # ROUGE-L: sequências estruturadas (ordem importa)
        'campos_rouge': [
            '(global)',                   # Visão geral do documento
            'jurisprudenciaCitada',       # Citações estruturadas + ROUGE-2 (estrutura + bigramas)
            'informacoesComplementares',  # Informações adicionais (texto livre)
            'referenciasLegislativas',    # Referências legais (estrutura Lei/Art/§)
            'notas',                     # Textos descritivos (admite parafraseamento)
            'teseJuridica',               # + BERTScore (valida fraseamento legal exato)
        ],
        
        # ROUGE-2: precision de bigramas (fraseamento técnico e termos exatos)
        'campos_rouge2': [
            'termosAuxiliares',          # + BERTScore (bigramas técnicos)
            'tema',                      # Temas como frases curtas
            'jurisprudenciaCitada',       # Citações estruturadas + ROUGE-2 (estrutura + bigramas)
            # (global) será adicionado automaticamente aqui se não estiver em outra métrica
        ],
        
        # ═══════════════════════════════════════════════════════════════════════
        # OBSERVAÇÃO: (global) e (estrutura) recebem métricas padrão automáticas:
        # - (global) → campos_rouge2 (se não especificado em outra lista)
        # - (estrutura) → campos_rouge1 (se não especificado em outra lista)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Configurações de processamento
        'padronizar_simbolos': True,    # Normaliza aspas, espaços, case
        'rouge_stemmer': True           # Usa stemmer no ROUGE para variações morfológicas
    }
    
    if TESTE:
        # Configuração rápida para testes (sem BERTScore)
        from util import Util
        campos_bertscore = CONFIG_COMPARACAO.get('campos_bertscore', [])
        CONFIG_COMPARACAO['campos_bertscore'] = []
        CONFIG_COMPARACAO['campos_levenshtein'] = ['termosAuxiliares', 'referenciasLegislativas']
        campos_bertscore = [_ for _ in campos_bertscore if _ not in CONFIG_COMPARACAO['campos_rouge2']]
        _linha = '⚠️  ' * 20
        print(f'\n{_linha}\nModo TESTE ativado: BERTScore desabilitado:\n - campos movidos para Rouge 2: {campos_bertscore}\n{_linha}\n')
        CONFIG_COMPARACAO['campos_rouge2'] += campos_bertscore
        Util.pausa(3)
    
    return CONFIG_COMPARACAO

# Variável global que será inicializada no __main__
CONFIG_COMPARACAO = None


def _buscar_metricas_globais(stats):
    """
    Busca métricas globais F1 nas estatísticas, com fallback inteligente.
    
    Args:
        stats: DataFrame de estatísticas do analisador
    
    Returns:
        DataFrame filtrado com métricas globais F1, ou DataFrame vazio se não encontrar
    """
    # Tenta ROUGE-2 primeiro (métrica padrão preferida)
    f1_global = stats[stats['metrica'] == '(global)_rouge2_F1']
    
    if len(f1_global) == 0:
        # Fallback: tenta qualquer (global)_*_F1
        f1_global = stats[stats['metrica'].str.contains(r'\(global\)_.*_F1', regex=True)]
    
    return f1_global


def processar_analise_estatistica(dados_analise, pasta_saida):
    """
    Executa a análise estatística (LLM-as-a-Judge) usando a classe AnaliseEstatistica.
    """
    print("\n📊 Iniciando Análise Estatística (LLM-as-a-Judge)...")
    # Importa da nova localização em src (já no path)
    try:
        from util_analise_estatistica import AnaliseEstatistica
    except ImportError:
        # Fallback se não encontrar no path padrão, tenta adicionar ../src
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
        from util_analise_estatistica import AnaliseEstatistica
        
    import pandas as pd
    
    lista_dados = []
    
    # Lookups agora são feitos via métodos do objeto dados_analise
    pk = dados_analise.config.nome_campo_id

    # Definição EXPLICITA dos pares para análise (Base vs Agentes da mesma família)
    # Tuplas: (Nome Família/Relatório, Rótulo Base, Rótulo Agente)
    PARES_ANALISE = [
        ('GPT-5',            'base_gpt5',        'agentes_gpt5'),
        ('Gemma 3 12b',      'base_gemma3(12)',  'agentes_gemma3(12)'),
        ('Gemma 3 27b',      'base_gemma3(27)',  'agentes_gemma3(27)')
    ]
    
    # Iterar sobre cada PAR definido
    for nome_familia, rotulo_base, rotulo_agente in PARES_ANALISE:
        print(f"   Processando família: {nome_familia} ({rotulo_base} vs {rotulo_agente})...")
        
        # Verifica se os rótulos existem nos dados
        if rotulo_base not in dados_analise.rotulos or rotulo_agente not in dados_analise.rotulos:
            print(f"      ⚠️  Saltando {nome_familia}: rótulos não encontrados nos dados.")
            continue

        # Para cada peça carregada
        for item in dados_analise.dados:
            id_peca = item.get(pk)
            if not id_peca: continue
            
            # Busca dados nos lookups
            tokens = dados_analise.get_tokens(id_peca)
            evals = dados_analise.get_avaliacao(id_peca)
            
            # Extrai valores usando os rótulos do PAR
            v1 = evals.get(f'{rotulo_base}_F1')
            v2 = evals.get(f'{rotulo_agente}_F1')
            
            # Custo (Tokens Total)
            c1 = tokens.get(f'{rotulo_base}_total', 0)
            c2 = tokens.get(f'{rotulo_agente}_total', 0)
            
            # Fallback para evitar divisão por zero
            if c1 == 0: c1 = 1 
            if c2 == 0: c2 = 1

            # Só adiciona se tiver avaliação em ambos
            if v1 is not None and v2 is not None:
                lista_dados.append({
                    'valor1': v1, # Base (F1)
                    'valor2': v2, # Agente (F1)
                    'custo1': c1, # Base (Tokens)
                    'custo2': c2, # Agente (Tokens)
                    'familia': nome_familia
                })
    
    if not lista_dados:
        print("❌ Nenhum dado de avaliação (LLM-as-a-Judge) encontrado para análise estatística.")
        return
        
    print(f"   Total de pares recuperados: {len(lista_dados)}")
    df_stat = pd.DataFrame(lista_dados)
    
    # Configura análise com rótulos genéricos pois agora estamos agrupando corretamente
    analise = AnaliseEstatistica(df_stat, config={
        'rotulo1': 'Base',   # Genérico
        'rotulo2': 'Agente', # Genérico
        'arquivo_saida': os.path.join(pasta_saida, 'relatorio_analise_estatistica.md')
    })
    analise.processar_analise()
    analise.salvar_relatorio()
    print("\n✅ Análise Estatística concluída.")


if __name__ == '__main__':
    ''' realiza a comparação das extrações dos espelhos na pasta ORIGEM com as extrações nas pastas DESTINOS
        todas as pastas devem conter arquivos json nomeados com o id_peca.json, outros arquivos são ignorados
        caso o arquivo não exista em uma das pastas, é registrado como "Inexistente"
        caso exista uma chave "erro" no json, é registrado como "Erro na extração"
        caso o campo origem seja nulo ou vazio e no destino também, os campos podem ser removidos na comparação
        o resultado é salvo conforme exemplo no arquivo "exemplo_dataframe.py"
    '''
    
    # =========================================================================
    # INICIALIZAÇÃO DO AMBIENTE (APENAS NO PROCESSO PRINCIPAL)
    # =========================================================================
    # Isso evita que os workers do multiprocessing reimportem as configurações
    
    # 1. Inicializa ambiente (paths, .env, BERTScore workers)
    MAX_WORKERS_ANALISE, PASTA_ENTRADA_RAIZ = _inicializar_ambiente()
    
    # 2. Imports pesados - só após inicialização e apenas no processo principal
    from util_json import JsonAnaliseDataFrame
    from util_json_carga import CargaDadosComparacao
    
    # 3. Configura cenário (valida pastas, carrega CONFIG_COMPARACAO)
    CONFIG_COMPARACAO = _configurar_cenario()
    
    # =========================================================================
    # EXECUÇÃO PRINCIPAL
    # =========================================================================
    
    print("=" * 80)
    print("🔍 COMPARAÇÃO DE EXTRAÇÕES - Espelhos RAW vs Base vs Agentes")
    print("=" * 80)
    
    # tipo de arquivo \d{12}.\d+.\d*.json (padrão)
    RE_ARQUIVOS_JSON_PADRAO = re.compile(r'^(\d{12})\.\d+\.\d*\.json$')
    
        # Instancia a classe de carga de dados
    carga = CargaDadosComparacao(
        pasta_origem=ORIGEM,
        pastas_destinos=DESTINOS,
        rotulo_id=ROTULO_ID,
        rotulo_origem=ROTULO_ORIGEM,
        rotulos_destinos=D_ROTULOS,
        campos_comparacao=CAMPOS_COMPARACAO,
        regex_arquivos=RE_ARQUIVOS_JSON_PADRAO
    )
    
    # Carrega os dados - agora retorna JsonAnaliseDados completo
    dados_analise = carga.carregar()
    
    # Exibe resumo dos dados
    print(dados_analise.resumo())

    if not dados_analise.dados:
        print("\n❌ Nenhum dado encontrado para comparação!")
        sys.exit(1)
    
    SO_ANALISE_ESTATISTICA = False # Configurar conforme necessidade
    
    if SO_ANALISE_ESTATISTICA:
        processar_analise_estatistica(dados_analise, PASTA_SAIDA_COMPARACAO)
        sys.exit(0)

    print(f"\n⚙️  Configuração de comparação:")
    print(f"   Campos analisados: {len(CAMPOS_COMPARACAO)}")
    print(f"   Nível de campos: {CONFIG_COMPARACAO.get('nivel_campos')}")
    print(f"   Campos BERTScore: {len(CONFIG_COMPARACAO.get('campos_bertscore', []))} → {CONFIG_COMPARACAO.get('campos_bertscore', [])}")
    print(f"   Campos ROUGE-L: {len(CONFIG_COMPARACAO.get('campos_rouge', []))} → {CONFIG_COMPARACAO.get('campos_rouge', [])}")
    print(f"   Campos ROUGE-1: {len(CONFIG_COMPARACAO.get('campos_rouge1', []))} → {CONFIG_COMPARACAO.get('campos_rouge1', [])}")
    print(f"   Campos ROUGE-2: {len(CONFIG_COMPARACAO.get('campos_rouge2', []))} → {CONFIG_COMPARACAO.get('campos_rouge2', [])} (+ (global) automático)")
    print(f"   📌 Nota: (estrutura) será adicionado automaticamente em ROUGE-1")

    # Cria analisador
    print(f"\n🚀 Iniciando análise com JsonAnaliseDataFrame...")
    analisador = JsonAnaliseDataFrame(
        dados_analise,  # Nova interface: passa JsonAnaliseDados
        config=CONFIG_COMPARACAO,
        pasta_analises=PASTA_SAIDA_COMPARACAO,
        max_workers=MAX_WORKERS_ANALISE,
        incluir_valores_analise=True,  # incluir valores nos JSONs de análise
        gerar_exemplos_md=True,  # Gera arquivo Markdown com exemplos
        max_exemplos_md_por_metrica=5,  # Máximo de 5 exemplos por métrica
        gerar_relatorio=True  # Gera relatório markdown
    )
    
    # Configura informações do relatório
    if analisador.relatorio:
        titulo_experimento = f"Comparação {ROTULO_ORIGEM} vs Modelos"
        descricao_experimento = f"Análise comparativa de extrações JSON usando múltiplas métricas (BERTScore, ROUGE, Levenshtein)"
        analisador.relatorio.set_overview(
            titulo=titulo_experimento,
            descricao=descricao_experimento,
            rotulos=analisador.rotulos,
            total_documentos=len(dados_analise.dados),
            campos_comparacao=CAMPOS_COMPARACAO
        )
        analisador.relatorio.set_config(CONFIG_COMPARACAO, CAMPOS_COMPARACAO)

    # Define nome base dos arquivos (usado pelos métodos de exportação)
    nome_arquivo_base = 'comparacao_extracoes'
    arquivo_excel = os.path.join(PASTA_SAIDA_COMPARACAO, f'{nome_arquivo_base}.xlsx')

    SO_GRAFICOS = False  # Define como True para gerar apenas gráficos de Excel existente
    if SO_GRAFICOS:
       # Apenas atualiza os gráficos do excel já existente 
        if os.path.isfile(arquivo_excel):
            print(f"\n⚠️  Aviso: O arquivo Excel de comparação já existe: {arquivo_excel}\nGerando gráficos...")
            analisador.gerar_graficos_de_excel(arquivo_excel, pasta_saida=PASTA_SAIDA_COMPARACAO)
            exit(0)
    
    SO_LLM_AS_A_JUDGE = False  # Define como True para usar LLM as a Judge
    if SO_LLM_AS_A_JUDGE:
        # Apenas atualiza as análises de LLM as a Judge do Excel existente
        if os.path.isfile(arquivo_excel):
            print(f"\n⚠️  Aviso: O arquivo Excel de comparação já existe: {arquivo_excel}\nAtualizando com análises de LLM as a Judge...")
            # Atualiza apenas a aba de avaliação LLM
            analisador.atualizar_avaliacao_llm_no_excel(arquivo_excel, gerar_graficos=True)
            print(f"\n✅ Aba 'Avaliação LLM' atualizada com sucesso!")
            print(f"📁 Arquivo: {arquivo_excel}")
            exit(0)
        else:
            print(f"\n❌ Erro: Arquivo Excel não encontrado: {arquivo_excel}")
            print(f"   Execute primeiro sem SO_LLM_AS_A_JUDGE=True para gerar o arquivo base.")
            exit(1)
    
    # Gera DataFrame
    print("📊 Gerando DataFrame...")
    df = analisador.to_df()
    
    print(f"\n✅ Análise concluída!")
    print(f"   Documentos processados: {len(df)}")
    print(f"   Colunas geradas: {len(df.columns)}")
    
    # Mostra estatísticas globais
    print("\n📈 Estatísticas Globais:")
    stats = analisador.estatisticas_globais()
    
    # NOVA ESTRUTURA: Múltiplas técnicas para (global)
    # Exemplos: (global)_rouge2_F1, (global)_rouge1_F1, etc.
    print("\n   📊 F1-Score por Técnica (campo global):")
    
    # Agrupa por técnica
    tecnicas_disponiveis = stats['tecnica'].unique()
    for tecnica in sorted(tecnicas_disponiveis):
        # Busca F1 global dessa técnica
        metrica_busca = f'(global)_{tecnica.lower().replace("-", "")}_F1'
        f1_tecnica = stats[stats['metrica'] == metrica_busca]
        
        if len(f1_tecnica) > 0:
            print(f"\n   {tecnica}:")
            for _, row in f1_tecnica.iterrows():
                print(f"      {row['modelo']:15s}: Mean={row['mean']:.4f}, Median={row['median']:.4f}, Std={row['std']:.4f}")
    
    # Busca métricas globais usando função auxiliar
    f1_global = _buscar_metricas_globais(stats)
    
    # Mostra comparação de modelos (usa métrica disponível)
    if len(f1_global) > 0:
        metrica_comparacao = f1_global.iloc[0]['metrica']
        print(f"\n🔍 Comparação por documento (métrica: {metrica_comparacao}):")
        try:
            comp_f1 = analisador.comparar_modelos(metrica_comparacao)
            print(comp_f1.head(10).to_string(index=False))
        except ValueError as e:
            print(f"\n   ⚠️  Erro ao comparar modelos: {e}")
    else:
        print("\n   ⚠️  Nenhuma métrica global F1 disponível para comparação")
    
    # Exporta resultados
    print("\n💾 Exportando resultados...")
    
    # CSV (método retorna o caminho do arquivo gerado)
    arquivo_csv = analisador.exportar_csv(nome_arquivo_base)
    arquivo_estatisticas = arquivo_csv.replace('.csv', '.estatisticas.csv')
    print(f"   ✓ CSV: {arquivo_csv}")
    print(f"   ✓ Estatísticas CSV: {arquivo_estatisticas}")
    
    # Excel com formatação avançada (mapas de calor)
    print("\n   Gerando Excel formatado com mapas de calor...")
    arquivo_excel = analisador.exportar_excel(
        nome_arquivo_base,  # Método adiciona .xlsx automaticamente
        incluir_estatisticas=True,
        usar_formatacao_avancada=True,  # Usa UtilPandasExcel com mapas de calor
        congelar_paineis=True,
        gerar_graficos=True  # Gráficos gerados separadamente
    )
    print(f"   ✓ Excel formatado: {arquivo_excel}")
    print(f"      • Aba 'Resultados': métricas por documento com mapa de calor")
    print(f"      • Aba 'Estatísticas': agregações globais")
    print(f"      • Aba 'Comparação_F1': comparação de modelos")
    
    # Resumo final
    print("\n📊 Resumo Final:")
    print(f"   Total de campos comparados: {len(CAMPOS_COMPARACAO)}")
    print(f"   Campos: {', '.join(CAMPOS_COMPARACAO[:3])}...")
    
    # Melhor modelo por F1 (reutiliza f1_global já calculado)
    if len(f1_global) > 0:
        idx_vencedor = f1_global['mean'].idxmax()
        modelo_vencedor = f1_global.loc[idx_vencedor, 'modelo']
        f1_vencedor = f1_global.loc[idx_vencedor, 'mean']
        metrica_vencedor = f1_global.iloc[0]['metrica']
        print(f"\n🏆 Melhor modelo ({metrica_vencedor}): {modelo_vencedor} (Mean={f1_vencedor:.4f})")
        
        # Mostra todas as métricas globais do vencedor (reutiliza padrão de busca)
        print(f"\n   Todas as métricas do modelo vencedor ({modelo_vencedor}):")
        stats_vencedor = stats[(stats['modelo'] == modelo_vencedor) & (stats['metrica'].str.contains(r'\(global\)_.*_F1', regex=True))]
        for _, row in stats_vencedor.iterrows():
            tecnica = row['tecnica']
            print(f"      {tecnica:12s} F1: Mean={row['mean']:.4f}, Median={row['median']:.4f}, Std={row['std']:.4f}")
    else:
        print("\n   ⚠️  Estatísticas não disponíveis para exibir vencedor")
    

    # Gera estatística também se não for só estatística (já que se fosse True teria saído antes)
    processar_analise_estatistica(dados_analise, PASTA_SAIDA_COMPARACAO)

    print("\n" + "=" * 80)
    print("✅ Comparação concluída com sucesso!")
    print(f"📁 Resultados salvos em: {PASTA_SAIDA_COMPARACAO}")
    print("=" * 80)


