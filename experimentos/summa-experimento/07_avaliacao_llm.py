#!/usr/bin/env python3
'''
Autor: Luiz Anísio

Roda o prompt de avaliação LLM-as-a-judge nas respostas das saídas dos modelos.
Roda na lista de protocolos selecionados, buscando a pasta de saída informada.
Monta a saída plana do que se espera para a escala likert, igual ao que é feito para a avaliação humana.

Guarda a saída no formato esperado pelo pacote de comparação na coluna de avaliação.

Fluxo de execução:
1. Para cada protocolo listado em PROTOCOLOS, localiza em PASTA_SAIDAS o parquet
   de saída correspondente (ex: saida_qwen7b(b)_teste.parquet).
2. Para cada documento no parquet, monta o prompt do juiz LLM substituindo os
   placeholders <<--TEXTO_ACORDAO-->> e <<--TEXTO_EXTRACAO-->> no template
   (avaliacao_llm_humana/prompt_juiz_llm.txt) com a íntegra do acórdão e a
   extração formatada em texto plano.
3. Envia os prompts ao modelo juiz (via util_vllm_batch.py) em 1 rodada, obtendo um JSON
   {"nota": 1..4, "problemas": [...]}.
3.1. Se ocorrer erro, tenta novamente por QTD_TENTATIVAS vezes   
4. Gera, para cada protocolo, a coluna 'avaliacao' no parquet de saída no
   formato esperado pelo pipeline de comparação: JSON com o campo "nota" 
   (onde nota é o Likert 1-4). Isso produz os arquivos
   {id}.avaliacao.json que o comparar_extracoes.py (06_compara_todos.yaml)
   consome para a análise bayesiana Likert do juiz LLM.
4.1. incluir no yaml de comparação:
   llm_as_a_judge: true
      campos_parquet:   
         avaliacao: "avaliacao" # nome da coluna do parquet
5. A saída plana replica o formato da avaliação humana (converter_label_studio_
   para_parquet.py), permitindo rodar realizar_avaliacoes.py sobre ambas as
   fontes com a mesma interface.

Parâmetros de linha de comando:

--refazer = ignora as colunas de avaliação e envia o prompt novamente, substituindo os valores existentes
> sem o --refazer = ignora as instâncieas com a coluna já preenchida

--protocolos "d24" "d25" seleciona os protocolos a serem processados
  Suporta também protocolos especiais:
  - "a": zero-shot (extrai teste de saida_qwen7b.parquet e salva em saida_qwen7b(a)_teste.parquet)
  - "prof": professor (extrai teste de saida_or_235b.parquet e salva em saida_or_235b(prof)_teste.parquet)
'''

import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
import statistics
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PASTA_SAIDA_EXTRACAO = './saida'
PASTA_SAIDA_AVALIACAO = './avaliacao_llm'
PROTOCOLOS = ['prof', 'a','b', 'd1', 'd24', 'd7', 'd8', 'c']
QTD_TENTATIVAS = 20
WORKERS_YAML = 20
#MODELO_JUIZ = 'oa:gpt5-chat:m:l' # h> 20000 | m>800 | l> 400 || sabia-4 aprox. R$ 85 por protocolo
MODELO_JUIZ = 'oa:gpt5:m:l' # h> 20000 | m>800 | l> 400 || sabia-4 aprox. R$ 85 por protocolo

# Caminhos derivados
ARQUIVO_INTEGRAS = os.path.join(SCRIPT_DIR, 'dados', 'integras_experimento_summa_novos.parquet')
ARQUIVO_DIVISAO = os.path.join(SCRIPT_DIR, 'dados', 'divisao_Qwen235b_Qwen7b.csv')
PROMPT_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, 'avaliacao_llm_humana', 'prompt_juiz_llm.txt')
UTIL_VLLM_BATCH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'src', 'util_vllm_batch.py'))

# Padrão de nome dos parquets de saída
PADRAO_PARQUET = 'saida_qwen7b({protocolo})_teste.parquet'

# Protocolos especiais que partem de extrações completas (22k) e precisam ser
# filtrados para o conjunto de teste canônico (3.948 instâncias):
PROTOCOLOS_ESPECIAIS = {
    'a': {
        'arquivo_origem': 'saida_qwen7b.parquet',
        'arquivo_teste': 'saida_qwen7b(a)_teste.parquet',
        'rotulo': 'Qwen7B (Zero-Shot)',
        'aliases': ['a', 'zero-shot', 'zeroshot'],
    },
    'prof': {
        'arquivo_origem': 'saida_or_235b.parquet',
        'arquivo_teste': 'saida_or_235b(prof)_teste.parquet',
        'rotulo': 'Qwen235B (Professor)',
        'aliases': ['prof', 'professor', 'or_235b', '235b', 'qwen235b'],
    },
}

# Placeholders do template do juiz
PH_ACORDAO = '<<--TEXTO_ACORDAO-->>'
PH_EXTRACAO = '<<--TEXTO_EXTRACAO-->>'

# Avaliação atribuída a extrações com JSON inválido (nota 1 = Inaceitável)
AVALIACAO_INVALIDA = json.dumps(
    {"nota": 1, "problemas": ["json_invalido"]}, ensure_ascii=False
)

# Estimativa de tokens de saída por instância (reasoning medium + resposta JSON)
# A resposta JSON é ~50 tokens, mas reasoning_tokens pode ser significativo.
# high ~2000, medium ~800, low ~400 tokens/instância.
TOKENS_SAIDA_ESTIMADO_POR_INSTANCIA = 800
MAX_TOKENS_SAIDA = 8192

# ROPE padrão para a análise bayesiana Likert (margem de equivalência prática)
# Calibrada a partir do controle negativo — corresponde ao rope_likert dos YAMLs de comparação.
ROPE_LIKERT = 0.1579  # calibrada pela divergência média entre os 3 avaliadores do grupo


# ---------------------------------------------------------------------------
# Formatação: JSON da extração → texto plano estruturado
# (reutilizado de avaliacao_llm_humana/01_gerar_entrada_juiz_llm.py)
# ---------------------------------------------------------------------------

def _formatar_lista(itens, rotulo_vazio="Não consta"):
    """Formata uma lista como itens de texto plano (um por linha com '- ')."""
    if not itens:
        return rotulo_vazio
    if isinstance(itens, str):
        return itens.strip() if itens.strip() else rotulo_vazio
    return "\n".join(f"- {item}" for item in itens)


def formatar_extracao_texto_plano(dados, chave_debug):
    """Converte o dict de extração SUMMA em texto plano para o prompt do juiz.

    Campos formatados (na ordem do prompt do juiz):
        MATÉRIA, TEMAS (PONTO, ARGUMENTOS, DOUTRINA, CONCEITOS), RESUMO.

    Raises:
        ValueError: se faltar campos obrigatórios.
    """
    if not isinstance(dados, dict):
        raise ValueError(f"[{chave_debug}] Extração não é um dicionário.")

    # --- Matéria ---
    materia = dados.get("Materia") or dados.get("Matéria", "")
    if not materia or not str(materia).strip():
        materia = "Não consta"

    # --- Temas ---
    temas = dados.get("Temas", [])
    if not isinstance(temas, list):
        raise ValueError(f"[{chave_debug}] Campo 'Temas' não é uma lista.")
    if len(temas) == 0:
        raise ValueError(f"[{chave_debug}] Campo 'Temas' é uma lista vazia.")

    # --- Resumo ---
    resumo = dados.get("Resumo", "")
    if not resumo or not str(resumo).strip():
        resumo = dados.get("Dispositivo", "")
    if not resumo or not str(resumo).strip():
        resumo = "Não consta"

    # --- Montar texto plano ---
    partes = [f"MATÉRIA:\n{materia}"]

    for i, tema in enumerate(temas, 1):
        if not isinstance(tema, dict):
            raise ValueError(f"[{chave_debug}] Tema {i} não é um dicionário.")

        partes.append(f"\n--- Tema {i} ---")
        partes.append(f"PONTO: {tema.get('Ponto', 'Não consta')}")
        partes.append(f"ARGUMENTOS:\n{_formatar_lista(tema.get('Argumentos', []))}")
        partes.append(f"DOUTRINA:\n{_formatar_lista(tema.get('Doutrina', []))}")
        partes.append(f"CONCEITOS:\n{_formatar_lista(tema.get('Conceitos', []))}")

    partes.append(f"\nRESUMO:\n{resumo}")
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Utilitários e Tratamento de Protocolos
# ---------------------------------------------------------------------------

def normalizar_protocolo(protocolo):
    """Normaliza o identificador do protocolo (ex: 'A' -> 'a', 'professor' -> 'prof')."""
    p = str(protocolo).strip().lower()
    for proto_key, cfg in PROTOCOLOS_ESPECIAIS.items():
        if p == proto_key or p in cfg.get('aliases', []):
            return proto_key
    return p


def eh_protocolo_especial(protocolo):
    """Retorna a configuração se o protocolo for especial ('a', 'prof'), ou None."""
    p = normalizar_protocolo(protocolo)
    return PROTOCOLOS_ESPECIAIS.get(p)


def caminho_parquet_leitura(protocolo):
    """Retorna o caminho do parquet para leitura do protocolo.

    Para protocolos especiais:
      1. Se o arquivo _teste já existir, prioriza ele (permite retomada).
      2. Senão, retorna o arquivo completo de origem (para filtrar).
    Para protocolos normais:
      Retorna saida_qwen7b({protocolo})_teste.parquet.
    """
    p = normalizar_protocolo(protocolo)
    cfg = eh_protocolo_especial(p)
    if cfg:
        caminho_teste = os.path.abspath(os.path.join(SCRIPT_DIR, PASTA_SAIDA_EXTRACAO, cfg['arquivo_teste']))
        if os.path.isfile(caminho_teste):
            return caminho_teste
        return os.path.abspath(os.path.join(SCRIPT_DIR, PASTA_SAIDA_EXTRACAO, cfg['arquivo_origem']))
    nome = PADRAO_PARQUET.format(protocolo=protocolo)
    return os.path.abspath(os.path.join(SCRIPT_DIR, PASTA_SAIDA_EXTRACAO, nome))


def caminho_parquet_destino(protocolo):
    """Retorna o caminho do parquet onde a avaliação deste protocolo deve ser salva.

    Para 'a': saida_qwen7b(a)_teste.parquet
    Para 'prof': saida_or_235b(prof)_teste.parquet
    Para demais protocolos: saida_qwen7b({protocolo})_teste.parquet
    """
    p = normalizar_protocolo(protocolo)
    cfg = eh_protocolo_especial(p)
    if cfg:
        return os.path.abspath(os.path.join(SCRIPT_DIR, PASTA_SAIDA_EXTRACAO, cfg['arquivo_teste']))
    nome = PADRAO_PARQUET.format(protocolo=protocolo)
    return os.path.abspath(os.path.join(SCRIPT_DIR, PASTA_SAIDA_EXTRACAO, nome))


def caminho_parquet_origem(protocolo):
    """Retorna o caminho do arquivo completo de origem se for especial, ou None."""
    p = normalizar_protocolo(protocolo)
    cfg = eh_protocolo_especial(p)
    if cfg:
        return os.path.abspath(os.path.join(SCRIPT_DIR, PASTA_SAIDA_EXTRACAO, cfg['arquivo_origem']))
    return None


def caminho_parquet_protocolo(protocolo):
    """Retorna o caminho do parquet para o protocolo (compatibilidade)."""
    return caminho_parquet_leitura(protocolo)


_CHAVES_TESTE_CACHE = None

def obter_chaves_teste():
    """Retorna o conjunto de chaves (strings) do split de teste canônico (3.948 instâncias).

    Estratégia:
      1. Tenta extrair diretamente da coluna 'chave' de qualquer parquet de teste
         já existente no diretório (ex: 'b', 'c', 'd24', 'd1'). Isso garante
         alinhamento 1:1 exato com os outros modelos.
      2. Fallback: lê ARQUIVO_DIVISAO ('alvo' == 'teste') cruzado com
         ARQUIVO_INTEGRAS ('fold' <= 10).
    """
    global _CHAVES_TESTE_CACHE
    if _CHAVES_TESTE_CACHE is not None:
        return _CHAVES_TESTE_CACHE

    # 1. Tentar ler de qualquer _teste.parquet existente
    for proto_ref in ['b', 'c', 'd24', 'd25', 'd1', 'd2', 'd7', 'd8']:
        nome_ref = PADRAO_PARQUET.format(protocolo=proto_ref)
        caminho_ref = os.path.abspath(os.path.join(SCRIPT_DIR, PASTA_SAIDA_EXTRACAO, nome_ref))
        if os.path.isfile(caminho_ref):
            try:
                df_ref = pd.read_parquet(caminho_ref, columns=['chave'])
                chaves = set(df_ref['chave'].astype(str).str.strip())
                logging.info(f"  → Chaves de teste obtidas de {nome_ref}: {len(chaves)} itens")
                _CHAVES_TESTE_CACHE = chaves
                return chaves
            except Exception as e:
                logging.warning(f"Falha ao ler chaves de {caminho_ref}: {e}")

    # 2. Fallback via ARQUIVO_DIVISAO + fold <= 10
    if os.path.isfile(ARQUIVO_DIVISAO) and os.path.isfile(ARQUIVO_INTEGRAS):
        try:
            df_div = pd.read_csv(ARQUIVO_DIVISAO, usecols=['id', 'alvo'])
            ids_teste = set(df_div[df_div['alvo'] == 'teste']['id'].astype(str).str.strip())
            df_int = pd.read_parquet(ARQUIVO_INTEGRAS, columns=['seq_documento_acordao', 'fold'])
            ids_fold = set(df_int[df_int['fold'] <= 10]['seq_documento_acordao'].astype(str).str.strip())
            chaves = ids_teste & ids_fold
            logging.info(f"  → Chaves de teste obtidas via split CSV + fold<=10: {len(chaves)} itens")
            _CHAVES_TESTE_CACHE = chaves
            return chaves
        except Exception as e:
            logging.warning(f"Falha no fallback de chaves de teste: {e}")

    raise RuntimeError("Não foi possível determinar as chaves de teste para filtrar o protocolo.")


def carregar_dataframe_protocolo(protocolo):
    """Carrega o DataFrame do protocolo, aplicando o filtro de teste se necessário.

    Retorna (df, caminho_leitura, caminho_destino, foi_filtrado).
    """
    p = normalizar_protocolo(protocolo)
    caminho_leitura = caminho_parquet_leitura(p)
    caminho_destino = caminho_parquet_destino(p)

    if not os.path.isfile(caminho_leitura):
        return None, caminho_leitura, caminho_destino, False

    df = pd.read_parquet(caminho_leitura)
    foi_filtrado = False

    # Se for protocolo especial e estiver lendo o arquivo completo (> 5000 linhas)
    cfg = eh_protocolo_especial(p)
    if cfg and len(df) > 5000:
        chaves_teste = obter_chaves_teste()
        antes = len(df)
        df = df[df['chave'].astype(str).str.strip().isin(chaves_teste)].copy()
        foi_filtrado = True
        logging.info(f"  [{cfg['rotulo']}] Filtrado de {antes} para {len(df)} instâncias de teste.")

    return df, caminho_leitura, caminho_destino, foi_filtrado


def sincronizar_parquet_origem(protocolo, df_avaliado):
    """Se for protocolo especial, sincroniza a coluna 'avaliacao' de volta no parquet completo original."""
    caminho_orig = caminho_parquet_origem(protocolo)
    caminho_dest = caminho_parquet_destino(protocolo)
    if not caminho_orig or not os.path.isfile(caminho_orig) or caminho_orig == caminho_dest:
        return

    try:
        logging.info(f"  → Sincronizando avaliações com o arquivo completo: {os.path.basename(caminho_orig)}...")
        df_orig = pd.read_parquet(caminho_orig)
        if 'avaliacao' not in df_orig.columns:
            df_orig['avaliacao'] = ''

        mapa_aval = dict(zip(df_avaliado['chave'].astype(str).str.strip(), df_avaliado['avaliacao']))
        mask = df_orig['chave'].astype(str).str.strip().isin(mapa_aval)
        df_orig.loc[mask, 'avaliacao'] = df_orig.loc[mask, 'chave'].astype(str).str.strip().map(mapa_aval)

        backup_parquet(caminho_orig)
        df_orig.to_parquet(caminho_orig, index=False)
        logging.info(f"  ✓ Parquet completo sincronizado: {os.path.basename(caminho_orig)} ({mask.sum()} linhas atualizadas)")
    except Exception as e:
        logging.warning(f"  ⚠️  Não foi possível sincronizar avaliações com {os.path.basename(caminho_orig)}: {e}")


def resposta_para_dict(resposta):
    """Converte a coluna resposta (str JSON ou dict) para dict. Retorna None se inválido."""
    if resposta is None or (isinstance(resposta, float) and pd.isna(resposta)):
        return None
    if isinstance(resposta, dict):
        return resposta
    if isinstance(resposta, str):
        resposta = resposta.strip()
        if not resposta:
            return None
        try:
            d = json.loads(resposta)
            return d if isinstance(d, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def extracao_valida(dados):
    """Verifica se o dict da extração tem os campos mínimos para avaliação."""
    if not isinstance(dados, dict):
        return False
    temas = dados.get("Temas", [])
    return isinstance(temas, list) and len(temas) > 0


def _tem_avaliacao(valor):
    """Verifica se o valor da coluna avaliacao é um preenchimento válido."""
    if pd.isna(valor):
        return False
    s = str(valor).strip()
    return s not in ('', '{}')


def _tem_erro_extracao(row):
    """Verifica se a linha do parquet teve erro na extração."""
    if 'erro' not in row.index:
        return False
    erro = row.get('erro')
    return pd.notna(erro) and str(erro).strip() != ''


def backup_parquet(caminho):
    """Cria backup .bak do parquet antes de modificar."""
    bak = caminho + '.bak'
    if os.path.exists(caminho):
        shutil.copy2(caminho, bak)
        logging.info(f"  Backup: {os.path.basename(bak)}")


# ---------------------------------------------------------------------------
# Geração do YAML para util_vllm_batch.py
# ---------------------------------------------------------------------------

def gerar_yaml(protocolo, arquivo_entrada, arquivo_saida):
    """Gera o conteúdo YAML para chamar o util_vllm_batch.py como juiz."""
    pasta_base = os.path.abspath(os.path.join(SCRIPT_DIR, PASTA_SAIDA_AVALIACAO))
    return f"""\
# Gerado automaticamente por 07_avaliacao_llm.py — protocolo {protocolo}

misc:
  pastas_base:
    - {pasta_base}

modelo:
  caminho: "{MODELO_JUIZ}"
  lora: ""

geracao:
  max_tokens: 8192
  temperature: 0.01
  top_k: 2
  top_p: 0.9
  batch_size: {WORKERS_YAML}
  tentativas: {QTD_TENTATIVAS}
  pausa_tentativas: 5

entrada:
  arquivo: "{arquivo_entrada}"
  campo_chave: "chave"
  campo_texto: "texto"
  prompt_template: ""
  variavel_texto: ""
  system_prompt: ""

saida:
  arquivo: "{arquivo_saida}"
  tipo_saida: "json"
"""


# ---------------------------------------------------------------------------
# Análise pré-execução (resumo para confirmação)
# ---------------------------------------------------------------------------

def analisar_protocolo(protocolo, df, refazer):
    """Analisa o estado de avaliação de um protocolo e retorna estatísticas."""
    total = len(df)

    # Contar já avaliados
    tem_col = 'avaliacao' in df.columns
    ja_avaliados = 0
    if tem_col and not refazer:
        ja_avaliados = sum(1 for v in df['avaliacao'] if _tem_avaliacao(v))

    # Identificar pendentes e contar inválidos entre eles
    json_invalidos = 0
    pendentes = 0

    for _, row in df.iterrows():
        # Pular já avaliados (se não --refazer)
        if tem_col and not refazer and _tem_avaliacao(row.get('avaliacao')):
            continue

        pendentes += 1

        # Verificar erro de extração
        if _tem_erro_extracao(row):
            json_invalidos += 1
            continue

        # Verificar JSON válido
        dados = resposta_para_dict(row.get('resposta'))
        if dados is None or not extracao_valida(dados):
            json_invalidos += 1
            continue

        # Verificar se formata sem erro
        try:
            formatar_extracao_texto_plano(dados, str(row.get('chave', '?')))
        except ValueError:
            json_invalidos += 1

    return {
        'protocolo': protocolo,
        'total': total,
        'ja_avaliados': ja_avaliados,
        'pendentes': pendentes,
        'json_invalidos': json_invalidos,
        'a_enviar': pendentes - json_invalidos,
    }


# ---------------------------------------------------------------------------
# Estimativa de tokens
# ---------------------------------------------------------------------------

def _contar_tokens_texto(texto):
    """Estima tokens a partir do texto (heurística: ~4 caracteres por token para GPT)."""
    return len(texto) // 4


def estimar_tokens_protocolos(protocolos, integras_idx, template, refazer):
    """Monta os prompts de cada protocolo e calcula estimativas de tokens.

    Não faz chamadas à API — apenas contabiliza os tokens de entrada e
    estima os de saída com base na configuração de max_tokens.
    """
    resultados = []

    for proto in protocolos:
        df, caminho_leitura, _, _ = carregar_dataframe_protocolo(proto)
        if df is None:
            print(f"  ✗ {proto:>5}:  ARQUIVO NÃO ENCONTRADO — {os.path.basename(caminho_leitura)}")
            continue

        tem_col = 'avaliacao' in df.columns

        tokens_entrada = []
        instancias_validas = 0
        instancias_invalidas = 0

        for _, row in df.iterrows():
            chave = str(row['chave'])

            # Pular já avaliados (se não --refazer)
            if tem_col and not refazer and _tem_avaliacao(row.get('avaliacao')):
                continue

            # Verificar erro ou JSON inválido
            if _tem_erro_extracao(row):
                instancias_invalidas += 1
                continue

            dados = resposta_para_dict(row.get('resposta'))
            if dados is None or not extracao_valida(dados):
                instancias_invalidas += 1
                continue

            try:
                texto_extracao = formatar_extracao_texto_plano(dados, chave)
            except ValueError:
                instancias_invalidas += 1
                continue

            integra = integras_idx.get(chave)
            if not integra or not str(integra).strip():
                instancias_invalidas += 1
                continue

            # Montar prompt e contar tokens
            prompt = template.replace(PH_ACORDAO, str(integra))
            prompt = prompt.replace(PH_EXTRACAO, texto_extracao)
            tokens_entrada.append(_contar_tokens_texto(prompt))
            instancias_validas += 1

        total_entrada = sum(tokens_entrada)
        media_entrada = total_entrada // len(tokens_entrada) if tokens_entrada else 0
        min_entrada = min(tokens_entrada) if tokens_entrada else 0
        max_entrada = max(tokens_entrada) if tokens_entrada else 0
        saida_estimada = instancias_validas * TOKENS_SAIDA_ESTIMADO_POR_INSTANCIA
        saida_maxima = instancias_validas * MAX_TOKENS_SAIDA

        resultados.append({
            'protocolo': proto,
            'instancias': instancias_validas,
            'invalidas': instancias_invalidas,
            'tokens_entrada_total': total_entrada,
            'tokens_entrada_media': media_entrada,
            'tokens_entrada_min': min_entrada,
            'tokens_entrada_max': max_entrada,
            'tokens_saida_estimado': saida_estimada,
            'tokens_saida_max': saida_maxima,
        })

    # ----- Exibir tabela -----
    if not resultados:
        print("  Nenhum protocolo válido para estimar.")
        return

    def _fmt(n):
        """Formata número com separador de milhar."""
        return f"{n:,.0f}".replace(',', '.')

    print(f"\n{'='*90}")
    print(f"  Estimativa de tokens (heurística: ~4 chars/token)")
    print(f"{'='*90}")
    print(f"  {'Proto':>6}  {'Inst':>6}  {'Inv':>5}  "
          f"{'Entrada Total':>15}  {'Média':>8}  {'Min':>8}  {'Max':>8}  "
          f"{'Saída Est.':>12}  {'Saída Máx.':>12}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*5}  "
          f"{'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*12}  {'-'*12}")

    total_inst = 0
    total_inv = 0
    total_entr = 0
    total_saida_est = 0
    total_saida_max = 0

    for r in resultados:
        print(
            f"  {r['protocolo']:>6}  {r['instancias']:>6}  {r['invalidas']:>5}  "
            f"{_fmt(r['tokens_entrada_total']):>15}  "
            f"{_fmt(r['tokens_entrada_media']):>8}  "
            f"{_fmt(r['tokens_entrada_min']):>8}  "
            f"{_fmt(r['tokens_entrada_max']):>8}  "
            f"{_fmt(r['tokens_saida_estimado']):>12}  "
            f"{_fmt(r['tokens_saida_max']):>12}"
        )
        total_inst += r['instancias']
        total_inv += r['invalidas']
        total_entr += r['tokens_entrada_total']
        total_saida_est += r['tokens_saida_estimado']
        total_saida_max += r['tokens_saida_max']

    print(f"  {'-'*6}  {'-'*6}  {'-'*5}  "
          f"{'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*12}  {'-'*12}")
    print(
        f"  {'TOTAL':>6}  {total_inst:>6}  {total_inv:>5}  "
        f"{_fmt(total_entr):>15}  {'':>8}  {'':>8}  {'':>8}  "
        f"{_fmt(total_saida_est):>12}  {_fmt(total_saida_max):>12}"
    )
    print(f"{'='*90}")
    print(f"\n  Resumo geral:")
    print(f"    Tokens de entrada:         {_fmt(total_entr)}")
    print(f"    Tokens de saída (estimado): {_fmt(total_saida_est)}  (~{TOKENS_SAIDA_ESTIMADO_POR_INSTANCIA} por instância)")
    print(f"    Tokens de saída (máximo):   {_fmt(total_saida_max)}  ({MAX_TOKENS_SAIDA} por instância)")
    print(f"    Total estimado:             {_fmt(total_entr + total_saida_est)}")
    print(f"    Total máximo:               {_fmt(total_entr + total_saida_max)}")
    print()


# ---------------------------------------------------------------------------
# Processamento de um protocolo
# ---------------------------------------------------------------------------

def processar_protocolo(protocolo, integras_idx, template, refazer):
    """Processa a avaliação LLM-as-a-judge para um protocolo completo.

    Fluxo:
        1. Carrega parquet de extração
        2. Separa instâncias: inválidas (nota 1 direta) vs. válidas (enviar ao juiz)
        3. Gera parquet de entrada e YAML, chama util_vllm_batch.py
        4. Mergeia resultados no parquet original e salva com backup

    Returns:
        True se processado com sucesso, False em caso de erro.
    """
    df, caminho_leitura, caminho_destino, foi_filtrado = carregar_dataframe_protocolo(protocolo)
    if df is None:
        logging.error(f"[{protocolo}] Parquet não encontrado: {caminho_leitura}")
        return False

    logging.info(f"\n{'='*60}")
    rotulo_extra = ""
    cfg = eh_protocolo_especial(protocolo)
    if cfg:
        rotulo_extra = f" ({cfg['rotulo']})"
    logging.info(f"  Protocolo: {protocolo}{rotulo_extra}")
    logging.info(f"  Origem : {os.path.basename(caminho_leitura)}")
    logging.info(f"  Destino: {os.path.basename(caminho_destino)}")
    logging.info(f"{'='*60}")

    if 'avaliacao' not in df.columns:
        df['avaliacao'] = ''

    # ----- Separar instâncias -----
    entradas_juiz = []          # [{chave, texto}] → enviar ao juiz
    avaliacoes_diretas = {}     # {chave: json_str} → nota 1 direta
    contagem_sem_integra = 0

    for _, row in df.iterrows():
        chave = str(row['chave'])

        # Pular já avaliados (se não --refazer)
        if not refazer and _tem_avaliacao(row.get('avaliacao')):
            continue

        # Erro de extração → nota 1
        if _tem_erro_extracao(row):
            avaliacoes_diretas[chave] = AVALIACAO_INVALIDA
            continue

        # Validar JSON da extração
        dados = resposta_para_dict(row.get('resposta'))
        if dados is None or not extracao_valida(dados):
            avaliacoes_diretas[chave] = AVALIACAO_INVALIDA
            continue

        # Formatar extração em texto plano
        try:
            texto_extracao = formatar_extracao_texto_plano(dados, chave)
        except ValueError as e:
            logging.warning(str(e))
            avaliacoes_diretas[chave] = AVALIACAO_INVALIDA
            continue

        # Buscar íntegra do acórdão
        integra = integras_idx.get(chave)
        if not integra or not str(integra).strip():
            contagem_sem_integra += 1
            avaliacoes_diretas[chave] = AVALIACAO_INVALIDA
            continue

        # Montar prompt do juiz
        prompt = template.replace(PH_ACORDAO, str(integra))
        prompt = prompt.replace(PH_EXTRACAO, texto_extracao)

        entradas_juiz.append({'chave': chave, 'texto': prompt})

    # ----- Gravar avaliações diretas (inválidos) -----
    if avaliacoes_diretas:
        logging.info(f"  → {len(avaliacoes_diretas)} instâncias inválidas → nota 1 direta")
        if contagem_sem_integra > 0:
            logging.warning(f"    ({contagem_sem_integra} sem íntegra encontrada)")
        chave_para_aval = avaliacoes_diretas
        for chave, aval in chave_para_aval.items():
            mask = df['chave'].astype(str) == chave
            df.loc[mask, 'avaliacao'] = aval

    if not entradas_juiz:
        logging.info("  → Nenhuma instância para enviar ao juiz.")
        if avaliacoes_diretas or foi_filtrado:
            backup_parquet(caminho_destino)
            df.to_parquet(caminho_destino, index=False)
            logging.info(f"  ✓ Parquet atualizado: {os.path.basename(caminho_destino)}")
            sincronizar_parquet_origem(protocolo, df)
        return True

    logging.info(f"  → {len(entradas_juiz)} instâncias para enviar ao juiz LLM")

    # ----- Gerar artefatos para util_vllm_batch.py -----
    pasta_trabalho = os.path.abspath(os.path.join(SCRIPT_DIR, PASTA_SAIDA_AVALIACAO))
    os.makedirs(pasta_trabalho, exist_ok=True)

    # Parquet de entrada do juiz
    arquivo_entrada = os.path.join(pasta_trabalho, f'entrada_juiz_{protocolo}.parquet')
    df_entrada = pd.DataFrame(entradas_juiz)
    df_entrada.to_parquet(arquivo_entrada, index=False)
    logging.info(f"  → Entrada: {os.path.basename(arquivo_entrada)} ({len(df_entrada)} linhas)")

    # Saída do juiz
    arquivo_saida_juiz = os.path.join(pasta_trabalho, f'saida_juiz_{protocolo}.parquet')
    if os.path.exists(arquivo_saida_juiz):
        logging.info(f"  → Saída anterior encontrada — itens já processados serão reaproveitados")

    # YAML de configuração
    yaml_content = gerar_yaml(protocolo, arquivo_entrada, arquivo_saida_juiz)
    yaml_path = os.path.join(pasta_trabalho, f'config_juiz_{protocolo}.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    logging.info(f"  → YAML: {os.path.basename(yaml_path)}")

    # ----- Chamar util_vllm_batch.py -----
    logging.info(f"  → Executando util_vllm_batch.py ({MODELO_JUIZ}, {WORKERS_YAML} workers)...")
    cmd = [sys.executable, UTIL_VLLM_BATCH, '--config', yaml_path]
    resultado = subprocess.run(cmd, cwd=pasta_trabalho)

    if resultado.returncode != 0:
        logging.error(f"  ✗ util_vllm_batch.py retornou código {resultado.returncode}")
        return False

    # ----- Ler saída do juiz e mergear -----
    if not os.path.isfile(arquivo_saida_juiz):
        logging.error(f"  ✗ Saída do juiz não encontrada: {arquivo_saida_juiz}")
        return False

    df_juiz = pd.read_parquet(arquivo_saida_juiz)
    logging.info(f"  → Saída do juiz carregada: {len(df_juiz)} linhas")

    # Indexar respostas do juiz por chave
    avaliacoes_ok = 0
    avaliacoes_erro_juiz = 0

    for _, row_juiz in df_juiz.iterrows():
        chave_juiz = str(row_juiz['chave'])
        mask = df['chave'].astype(str) == chave_juiz

        if not mask.any():
            continue

        # Verificar se o juiz teve erro
        erro_juiz = row_juiz.get('erro', '')
        if pd.notna(erro_juiz) and str(erro_juiz).strip():
            df.loc[mask, 'avaliacao'] = AVALIACAO_INVALIDA
            avaliacoes_erro_juiz += 1
            continue

        # Pegar resposta do juiz: {"nota": 1..4, "problemas": [...]}
        resp_juiz = row_juiz.get('resposta', '')
        juiz_dict = resposta_para_dict(resp_juiz)

        if juiz_dict and 'nota' in juiz_dict:
            # Garantir que é string JSON para a coluna do parquet
            if isinstance(resp_juiz, dict):
                aval_str = json.dumps(resp_juiz, ensure_ascii=False)
            else:
                aval_str = str(resp_juiz).strip()
            df.loc[mask, 'avaliacao'] = aval_str
            avaliacoes_ok += 1
        else:
            df.loc[mask, 'avaliacao'] = AVALIACAO_INVALIDA
            avaliacoes_erro_juiz += 1

    logging.info(f"  → Avaliações: {avaliacoes_ok} OK, {avaliacoes_erro_juiz} com erro do juiz")

    # ----- Salvar parquet atualizado no destino de teste -----
    backup_parquet(caminho_destino)
    df.to_parquet(caminho_destino, index=False)
    logging.info(f"  ✓ Parquet de teste salvo: {os.path.basename(caminho_destino)} ({len(df)} linhas)")

    # ----- Sincronizar com arquivo original se for protocolo especial -----
    sincronizar_parquet_origem(protocolo, df)

    return True


# ---------------------------------------------------------------------------
# Geração do resumo (--resumo)
# ---------------------------------------------------------------------------

def _extrair_nota(valor_avaliacao):
    """Extrai a nota numérica e se é json_invalido a partir do valor da coluna avaliacao.

    Returns:
        (nota, eh_invalido): nota int ou None, e se a nota 1 veio de json_invalido.
    """
    if pd.isna(valor_avaliacao):
        return None, False
    s = str(valor_avaliacao).strip()
    if not s or s == '{}':
        return None, False
    try:
        d = json.loads(s)
        nota = d.get('nota')
        if nota is None:
            return None, False
        nota = int(nota)
        problemas = d.get('problemas', [])
        eh_inv = nota == 1 and 'json_invalido' in problemas
        return nota, eh_inv
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, False


def _analisar_notas_protocolo(proto, df):
    """Analisa as notas da coluna avaliacao de um DataFrame.

    Returns:
        dict com métricas do protocolo ou None se não houver coluna avaliacao.
    """
    if 'avaliacao' not in df.columns:
        return None

    total = len(df)
    notas = []
    n1_invalido = 0
    n1_juiz = 0
    n2 = 0
    n3 = 0
    n4 = 0
    pendentes = 0

    for val in df['avaliacao']:
        nota, eh_inv = _extrair_nota(val)
        if nota is None:
            pendentes += 1
            continue
        notas.append(nota)
        if nota == 1:
            if eh_inv:
                n1_invalido += 1
            else:
                n1_juiz += 1
        elif nota == 2:
            n2 += 1
        elif nota == 3:
            n3 += 1
        elif nota == 4:
            n4 += 1

    avaliados = len(notas)
    pct_bom = (n3 + n4) / avaliados * 100 if avaliados > 0 else 0.0
    media = statistics.mean(notas) if notas else 0.0
    mediana = statistics.median(notas) if notas else 0.0

    return {
        'protocolo': proto,
        'total': total,
        'avaliados': avaliados,
        'pendentes': pendentes,
        'n1_inv': n1_invalido,
        'n1_juiz': n1_juiz,
        'n2': n2,
        'n3': n3,
        'n4': n4,
        'pct_gte3': pct_bom,
        'media': media,
        'mediana': mediana,
        'notas': notas,  # lista bruta para análise bayesiana
    }


def _ler_resumo_juiz(proto):
    """Lê o resumo.json do juiz para um protocolo, se existir."""
    caminho = os.path.join(SCRIPT_DIR, PASTA_SAIDA_AVALIACAO, f'saida_juiz_{proto}_resumo.json')
    if not os.path.isfile(caminho):
        return None
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _fmt_num(n):
    """Formata número inteiro com separador de milhar (ponto)."""
    return f"{n:,.0f}".replace(',', '.')


def gerar_resumo_avaliacao(protocolos, rope):
    """Gera o RESUMO_AVALIACAO_LLM.md com tabelas, CSVs e análise bayesiana."""
    from datetime import datetime

    pasta = os.path.join(SCRIPT_DIR, PASTA_SAIDA_AVALIACAO)
    os.makedirs(pasta, exist_ok=True)

    dados_protocolos = []  # lista de dicts com métricas por protocolo
    dados_notas = {}       # {proto: [notas]} para análise bayesiana
    dados_chave_nota = {}  # {proto: {chave: nota}} para pareamento

    print(f"\n{'='*70}")
    print(f"  Gerando resumo de avaliações")
    print(f"{'='*70}\n")

    for proto in protocolos:
        df, caminho_leitura, _, _ = carregar_dataframe_protocolo(proto)
        if df is None:
            print(f"  ✗ {proto:>5}: ARQUIVO NÃO ENCONTRADO")
            continue

        info = _analisar_notas_protocolo(proto, df)
        if info is None:
            print(f"  ✗ {proto:>5}: sem coluna 'avaliacao'")
            continue

        dados_protocolos.append(info)
        dados_notas[proto] = info['notas']

        # Guardar mapeamento chave→nota para pareamento bayesiano
        if 'chave' in df.columns and 'avaliacao' in df.columns:
            mapa = {}
            for _, row in df[['chave', 'avaliacao']].iterrows():
                nota, _ = _extrair_nota(row['avaliacao'])
                if nota is not None:
                    mapa[str(row['chave']).strip()] = nota
            dados_chave_nota[proto] = mapa

        status = '✓' if info['pendentes'] == 0 else '→'
        print(f"  {status} {proto:>5}: {info['avaliados']}/{info['total']} avaliados "
              f"(média {info['media']:.2f}, %≥3 {info['pct_gte3']:.1f}%)")

    if not dados_protocolos:
        logging.error("Nenhum protocolo com avaliações encontrado.")
        return

    # =====================================================================
    # 1. Tabela de avaliações (markdown + CSV)
    # =====================================================================
    md = []
    md.append('# Avaliação LLM-as-a-Judge — Resumo\n')
    md.append(f'> Gerado em: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  ')
    md.append(f'> ROPE Likert: {rope}  \n')

    md.append('## 1. Tabela de Avaliações por Protocolo\n')
    md.append('| Proto | Total | Avaliados | Pend. | N1 (inv.) | N1 (juiz) | N2 | N3 | N4 | %≥3 | Média | Mediana |')
    md.append('|-------|------:|----------:|------:|----------:|----------:|---:|---:|---:|----:|------:|--------:|')

    csv_rows = []
    for d in dados_protocolos:
        md.append(
            f"| {d['protocolo']} | {d['total']} | {d['avaliados']} | {d['pendentes']} | "
            f"{d['n1_inv']} | {d['n1_juiz']} | {d['n2']} | {d['n3']} | {d['n4']} | "
            f"{d['pct_gte3']:.1f}% | {d['media']:.2f} | {d['mediana']:.0f} |"
        )
        csv_rows.append(d)

    # Salvar CSV de avaliações
    df_csv = pd.DataFrame([{k: v for k, v in r.items() if k != 'notas'} for r in csv_rows])
    csv_path = os.path.join(pasta, 'resumo_avaliacoes.csv')
    df_csv.to_csv(csv_path, index=False, encoding='utf-8')
    md.append(f'\n> Dados exportados: `resumo_avaliacoes.csv`\n')

    # =====================================================================
    # 2. Custos e Tokens
    # =====================================================================
    custos = []
    for d in dados_protocolos:
        resumo_juiz = _ler_resumo_juiz(d['protocolo'])
        if resumo_juiz:
            custos.append({
                'protocolo': d['protocolo'],
                'tokens_entrada': resumo_juiz.get('input_tokens_total', 0),
                'tokens_saida': resumo_juiz.get('output_tokens_total', 0),
                'tempo_s': resumo_juiz.get('tempo_total_s', 0),
                'modelo': resumo_juiz.get('modelo_usado', ''),
            })

    if custos:
        md.append('## 2. Custos e Tokens\n')
        md.append('| Proto | Tokens Entrada | Tokens Saída | Tempo (h) | Modelo |')
        md.append('|-------|---------------:|-------------:|----------:|--------|')
        for c in custos:
            horas = c['tempo_s'] / 3600 if c['tempo_s'] else 0
            md.append(
                f"| {c['protocolo']} | {_fmt_num(c['tokens_entrada'])} | "
                f"{_fmt_num(c['tokens_saida'])} | {horas:.1f} | {c['modelo']} |"
            )
        md.append('')

    # =====================================================================
    # 3. Análise Bayesiana
    # =====================================================================
    # Filtrar protocolos completos (sem pendentes) para análise bayesiana
    protos_completos = [d['protocolo'] for d in dados_protocolos if d['pendentes'] == 0 and d['avaliados'] > 0]
    protos_incompletos = [d['protocolo'] for d in dados_protocolos if d['pendentes'] > 0]

    if len(protos_completos) >= 2:
        md.append('## 3. Análise Bayesiana (Likert)\n')
        md.append(f'> Protocolos incluídos: {", ".join(protos_completos)} (100% avaliados)  ')
        if protos_incompletos:
            md.append(f'> Protocolos excluídos: {", ".join(protos_incompletos)} (em andamento)  ')
        md.append('')

        try:
            # Importar o módulo de análise bayesiana
            sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'src')))
            from util_est_bayesiana import matriz_pares, sintese, heatmap, grafico_diferencas

            # Montar DataFrame pareado: alinhar por chave
            chaves_comuns = None
            for proto in protos_completos:
                if proto in dados_chave_nota:
                    chaves_proto = set(dados_chave_nota[proto].keys())
                    chaves_comuns = chaves_proto if chaves_comuns is None else chaves_comuns & chaves_proto

            if chaves_comuns and len(chaves_comuns) > 0:
                chaves_ordenadas = sorted(chaves_comuns)
                dados_pareados = {}
                for proto in protos_completos:
                    dados_pareados[proto] = [dados_chave_nota[proto][c] for c in chaves_ordenadas]

                df_pareado = pd.DataFrame(dados_pareados)
                n_pareados = len(chaves_ordenadas)

                md.append(f'> Instâncias pareadas: {n_pareados}  \n')

                # Calcular matriz de pares
                matriz = matriz_pares(df_pareado, rope=rope, nomes=protos_completos)

                # --- 3.1 Tabela de síntese ---
                tabela_sint = sintese(matriz)
                ciclos = tabela_sint.attrs.get('ciclos', [])

                md.append('### 3.1 Síntese por Protocolo\n')
                md.append('| Protocolo | Média | Superior a | Equivalente a | Inferior a | Incerto |')
                md.append('|-----------|------:|-----------:|--------------:|-----------:|--------:|')
                for proto_nome in tabela_sint.index:
                    row = tabela_sint.loc[proto_nome]
                    md.append(
                        f"| {proto_nome} | {row['média']:.4f} | "
                        f"{int(row.get('superior a', 0))} | "
                        f"{int(row.get('equivalente a', 0))} | "
                        f"{int(row.get('inferior a', 0))} | "
                        f"{int(row.get('incerto', 0))} |"
                    )

                ciclos_txt = 'nenhum' if not ciclos else str(ciclos)
                md.append(f'\n> Ciclos de transitividade: {ciclos_txt}  ')

                # Salvar CSV de síntese
                csv_sint_path = os.path.join(pasta, 'resumo_bayesiana_sintese.csv')
                tabela_sint.to_csv(csv_sint_path, encoding='utf-8')
                md.append(f'> Dados exportados: `resumo_bayesiana_sintese.csv`\n')

                # --- 3.2 Tabela de pares (simplificada) ---
                # Apenas pares não duplicados (i < j)
                pares_vistos = set()
                pares_unic = []
                for _, row in matriz.iterrows():
                    par = tuple(sorted([row['linha'], row['coluna']]))
                    if par not in pares_vistos:
                        pares_vistos.add(par)
                        pares_unic.append(row)

                md.append('### 3.2 Comparação entre Pares\n')
                md.append('| Par | P(sup) | P(equiv) | P(inf) | Δ média | IC 95% | Classif. |')
                md.append('|-----|-------:|---------:|-------:|--------:|--------|----------|')
                for row in pares_unic:
                    par_nome = f"{row['linha']} − {row['coluna']}"
                    ic = f"[{row['ic_inf']:+.4f}; {row['ic_sup']:+.4f}]"
                    md.append(
                        f"| {par_nome} | {row['p_esquerda']:.3f} | {row['p_rope']:.3f} | "
                        f"{row['p_direita']:.3f} | {row['diferenca_media']:+.4f} | {ic} | "
                        f"{row['classificacao']} |"
                    )

                # Salvar CSV de pares completo
                csv_pares_path = os.path.join(pasta, 'resumo_bayesiana_pares.csv')
                colunas_csv = ['linha', 'coluna', 'n', 'rope',
                               'p_esquerda', 'p_rope', 'p_direita',
                               'x_melhor', 'empate', 'y_melhor',
                               'diferenca_media', 'variancia', 'gl',
                               'ic_inf', 'ic_sup',
                               'media_linha', 'media_coluna',
                               'rope_minima', 'classificacao', 'probabilidade']
                colunas_presentes = [c for c in colunas_csv if c in matriz.columns]
                matriz[colunas_presentes].to_csv(csv_pares_path, index=False, encoding='utf-8')
                md.append(f'\n> Dados exportados: `resumo_bayesiana_pares.csv`\n')

                # --- 3.3 Heatmap ---
                try:
                    import matplotlib
                    matplotlib.use('Agg')  # backend não-interativo
                    heatmap_path = os.path.join(pasta, 'avaliacao_llm_heatmap.png')
                    heatmap(matriz, arquivo_saida=heatmap_path,
                            titulo='Avaliação LLM — Heatmap Bayesiano')
                    md.append('### 3.3 Heatmap\n')
                    md.append(f'![Heatmap bayesiano](avaliacao_llm_heatmap.png)\n')
                    print(f"  ✓ Heatmap salvo: avaliacao_llm_heatmap.png")
                except Exception as e:
                    logging.warning(f"  ⚠️  Erro ao gerar heatmap: {e}")

                # --- 3.4 Forest Plot ---
                try:
                    forest_path = os.path.join(pasta, 'avaliacao_llm_diferencas.png')
                    grafico_diferencas(matriz, arquivo_saida=forest_path,
                                       titulo='Avaliação LLM — Medindo as Diferenças')
                    md.append('### 3.4 Medindo as Diferenças (Forest Plot)\n')
                    md.append(f'![Forest plot bayesiano](avaliacao_llm_diferencas.png)\n')
                    print(f"  ✓ Forest plot salvo: avaliacao_llm_diferencas.png")
                except Exception as e:
                    logging.warning(f"  ⚠️  Erro ao gerar forest plot: {e}")

                import matplotlib.pyplot as plt
                plt.close('all')

            else:
                md.append('> ⚠️ Não foi possível parear os protocolos (chaves incompatíveis).\n')

        except ImportError as e:
            logging.warning(f"  ⚠️  util_est_bayesiana não disponível: {e}")
            md.append(f'> ⚠️ Análise bayesiana indisponível: {e}\n')
        except Exception as e:
            logging.warning(f"  ⚠️  Erro na análise bayesiana: {e}")
            md.append(f'> ⚠️ Erro na análise bayesiana: {e}\n')
    else:
        if len(protos_completos) == 1:
            md.append('## 3. Análise Bayesiana\n')
            md.append(f'> Apenas 1 protocolo completo ({protos_completos[0]}). '
                      f'São necessários ao menos 2 para a comparação bayesiana.\n')
        else:
            md.append('## 3. Análise Bayesiana\n')
            md.append('> Nenhum protocolo com 100% das avaliações concluídas. '
                      'A análise bayesiana será gerada quando houver ao menos 2 protocolos completos.\n')

    # =====================================================================
    # Salvar markdown
    # =====================================================================
    md_path = os.path.join(pasta, 'RESUMO_AVALIACAO_LLM.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')

    print(f"\n{'='*70}")
    print(f"  ✓ Resumo salvo: {md_path}")
    print(f"  ✓ CSV avaliações: {csv_path}")
    if len(protos_completos) >= 2:
        print(f"  ✓ CSV síntese bayesiana: {os.path.join(pasta, 'resumo_bayesiana_sintese.csv')}")
        print(f"  ✓ CSV pares bayesianos: {os.path.join(pasta, 'resumo_bayesiana_pares.csv')}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Avaliação LLM-as-a-judge para os protocolos de treinamento SUMMA'
    )
    parser.add_argument(
        '--refazer', action='store_true',
        help='Ignora avaliações existentes e reenvia tudo ao juiz'
    )
    parser.add_argument(
        '--protocolos', nargs='+', default=None,
        help=f'Protocolos a processar (padrão: {PROTOCOLOS})'
    )
    parser.add_argument(
        '--tokens', action='store_true',
        help='Estima tokens de entrada e saída por protocolo (sem chamar a API)'
    )
    parser.add_argument(
        '--resumo', action='store_true',
        help='Gera RESUMO_AVALIACAO_LLM.md com estatísticas, tabelas e análise bayesiana'
    )
    parser.add_argument(
        '--rope', type=float, default=ROPE_LIKERT,
        help=f'ROPE para análise bayesiana Likert (padrão: {ROPE_LIKERT})'
    )
    args = parser.parse_args()

    protocolos_raw = args.protocolos if args.protocolos else PROTOCOLOS
    protocolos = [normalizar_protocolo(p) for p in protocolos_raw]
    protocolos = list(dict.fromkeys(protocolos))

    # ----- Modo --resumo: gerar relatório e sair -----
    if args.resumo:
        gerar_resumo_avaliacao(protocolos, args.rope)
        return

    # ----- Validar arquivos essenciais -----
    arquivos_essenciais = [
        (ARQUIVO_INTEGRAS, "Parquet de íntegras"),
        (PROMPT_TEMPLATE_PATH, "Template do prompt do juiz"),
    ]
    if not args.tokens:
        arquivos_essenciais.append((UTIL_VLLM_BATCH, "util_vllm_batch.py"))
    for caminho, desc in arquivos_essenciais:
        if not os.path.isfile(caminho):
            logging.error(f"{desc} não encontrado: {caminho}")
            sys.exit(1)

    # ----- Carregar template do prompt -----
    with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template = f.read()

    if PH_ACORDAO not in template or PH_EXTRACAO not in template:
        logging.error("Placeholders não encontrados no template do juiz.")
        sys.exit(1)

    # ----- Carregar íntegras (indexar por chave) -----
    logging.info("Carregando íntegras...")
    df_integras = pd.read_parquet(ARQUIVO_INTEGRAS)
    integras_idx = {}
    for _, row in df_integras.iterrows():
        chave = str(row['seq_documento_acordao'])
        integra = str(row['integra']) if pd.notna(row['integra']) else ''
        integras_idx[chave] = integra
    logging.info(f"  → {len(integras_idx)} íntegras carregadas.")
    del df_integras  # liberar memória

    # ----- Modo --tokens: apenas estimar e sair -----
    if args.tokens:
        estimar_tokens_protocolos(protocolos, integras_idx, template, args.refazer)
        return

    # ----- Analisar protocolos e mostrar resumo -----
    print(f"\n{'='*70}")
    print(f"  Avaliação LLM-as-a-judge")
    print(f"{'='*70}")
    print(f"  Modelo juiz:    {MODELO_JUIZ}")
    print(f"  Workers:        {WORKERS_YAML}")
    print(f"  Tentativas:     {QTD_TENTATIVAS}")
    print(f"  Refazer:        {'Sim' if args.refazer else 'Não'}")
    print(f"  Protocolos:     {protocolos}")
    print(f"{'='*70}\n")

    resumos = []
    protocolos_validos = []

    for proto in protocolos:
        df, caminho_leitura, caminho_destino, foi_filtrado = carregar_dataframe_protocolo(proto)
        if df is None:
            print(f"  ✗ {proto:>5}:  ARQUIVO NÃO ENCONTRADO — {os.path.basename(caminho_leitura)}")
            continue

        info = analisar_protocolo(proto, df, args.refazer)
        resumos.append(info)
        protocolos_validos.append(proto)

        marcador = "→" if info['a_enviar'] > 0 else "✓"
        extra_desc = f" [{os.path.basename(caminho_destino)}]" if foi_filtrado else ""
        print(
            f"  {marcador} {proto:>5}:  "
            f"{info['total']:>5} total  |  "
            f"{info['ja_avaliados']:>5} avaliados  |  "
            f"{info['pendentes']:>5} pendentes  "
            f"({info['json_invalidos']} inválidos → nota 1,  "
            f"{info['a_enviar']} para o juiz){extra_desc}"
        )

    if not protocolos_validos:
        logging.error("Nenhum protocolo válido encontrado.")
        sys.exit(1)

    total_enviar = sum(r['a_enviar'] for r in resumos)
    total_invalidos = sum(r['json_invalidos'] for r in resumos)

    print(f"\n  Resumo: {total_enviar} chamadas ao juiz  +  {total_invalidos} notas diretas (inválidos)")

    if total_enviar == 0 and total_invalidos == 0:
        print("\n  ✓ Nada a fazer — todos os protocolos já estão avaliados.")
        return

    # ----- Confirmação -----
    resposta = input("\n  Continuar? [S/n]: ").strip().lower()
    if resposta and resposta not in ('s', 'sim', 'y', 'yes', ''):
        print("  Abortado pelo usuário.")
        return

    # ----- Processar cada protocolo sequencialmente -----
    for proto in protocolos_validos:
        ok = processar_protocolo(proto, integras_idx, template, args.refazer)
        if not ok:
            logging.error(f"Falha no protocolo '{proto}'. Abortando os próximos.")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  ✓ Avaliação concluída para {len(protocolos_validos)} protocolo(s).")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
