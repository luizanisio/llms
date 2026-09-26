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
'''

import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PASTA_SAIDA_EXTRACAO = './saida'
PASTA_SAIDA_AVALIACAO = './avaliacao_llm'
PROTOCOLOS = ['b', 'd24', 'd25', 'c']
QTD_TENTATIVAS = 20
WORKERS_YAML = 10
MODELO_JUIZ = 'oa:gpt5:m:l' # h> 20000 | m>800 | l> 400 || sabia-4 aprox. R$ 85 por protocolo

# Caminhos derivados
ARQUIVO_INTEGRAS = os.path.join(SCRIPT_DIR, 'dados', 'integras_experimento_summa_novos.parquet')
PROMPT_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, 'avaliacao_llm_humana', 'prompt_juiz_llm.txt')
UTIL_VLLM_BATCH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'src', 'util_vllm_batch.py'))

# Padrão de nome dos parquets de saída
PADRAO_PARQUET = 'saida_qwen7b({protocolo})_teste.parquet'

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
# Utilitários
# ---------------------------------------------------------------------------

def caminho_parquet_protocolo(protocolo):
    """Retorna o caminho absoluto do parquet de saída do protocolo."""
    nome = PADRAO_PARQUET.format(protocolo=protocolo)
    return os.path.abspath(os.path.join(SCRIPT_DIR, PASTA_SAIDA_EXTRACAO, nome))


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
        caminho = caminho_parquet_protocolo(proto)
        if not os.path.isfile(caminho):
            print(f"  ✗ {proto:>5}:  ARQUIVO NÃO ENCONTRADO")
            continue

        df = pd.read_parquet(caminho)
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
    caminho_parquet = caminho_parquet_protocolo(protocolo)
    if not os.path.isfile(caminho_parquet):
        logging.error(f"[{protocolo}] Parquet não encontrado: {caminho_parquet}")
        return False

    logging.info(f"\n{'='*60}")
    logging.info(f"  Protocolo: {protocolo}")
    logging.info(f"{'='*60}")

    df = pd.read_parquet(caminho_parquet)
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
        if avaliacoes_diretas:
            backup_parquet(caminho_parquet)
            df.to_parquet(caminho_parquet, index=False)
            logging.info(f"  ✓ Parquet atualizado: {os.path.basename(caminho_parquet)}")
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

    # ----- Salvar parquet atualizado -----
    backup_parquet(caminho_parquet)
    df.to_parquet(caminho_parquet, index=False)
    logging.info(f"  ✓ Parquet atualizado: {os.path.basename(caminho_parquet)}")

    return True


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
    args = parser.parse_args()

    protocolos = args.protocolos if args.protocolos else PROTOCOLOS

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
        caminho = caminho_parquet_protocolo(proto)
        if not os.path.isfile(caminho):
            print(f"  ✗ {proto:>5}:  ARQUIVO NÃO ENCONTRADO — {os.path.basename(caminho)}")
            continue

        df = pd.read_parquet(caminho)
        info = analisar_protocolo(proto, df, args.refazer)
        resumos.append(info)
        protocolos_validos.append(proto)

        marcador = "→" if info['a_enviar'] > 0 else "✓"
        print(
            f"  {marcador} {proto:>5}:  "
            f"{info['total']:>5} total  |  "
            f"{info['ja_avaliados']:>5} avaliados  |  "
            f"{info['pendentes']:>5} pendentes  "
            f"({info['json_invalidos']} inválidos → nota 1,  "
            f"{info['a_enviar']} para o juiz)"
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
