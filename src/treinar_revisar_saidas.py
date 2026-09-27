#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Script utilitário para revisar saídas de extração de LLMs (arquivos .parquet).
Inspeciona uma pasta de saída e exibe no console um resumo objetivo por
protocolo, além de gerar um arquivo RESUMO_EXTRACOES.md com tabela formatada
adequada para artigos ou dissertações.

Exemplo de uso dentro da pasta do experimento:
    python ../../src/treinar_revisar_saidas.py ./saida
    python ../../src/treinar_revisar_saidas.py ./saidas -d
"""

import os
import sys
import re
import glob
import json
import argparse
from datetime import datetime

# Garante que módulos em src/ sejam importáveis
_DIR_SRC = os.path.dirname(os.path.abspath(__file__))
if _DIR_SRC not in sys.path:
    sys.path.insert(0, _DIR_SRC)

try:
    from util_print import print_cores
except ImportError:
    def print_cores(*args, color_auto: bool = True, sep: str = ' ', end: str = '\n', file=None, flush: bool = False):
        msg = sep.join(str(a) for a in args)
        msg_limpa = re.sub(r'<[/]?[a-zA-Z0-9_]+>', '', msg)
        print(msg_limpa, end=end, file=file, flush=flush)

try:
    import pandas as pd
except ImportError:
    print_cores("<vermelho>❌ Erro: O pacote 'pandas' é obrigatório. Instale com: pip install pandas pyarrow</vermelho>")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Limiar (%) a partir do qual um protocolo é sinalizado para re-extração
# ---------------------------------------------------------------------------
_LIMIAR_REEXTRACAO = 30.0


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _chave_ordenacao_natural(texto: str):
    """Ordenação natural: d2 antes de d10."""
    partes = re.split(r'(\d+)', texto)
    return [int(p) if p.isdigit() else p.lower() for p in partes]


def _extrair_protocolo_e_modelo(nome_arquivo: str) -> tuple[str, str]:
    """
    Extrai (protocolo, modelo) a partir do nome do arquivo.
    Ex: saida_qwen7b(d11)_teste.parquet -> ('d11', 'qwen7b')
    """
    base = os.path.basename(nome_arquivo)
    base_sem_ext = re.sub(r'\.(parquet|parquet\.bak|csv|json)$', '', base, flags=re.IGNORECASE)

    match_proto = re.search(r'\(([^)]+)\)', base_sem_ext)
    if match_proto:
        proto = match_proto.group(1).strip()
        modelo_parte = base_sem_ext[:match_proto.start()].strip('_')
        modelo = re.sub(r'^saida_', '', modelo_parte, flags=re.IGNORECASE)
        return proto, modelo

    m_saida = re.sub(r'^saida_', '', base_sem_ext, flags=re.IGNORECASE)
    m_saida = re.sub(r'_teste$', '', m_saida, flags=re.IGNORECASE)
    return m_saida, m_saida


def _formatar_tempo(segundos) -> str:
    """Formata segundos em HHh MMm ou MMm SSs."""
    if segundos is None:
        return "-"
    segundos = round(float(segundos))
    m, s = divmod(segundos, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m:02d}m {s:02d}s"


def _formatar_numero(n) -> str:
    """Formata inteiro com separador de milhar (ponto)."""
    return f"{n:,}".replace(',', '.')


# ---------------------------------------------------------------------------
# Análise de um arquivo parquet
# ---------------------------------------------------------------------------
def _extrair_tempo_e_data_execucao(caminho_parquet: str) -> dict:
    """
    Recupera tempo total e data de extração de forma resiliente.
    Cruza o *_resumo.json com os logs de processamento (*_processamento.log ou *.parquet.log)
    para evitar valores parciais resultantes de re-execuções de erro ou checagem rápida.
    """
    base_nome = re.sub(r'\.parquet$', '', caminho_parquet, flags=re.IGNORECASE)
    res = {
        'tempo_total_s': None,
        'data_extracao': None,
        'tokens_entrada': None,
        'tokens_saida': None,
    }

    # 1. Procura e analisa log de processamento
    tempo_log = None
    data_fim_log = None
    candidatos_log = [
        f"{base_nome}_processamento.log",
        f"{caminho_parquet}.log",
        f"{base_nome}.log"
    ]
    for cand in candidatos_log:
        if os.path.exists(cand):
            try:
                with open(cand, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                inicios = re.findall(r'Processamento iniciado em:\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})', content)
                fins = re.findall(r'Processamento finalizado em:\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})', content)
                if not fins:
                    fins = re.findall(r'Batch finalizado em:\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})', content)
                if inicios and fins:
                    dt_ini = datetime.strptime(inicios[0], '%d/%m/%Y %H:%M:%S')
                    dt_fim = datetime.strptime(fins[-1], '%d/%m/%Y %H:%M:%S')
                    tempo_log = max(0.0, (dt_fim - dt_ini).total_seconds())
                    data_fim_log = dt_fim.strftime('%Y-%m-%d %H:%M:%S')
                    break
            except Exception:
                pass

    # 2. Procura e analisa _resumo.json
    caminho_resumo = f"{base_nome}_resumo.json"
    tempo_json = None
    data_json = None
    eh_parcial = False
    if os.path.exists(caminho_resumo):
        try:
            with open(caminho_resumo, 'r', encoding='utf-8') as f:
                rj = json.load(f)
            tempo_json = rj.get('tempo_total_s')
            data_json = rj.get('data_geracao')
            res['tokens_entrada'] = rj.get('input_tokens_total')
            res['tokens_saida'] = rj.get('output_tokens_total')

            p_ok = rj.get('processados_ok', 0) or 0
            p_err = rj.get('processados_erro', 0) or 0
            tot = rj.get('total_registros', 0) or 0
            if (tot > 50 and (p_ok + p_err) < tot * 0.5) or tempo_json == 0:
                eh_parcial = True
        except Exception:
            pass

    # 3. Consolidação inteligente:
    # Se houver log com duração relevante e o JSON for parcial/zerado ou significativamente menor, usa o log
    if tempo_log is not None:
        if tempo_json is None or eh_parcial or (tempo_log > 60 and tempo_json < (tempo_log * 0.5)):
            res['tempo_total_s'] = tempo_log
        else:
            res['tempo_total_s'] = tempo_json
    else:
        res['tempo_total_s'] = tempo_json

    res['data_extracao'] = data_fim_log or data_json
    return res


# ---------------------------------------------------------------------------
# Análise de um arquivo parquet
# ---------------------------------------------------------------------------
def _analisar_parquet(caminho: str) -> dict:
    """Lê o parquet e retorna um dicionário com todas as métricas relevantes."""
    info = {
        'arquivo': caminho,
        'nome_arquivo': os.path.basename(caminho),
        'total': 0, 'ok': 0, 'erro': 0,
        'pct_erro': 0.0,
        'detalhes_erros': {},
        'data_modificacao': '',
        'tempo_total_s': None,
        'tokens_entrada': None,
        'tokens_saida': None,
        'data_geracao': None,
        'erro_leitura': None,
    }

    if not os.path.exists(caminho):
        info['erro_leitura'] = "Arquivo não encontrado"
        return info

    stat = os.stat(caminho)
    info['data_modificacao'] = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')

    # Recupera métricas consolidadas de tempo e execução (log + json)
    exec_info = _extrair_tempo_e_data_execucao(caminho)
    info['tempo_total_s'] = exec_info['tempo_total_s']
    info['tokens_entrada'] = exec_info['tokens_entrada']
    info['tokens_saida'] = exec_info['tokens_saida']
    info['data_geracao'] = exec_info['data_extracao']

    # Leitura do DataFrame
    try:
        df = pd.read_parquet(caminho)
    except Exception as e:
        info['erro_leitura'] = f"Falha ao ler parquet: {e}"
        return info

    total = len(df)
    info['total'] = total
    if total == 0:
        return info

    # Máscara de erro
    tem_col_erro = 'erro' in df.columns
    tem_col_resp = 'resposta' in df.columns

    if tem_col_erro:
        s_erro = df['erro'].astype(str).str.strip()
        mask_erro = df['erro'].notna() & (~s_erro.isin(['', 'None', 'nan', 'False', '0']))
    else:
        mask_erro = pd.Series(False, index=df.index)

    if tem_col_resp:
        s_resp = df['resposta'].astype(str).str.strip()
        mask_sem_resp = df['resposta'].isna() | s_resp.isin(['', 'None', 'nan'])
    else:
        mask_sem_resp = pd.Series(False, index=df.index)

    mask_invalido = mask_erro | mask_sem_resp
    qtd_erros = int(mask_invalido.sum())
    qtd_ok = total - qtd_erros

    info['ok'] = qtd_ok
    info['erro'] = qtd_erros
    info['pct_erro'] = (qtd_erros / total) * 100.0 if total > 0 else 0.0

    # Agrupa tipos de erro
    if qtd_erros > 0:
        tipos = {}
        for _, row in df[mask_invalido].iterrows():
            motivo = ""
            if tem_col_erro and pd.notna(row['erro']) and str(row['erro']).strip() not in ['', 'None', 'nan']:
                motivo = str(row['erro']).strip()
            else:
                motivo = "Resposta vazia/nula"
            if len(motivo) > 80:
                motivo = motivo[:77] + "..."
            tipos[motivo] = tipos.get(motivo, 0) + 1
        info['detalhes_erros'] = tipos

    return info


# ---------------------------------------------------------------------------
# Descoberta da pasta de saída
# ---------------------------------------------------------------------------
def _encontrar_pasta(caminho_arg: str | None) -> str:
    if caminho_arg and os.path.exists(caminho_arg):
        return caminho_arg
    for c in ['./saida', './saidas', '.']:
        if os.path.isdir(c) and glob.glob(os.path.join(c, '*.parquet')):
            return c
    return caminho_arg or './saida'


# ---------------------------------------------------------------------------
# Geração do RESUMO_EXTRACOES.md
# ---------------------------------------------------------------------------
def _gerar_resumo_md(pasta_saida: str, registros: list, agora: str):
    """
    Gera (ou sobrescreve) RESUMO_EXTRACOES.md na pasta de saída com:
      - Cabeçalho com data/hora
      - Tabela principal de resultados (formato Markdown, acadêmico)
      - Tabela de cronologia (data e tempo de extração)
      - Seção de indicações de correção (protocolos com >30% de erro)
    """
    caminho_md = os.path.join(pasta_saida, "RESUMO_EXTRACOES.md")

    total_geral = sum(r['total'] for r in registros)
    total_ok = sum(r['ok'] for r in registros)
    total_erros = sum(r['erro'] for r in registros)
    pct_erro_geral = (total_erros / total_geral * 100.0) if total_geral > 0 else 0.0

    linhas = []
    linhas.append(f"# Resumo das Extrações")
    linhas.append("")
    linhas.append(f"- **Data do resumo:** {agora}")
    linhas.append(f"- **Pasta:** `{os.path.abspath(pasta_saida)}`")
    linhas.append(f"- **Total de protocolos:** {len(registros)}")
    linhas.append(f"- **Total de instâncias:** {_formatar_numero(total_geral)}")
    linhas.append(f"- **Sucesso geral:** {_formatar_numero(total_ok)} ({100.0 - pct_erro_geral:.2f}%)")
    linhas.append(f"- **Erros gerais:** {_formatar_numero(total_erros)} ({pct_erro_geral:.2f}%)")
    linhas.append("")

    # --- Tabela principal ---
    linhas.append("## Resultados por Protocolo")
    linhas.append("")
    linhas.append("| Protocolo | Total | Sucesso | Erros | % Erro | Obs |")
    linhas.append("|:----------|------:|--------:|------:|-------:|:----|")

    for r in registros:
        if r['erro_leitura']:
            linhas.append(f"| {r['protocolo']} | — | — | — | — | ❌ Erro de leitura |")
            continue

        obs = ""
        if r['pct_erro'] >= _LIMIAR_REEXTRACAO:
            obs = "⚠️ Re-extrair"
        elif r['erro'] == 0:
            obs = "✅"

        linhas.append(
            f"| {r['protocolo']} "
            f"| {_formatar_numero(r['total'])} "
            f"| {_formatar_numero(r['ok'])} "
            f"| {_formatar_numero(r['erro'])} "
            f"| {r['pct_erro']:.2f}% "
            f"| {obs} |"
        )

    # Linha de totais
    linhas.append(
        f"| **Total** "
        f"| **{_formatar_numero(total_geral)}** "
        f"| **{_formatar_numero(total_ok)}** "
        f"| **{_formatar_numero(total_erros)}** "
        f"| **{pct_erro_geral:.2f}%** "
        f"| |"
    )
    linhas.append("")

    # --- Tabela de cronologia ---
    registros_com_data = [r for r in registros if r['data_modificacao'] and not r['erro_leitura']]
    if registros_com_data:
        linhas.append("## Cronologia das Extrações")
        linhas.append("")
        linhas.append("| Protocolo | Data da Extração | Tempo |")
        linhas.append("|:----------|:-----------------|:------|")
        for r in registros_com_data:
            data_ext = r.get('data_geracao') or r['data_modificacao']
            tempo = _formatar_tempo(r['tempo_total_s'])
            linhas.append(f"| {r['protocolo']} | {data_ext} | {tempo} |")
        linhas.append("")

    # --- Seção de indicações de correção ---
    criticos = [r for r in registros if r['pct_erro'] >= _LIMIAR_REEXTRACAO and not r['erro_leitura']]
    if criticos:
        linhas.append("## ⚠️ Protocolos que Necessitam de Nova Extração")
        linhas.append("")
        linhas.append(f"Os protocolos listados abaixo apresentaram taxa de erro superior a {_LIMIAR_REEXTRACAO:.0f}%")
        linhas.append("e devem ser re-extraídos antes de serem incluídos em análises comparativas.")
        linhas.append("")
        for r in criticos:
            linhas.append(f"- **{r['protocolo']}** — {_formatar_numero(r['erro'])} erros de {_formatar_numero(r['total'])} ({r['pct_erro']:.2f}%)")
            if r['detalhes_erros']:
                for motivo, qtd in r['detalhes_erros'].items():
                    linhas.append(f"  - {_formatar_numero(qtd)}× {motivo}")
        linhas.append("")

    # --- Seção de detalhamento de falhas (sempre registrada no arquivo) ---
    com_falhas = [r for r in registros if (r['erro'] > 0 or r['erro_leitura'])]
    if com_falhas:
        linhas.append("## Detalhamento das Falhas por Protocolo")
        linhas.append("")
        linhas.append("Relação detalhada de todos os erros e inconsistências identificados por protocolo:")
        linhas.append("")
        for r in com_falhas:
            if r['erro_leitura']:
                linhas.append(f"- **{r['protocolo']}**: ❌ Erro de leitura ({r['erro_leitura']})")
                continue
            linhas.append(f"- **{r['protocolo']}** ({_formatar_numero(r['erro'])} falhas — {r['pct_erro']:.2f}% de erro):")
            if r['detalhes_erros']:
                for motivo, qtd in r['detalhes_erros'].items():
                    linhas.append(f"  - {_formatar_numero(qtd)}× {motivo}")
            else:
                linhas.append("  - Motivo não informado")
        linhas.append("")
    else:
        linhas.append("## Detalhamento das Falhas por Protocolo")
        linhas.append("")
        linhas.append("Nenhuma falha foi identificada nas extrações analisadas.")
        linhas.append("")

    # Escreve o arquivo
    with open(caminho_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))

    return caminho_md


# ---------------------------------------------------------------------------
# Exibição no console
# ---------------------------------------------------------------------------
def _exibir_console(registros: list, pasta_base: str, agora: str, detalhes: bool = False):
    """Imprime a tabela resumo no console com cores."""

    print()
    print_cores("<azul>════════════════════════════════════════════════════════════════════════════</azul>")
    print_cores("<negrito>📊 REVISÃO DE SAÍDAS DE EXTRAÇÃO</negrito>")
    print_cores(f"<cinza>Pasta: {os.path.abspath(pasta_base)}</cinza>")
    print_cores(f"<cinza>Data:  {agora}</cinza>")
    print_cores("<azul>════════════════════════════════════════════════════════════════════════════</azul>")
    print()

    # Largura dinâmica da coluna protocolo
    col_p = max(10, max((len(r['protocolo']) for r in registros), default=10))

    cabecalho = (
        f"{'Protocolo':<{col_p}}  "
        f"{'Total':>6}  "
        f"{'Sucesso':>7}  "
        f"{'Erros':>6}  "
        f"{'% Erro':>7}  "
        f"{'Tempo':>8}  "
        f"{'Extração':<19}"
    )
    print_cores(f"<negrito>{cabecalho}</negrito>")
    print_cores("─" * len(cabecalho))

    total_g, ok_g, erros_g = 0, 0, 0

    for r in registros:
        total_g += r['total']
        ok_g += r['ok']
        erros_g += r['erro']

        if r['erro_leitura']:
            print_cores(
                f"{r['protocolo']:<{col_p}}  "
                f"{'—':>6}  {'—':>7}  {'—':>6}  {'—':>7}  {'—':>8}  "
                f"<vermelho>❌ Erro de leitura</vermelho>"
            , color_auto=False)
            continue

        pct = r['pct_erro']
        tempo = _formatar_tempo(r['tempo_total_s'])
        data_ext = r.get('data_geracao') or r['data_modificacao']

        # Ícone de status
        if pct >= _LIMIAR_REEXTRACAO:
            icone = " <vermelho>⚠️ RE-EXTRAIR</vermelho>"
        elif r['erro'] == 0:
            icone = ""
        else:
            icone = ""

        # Cores para o percentual de erro
        if pct >= _LIMIAR_REEXTRACAO:
            pct_str = f"<vermelho>{pct:>6.2f}%</vermelho>"
            erros_str = f"<vermelho>{r['erro']:>6}</vermelho>"
        elif pct > 0:
            pct_str = f"<amarelo>{pct:>6.2f}%</amarelo>"
            erros_str = f"<amarelo>{r['erro']:>6}</amarelo>"
        else:
            pct_str = f"<cinza>{pct:>6.2f}%</cinza>"
            erros_str = f"<cinza>{r['erro']:>6}</cinza>"

        linha = (
            f"<negrito>{r['protocolo']:<{col_p}}</negrito>  "
            f"{r['total']:>6}  "
            f"{r['ok']:>7}  "
            f"{erros_str}  "
            f"{pct_str}  "
            f"{tempo:>8}  "
            f"{data_ext:<19}"
            f"{icone}"
        )
        print_cores(linha, color_auto=False)

    print_cores("─" * len(cabecalho))

    pct_g = (erros_g / total_g * 100.0) if total_g > 0 else 0.0
    print_cores(
        f"<negrito>{'TOTAL':<{col_p}}  "
        f"{total_g:>6}  "
        f"{ok_g:>7}  "
        f"{erros_g:>6}  "
        f"{pct_g:>6.2f}%</negrito>"
    )
    print()

    # Detalhamento de erros (protocolos com >30% ou com -d)
    criticos = [r for r in registros if r['pct_erro'] >= _LIMIAR_REEXTRACAO and not r['erro_leitura']]
    mostrar_detalhes = [r for r in registros if (r['erro'] > 0 or r['erro_leitura'])]

    if detalhes and mostrar_detalhes:
        print_cores("<azul>🔍 DETALHAMENTO DAS FALHAS (CONSOLE):</azul>")
        for r in mostrar_detalhes:
            if r['erro_leitura']:
                print_cores(f"   <vermelho>[{r['protocolo']}]</vermelho> Erro de leitura: {r['erro_leitura']}")
                continue
            pct = r['pct_erro']
            cor = 'vermelho' if pct >= _LIMIAR_REEXTRACAO else 'amarelo'
            print_cores(f"   <{cor}>[{r['protocolo']}]</{cor}> {r['erro']} erro(s) ({pct:.2f}%):")
            for motivo, qtd in r['detalhes_erros'].items():
                print_cores(f"      • {qtd}×: <cinza>{motivo}</cinza>")
        print()
    elif criticos:
        print_cores("<vermelho>⚠️  PROTOCOLOS COM TAXA DE ERRO CRÍTICA (≥30%):</vermelho>")
        for r in criticos:
            print_cores(f"   • <negrito>{r['protocolo']}</negrito> — {r['erro']} erros de {r['total']} ({r['pct_erro']:.2f}%)")
            for motivo, qtd in r['detalhes_erros'].items():
                print_cores(f"      {qtd}×: <cinza>{motivo}</cinza>")
        print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Revisor de saídas de extração (.parquet). Gera resumo no console e RESUMO_EXTRACOES.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python treinar_revisar_saidas.py ./saida
  python treinar_revisar_saidas.py ./saidas -d
  python treinar_revisar_saidas.py ./saida/saida_qwen7b(d2)_teste.parquet
        """
    )
    parser.add_argument(
        'caminho', nargs='?', default=None,
        help="Pasta de saídas (ex: ./saida) ou arquivo .parquet específico."
    )
    parser.add_argument(
        '-d', '--detalhes', action='store_true',
        help="Exibe no console o detalhamento dos tipos de erro para todos os protocolos com falhas."
    )

    args = parser.parse_args()
    caminho = _encontrar_pasta(args.caminho)

    # Identifica arquivos
    arquivos = []
    pasta_base = ""
    if os.path.isfile(caminho):
        arquivos = [caminho]
        pasta_base = os.path.dirname(caminho) or "."
    elif os.path.isdir(caminho):
        pasta_base = caminho
        todos = sorted(glob.glob(os.path.join(caminho, "*.parquet")))
        arquivos = [f for f in todos if not f.endswith('.parquet.bak')]
    else:
        print_cores(f"<vermelho>❌ Erro: O caminho '{caminho}' não foi encontrado.</vermelho>")
        sys.exit(1)

    if not arquivos:
        print_cores(f"<amarelo>⚠️  Nenhum arquivo .parquet encontrado em: '{caminho}'</amarelo>")
        sys.exit(0)

    # Análise
    registros = []
    for arq in arquivos:
        info = _analisar_parquet(arq)
        proto, modelo = _extrair_protocolo_e_modelo(arq)
        info['protocolo'] = proto
        info['modelo'] = modelo
        registros.append(info)

    registros.sort(key=lambda r: _chave_ordenacao_natural(r['protocolo']))

    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Console
    _exibir_console(registros, pasta_base, agora, detalhes=args.detalhes)

    # Markdown
    caminho_md = _gerar_resumo_md(pasta_base, registros, agora)
    print_cores(f"<verde>💾 Resumo salvo em: {caminho_md}</verde>")
    print()


if __name__ == '__main__':
    main()