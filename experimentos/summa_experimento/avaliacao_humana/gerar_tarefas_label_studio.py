import sys
import sys
sys.path.append('../../../src')
from util_print import print_cores
import os
import json
import random
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

QTD_ITENS = 70

# Caminhos dos arquivos
# ARQUIVO_DIVISOES_QWEN: CSV gerado na fase de curadoria/avaliação, que contém o ID do documento ('id' ou 'seq_documento_acordao')
# e a coluna 'dificuldade' calculada previamente a partir das extrações do modelo qwen 7b (ex: percentis P30/P70 do score S_i).
ARQUIVO_DIVISOES_QWEN = '../compara/analises_comparacao_summa_q235 (full-base)/divisoes/divisao_Qwen235b_Qwen7b.csv'
ARQUIVO_DIVISAO_AMOSTRA = './arquivo_avaliacao.csv'
ARQUIVO_TAREFAS = './tarefas_avaliacao_label_studio.json'

# ARQUIVO_INTEGRAS: Parquet contendo os metadados brutos e o texto original de cada acórdão.
# Utilizado para extrair o 'fold' de cada documento, a íntegra e informações como 'sg_classe' e 'dt_publicacao'.
ARQUIVO_INTEGRAS = '../dados/integras_experimento_summa_novos.parquet'

MODELOS_ARQUIVOS = {
    'qwen7b': '../saida/saida_qwen7b.parquet',
    'qwen235b': '../saida/saida_or_235b.parquet',
    'gpt5': '../saida/saida_oa_gpt5.parquet'
}

# Regras de amostragem conforme README_AVALAICAO.md
REGRAS_AMOSTRAGEM = {
    'treino': {'Facil': 6, 'Medio': 6, 'Dificil': 6},
    'validacao': {'Facil': 5, 'Medio': 5, 'Dificil': 5},
    'teste': {'Facil': 5, 'Medio': 5, 'Dificil': 5},
    'ineditos': {'Facil': 7, 'Medio': 7, 'Dificil': 8}
}

def formatar_lista(lista):
    if not lista or (isinstance(lista, list) and len(lista) == 0):
        return "<em style='color:#999;'>Não consta</em>"
    if isinstance(lista, str):
        if lista.strip() == "":
            return "<em style='color:#999;'>Não consta</em>"
        return f"<ul style='margin:0;padding-left:20px;'><li>{lista}</li></ul>"
    
    html = "<ul style='margin:0;padding-left:20px;'>"
    for item in lista:
        html += f"<li>{item}</li>"
    html += "</ul>"
    return html

def formatar_html(json_str):
    try:
        dados = json.loads(json_str)
    except Exception:
        # Se falhar o parse, retorna string pura num div
        return f"<div style='color:red;'>Erro ao ler JSON:</div><pre>{json_str}</pre>"

    materia = dados.get('Materia', "<em style='color:#999;'>Não consta</em>")
    resumo = dados.get('Dispositivo', "<em style='color:#999;'>Não consta</em>")
    if 'Resumo' in dados: # pode chamar de Resumo em outros modelos
        resumo = dados['Resumo']

    temas_html = ""
    temas = dados.get('Temas', [])
    if isinstance(temas, list):
        for i, tema in enumerate(temas, 1):
            ponto = tema.get('Ponto', "<em style='color:#999;'>Não consta</em>")
            argumentos = tema.get('Argumentos', [])
            doutrina = tema.get('Doutrina', [])
            conceitos = tema.get('Conceitos', [])

            doutrina_str = doutrina
            if isinstance(doutrina, list):
                if len(doutrina) == 0:
                    doutrina_str = "<em style='color:#999;'>Não consta</em>"
                else:
                    doutrina_str = "<ul style='margin:0;padding-left:20px;'>" + "".join(f"<li>{d}</li>" for d in doutrina) + "</ul>"

            temas_html += f"""
<div style='border-left:3px solid #c06000;padding:8px 12px;margin-bottom:12px;background:#fff8f2;border-radius:0 4px 4px 0;'>
  <div style='font-weight:bold;color:#8a3a00;margin-bottom:6px;'>— Tema {i} —</div>
  <p style='margin:3px 0;'><strong>PONTO:</strong> {ponto}</p>
  <p style='margin:8px 0 3px;'><strong>ARGUMENTOS:</strong></p>
  {formatar_lista(argumentos)}
  <p style='margin:8px 0 3px;'><strong>DOUTRINA:</strong></p>
  {formatar_lista(doutrina) if isinstance(doutrina, list) else f"<p style='margin:3px 0;'>{doutrina_str}</p>"}
  <p style='margin:8px 0 3px;'><strong>CONCEITOS:</strong></p>
  {formatar_lista(conceitos)}
</div>"""

    html = f"""<div style='font-family:Georgia,serif;line-height:1.7;font-size:0.93em;color:#222;'>
<div style='background:#ddeedd;padding:7px 12px;border-radius:5px;margin-bottom:12px;'>
  <strong style='color:#1a4a1a;'>MATÉRIA</strong>
  <p style='margin:5px 0 0 0;'>{materia}</p>
</div>
<div style='background:#ddeedd;padding:5px 12px;border-radius:5px;margin-bottom:12px;'>
  <strong style='color:#1a4a1a;'>TEMAS</strong>
</div>
{temas_html}
<div style='background:#ddeedd;padding:7px 12px;border-radius:5px;'>
  <strong style='color:#1a4a1a;'>RESUMO</strong>
  <p style='margin:5px 0 0 0;'>{resumo}</p>
</div>
</div>"""
    return html.replace('\n', '')


def criar_arquivo_divisao():
    ''' Se o arquivo de amostra já existir, carrega-o. 
        Caso contrário, lê ARQUIVO_DIVISOES_QWEN e ARQUIVO_INTEGRAS,
        faz o cruzamento, aplica amostragem estratificada e cria a amostra.
    '''
    if os.path.exists(ARQUIVO_DIVISAO_AMOSTRA):
        logging.info(f"O arquivo '{ARQUIVO_DIVISAO_AMOSTRA}' já existe. Carregando...")
        return pd.read_csv(ARQUIVO_DIVISAO_AMOSTRA)
    
    if not os.path.exists(ARQUIVO_DIVISOES_QWEN) or not os.path.exists(ARQUIVO_INTEGRAS):
        logging.error("Arquivos de divisão ou de íntegras não encontrados.")
        return None

    logging.info("Lendo bases de dados para gerar a divisão...")
    # Lê a dificuldade das extrações qwen7b
    df_div = pd.read_csv(ARQUIVO_DIVISOES_QWEN)
    # Lê as íntegras para obter os agrupamentos de folds
    df_int = pd.read_parquet(ARQUIVO_INTEGRAS, columns=['seq_documento_acordao', 'fold'])
    
    # Padronizar a coluna de identificador primário para permitir o merge
    if 'id' in df_div.columns:
        df_div = df_div.rename(columns={'id': 'seq_documento_acordao'})
        
    df = pd.merge(df_div, df_int, on='seq_documento_acordao', how='inner')
    
    # Normalizar a dificuldade
    if 'dificuldade' in df.columns:
        df['dificuldade'] = df['dificuldade'].astype(str).str.title().str.strip()
        df['dificuldade'] = df['dificuldade'].replace({'Fácil': 'Facil', 'Médio': 'Medio', 'Difícil': 'Dificil'})
    else:
        logging.error("Coluna 'dificuldade' não encontrada no arquivo de divisões.")
        return None

    # Regra de conjuntos (Folds e Alvo):
    # - Fold 12 é reservado para o conjunto de Inéditos (sobrescrevemos o alvo original).
    # - Folds <= 10 são usados de acordo com a coluna 'alvo' original (treino, validacao, teste).
    df['alvo'] = df['alvo'].astype(str).str.lower().str.strip()
    df.loc[df['fold'] == 12, 'alvo'] = 'ineditos'
    
    pool_ineditos = df[df['alvo'] == 'ineditos']
    pool_antigos = df[df['fold'] <= 10]
    
    amostras = []
    # Set seed
    random_state = 42
    
    for alvo_str, qtds in REGRAS_AMOSTRAGEM.items():
        for dif, qtd_desejada in qtds.items():
            if alvo_str == 'ineditos':
                # Inéditos vêm exclusivamente do fold 12
                filtro = pool_ineditos[pool_ineditos['dificuldade'] == dif]
            else:
                # Para treino, validacao e teste, usamos o pool de folds antigos respeitando a coluna 'alvo' original
                filtro = pool_antigos[(pool_antigos['dificuldade'] == dif) & (pool_antigos['alvo'] == alvo_str)]
            
            if len(filtro) < qtd_desejada:
                logging.warning(f"Faltam documentos para {alvo_str} - {dif}. Desejado: {qtd_desejada}, Disponível: {len(filtro)}")
                amostra = filtro.copy()
            else:
                # Amostragem determinística usando random_state=42 para garantia de reprodutibilidade
                amostra = filtro.sample(n=qtd_desejada, random_state=random_state)
                
            amostras.append(amostra)
            
    df_final = pd.concat(amostras).reset_index(drop=True)
    
    # Manter apenas as colunas essenciais para o CSV de avaliação
    colunas_essenciais = ['seq_documento_acordao', 'dificuldade', 'alvo', 'fold']
    # Caso 'alvo' não exista na base, garantimos que não ocorra erro
    colunas_presentes = [c for c in colunas_essenciais if c in df_final.columns]
    df_final = df_final[colunas_presentes]
    
    df_final.to_csv(ARQUIVO_DIVISAO_AMOSTRA, index=False)
    logging.info(f"Arquivo de divisão criado e salvo em '{ARQUIVO_DIVISAO_AMOSTRA}' com {len(df_final)} itens.")
    return df_final


def gerar_tarefas_label_studio(divisao):
    if divisao is None or divisao.empty:
        logging.error("Nenhuma divisão fornecida.")
        return

    if os.path.exists(ARQUIVO_TAREFAS):
        resp = input(f"O arquivo {ARQUIVO_TAREFAS} já existe. Deseja recriá-lo? (s/n): ")
        if resp.lower() != 's':
            logging.info("Operação cancelada pelo usuário.")
            return

    logging.info("Carregando arquivo de íntegras e saídas dos modelos...")
    df_integras = pd.read_parquet(ARQUIVO_INTEGRAS)
    
    df_modelos = {}
    for nome_modelo, caminho in MODELOS_ARQUIVOS.items():
        if os.path.exists(caminho):
            df_modelos[nome_modelo] = pd.read_parquet(caminho)
        else:
            logging.warning(f"Arquivo de modelo não encontrado: {caminho}")

    tarefas = []
    
    # Processar cada documento da divisão
    for idx, row in divisao.iterrows():
        seq_doc = row.get('seq_documento_acordao')
        
        # Obter os metadados
        doc_integra = df_integras[df_integras['seq_documento_acordao'] == seq_doc]
        if doc_integra.empty:
            logging.warning(f"Documento {seq_doc} não encontrado em {ARQUIVO_INTEGRAS}. Ignorando.")
            continue
        
        doc_integra = doc_integra.iloc[0]
        
        # Construir bloco base do documento
        data = {
            "task_id": f"principal_{idx+1:04d}",
            "alvo": row.get('alvo', ''),
            "doc_id": str(seq_doc),
            "fold": row.get('fold', ''),
            "dificuldade": row.get('dificuldade', ''),
            "metadados_doc_classe": doc_integra.get('sg_classe', ''),
            "metadados_doc_ramo": doc_integra.get('sg_ramo_direito', ''),
            "metadados_doc_data": str(doc_integra.get('dt_publicacao', '')),
            "url_acordao_fonte": f"https://scon.stj.jus.br/SCON/pesquisa.jsp?b=ACOR&p=true&tp=T&processo={doc_integra.get('num_registro', '')}",
            "texto_acordao_fonte": doc_integra.get('integra', '')
        }
        
        # O teste é cego (blind test), portanto a ordem dos modelos apresentados ao avaliador (col1, col2, col3)
        # deve ser sorteada aleatoriamente a cada tarefa para evitar viés de posição na Label Studio.
        modelos_disponiveis = list(MODELOS_ARQUIVOS.keys())
        seed_rand = random.randint(1, 10000)
        random.seed(seed_rand) # Nova seed por documento para garantir embaralhamento independente
        random.shuffle(modelos_disponiveis)
        
        # Guardamos a seed e o rastro da coluna para auditoria futura e conferência do gabarito
        data['seed_randomizacao'] = seed_rand
        
        # Popular as 3 colunas baseando-se na ordem que foi sorteada
        for i, nome_modelo in enumerate(modelos_disponiveis, 1):
            data[f"fonte_real_col{i}"] = nome_modelo
            df_m = df_modelos.get(nome_modelo)
            
            bloco_html = "<div style='color:red;'>Nenhuma previsão disponível</div>"
            if df_m is not None:
                # Na saída, a chave bate com o seq_documento_acordao.
                # Converter ambos para string para evitar mismatch de tipagem (int vs str)
                pred_row = df_m[df_m['chave'].astype(str) == str(seq_doc)]
                if not pred_row.empty:
                    resposta_json = pred_row.iloc[0].get('resposta', '{}')
                    bloco_html = formatar_html(resposta_json)
            
            data[f"bloco_html_col{i}"] = bloco_html

        tarefas.append({"data": data})
        
    with open(ARQUIVO_TAREFAS, 'w', encoding='utf-8') as f:
        json.dump(tarefas, f, ensure_ascii=False, indent=2)
        
    logging.info(f"Arquivo gerado com sucesso: {ARQUIVO_TAREFAS} ({len(tarefas)} tarefas)")


if __name__ == '__main__':
    # TODO: se receber o parâmetro --reset vai remover os arquivos de saída antes de rodar a lógica
    if len(sys.argv) > 1 and sys.argv[1].lower() == '--reset':
        if os.path.exists(ARQUIVO_DIVISAO_AMOSTRA):
            os.remove(ARQUIVO_DIVISAO_AMOSTRA)
        if os.path.exists(ARQUIVO_TAREFAS):
            os.remove(ARQUIVO_TAREFAS)  
        print_cores('<verde>Arquivos de saída removidos.</verde>')
    
    divisao = criar_arquivo_divisao()
    if divisao is not None:
        gerar_tarefas_label_studio(divisao)