# pip install biopython pandas fastparquet


from Bio import Entrez
import sys
import os
import time
import pandas as pd
from tqdm import tqdm

sys.path.append("../../src")
from util import UtilEnv, Util

# ATENÇÃO: preencher o email na variável EMAIL_PUBMED do arquivo .env em src ou na pasta atual
if not UtilEnv.carregar_env(pastas = ['../', './', '../../src']):
    print('-' * 50)
    print('🚩 Erro: Não foi possível encontrar o arquivo .env')
    print()
    exit(1)
    
email_pubmed = UtilEnv.get_str("EMAIL_PUBMED")

if not email_pubmed:
    print('-' * 50)
    print('🚩 Erro: Não foi possível encontrar o email cadastrado, preencha a variável de ambiente EMAIL_PUBMED com um email válido!')
    print()
    exit(1)
Entrez.email = email_pubmed

PASTA_CACHE = '.cache_pubmed'

def _buscar_cache(pmid: str) -> dict:
    arquivo = os.path.join(PASTA_CACHE, f'{pmid}.json')
    if os.path.isfile(arquivo):
        res = Util.ler_json(arquivo, {})
        if res.get('titulo'):
            return res
    return {}

def _salvar_cache(pmid: str, dados: dict):
    os.makedirs(PASTA_CACHE, exist_ok=True)
    arquivo = os.path.join(PASTA_CACHE, f'{pmid}.json')
    Util.gravar_json(arquivo, dados)

def enrich_abstract(pmid: str) -> dict:
    cached = _buscar_cache(pmid)
    if cached:
        return cached

    try:
        handle = Entrez.efetch(db="pubmed", id=str(pmid), rettype="xml", retmode="xml")
        record = Entrez.read(handle)["PubmedArticle"][0]
        article = record["MedlineCitation"]["Article"]

        titulo = str(article.get("ArticleTitle", ""))
        
        keywords = [
            str(kw)
            for kw_list in record["MedlineCitation"].get("KeywordList", [])
            for kw in kw_list
        ]
        
        journal_info = article.get("Journal", {})
        journal = str(journal_info.get("Title", ""))
        
        pubdate_info = journal_info.get("JournalIssue", {}).get("PubDate", {})
        ano = str(pubdate_info.get("Year", ""))
        mes = str(pubdate_info.get("Month", ""))
        if mes:
            data_publicacao = f"{ano}-{mes}"
        else:
            data_publicacao = ano
        
        dados = {
            "pmid": str(pmid), 
            "titulo": titulo, 
            "palavras_chave": keywords,
            "journal": journal,
            "data_publicacao": data_publicacao
        }
        _salvar_cache(pmid, dados)
        return dados
    except Exception as e:
        print(f"Erro ao buscar pmid {pmid}: {e}")
        return {
            "pmid": str(pmid), 
            "titulo": "", 
            "palavras_chave": [],
            "journal": "",
            "data_publicacao": ""
        }
    
def enrich_abstract_lote(pmids: list):
    """ Busca uma lista de PMIDs em lote na API, salvando no cache local. Retorna dict {pmid: dados}. """
    faltantes = []
    resultados = {}
    for p in pmids:
        c = _buscar_cache(str(p))
        if c.get("titulo"):
            resultados[str(p)] = c
        else:
            faltantes.append(str(p))
    
    if not faltantes:
        return resultados
    
    # NCBI recomenda blocos (lotes de até 200) para buscas múltiplas
    tamanho_lote = 200
    for i in tqdm(range(0, len(faltantes), tamanho_lote), desc="Buscando na API (Lotes)"):
        lote = faltantes[i:i+tamanho_lote]
        ids_str = ",".join(lote)
        
        tentativas = 3
        while tentativas > 0:
            try:
                handle = Entrez.efetch(db="pubmed", id=ids_str, rettype="xml", retmode="xml")
                records = Entrez.read(handle)
                
                artigos = records.get("PubmedArticle", [])
                
                for record in artigos:
                    pmid = str(record["MedlineCitation"]["PMID"])
                    article = record["MedlineCitation"]["Article"]
                    
                    titulo = str(article.get("ArticleTitle", ""))
                    
                    keywords = [
                        str(kw)
                        for kw_list in record["MedlineCitation"].get("KeywordList", [])
                        for kw in kw_list
                    ]
                    
                    journal_info = article.get("Journal", {})
                    journal = str(journal_info.get("Title", ""))
                    
                    pubdate_info = journal_info.get("JournalIssue", {}).get("PubDate", {})
                    ano = str(pubdate_info.get("Year", ""))
                    mes = str(pubdate_info.get("Month", ""))
                    if mes:
                        data_publicacao = f"{ano}-{mes}"
                    else:
                        data_publicacao = ano
                    
                    dados = {
                        "pmid": pmid, 
                        "titulo": titulo, 
                        "palavras_chave": keywords,
                        "journal": journal,
                        "data_publicacao": data_publicacao
                    }
                    _salvar_cache(pmid, dados)
                    resultados[pmid] = dados
                
                # Pausa respeitosa de 0.35s à API (limite de 3 req/s)
                time.sleep(0.35) 
                break
                
            except Exception as e:
                tentativas -= 1
                print(f"\nErro no lote de {i} a {i+tamanho_lote}: {e}. Tentativas restantes: {tentativas}")
                if tentativas == 0:
                    print("Falha definitiva neste lote.")
                else:
                    time.sleep(2) # Pausa maior antes de tentar novamente
                
    return resultados

def buscar_titulo(pmid):
    dados = enrich_abstract(str(pmid))
    return dados.get('titulo', '')

def criar_parquet_experimento(pasta_dataset, arquivo_saida, max_itens_por_split=None):
    ''' recebe o nome da pasta com o dataset PubMed-200k-RCT ou PubMed-20k-RCT
        a pasta contém os arquivos dev.csv, test.csv e train.csv
        e gera um arquivo parquet com os dados do experimento
        Campos do arquivo parquet:
        - pmid: id do artigo
        - article: texto montado para o experimento contendo o título no início, um texto corrido concatenando os dados em frases, o nome do jornal com data de publicação e as palavras-chave
        - split: conjunto a que pertence (dev, test ou train)
        - response: {   "title":             "string — extraído do cabeçalho",
                        "publication":       "string — formato YYYY-MM",
                        "journal":           "string — nome do periódico",
                        "keywords":          ["lista", "de", "strings"],
                        "background":        "string — seção BACKGROUND concatenada",
                        "objective":         "string — seção OBJECTIVE concatenada",
                        "methods":           "string — seção METHODS concatenada",
                        "results":           "string — seção RESULTS concatenada",
                        "conclusions":       "string — seção CONCLUSIONS concatenada"} 
                    }  

        Também são criados os arquivos com csv lista_treino.csv, lista_validacao.csv e lista_teste.csv que vão alimentar as comparações por grupos no futuro.
        cada arquivo precisa conter os campos: pmid, publication e split
    '''
    arquivos_csv = ['dev.csv', 'test.csv', 'train.csv']
    conjuntos = ['dev', 'test', 'train']
    
    pmids_unicos = set()
    dfs = {}
    for arq, conj in zip(arquivos_csv, conjuntos):
        caminho_arq = os.path.join(pasta_dataset, arq)
        if not os.path.exists(caminho_arq):
            print(f"Aviso: Arquivo {caminho_arq} não encontrado.")
            continue
        df_tmp = pd.read_csv(caminho_arq)
        
        if max_itens_por_split is not None:
            pmids_limitados = df_tmp['abstract_id'].unique()[:max_itens_por_split]
            df_tmp = df_tmp[df_tmp['abstract_id'].isin(pmids_limitados)]
            
        dfs[conj] = df_tmp
        pmids_unicos.update(df_tmp['abstract_id'].unique())
    
    lista_pmids = list(pmids_unicos)
    print(f"Total de {len(lista_pmids)} PMIDs únicos encontrados. Iniciando extração (em lotes, se necessário)...")
    
    # Processa todos de uma vez (baixando os que faltam e lendo os existentes do cache)
    enrich_abstract_lote(lista_pmids)

    dados_parquet = []
    lista_treino = []
    lista_validacao = []
    lista_teste = []
    
    estatisticas = {
        'ids_com_erro': [],
        'campos_preenchidos': [],
        'caracteres_registro': []
    }
    
    print("\nProcessando os datasets...")
    for conj, df in dfs.items():
        df = df.sort_values(by=['abstract_id', 'line_number'])
        agrupado = df.groupby('abstract_id')
        
        for pmid, grupo in tqdm(agrupado, desc=f"Processando {conj}"):
                
            info_api = enrich_abstract(str(pmid))
            
            titulo = info_api.get('titulo', '')
            if not titulo:
                estatisticas['ids_com_erro'].append(str(pmid))
                continue
            journal = info_api.get('journal', '')
            data_publicacao = info_api.get('data_publicacao', '')
            keywords = info_api.get('palavras_chave', [])
            
            secoes = {
                "BACKGROUND": [],
                "OBJECTIVE": [],
                "METHODS": [],
                "RESULTS": [],
                "CONCLUSIONS": []
            }
            resumo_frases = []
            
            for _, row in grupo.iterrows():
                frase = str(row['abstract_text']).strip()
                target = str(row['target']).strip()
                
                resumo_frases.append(frase)
                if target in secoes:
                    secoes[target].append(frase)
            
            resumo_completo = " ".join(resumo_frases)
            
            keywords_str = ", ".join(keywords)
            artigo = f"{titulo}\n\n{resumo_completo}\n\n{journal} ({data_publicacao})\nKeywords: {keywords_str}"
            
            resposta = {
                "title": titulo,
                "publication": data_publicacao,
                "journal": journal,
                "keywords": keywords,
                "background": " ".join(secoes["BACKGROUND"]),
                "objective": " ".join(secoes["OBJECTIVE"]),
                "methods": " ".join(secoes["METHODS"]),
                "results": " ".join(secoes["RESULTS"]),
                "conclusions": " ".join(secoes["CONCLUSIONS"])
            }
            
            campos_com_dados = sum(1 for v in resposta.values() if (isinstance(v, str) and len(v.strip()) > 0) or (isinstance(v, list) and len(v) > 0))
            estatisticas['campos_preenchidos'].append(campos_com_dados)
            estatisticas['caracteres_registro'].append(len(artigo))
            
            registro = {
                "pmid": str(pmid),
                "article": artigo,
                "split": conj,
                "response": resposta
            }
            
            dados_parquet.append(registro)
            
            registro_lista = {
                "pmid": str(pmid),
                "publication": data_publicacao,
                "split": conj
            }
            
            if conj == 'train':
                lista_treino.append(registro_lista)
            elif conj == 'dev':
                lista_validacao.append(registro_lista)
            elif conj == 'test':
                lista_teste.append(registro_lista)
    
    print("Gerando arquivos de saída...")
    
    df_parquet = pd.DataFrame(dados_parquet)
    if os.path.dirname(arquivo_saida):
        os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)
    df_parquet.to_parquet(arquivo_saida, index=False)
    
    base_saida = arquivo_saida.replace('.parquet', '')
    if lista_treino:
        pd.DataFrame(lista_treino).to_csv(f'{base_saida}-train.csv', index=False)
    if lista_validacao:
        pd.DataFrame(lista_validacao).to_csv(f'{base_saida}-dev.csv', index=False)
    if lista_teste:
        pd.DataFrame(lista_teste).to_csv(f'{base_saida}-test.csv', index=False)
        
    arquivo_log = f"{base_saida}.log"
    qtd_ids_com_erro = len(set(estatisticas['ids_com_erro']))
    
    with open(arquivo_log, 'w', encoding='utf-8') as f:
        f.write("=== Relatório de Extração ===\n")
        f.write(f"Arquivo de saída: {arquivo_saida}\n\n")
        f.write(f"Total de registros de Treino: {len(lista_treino)}\n")
        f.write(f"Total de registros de Validação (Dev): {len(lista_validacao)}\n")
        f.write(f"Total de registros de Teste: {len(lista_teste)}\n")
        f.write(f"Total de registros processados: {len(dados_parquet)}\n\n")
        
        f.write(f"PMIDs únicos totais enviados à API: {len(lista_pmids)}\n")
        f.write(f"PMIDs com erro na extração (sem título): {qtd_ids_com_erro}\n\n")
        
        if estatisticas['campos_preenchidos']:
            f.write("Estatísticas de campos preenchidos por registro (de 9 campos):\n")
            f.write(f"  Mínimo: {min(estatisticas['campos_preenchidos'])}\n")
            f.write(f"  Máximo: {max(estatisticas['campos_preenchidos'])}\n")
            f.write(f"  Média : {sum(estatisticas['campos_preenchidos']) / len(estatisticas['campos_preenchidos']):.2f}\n\n")
            
            f.write("Estatísticas de caracteres por registro (tamanho do artigo):\n")
            f.write(f"  Mínimo: {min(estatisticas['caracteres_registro'])}\n")
            f.write(f"  Máximo: {max(estatisticas['caracteres_registro'])}\n")
            f.write(f"  Média : {sum(estatisticas['caracteres_registro']) / len(estatisticas['caracteres_registro']):.2f}\n\n")

        if qtd_ids_com_erro > 0:
            f.write(f"=== Lista de PMIDs com erro ({qtd_ids_com_erro}) ===\n")
            f.write(f"{', '.join(sorted(set(estatisticas['ids_com_erro'])))}\n")

    print(f"Processo concluído com sucesso! Log salvo em {arquivo_log}")

if __name__=='__main__':
    import json
    
    # Exemplo simples de teste apenas com uma chamada:
    _id = 24845963
    dados = enrich_abstract(_id)

    print(f'Dados do id: {_id}')
    print(json.dumps(dados, ensure_ascii=False, indent=2))

    # Para rodar o processo completo:
    arquivo = './dados/pubmed-rct-20k.parquet'
    pasta_dataset = './PubMed_20k_RCT'
    
    if not os.path.isfile(arquivo):
        print(f'Criando dataset parquet a partir de {pasta_dataset} e saída para {arquivo}')
        criar_parquet_experimento(pasta_dataset, arquivo)
        
    arquivo_mini = arquivo.replace('.parquet', '-mini.parquet')
    if not os.path.isfile(arquivo_mini):
        print(f'Criando dataset parquet mini a partir de {pasta_dataset} e saída para {arquivo_mini}')
        criar_parquet_experimento(pasta_dataset, arquivo_mini, max_itens_por_split=20)
    
    print('-' * 50)
    print('Alguns dados do dataframe:')
    df = pd.read_parquet(arquivo)
    print(df.head())
    
    print('\n' + '=' * 50)
    print('Exemplo de um registro completo (artigo e json de resposta):')
    primeiro = df.iloc[0]
    
    print(f"PMID: {primeiro['pmid']} | Split: {primeiro['split']}")
    print("\n--- Artigo Completo ---")
    print(primeiro['article'])
    print("\n--- JSON Resposta ---")
    
    # Tratamento para serializar numpy arrays dentro do dicionário
    def conversor_numpy(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError
        
    if isinstance(primeiro['response'], dict):
        print(json.dumps(primeiro['response'], ensure_ascii=False, indent=2, default=conversor_numpy))
    elif hasattr(primeiro['response'], "tolist"):
        print(json.dumps(primeiro['response'].tolist(), ensure_ascii=False, indent=2, default=conversor_numpy))
    else:
        print(primeiro['response'])