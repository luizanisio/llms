# Extração e Preparação do Dataset PubMed 20k RCT

Este documento descreve de forma objetiva como a extração e o preparo do dataset foram realizados utilizando o script `util_pubmed.py`.

A base de dados original "PubMed 20k RCT" disponibiliza o texto dos *abstracts* (resumos) divididos em sentenças, mas não fornece dados bibliográficos importantes como o título, jornal, data de publicação e as palavras-chave originais do artigo. 

Para reconstituir o documento em um formato consolidado semelhante a um artigo real e possibilitar experimentos de extração estruturada de informação com LLMs, o script `util_pubmed.py` executa os seguintes passos:

1. **Leitura dos CSVs Originais:** Carrega as divisões oficiais do dataset (`train.csv`, `dev.csv` e `test.csv`).
2. **Busca na API NCBI (Biopython):** Extrai a lista de PMIDs únicos e realiza requisições em lote à API pública do PubMed para capturar os metadados bibliográficos faltantes (título, revista, data e palavras-chave). 
3. **Uso de Cache Local:** Todas as respostas da API são imediatamente salvas no formato JSON dentro de uma pasta local `.cache_pubmed`. Isso garante que as informações não precisem ser consultadas duas vezes, evitando limites de taxa da API e acelerando execuções futuras.
4. **Reconstrução do Documento:** O texto do artigo é reconstituído em texto corrido (concatenando as frases marcadas no CSV) e envelopado com os metadados resgatados pela API (formando um cabeçalho e rodapé para simular o formato de texto final).
5. **Criação do Gabarito Estruturado (Target):** Utilizando as _labels_ marcadas frase a frase (ex: BACKGROUND, METHODS, RESULTS, etc.), o script processa o mapeamento e gera um dicionário de resposta com todas as seções unidas.
6. **Exportação Final:** As divisões consolidadas (train, dev, test) são exportadas para um arquivo único `.parquet` (`./dados/pubmed-rct-20k.parquet`), facilitando o carregamento rápido via pacote `util_vllm_batch.py` ou rotinas de fine-tuning. Adicionalmente, são criadas listas simplificadas em formato CSV para facilitar segmentações posteriores nos grupos experimentais.
