# -*- coding: utf-8 -*-

"""
Utilitário de pré-processamento para comparação de extrações.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Módulo responsável por converter arquivos .parquet em diretórios de arquivos JSON
compatíveis com o fluxo existente de CargaDadosComparacao (util_json_carga.py).

A extração ocorre uma única vez e é cacheada via arquivo de controle
'extracao_finalizada.md'. Para forçar re-extração, basta remover esse arquivo
ou o diretório de destino inteiro.

Uso:
    from comparar_extracoes_util import ExtracaoDataset

    extrator = ExtracaoDataset(
        arquivo_dataset='./saida/saida_qwen7b.parquet',
        pasta_destino='./compara/saida_qwen7b/',
        campos_dataset={'id': 'chave', 'resposta': 'resposta', 'resumo_tokens': 'resumo', 'avaliacao': '', 'erro': 'erro'}
    )
    erros = extrator.validar_colunas()
    if erros:
        print(f"Erros: {erros}")
    else:
        pasta = extrator.extrair()
        print(f"Dados extraídos em: {pasta}")
"""

import os
import json
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# Nome do arquivo de controle de cache
ARQUIVO_CONTROLE = 'extracao_finalizada.md'


class ExtracaoDataset:
    """
    Converte um arquivo .parquet ou .csv em um diretório de arquivos JSON para
    uso pelo pipeline de comparação de extrações.

    Arquivos gerados por registro:
        - {id}.json          — conteúdo da coluna 'resposta' (JSON da extração)
        - {id}.tokens.json   — conteúdo da coluna 'resumo_tokens' (metadados de tokens)
        - {id}.avaliacao.json — conteúdo da coluna 'avaliacao' (se mapeada e não vazia)

    Parâmetros:
    -----------
    arquivo_dataset : str
        Caminho do arquivo .parquet ou .csv de entrada
    pasta_destino : str
        Pasta onde os JSONs serão extraídos
    campos_dataset : dict
        Mapeamento de campos do dataset. Chaves esperadas:
            - 'id': nome da coluna com o ID do documento
            - 'resposta': nome da coluna com o JSON da extração
            - 'resumo_tokens': nome da coluna com o JSON de tokens (opcional)
            - 'avaliacao': nome da coluna com avaliação LLM (opcional, vazio = ignorar)
            - 'erro': nome da coluna com mensagem de erro (opcional)
    """

    def __init__(self, arquivo_dataset: str, pasta_destino: str, campos_dataset: dict, ids_filtro: set = None, saida_json: bool = True):
        self.arquivo_dataset = arquivo_dataset
        self.pasta_destino = pasta_destino
        self.campos_dataset = campos_dataset or {}
        self.ids_filtro = ids_filtro
        self.saida_json = saida_json

        # Mapeamento de campos (com defaults seguros)
        self._col_id = self.campos_dataset.get('id', 'chave')
        self._col_resposta = self.campos_dataset.get('resposta', 'resposta')
        self._col_tokens = self.campos_dataset.get('resumo_tokens', '')
        self._col_avaliacao = self.campos_dataset.get('avaliacao', '')
        self._col_erro = self.campos_dataset.get('erro', '')

        self._df = None  # Carregado sob demanda

    def _carregar_df(self):
        """Carrega o DataFrame do dataset se ainda não carregado."""
        if self._df is None:
            if not os.path.exists(self.arquivo_dataset):
                raise FileNotFoundError(f"Arquivo dataset não encontrado: {self.arquivo_dataset}")
            from util_pandas import ler_dataset
            self._df = ler_dataset(self.arquivo_dataset)
        return self._df

    def validar_colunas(self) -> list:
        """
        Valida se as colunas mapeadas em campos_dataset existem no DataFrame.

        Returns:
            list: Lista de mensagens de erro. Vazia se tudo OK.
        """
        try:
            df = self._carregar_df()
        except FileNotFoundError as e:
            return [str(e)]

        colunas_df = set(df.columns)
        erros = []

        # Campos obrigatórios
        for campo_nome, col_nome in [('id', self._col_id), ('resposta', self._col_resposta)]:
            if not col_nome:
                erros.append(f"Campo obrigatório '{campo_nome}' não definido em campos_dataset")
            elif col_nome not in colunas_df:
                erros.append(f"Coluna '{col_nome}' (campo '{campo_nome}') não encontrada no dataset. "
                             f"Colunas disponíveis: {sorted(colunas_df)}")

        # Campos opcionais (só valida se definidos)
        for campo_nome, col_nome in [('resumo_tokens', self._col_tokens),
                                      ('avaliacao', self._col_avaliacao),
                                      ('erro', self._col_erro)]:
            if col_nome and col_nome not in colunas_df:
                erros.append(f"Coluna '{col_nome}' (campo '{campo_nome}') não encontrada no dataset. "
                             f"Colunas disponíveis: {sorted(colunas_df)}")

        return erros

    def ja_extraido(self) -> bool:
        """Verifica se a extração já foi feita (cache via arquivo de controle) e está atualizada."""
        caminho_controle = os.path.join(self.pasta_destino, ARQUIVO_CONTROLE)
        if not os.path.isfile(caminho_controle):
            return False
            
        if os.path.isfile(self.arquivo_dataset):
            return os.path.getmtime(caminho_controle) >= os.path.getmtime(self.arquivo_dataset)
        return True

    def extrair(self, forcar: bool = False) -> str:
        """
        Executa a extração do dataset para diretório de JSONs.

        Se a extração já foi feita (arquivo de controle existe) e forcar=False,
        retorna o caminho da pasta sem re-extrair.

        Args:
            forcar: Se True, ignora o cache e re-extrai tudo.

        Returns:
            str: Caminho absoluto da pasta com os JSONs extraídos.
        """
        # Verifica cache
        if not forcar and self.ja_extraido():
            print(f"✅ Extração já realizada e atualizada (cache). Pasta: {self.pasta_destino}")
            print(f"   Para re-extrair, remova o arquivo '{ARQUIVO_CONTROLE}' da pasta.")
            return self.pasta_destino
            
        caminho_controle = os.path.join(self.pasta_destino, ARQUIVO_CONTROLE)
        desatualizado = os.path.isfile(caminho_controle)

        if desatualizado:
            print(f"🔄 Arquivo dataset foi modificado após a última extração. Atualizando JSONs...")
            # Remove o controle temporariamente para que em caso de interrupção não fique sujo
            os.remove(caminho_controle)
            print(f"   🗑️  Limpando pasta anterior para evitar arquivos órfãos (garantindo consistência)...")
            self._limpar_pasta()
        # Verifica integridade: pasta existe mas sem arquivo de controle (extração parcial/interrompida)
        elif os.path.isdir(self.pasta_destino) and os.listdir(self.pasta_destino):
            print(f"\n⚠️  A pasta de destino já existe mas não possui arquivo de controle '{ARQUIVO_CONTROLE}'.")
            print(f"   Pasta: {self.pasta_destino}")
            print(f"   Isso indica uma extração anterior incompleta ou interrompida.")
            print(f"   Para garantir a integridade dos dados, é necessário limpar a pasta antes de re-extrair.")
            resposta = input(f"\n   Deseja remover os arquivos existentes e re-extrair? (s/N): ").strip().lower()
            if resposta in ('s', 'sim', 'y', 'yes'):
                print(f"   🗑️  Limpando pasta incompleta...")
                self._limpar_pasta()
            else:
                print('-=' * 35)
                print(f"\n❌ Extração cancelada. Resolva a inconsistência manualmente:")
                print(f"   - Remova a pasta: {self.pasta_destino}")
                print(f"   - Ou renomeie-a para preservar os dados parciais.")
                print(f"\n ⚠️  Extração cancelada pelo usuário. Pasta inconsistente: {self.pasta_destino}\n")
                exit(1)

        # Carrega DataFrame
        df = self._carregar_df()


        df = self._carregar_df()
        total = len(df)

        # Cria pasta de destino
        os.makedirs(self.pasta_destino, exist_ok=True)

        print(f"\n📦 Extraindo dataset para JSONs...")
        print(f"   Fonte: {self.arquivo_dataset}")
        print(f"   Destino: {self.pasta_destino}")
        print(f"   Total de registros: {total}")

        # Contadores para o relatório
        stats = {
            'total': total,
            'filtrados': 0,
            'extraidos_json': 0,
            'extraidos_tokens': 0,
            'extraidos_avaliacao': 0,
            'com_erro': 0,
            'json_invalido': 0,
        }

        for _, row in tqdm(df.iterrows(), total=total, desc="Extraindo"):
            id_doc = str(row[self._col_id]).strip()
            if not id_doc:
                continue

            if self.ids_filtro is not None and id_doc not in self.ids_filtro:
                stats['filtrados'] += 1
                continue

            # --- Coluna de erro ---
            erro_msg = ''
            if self._col_erro and self._col_erro in row.index:
                erro_msg = str(row[self._col_erro]).strip() if pd.notna(row[self._col_erro]) else ''

            # --- Extração da resposta (JSON principal ou Texto Livre) ---
            resposta_raw = row[self._col_resposta] if pd.notna(row[self._col_resposta]) else ''
            resposta_raw = str(resposta_raw).strip() if resposta_raw else ''

            json_resposta = None
            if resposta_raw:
                if self.saida_json:
                    try:
                        json_resposta = json.loads(resposta_raw)
                    except (json.JSONDecodeError, ValueError):
                        json_resposta = {'erro': f'JSON inválido na resposta: {resposta_raw[:200]}...'}
                        stats['json_invalido'] += 1
                else:
                    json_resposta = {'resposta': resposta_raw}

            if json_resposta is None:
                json_resposta = {'erro': 'Resposta vazia'}

            # Se há erro no parquet, adiciona a chave "erro" no JSON (se ainda não tem)
            if erro_msg and 'erro' not in json_resposta:
                json_resposta['erro'] = erro_msg
                stats['com_erro'] += 1
            elif erro_msg:
                stats['com_erro'] += 1

            # Salva {id}.json
            caminho_json = os.path.join(self.pasta_destino, f'{id_doc}.json')
            with open(caminho_json, 'w', encoding='utf-8') as f:
                json.dump(json_resposta, f, ensure_ascii=False, indent=2)
            stats['extraidos_json'] += 1

            # --- Extração dos tokens ---
            if self._col_tokens and self._col_tokens in row.index:
                tokens_raw = row[self._col_tokens] if pd.notna(row[self._col_tokens]) else ''
                tokens_raw = str(tokens_raw).strip() if tokens_raw else ''
                if tokens_raw:
                    try:
                        json_tokens = json.loads(tokens_raw)
                        caminho_tokens = os.path.join(self.pasta_destino, f'{id_doc}.tokens.json')
                        with open(caminho_tokens, 'w', encoding='utf-8') as f:
                            json.dump(json_tokens, f, ensure_ascii=False, indent=2)
                        stats['extraidos_tokens'] += 1
                    except (json.JSONDecodeError, ValueError):
                        pass  # Tokens inválidos não são críticos

            # --- Extração da avaliação ---
            if self._col_avaliacao and self._col_avaliacao in row.index:
                aval_raw = row[self._col_avaliacao] if pd.notna(row[self._col_avaliacao]) else ''
                aval_raw = str(aval_raw).strip() if aval_raw else ''
                if aval_raw:
                    try:
                        json_aval = json.loads(aval_raw)
                        caminho_aval = os.path.join(self.pasta_destino, f'{id_doc}.avaliacao.json')
                        with open(caminho_aval, 'w', encoding='utf-8') as f:
                            json.dump(json_aval, f, ensure_ascii=False, indent=2)
                        stats['extraidos_avaliacao'] += 1
                    except (json.JSONDecodeError, ValueError):
                        pass  # Avaliação inválida não é crítica

        # Gera arquivo de controle
        self._gerar_arquivo_controle(stats)

        # Resumo
        print(f"\n📋 Extração concluída:")
        print(f"   ✅ Arquivos JSON gerados: {stats['extraidos_json']}")
        if stats['extraidos_tokens'] > 0:
            print(f"   📊 Arquivos de tokens gerados: {stats['extraidos_tokens']}")
        if stats['extraidos_avaliacao'] > 0:
            print(f"   ⚖️  Arquivos de avaliação gerados: {stats['extraidos_avaliacao']}")
        if stats['com_erro'] > 0:
            print(f"   ⚠️  Registros com erro (extraídos com flag): {stats['com_erro']}")
        if stats['json_invalido'] > 0:
            print(f"   ❌ Registros com JSON inválido na resposta: {stats['json_invalido']}")

        return self.pasta_destino

    def _limpar_pasta(self):
        """Remove todos os arquivos e subpastas do diretório de destino para garantir consistência."""
        import sys
        from concurrent.futures import ThreadPoolExecutor
        
        arquivos_remover = []
        pastas_remover = []
        for root, dirs, files in os.walk(self.pasta_destino, topdown=False):
            for name in files:
                arquivos_remover.append(os.path.join(root, name))
            for name in dirs:
                pastas_remover.append(os.path.join(root, name))
        pastas_remover.append(self.pasta_destino)
        
        def deletar_arquivo(caminho):
            try:
                if os.path.isfile(caminho) or os.path.islink(caminho):
                    os.remove(caminho)
            except FileNotFoundError:
                pass
            except OSError as e:
                return e
            return None
        
        if arquivos_remover:
            with ThreadPoolExecutor(max_workers=10) as executor:
                for result in tqdm(executor.map(deletar_arquivo, arquivos_remover), 
                                   total=len(arquivos_remover), desc="Removendo Arquivos", leave=False):
                    if isinstance(result, Exception):
                        print('-=' * 35)
                        print(f"\n❌ Houve um erro na exclusão de um arquivo: {result}")
                        sys.exit(1)
                        
        if pastas_remover:
            for d in tqdm(pastas_remover, desc="Removendo Pastas", leave=False):
                try:
                    if os.path.exists(d):
                        os.rmdir(d)
                except OSError as e:
                    print('-=' * 35)
                    print(f"\n❌ Houve um erro na exclusão da pasta {d} erro: {e}")
                    sys.exit(1)

        print(f"   🗑️  Pasta limpa: {self.pasta_destino}")


    def _gerar_arquivo_controle(self, stats: dict):
        """
        Gera o arquivo de controle 'extracao_finalizada.md' com metadados da extração.

        Args:
            stats: Dicionário com contadores da extração
        """
        agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Monta lista de campos mapeados para exibição
        campos_info = []
        if self._col_id: campos_info.append(f"id={self._col_id}")
        if self._col_resposta: campos_info.append(f"resposta={self._col_resposta}")
        if self._col_tokens: campos_info.append(f"resumo_tokens={self._col_tokens}")
        if self._col_avaliacao: campos_info.append(f"avaliacao={self._col_avaliacao}")
        if self._col_erro: campos_info.append(f"erro={self._col_erro}")

        conteudo = f"""# Extração de Dataset Finalizada

- **Arquivo fonte:** {self.arquivo_dataset}
- **Data da extração:** {agora}
- **Total de registros:** {stats['total']}
- **Registros filtrados:** {stats['filtrados']}
- **Registros com erro:** {stats['com_erro']}
- **Registros com JSON inválido:** {stats['json_invalido']}
- **Registros extraídos (JSON):** {stats['extraidos_json']}
- **Registros com tokens:** {stats['extraidos_tokens']}
- **Registros com avaliação:** {stats['extraidos_avaliacao']}
- **Campos mapeados:** {', '.join(campos_info)}

> Para forçar re-extração, remova este arquivo ou o diretório inteiro.
"""
        caminho = os.path.join(self.pasta_destino, ARQUIVO_CONTROLE)
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(conteudo)


def resolver_pasta_dataset(arquivo_dataset: str, pasta_base: str) -> str:
    """
    Calcula a pasta de destino para extração de um dataset.

    A pasta é: pasta_base / nome_do_arquivo_sem_extensão

    Args:
        arquivo_dataset: Caminho do arquivo
        pasta_base: Pasta base definida em saida.pasta_parquet (ou similar)

    Returns:
        str: Caminho da pasta de destino
    """
    nome_base = os.path.splitext(os.path.basename(arquivo_dataset))[0]
    return os.path.join(pasta_base, nome_base)
