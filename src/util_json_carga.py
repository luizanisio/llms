# -*- coding: utf-8 -*-

"""
Carga de dados de comparação de JSONs.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Classe genérica para carga e organização de dados de comparação de JSONs,
incluindo tokens, avaliações LLM e métricas de observabilidade.
"""

from genericpath import isdir
import os
import json
import sys
import regex as re

from tqdm import tqdm
import json
import regex as re
from statistics import harmonic_mean

# Importa JsonAnaliseDados para retornar objeto completo
sys.path.extend(['./utils', './src'])
from util_json_dados import JsonAnaliseDados
from util import Util, UtilTextos

class CargaDadosComparacao():
    """
    Classe genérica para carga de dados de comparação de JSONs.
    
    Esta classe organiza e carrega dados de diferentes fontes (modelos) para
    comparação, incluindo JSONs principais, tokens, avaliações LLM e métricas
    de observabilidade.
    
    Premissas da estrutura de dados:
    --------------------------------
    - Arquivos JSON nomeados como: <id_peca>.json (padrão: r'\\d{12}\\.\\d+\\.\\d*\\.json')
    - Arquivos opcionais de tokens: <id_peca>.tokens.json ou <id_peca>.resumo.json
    - Arquivos opcionais de avaliação LLM: <id_peca>.avaliacao.json
    - Uma pasta de origem (ground truth) e N pastas de destino (modelos)
    
    Parâmetros:
    -----------
    pasta_origem : str
        Pasta com JSONs de referência (ground truth)
    pastas_destinos : list
        Lista de pastas com JSONs a comparar
    rotulo_id : str
        Nome do campo ID (ex: 'id', 'doc_id')
    rotulo_origem : str
        Rótulo do modelo de origem/ground truth (ex: 'True', 'base_gpt5')
    rotulos_destinos : list
        Lista de rótulos para cada pasta destino
    campos_comparacao : list
        Lista de campos a serem comparados
    regex_arquivos : str, opcional
        Regex customizada para identificar arquivos válidos
    
    Atributos públicos (após carregar()):
    --------------------------------------
    dados : list
        Lista de dicts com JSONs carregados
    rotulos : list
        Lista de rótulos ['id', 'True', 'Modelo1', 'Modelo2', ...]
    tokens : list
        Lista de dicts com contagem de tokens por modelo
    avaliacao_llm : list
        Lista de dicts com avaliações LLM
    observabilidade : list
        Lista de dicts com métricas de observabilidade
    
    Exemplo:
    --------
    >>> carga = CargaDadosComparacao(
    ...     pasta_origem='./ground_truth',
    ...     pastas_destinos=['./modelo1', './modelo2'],
    ...     rotulo_id='id_peca',
    ...     rotulo_origem='True',
    ...     rotulos_destinos=['Modelo1', 'Modelo2'],
    ...     campos_comparacao=['campo1', 'campo2']
    ... )
    >>> carga.carregar()
    >>> dados_analise = JsonAnaliseDados(
    ...     dados=carga.dados,
    ...     rotulos=carga.rotulos,
    ...     tokens=carga.tokens
    ... )
    """
    
    def __init__(self, 
                 pasta_origem: str,
                 pastas_destinos: list,
                 rotulo_id: str,
                 rotulo_origem: str,
                 rotulos_destinos: list,
                 campos_comparacao: list,
                 mascara_extracao: str = r'^(.+)\.json$',
                 mascara_tokens: str = None,
                 mascara_avaliacao: str = None,
                 mascara_observabilidade: str = None,
                 pasta_log_erros: str = None,
                 ignorar_erro_extracao: bool = False):
        """
        Inicializa a classe de carga de dados com suporte a Regex.
        
        Args:
            pasta_origem: Pasta com arquivos de referência
            pastas_destinos: Lista de pastas com arquivos a comparar
            rotulo_id: Nome do campo ID (ex: 'id')
            rotulo_origem: Rótulo do modelo de origem
            rotulos_destinos: Lista de rótulos para destinos
            campos_comparacao: Lista de campos a serem comparados
            mascara_extracao: Regex para arquivos de extração (grupo 1 deve ser o ID)
            mascara_tokens: Regex para arquivos de tokens (grupo 1 deve ser o ID)
            mascara_avaliacao: Regex para arquivos de avaliação (grupo 1 deve ser o ID)
            mascara_observabilidade: Regex para arquivos de observabilidade (grupo 1 deve ser o ID)
        """
        assert len(pastas_destinos) == len(rotulos_destinos), \
            "Número de pastas_destinos deve ser igual ao número de rotulos_destinos"
        
        self.pasta_origem = pasta_origem
        self.pastas_destinos = pastas_destinos
        self.rotulo_id = rotulo_id
        self.rotulo_origem = rotulo_origem
        self._rotulos_destinos = rotulos_destinos
        self.campos_comparacao = campos_comparacao
        self.pasta_log_erros = pasta_log_erros
        self.ignorar_erro_extracao = ignorar_erro_extracao
        
        # Compilação das Regex
        self.re_extracao = re.compile(mascara_extracao)
        self.re_tokens = re.compile(mascara_tokens) if mascara_tokens else None
        self.re_avaliacao = re.compile(mascara_avaliacao) if mascara_avaliacao else None
        self.re_observabilidade = re.compile(mascara_observabilidade) if mascara_observabilidade else None
        
        # Atributos públicos
        self.dados = None
        self.rotulos = None
        self.tokens = None
        self.avaliacao_llm = None
        self.observabilidade = None
        
        # Controle de erros
        self._erros_carga = []

    def _mapear_pasta(self, pasta: str, regex: re.Pattern) -> dict:
        """
        Varre a pasta e retorna dicionário {id: caminho_completo} para arquivos que casam com o regex.
        O regex DEVE ter um grupo de captura para o ID.
        """
        if not os.path.exists(pasta) or not regex:
            return {}
        
        mapa = {}
        items = os.listdir(pasta)
        # Se for observabilidade, tenta incluir subpasta 'observabilidade' se existir
        sub_obs = os.path.join(pasta, 'observabilidade')
        if os.path.exists(sub_obs) and os.path.isdir(sub_obs):
            items.extend([os.path.join('observabilidade', f) for f in os.listdir(sub_obs)])
            
        for item in items:
            nome_arquivo = os.path.basename(item)
            match = regex.match(nome_arquivo)
            if match:
                id_peca = match.group(1)
                full_path = os.path.join(pasta, item)
                mapa[id_peca] = full_path
        
        return mapa

    def _carregar_json(self, caminho: str) -> dict:
        """Carrega arquivo JSON com suporte a múltiplos encodings.
        
        Args:
            caminho: Caminho do arquivo JSON
        """
        if not os.path.exists(caminho):
            return {'erro': 'Arquivo inexistente'}
        
        # Lista de encodings para tentar (ordem de preferência)
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        last_error = None
        for encoding in encodings:
            try:
                with open(caminho, 'r', encoding=encoding) as f:
                     #dados = json.load(f)
                     dados = UtilTextos.mensagem_to_json(f.read()) # tratamentos de correção para json
                
                # Se chegou aqui, conseguiu carregar
                # Verifica se já possui erro registrado no próprio JSON
                if 'erro' in dados:
                    return {'erro': f"Erro na extração: {dados['erro']}"}
                return dados
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                last_error = e
                continue
            except Exception as e:
                # Outros erros não são de encoding, devem ser re-raised
                raise
        
        # Se chegou aqui, nenhum encoding funcionou
        if isinstance(last_error, json.JSONDecodeError):
            # Apenas erros de parsing JSON são logados (arquivo corrompido/mal-formado)
            self._registrar_erro_carga(
                os.path.basename(caminho),
                caminho,
                'Erro de parsing JSON',
                str(last_error)
            )
        return {'erro': f'Erro ao ler JSON: {str(last_error)}'}

    def _registrar_erro_carga(self, id_peca: str, arquivo: str, tipo_erro: str, mensagem: str):
        """Registra erro de carga para geração posterior de log."""
        self._erros_carga.append({
            'id_peca': id_peca,
            'arquivo': arquivo,
            'tipo': tipo_erro,
            'mensagem': mensagem
        })

    def salvar_log_erros(self):
        """Salva log de erros em arquivo, se houver erros e se pasta_log_erros foi especificada."""
        if not self.pasta_log_erros or not self._erros_carga:
            return
        
        os.makedirs(self.pasta_log_erros, exist_ok=True)
        arquivo_log = os.path.join(self.pasta_log_erros, 'erros_carga.log')
        
        from datetime import datetime
        
        with open(arquivo_log, 'w', encoding='utf-8') as f:
            f.write('=' * 60 + '\n')
            f.write('RELATÓRIO DE ERROS DE CARGA\n')
            f.write('=' * 60 + '\n')
            f.write(f'Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Total de erros: {len(self._erros_carga)}\n')
            f.write('\n')
            f.write('=' * 60 + '\n')
            f.write('ERROS POR ARQUIVO\n')
            f.write('=' * 60 + '\n\n')
            
            for erro in self._erros_carga:
                f.write(f"ID: {erro['id_peca']}\n")
                f.write(f"Arquivo: {erro['arquivo']}\n")
                f.write(f"Tipo: {erro['tipo']}\n")
                f.write(f"Erro: {erro['mensagem']}\n")
                f.write('\n' + '-' * 60 + '\n\n')
        
        print(f"\n📝 Log de erros salvo em: {arquivo_log}")


    def _extrair_campo_subnivel(self, json_data: dict, campo_path: str):
        """
        Extrai valor de um campo com subnível (notação com ponto).
        
        Exemplos de campo_path:
            - "Temas.Ponto" -> busca 'Ponto' dentro de 'Temas'
            - "Temas.Ponto.Argumentos" -> busca 'Argumentos' dentro de 'Ponto' dentro de 'Temas'
        
        Comportamento:
            - Se o campo pai for um dicionário, retorna o subcampo diretamente
            - Se o campo pai for uma lista de dicionários, agrupa todas as ocorrências
              do subcampo em uma lista
            - Retorna None se o caminho não existir
        
        Args:
            json_data: dicionário JSON fonte
            campo_path: caminho do campo (ex: "Temas.Ponto")
        
        Returns:
            O valor do subcampo, uma lista de valores agrupados, ou None
        """
        partes = campo_path.split('.')
        valor_atual = json_data
        
        for i, parte in enumerate(partes):
            if valor_atual is None:
                return None
            
            if isinstance(valor_atual, dict):
                # Dicionário: busca a chave diretamente
                valor_atual = valor_atual.get(parte)
            
            elif isinstance(valor_atual, list):
                # Lista: agrupa valores do subcampo de cada item
                # Partes restantes formam o subcampo a extrair de cada item
                subcampo_restante = '.'.join(partes[i:])
                valores_agrupados = []
                
                for item in valor_atual:
                    if isinstance(item, dict):
                        # Extração recursiva para subcampos mais profundos
                        if '.' in subcampo_restante:
                            valor_item = self._extrair_campo_subnivel(item, subcampo_restante)
                        else:
                            valor_item = item.get(parte)
                        
                        if valor_item is not None:
                            # Se o valor extraído for uma lista, estende; senão, adiciona
                            if isinstance(valor_item, list):
                                valores_agrupados.extend(valor_item)
                            else:
                                valores_agrupados.append(valor_item)
                
                # Retorna a lista agrupada ou None se vazia
                return valores_agrupados if valores_agrupados else None
            
            else:
                # Valor escalar ou tipo não navegável: não pode continuar
                return None
        
        return valor_atual

    def _filtrar_campos(self, json_data: dict, campos: list) -> dict:
        """
        Filtra apenas os campos especificados para comparação.
        
        Suporta campos com subníveis usando notação com ponto (ex: "Temas.Ponto").
        Para campos com subnível:
            - Se o campo pai for um dicionário, extrai o subcampo diretamente
            - Se o campo pai for uma lista de dicionários, agrupa todas as ocorrências
              do subcampo em uma lista
        
        Args:
            json_data: dicionário JSON fonte
            campos: lista de campos a filtrar (pode incluir campos com ".")
        
        Returns:
            dict: dicionário com os campos filtrados
        """
        if 'erro' in json_data:
            return json_data
        
        resultado = {}
        for campo in campos:
            if '.' in campo:
                # Campo com subnível: usa extração especial
                valor = self._extrair_campo_subnivel(json_data, campo)
                if valor is not None:
                    resultado[campo] = valor
            else:
                # Campo simples: busca direta
                if campo in json_data:
                    resultado[campo] = json_data.get(campo)
        
        return resultado


    def _listar_arquivos_json(self, pasta: str) -> list:
        """Lista todos os arquivos .json da pasta, retorna lista de IDs (método interno)"""
        if not os.path.exists(pasta):
            return []

        arquivos = [f for f in os.listdir(pasta) if self._re_arquivos_json.match(f)]
        # Extrai o ID (nome sem extensão)
        ids = [os.path.splitext(f)[0] for f in arquivos]
        return sorted(ids)

    def _filtro_origem(self, dados:dict):
        """Filtra documentos vazios da origem (método interno)"""
        # se nenhum campo estiver preenchido (None ou vazio), retorna None
        if not dados:
            return None
        campos_preenchidos = [v for v in dados.values() if v not in (None, '', [], {})]
        if not campos_preenchidos:
            return None
        return dados

    def _ler_arquivo(self, caminho: str) -> dict:
        """Lê arquivo JSON genérico."""
        if not caminho or not os.path.exists(caminho):
            return None
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def _ler_avaliacao_llm(self, caminho_arquivo: str, rotulo:str) -> dict:
        """Lê e formata avaliação LLM dado um caminho de arquivo."""
        dados_original = self._ler_arquivo(caminho_arquivo)
        if not dados_original:
            return None

        # ===== MÉTRICAS GLOBAIS =====
        p = dados_original.get('precision', 0)
        r = dados_original.get('recall', 0)
        f1 = dados_original.get('f1', 0) or (harmonic_mean([p, r]) if (p + r) > 0 else 0)
        nota = dados_original.get('nota', 0)
        exp = dados_original.get('explicacao', '')
        
        dados = {
            f'{rotulo}_P': p,
            f'{rotulo}_R': r,
            f'{rotulo}_F1': f1,
            f'{rotulo}_nota': nota,
            f'{rotulo}_explicacao': exp
        }
        
        # ===== MÉTRICAS POR CAMPO =====
        metricas_por_campo = dados_original.get('metricas_por_campo', {})
        if metricas_por_campo and isinstance(metricas_por_campo, dict):
            for campo, metricas in metricas_por_campo.items():
                if not isinstance(metricas, dict):
                    continue
                p_campo = metricas.get('precision')
                r_campo = metricas.get('recall')
                
                dados[f'{rotulo}_{campo}_P'] = p_campo
                dados[f'{rotulo}_{campo}_R'] = r_campo
                
                if p_campo is not None and r_campo is not None:
                    f1_campo = harmonic_mean([p_campo, r_campo]) if (p_campo + r_campo) > 0 else 0
                    dados[f'{rotulo}_{campo}_F1'] = f1_campo
                else:
                    dados[f'{rotulo}_{campo}_F1'] = None
        return dados

    def _ler_observabilidade(self, caminho_arquivo: str, rotulo:str, resumo:dict) -> dict:
        """Lê dados de observabilidade."""
        dados_original = self._ler_arquivo(caminho_arquivo)
        dados = {}
        
        # Se tem arquivo, processa
        if dados_original:
            obs = dados_original.get('observabilidade', {})
            # ... (Lógica de processamento de observabilidade existente) ...
            # Simplificação: Copiando lógica original
            orquestracao = obs.get('OrquestracaoFinal', [])
            if orquestracao and len(orquestracao) > 0:
                orch = orquestracao[0]
                tempo_total = orch.get('duracao_total_segundos')
                if tempo_total is not None: dados[f'{rotulo}_SEG'] = int(round(tempo_total))
                loops = orch.get('loops_revisao')
                if loops is not None: dados[f'{rotulo}_REV'] = loops
                agentes = orch.get('total_agentes_executados')
                if agentes is not None: dados[f'{rotulo}_AGT'] = agentes
            
            # Campos/Agentes
            total_revisoes_global = 0
            max_iteracoes_global = 0
            for nome_agente, lista_execucoes in obs.items():
                if nome_agente == 'OrquestracaoFinal' or not isinstance(lista_execucoes, list): continue
                if not lista_execucoes: continue
                
                tempo_campo = 0.0
                max_iteracoes_campo = 0
                tem_revisao_campo = False
                sucesso_ultima = False
                
                for idx, execucao in enumerate(lista_execucoes):
                    iteracoes = execucao.get('iteracoes', 0)
                    max_iteracoes_campo = max(max_iteracoes_campo, iteracoes)
                    tempo_campo += execucao.get('duracao_segundos', 0.0)
                    if execucao.get('tem_revisao', False): tem_revisao_campo = True
                    if idx == len(lista_execucoes) - 1: sucesso_ultima = execucao.get('sucesso', False)
                
                if tempo_campo > 0: dados[f'{rotulo}_{nome_agente}_SEG'] = int(round(tempo_campo))
                if max_iteracoes_campo > 0: 
                    dados[f'{rotulo}_{nome_agente}_IT'] = int(max_iteracoes_campo)
                    max_iteracoes_global = max(max_iteracoes_global, max_iteracoes_campo)
                dados[f'{rotulo}_{nome_agente}_OK'] = 'sim' if sucesso_ultima else 'não'
                if tem_revisao_campo: total_revisoes_global += 1
            
            if max_iteracoes_global > 0: dados[f'{rotulo}_IT'] = int(max_iteracoes_global)
            if total_revisoes_global > 0: dados[f'{rotulo}_REV'] = int(total_revisoes_global)

        # Fallback de tempo se vazio
        if not dados and resumo:
            tempo_resumo = resumo.get('time') or resumo.get('tempo') or resumo.get(f'{rotulo}_tempo')
            if tempo_resumo is not None:
                dados = {f'{rotulo}_SEG': int(round(tempo_resumo)) if isinstance(tempo_resumo, (int, float)) else tempo_resumo}
        
        return dados if dados else None

    def _ler_resumo_tokens(self, caminho_arquivo: str, rotulo: str) -> dict:
        """Lê resumo de tokens."""
        dados_full = self._ler_arquivo(caminho_arquivo)
        if not dados_full: return None
        
        # Normalização (tokens, total_geral, etc)
        # Prioridade: 1. chaves de agrupamento conhecidas, 2. usage (padrão OpenAI/novo), 3. raiz
        dados = {}
        if 'total' in dados_full: dados = dados_full.get('total') or {}
        elif 'total_geral' in dados_full: dados = dados_full.get('total_geral') or {}
        elif 'tokens' in dados_full: dados = dados_full.get('tokens') or {}
        elif 'usage' in dados_full: dados = dados_full.get('usage') or {}
        else: dados = dados_full # Tenta ler da raiz
        
        input_tokens = dados.get('input_tokens') or dados.get('prompt_tokens') or 0
        output_tokens = dados.get('output_tokens') or dados.get('completion_tokens') or 0
        cached_tokens = dados.get('cached_tokens') or 0
        reasoning_tokens = dados.get('reasoning_tokens') or 0
        finish_reason = dados.get('finish_reason') or dados.get('finished_reason') or '-'
        total_tokens = dados.get('total_tokens') or 0
        
        # Tempo geralmente está na raiz, mas pode estar no bloco de tokens
        tempo = dados_full.get('time') or dados_full.get('tempo') or dados.get('time') or dados.get('tempo') or None
        
        return {
            f'{rotulo}_input': input_tokens,
            f'{rotulo}_output': output_tokens,
            f'{rotulo}_total': max(total_tokens, input_tokens + output_tokens + reasoning_tokens),
            f'{rotulo}_cache': cached_tokens,
            f'{rotulo}_reason': reasoning_tokens,
            f'{rotulo}_finish': finish_reason,
            f'_tempo': tempo
        }

    def carregar(self):
        """Carrega dados usando mapeamento via Regex."""
        # 1. Mapeamento da Origem
        print(f"📂 Mapeando origem: {self.pasta_origem}")
        map_ext_origem = self._mapear_pasta(self.pasta_origem, self.re_extracao)
        map_tokens_origem = self._mapear_pasta(self.pasta_origem, self.re_tokens)
        map_av_origem = self._mapear_pasta(self.pasta_origem, self.re_avaliacao)
        map_obs_origem = self._mapear_pasta(self.pasta_origem, self.re_observabilidade)
        
        ids_origem = set(map_ext_origem.keys())
        
        # 2. Mapeamento dos Destinos
        maps_destinos = [] # Lista de dicts: [{'ext': {}, 'tok': {}, ...}, ...]
        for p_dest in self.pastas_destinos:
            # print(f"📂 Mapeando destino: {p_dest}")
            maps_destinos.append({
                'ext': self._mapear_pasta(p_dest, self.re_extracao),
                'tok': self._mapear_pasta(p_dest, self.re_tokens),
                'av': self._mapear_pasta(p_dest, self.re_avaliacao),
                'obs': self._mapear_pasta(p_dest, self.re_observabilidade)
            })
            
        # Define rótulos
        self.rotulos = [self.rotulo_id, self.rotulo_origem] + self._rotulos_destinos
        
        print(f"\n📊 Resumo dos dados:")
        print(f"   IDs na origem (a processar): {len(ids_origem)}")
        
        # Inicializa containers
        self.dados = []
        tokens_dict = {}     # {id: {chave: valor}}
        avaliacoes_dict = {} # {id: {chave: valor}}
        obs_dict = {}        # {id: {chave: valor}}
        campos_dict = {}     # {id: {chave: valor}}
        
        erros_count = {'inexistente': 0, 'erro_extracao': 0, 'sucesso': 0, 'filtrado': 0}

        # Helper interno atualizado para usar maps passados
        def _processar_auxiliares(id_peca, rotulo, map_tok, map_av, map_obs, json_dados_para_campos=None):
            # Tokens
            if id_peca in map_tok:
                dados_tok = self._ler_resumo_tokens(map_tok[id_peca], rotulo)
                if dados_tok:
                    if id_peca not in tokens_dict: tokens_dict[id_peca] = {}
                    tokens_dict[id_peca].update(dados_tok)
            else:
                dados_tok = None # Necessário para fallback de tempo na obs

            # Avaliação
            if id_peca in map_av:
                dados_av = self._ler_avaliacao_llm(map_av[id_peca], rotulo)
                if dados_av:
                    if id_peca not in avaliacoes_dict: avaliacoes_dict[id_peca] = {}
                    avaliacoes_dict[id_peca].update(dados_av)
            
            # Observabilidade (usa token/tempo como fallback)
            path_obs = map_obs.get(id_peca)
            # Mesmo que não tenha arquivo obs, tenta pegar tempo do resumo
            dados_obs = self._ler_observabilidade(path_obs, rotulo, dados_tok)
            if dados_obs:
                if id_peca not in obs_dict: obs_dict[id_peca] = {}
                obs_dict[id_peca].update(dados_obs)
            
            # Métricas de Campos (QTD/BYTES) - apenas se json de dados foi carregado
            if json_dados_para_campos:
                dados_campos = self._get_metricas_campos(json_dados_para_campos, rotulo)
                if dados_campos:
                    if id_peca not in campos_dict: campos_dict[id_peca] = {}
                    campos_dict[id_peca].update(dados_campos)

        # Loop Principal
        print(f"\n🔄 Carregando e filtrando dados...")
        for id_peca in tqdm(sorted(ids_origem), desc="Processando"):
            # 1. Carrega Origem
            path_origem = map_ext_origem[id_peca]
            json_origem = self._carregar_json(path_origem)
            json_origem_filtrado = self._filtrar_campos(json_origem, self.campos_comparacao)
            json_origem_filtrado = self._filtro_origem(json_origem_filtrado)
            
            if json_origem_filtrado is None:
                erros_count['filtrado'] += 1
                continue
                
            # Processa auxiliares origem
            _processar_auxiliares(id_peca, self.rotulos[1], map_tokens_origem, map_av_origem, map_obs_origem, json_origem_filtrado)

            # 2. Carrega Destinos
            jsons_destinos = []
            tem_erro = 'erro' in json_origem_filtrado
            
            for i, map_dest in enumerate(maps_destinos):
                # Tenta achar arquivo de extração do destino
                path_dest = map_dest['ext'].get(id_peca)
                json_dest_filtrado = {}
                
                if path_dest:
                    json_dest = self._carregar_json(path_dest)
                    json_dest_filtrado = self._filtrar_campos(json_dest, self.campos_comparacao)
                else:
                    json_dest_filtrado = {'erro': 'Inexistente (Não encontrado no mapa)'}
                
                jsons_destinos.append(json_dest_filtrado)
                if 'erro' in json_dest_filtrado: tem_erro = True
                
                # Processa auxiliares destino
                _processar_auxiliares(id_peca, self.rotulos[2+i], map_dest['tok'], map_dest['av'], map_dest['obs'])

            # Contabiliza
            if tem_erro:
                if any('Inexistente' in j.get('erro', '') for j in [json_origem_filtrado] + jsons_destinos):
                     erros_count['inexistente'] += 1
                else:
                     erros_count['erro_extracao'] += 1
            else:
                erros_count['sucesso'] += 1

            # Monta linha de dados
            linha = {
                self.rotulos[0]: id_peca,
                self.rotulos[1]: json_origem_filtrado
            }
            for i, jd in enumerate(jsons_destinos):
                linha[self.rotulos[2+i]] = jd
            
            # Se ignorar_erro_extracao=True, ignora documentos com erro
            if self.ignorar_erro_extracao and tem_erro:
                continue
                
            self.dados.append(linha)

        # Finalização (Consolidação de listas)
        self._consolidar_listas(tokens_dict, avaliacoes_dict, obs_dict, campos_dict)
        
        # Salva log de erros se necessário
        self.salvar_log_erros()
        
        self._imprimir_resumo(erros_count, len(ids_origem))
        
        return JsonAnaliseDados(
            dados=self.dados,
            rotulos=self.rotulos,
            tokens=self.tokens,
            avaliacao_llm=self.avaliacao_llm,
            observabilidade=self.observabilidade,
            pasta_origem=self.pasta_origem,
            pastas_destinos=self.pastas_destinos,
            campos_comparacao=self.campos_comparacao,
            rotulos_destinos=self._rotulos_destinos,
            nome_campo_id=self.rotulo_id,
            rotulo_campo_id=self.rotulo_id,
            rotulo_origem=self.rotulo_origem
        )

    def _consolidar_listas(self, tokens_dict, avaliacoes_dict, obs_dict, campos_dict):
        """Helper para consolidar dicionários em listas finais."""
        # Tokens
        self.tokens = []
        for id_peca, dados in tokens_dict.items():
            d = {self.rotulo_id: id_peca}
            d.update(dados)
            self.tokens.append(d)
            
        # Avaliação LLM
        self.avaliacao_llm = []
        max_modelos = 0
        for id_peca, dados in avaliacoes_dict.items():
            d = {self.rotulo_id: id_peca}
            d.update(dados)
            self.avaliacao_llm.append(d)
            # Estimativa de completude (opcional)
        
        # Observabilidade (mescla com campos_dict)
        self.observabilidade = []
        # Uniao de chaves de obs e campos
        all_ids = set(obs_dict.keys()) | set(campos_dict.keys())
        for id_peca in all_ids:
            d = {self.rotulo_id: id_peca}
            d.update({k:v for k,v in obs_dict.get(id_peca, {}).items() if not k.startswith('_')})
            d.update(campos_dict.get(id_peca, {}))
            self.observabilidade.append(d)

    def _get_metricas_campos(self, json_dados: dict, rotulo: str) -> dict:
        """
        Calcula métricas de campos do JSON (QTD e BYTES).
        Separado de _get_observabilidade para melhor organização.
        
        Args:
            json_dados: Dict com dados JSON filtrados
            rotulo: Rótulo do modelo (ex: 'base_p')
        
        Returns:
            dict: {rotulo_QTD: val, rotulo_campo_BYTES: val, ...}
                  ou None se json_dados for None ou vazio
        """
        if json_dados is None:
            return None
        
        # Conta campos com valores não vazios
        campos_com_valor = 0
        dados = {}
        
        for campo, valor in json_dados.items():
            if campo == 'erro':  # Ignora campo de erro
                continue
            
            # Verifica se campo tem valor
            tem_valor = False
            bytes_campo = 0
            
            if valor is not None:
                if isinstance(valor, (list, dict)):
                    tem_valor = len(valor) > 0
                    bytes_campo = len(str(valor))
                elif isinstance(valor, str):
                    tem_valor = len(valor.strip()) > 0
                    bytes_campo = len(valor)
                else:
                    tem_valor = True
                    bytes_campo = len(str(valor))
            
            if tem_valor:
                campos_com_valor += 1
                # Adiciona métrica de bytes por campo com sufixo _campo_BYTES
                dados[f'{rotulo}_{campo}_BYTES'] = bytes_campo
        
        # Adiciona total de campos com valor
        if campos_com_valor > 0:
            dados[f'{rotulo}_QTD'] = campos_com_valor
        
        return dados if dados else None

    def _imprimir_resumo(self, erros_count, total_origem):
        print(f"\n📋 Status dos dados:")
        print(f"   ✅ Sucesso: {erros_count['sucesso']}")
        print(f"   🔍 Filtrados (origem vazia): {erros_count['filtrado']}")
        if erros_count['inexistente'] > 0: print(f"   ❌ Inexistente: {erros_count['inexistente']}")
        if erros_count['erro_extracao'] > 0: print(f"   ⚠️  Erro na extração: {erros_count['erro_extracao']}")
        print(f"   📊 Total comparado: {len(self.dados)}")

        
if __name__ == "__main__":
    # Exemplo de uso da classe CargaDadosComparacao
    for _raiz in ['./saidas', './espelho/saidas', '../saidas']:
        if os.path.isdir(_raiz):
            ORIGEM = os.path.join(_raiz, 'espelhos_base_p')
            DESTINOS = [os.path.join(_raiz, 'espelhos_agentes_gpt5/')]
            D_ROTULOS = ['agentes_gpt5']
            ROTULO_ID = 'id'
            ROTULO_ORIGEM = 'base_p'
            CAMPOS_COMPARACAO = [
                'jurisprudenciaCitada', 'notas', 'informacoesComplementares', 
                'termosAuxiliares', 'teseJuridica', 'tema', 'referenciasLegislativas'
            ]
            
            # Cria instância da classe
            carga = CargaDadosComparacao(
                pasta_origem=ORIGEM,
                pastas_destinos=DESTINOS,
                rotulo_id=ROTULO_ID,
                rotulo_origem=ROTULO_ORIGEM,
                rotulos_destinos=D_ROTULOS,
                campos_comparacao=CAMPOS_COMPARACAO
            )
            
            # Carrega os dados - agora retorna JsonAnaliseDados
            dados_analise = carga.carregar()
            
            print(f"\n✅ Dados carregados com sucesso!")
            print(dados_analise.resumo())
            
            break