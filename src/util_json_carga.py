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

class CargaDadosComparacao():
    """
    Classe genérica para carga de dados de comparação de JSONs.
    
    Esta classe organiza e carrega dados de diferentes fontes (modelos) para
    comparação, incluindo JSONs principais, tokens, avaliações LLM e métricas
    de observabilidade.
    
    Premissas da estrutura de dados:
    --------------------------------
    - Arquivos JSON nomeados como: <id_peca>.json (padrão: \d{12}.\d+.\d*.json)
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
                 regex_arquivos: str = None):
        """
        Inicializa a classe de carga de dados.
        
        Args:
            pasta_origem: Pasta com JSONs de referência (ground truth)
            pastas_destinos: Lista de pastas com JSONs a comparar
            rotulo_id: Nome do campo ID (ex: 'id', 'doc_id')
            rotulo_origem: Rótulo do modelo de origem/ground truth (ex: 'True', 'base_gpt5')
            rotulos_destinos: Lista de rótulos para cada pasta destino
            campos_comparacao: Lista de campos a serem comparados
            regex_arquivos: Regex customizada para identificar arquivos válidos (opcional)
        """
        assert len(pastas_destinos) == len(rotulos_destinos), \
            "Número de pastas_destinos deve ser igual ao número de rotulos_destinos"
        
        self.pasta_origem = pasta_origem
        self.pastas_destinos = pastas_destinos
        self.rotulo_id = rotulo_id
        self.rotulo_origem = rotulo_origem
        self._rotulos_destinos = rotulos_destinos
        self.campos_comparacao = campos_comparacao
        
        # Regex para identificar arquivos JSON válidos
        if regex_arquivos and str(regex_arquivos).strip() != '*':
            self._re_arquivos_json = re.compile(regex_arquivos)
        else:
            self._re_arquivos_json = re.compile(r'^((?!resumo|tokens|avaliacao).+)\.json$')
        
        # Atributos públicos (preenchidos após carregar())
        self.dados = None
        self.rotulos = None
        self.tokens = None
        self.avaliacao_llm = None
    
    def _carregar_json(self, caminho: str) -> dict:
        """Carrega arquivo JSON com tratamento de erros (método interno)"""
        if not os.path.exists(caminho):
            return {'erro': 'Inexistente'}
        
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                # Verifica se já possui erro registrado
                if 'erro' in dados:
                    return {'erro': f"Erro na extração: {dados['erro']}"}
                return dados
        except Exception as e:
            return {'erro': f'Erro ao ler JSON: {str(e)}'}


    def _filtrar_campos(self, json_data: dict, campos: list) -> dict:
        """Filtra apenas os campos especificados para comparação (método interno)"""
        if 'erro' in json_data:
            return json_data
        
        return {campo: json_data.get(campo) for campo in campos if campo in json_data}


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

    def _get_avaliacao_llm(self, arquivo_base:str, rotulo:str) -> dict:
        ''' 
        Busca o arquivo <id_peca>.avaliacao.json e retorna seu conteúdo ou None se não existir (método interno).
        
        Agora suporta métricas por campo em 'metricas_por_campo':
        - Métricas globais: rotulo_P, rotulo_R, rotulo_F1
        - Métricas por campo: rotulo_campo_P, rotulo_campo_R, rotulo_campo_F1
        - F1 é calculado como média harmônica quando não fornecido
        '''
        arquivo = f"{arquivo_base}.avaliacao.json"
        dados = None
        if os.path.exists(arquivo):
            with open(arquivo, 'r', encoding='utf-8') as f:
                dados_original = json.load(f)
            if dados_original:  # Verifica se tem conteúdo
                # ===== MÉTRICAS GLOBAIS =====
                p = dados_original.get('precision', 0)
                r = dados_original.get('recall', 0)
                # não tem f1, faz a média harmônica
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
                        
                        # Extrai precision e recall do campo
                        p_campo = metricas.get('precision')
                        r_campo = metricas.get('recall')
                        
                        # Armazena as métricas (None se não disponíveis)
                        dados[f'{rotulo}_{campo}_P'] = p_campo
                        dados[f'{rotulo}_{campo}_R'] = r_campo
                        
                        # Calcula F1 apenas se P e R não forem None
                        if p_campo is not None and r_campo is not None:
                            f1_campo = harmonic_mean([p_campo, r_campo]) if (p_campo + r_campo) > 0 else 0
                            dados[f'{rotulo}_{campo}_F1'] = f1_campo
                        else:
                            dados[f'{rotulo}_{campo}_F1'] = None
        return dados

    def _get_observabilidade(self, arquivo_base:str, rotulo:str, resumo:dict) -> dict:
        ''' 
        Busca o arquivo pasta(base)/observabilidade/arquivo_base.obs.json e retorna seu conteúdo ou None se não existir (método interno).
        - o arquivo pode estar na pasta do arquivo base também.
        
        Extrai métricas de observabilidade:
        - Tempo total global (sufixo _SEG: arredondado para inteiro em segundos)
        - Loops de revisão globais (sufixo _REV: contagem de campos com revisão)
        - Número de agentes executados (sufixo _AGT)
        - Por campo/agente: 
            * tempo total (sufixo _SEG: soma de todas iterações)
            * iterações (sufixo _IT: máximo entre todas execuções, não soma)
            * sucesso (sufixo _OK: sim/não da última execução)
        
        NOTA: Métricas de campos (QTD e BYTES) foram movidas para _get_metricas_campos()
              e são mescladas posteriormente no método carregar().
        
        Fallback: se não houver .obs.json mas houver tempo no resumo de tokens, usa esse valor.
        
        Args:
            arquivo_base: Caminho base do arquivo (sem extensão)
            rotulo: Rótulo do modelo (ex: 'agentes_gpt5')
            resumo: Dict com resumo de tokens (para fallback de tempo)
        
        Returns:
            dict: {rotulo_SEG: val, rotulo_REV: val, rotulo_AGT: val, rotulo_IT: val, ...}
                  ou None se não houver dados disponíveis
        '''
        pasta, arquivo = os.path.split(arquivo_base)
        arquivos = [f"{pasta}/observabilidade/{arquivo}.obs.json",
                  f"{arquivo_base}.obs.json"]
        
        dados = None
        for caminho_arquivo in arquivos:
            if not os.path.exists(caminho_arquivo):
                continue
                
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                dados_original = json.load(f)
            
            if not dados_original:  # Verifica se tem conteúdo
                continue
            
            # Extrai observabilidade
            obs = dados_original.get('observabilidade', {})
            if not obs:
                continue
            
            dados = {}
            
            # ===== MÉTRICAS GLOBAIS (OrquestracaoFinal) =====
            orquestracao = obs.get('OrquestracaoFinal', [])
            if orquestracao and len(orquestracao) > 0:
                orch = orquestracao[0]  # Pega o primeiro (geralmente único)
                
                # Tempo total de execução (arredondado para inteiro) - sufixo _SEG
                tempo_total = orch.get('duracao_total_segundos')
                if tempo_total is not None:
                    dados[f'{rotulo}_SEG'] = int(round(tempo_total))
                
                # Loops de revisão - sufixo _REV
                loops_revisao = orch.get('loops_revisao')
                if loops_revisao is not None:
                    dados[f'{rotulo}_REV'] = loops_revisao
                
                # Total de agentes executados - sufixo _AGT
                total_agentes = orch.get('total_agentes_executados')
                if total_agentes is not None:
                    dados[f'{rotulo}_AGT'] = total_agentes
            
            # ===== MÉTRICAS POR CAMPO/AGENTE =====
            # Agrupa por nome do agente (que representa o campo)
            max_iteracoes_global = 0
            total_revisoes_global = 0
            
            for nome_agente, lista_execucoes in obs.items():
                if nome_agente == 'OrquestracaoFinal':
                    continue
                
                if not isinstance(lista_execucoes, list) or len(lista_execucoes) == 0:
                    continue
                
                # Agrupa métricas deste campo/agente
                tempo_campo = 0.0
                max_iteracoes_campo = 0
                tem_revisao_campo = False
                sucesso_ultima_execucao = False
                
                for idx, execucao in enumerate(lista_execucoes):
                    # Pega o máximo de iterações (não soma)
                    iteracoes = execucao.get('iteracoes', 0)
                    max_iteracoes_campo = max(max_iteracoes_campo, iteracoes)
                    
                    # Soma tempo total
                    duracao = execucao.get('duracao_segundos', 0.0)
                    tempo_campo += duracao
                    
                    # Verifica se houve revisão em qualquer execução
                    if execucao.get('tem_revisao', False):
                        tem_revisao_campo = True
                    
                    # Sucesso da última execução
                    if idx == len(lista_execucoes) - 1:
                        sucesso_ultima_execucao = execucao.get('sucesso', False)
                
                # Adiciona métricas do campo (se houver dados)
                # Tempo com sufixo _SEG
                if tempo_campo > 0:
                    dados[f'{rotulo}_{nome_agente}_SEG'] = int(round(tempo_campo))
                
                # Iterações com sufixo _IT (máximo, não soma)
                if max_iteracoes_campo > 0:
                    dados[f'{rotulo}_{nome_agente}_IT'] = int(max_iteracoes_campo)
                    max_iteracoes_global = max(max_iteracoes_global, max_iteracoes_campo)
                
                # Sucesso com sufixo _OK (sim/não)
                dados[f'{rotulo}_{nome_agente}_OK'] = 'sim' if sucesso_ultima_execucao else 'não'
                
                # Contabiliza revisões
                if tem_revisao_campo:
                    total_revisoes_global += 1
            
            # Adiciona totais globais de iterações e revisões
            # Iterações globais = máximo entre todos os campos
            if max_iteracoes_global > 0:
                dados[f'{rotulo}_IT'] = int(max_iteracoes_global)
            
            if total_revisoes_global > 0:
                dados[f'{rotulo}_REV'] = int(total_revisoes_global)
            
            # Se encontrou dados, continua para adicionar métricas do modelo base
            if dados:
                break  # Sai do loop de arquivos
        
        # FALLBACK: Se não há .obs.json mas há tempo no resumo de tokens
        if dados is None and resumo is not None:
            # Prioriza 'time' (novo padrão), depois 'tempo' (antigo), depois com prefixo do rótulo
            tempo_resumo = resumo.get('time') or resumo.get('tempo') or resumo.get(f'{rotulo}_tempo')
            if tempo_resumo is not None:
                dados = {f'{rotulo}_SEG': int(round(tempo_resumo)) if isinstance(tempo_resumo, (int, float)) else tempo_resumo}
        
        # ===== MÉTRICAS DO MODELO BASE (se json_dados fornecido) =====
        # REMOVIDO: Agora essas métricas são calculadas em _get_metricas_campos
        # e mescladas posteriormente com observabilidade
        
        return dados

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

    def _get_resumo_tokens(self, arquivo_base:str, rotulo: str) -> dict:
        ''' 
        Busca o arquivo <id_peca>.tokens.json ou <id_peca>.resumo.json e retorna seu conteúdo ou None se não existir (método interno).
        
        Suporta múltiplas variações de nomes de campos:
        - input_tokens ou prompt_tokens
        - output_tokens ou completion_tokens
        - total_tokens (calculado se não existir ou se for maior que soma de input + output)
        - cached_tokens (opcional)
        - reasoning_tokens (opcional)
        - finish_reason (opcional)
        - tempo (opcional)
        '''
        for tipo in ['.tokens', '_tokens', '.resumo', '_resumo']:
            arquivo = f"{arquivo_base}{tipo}.json"
            if os.path.exists(arquivo):
                with open(arquivo, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                    
                    # Navega na estrutura para encontrar os totais
                    if 'total' in dados:
                        dados = dados.get('total') or {}
                    elif 'total_geral' in dados:
                        dados = dados.get('total_geral') or {}
                    elif 'tokens' in dados:
                        dados = dados.get('tokens') or {}
                    
                    # Normaliza nomes de campos com flexibilidade total
                    # Tenta diferentes variações para cada campo
                    input_tokens = (
                        dados.get('input_tokens') or 
                        dados.get('prompt_tokens') or 
                        0
                    )
                    
                    output_tokens = (
                        dados.get('output_tokens') or 
                        dados.get('completion_tokens') or 
                        0
                    )
                    
                    # Campos opcionais
                    cached_tokens = dados.get('cached_tokens') or 0
                    reasoning_tokens = dados.get('reasoning_tokens') or 0
                    finish_reason = dados.get('finish_reason') or '-'

                    # Total pode ser explícito ou calculado
                    total_tokens = dados.get('total_tokens') or 0
                    
                    # tempo - prioriza 'time' (novo padrão), depois 'tempo' (antigo)
                    tempo = dados.get('time') or dados.get('tempo') or None
                    return {
                        f'{rotulo}_input': input_tokens,
                        f'{rotulo}_output': output_tokens,
                        f'{rotulo}_total': max(total_tokens,
                                            input_tokens + output_tokens + reasoning_tokens),
                        f'{rotulo}_cache': cached_tokens,
                        f'{rotulo}_reason': reasoning_tokens,
                        f'{rotulo}_finish': finish_reason,
                        f'_tempo': tempo # não fica na planilha, usa na observabilidade
                    }
        #raise Exception(f'Arquivo de resumo/tokens não encontrado para {arquivo_base}')
        #print(f'⚠️  Arquivo de resumo/tokens não encontrado para {arquivo}')
        #Util.pausa(1)
        return None

    def carregar(self):
        """
        Carrega e processa dados, retornando um objeto JsonAnaliseDados completo.
        
        Returns:
            JsonAnaliseDados: Container com todos os dados processados, pronto para uso
                             em JsonAnaliseDataFrame
        
        IMPORTANTE: Só processa IDs que existem na ORIGEM.
        Se a origem for filtrada (retornar None), o ID não entra na comparação.
        
        Example:
            >>> carga = CargaDadosComparacao(...)
            >>> dados_analise = carga.carregar()  # Retorna JsonAnaliseDados
            >>> analisador = JsonAnaliseDataFrame(dados_analise, config={...})
        """
        # Coleta IDs da origem (apenas esses serão processados)
        ids_origem = set(self._listar_arquivos_json(self.pasta_origem))
        
        # Define rótulos primeiro (necessário para construir dicts)
        self.rotulos = [self.rotulo_id, self.rotulo_origem] + self._rotulos_destinos
        
        assert len(self.rotulos) == 2 + len(self.pastas_destinos), \
            'Número de rótulos deve ser igual a 2 + número de destinos!'
        
        # Coleta IDs dos destinos (apenas para estatísticas)
        ids_destinos = []
        for destino in self.pastas_destinos:
            ids_destinos.append(set(self._listar_arquivos_json(destino)))
        
        print(f"\n📊 Resumo dos dados:")
        print(f"   IDs na origem (a processar): {len(ids_origem)}")
        for i, ids_dest in enumerate(ids_destinos, 1):
            nome_destino = os.path.basename(self.pastas_destinos[i-1].rstrip('/')).replace('espelhos_', '').replace('_gpt5', '')
            # Calcula interseção e diferença
            em_ambos = len(ids_origem & ids_dest)
            so_destino = len(ids_dest - ids_origem)
            so_origem = len(ids_origem - ids_dest)
            print(f"   Destino {i} ({nome_destino}): {len(ids_dest)} IDs ({em_ambos} em comum, {so_origem} só na origem, {so_destino} só no destino)")
        
        # Prepara dados para comparação (apenas IDs da origem)
        self.dados = []
        tokens_dict = {}  # Temporário: {'id_peca': {modelo_input: qtd, modelo_output: qtd, ...}}
        avaliacoes_dict = {}  # Temporário: {'id_peca': {modelo_precision: val, modelo_recall: val, ...}}
        obs_dict = {}  # Temporário: {'id_peca': {tempo, iteracoes, aprovado, ...}}
        campos_dict = {}  # Temporário: {'id_peca': {rotulo_QTD: val, rotulo_campo_BYTES: val, ...}}
        erros_count = {'inexistente': 0, 'erro_extracao': 0, 'sucesso': 0, 'filtrado': 0}

        def _add_tokens_avaliacoes(id_peca, arquivo_base, rotulo, json_dados=None):
            """Adiciona tokens, avaliações e observabilidade de um modelo ao dicionário consolidado"""
            tokens_peca = self._get_resumo_tokens(arquivo_base, rotulo=rotulo)
            if tokens_peca is not None:
                if id_peca not in tokens_dict:
                    tokens_dict[id_peca] = {}
                # Mescla os dados de tokens do modelo no dicionário do id_peca
                tokens_dict[id_peca].update(tokens_peca)
            
            avaliacao_peca = self._get_avaliacao_llm(arquivo_base, rotulo=rotulo)
            if avaliacao_peca is not None:
                if id_peca not in avaliacoes_dict:
                    avaliacoes_dict[id_peca] = {}
                avaliacoes_dict[id_peca].update(avaliacao_peca)
            
            obs_peca = self._get_observabilidade(arquivo_base, rotulo=rotulo, resumo=tokens_peca)
            if obs_peca is not None:
                if id_peca not in obs_dict:
                    obs_dict[id_peca] = {}
                obs_dict[id_peca].update(obs_peca)
            
            # Métricas de campos (QTD e BYTES) - apenas se json_dados fornecido
            if json_dados is not None:
                campos_peca = self._get_metricas_campos(json_dados, rotulo=rotulo)
                if campos_peca is not None:
                    if id_peca not in campos_dict:
                        campos_dict[id_peca] = {}
                    campos_dict[id_peca].update(campos_peca)
        
        print(f"\n🔄 Carregando e filtrando dados (processando apenas IDs da origem)...")
        for id_peca in tqdm(sorted(ids_origem), desc="Processando arquivos"):
            # Carrega origem
            arquivo_base = os.path.join(self.pasta_origem, f'{id_peca}')
            arquivo_origem = f'{arquivo_base}.json'
            json_origem = self._carregar_json(arquivo_origem)
            json_origem_filtrado = self._filtrar_campos(json_origem, self.campos_comparacao)
            json_origem_filtrado = self._filtro_origem(json_origem_filtrado)
            if json_origem_filtrado is None:
                # Se o filtro retornar None, NÃO adiciona aos dados (pula completamente)
                erros_count['filtrado'] += 1
                continue
            # Origem: passa json_origem_filtrado para calcular métricas de campos (QTD e BYTES)
            _add_tokens_avaliacoes(id_peca, arquivo_base, self.rotulos[1], json_dados=json_origem_filtrado)

            # Carrega destinos
            jsons_destinos = []
            tem_erro = 'erro' in json_origem_filtrado

            for i, destino in enumerate(self.pastas_destinos):
                arquivo_base_dest = os.path.join(destino, f'{id_peca}')
                arquivo_destino = f'{arquivo_base_dest}.json'
                json_dest = self._carregar_json(arquivo_destino)
                json_dest_filtrado = self._filtrar_campos(json_dest, self.campos_comparacao)
                jsons_destinos.append(json_dest_filtrado)
                # Destinos: não passa json_dados (métricas de campos só da origem)
                _add_tokens_avaliacoes(id_peca, arquivo_base_dest, self.rotulos[2 + i])
                
                if 'erro' in json_dest_filtrado:
                    tem_erro = True
            
            # Contabiliza status
            if tem_erro:
                if any('Inexistente' in json.get('erro', '') for json in [json_origem_filtrado] + jsons_destinos):
                    erros_count['inexistente'] += 1
                else:
                    erros_count['erro_extracao'] += 1
            else:
                erros_count['sucesso'] += 1
            
            # Cria dict para nova estrutura
            linha_dict = {
                self.rotulos[0]: id_peca,  # 'id'
                self.rotulos[1]: json_origem_filtrado  # 'True'
            }
            # Adiciona destinos
            for i, json_dest in enumerate(jsons_destinos):
                linha_dict[self.rotulos[2 + i]] = json_dest
            
            self.dados.append(linha_dict)
        
        # Consolida o resumo dos tokens em uma lista com id_peca e as outras chaves
        resumo_tokens = []
        lista_avaliacao_llm = []
        lista_observabilidade = []
        max_modelos = 0
        
        # Usa nome do campo ID configurado
        nome_campo_id = self.rotulo_id
        
        for id_peca, avaliacao in avaliacoes_dict.items():
            _dados = {nome_campo_id: id_peca}
            _dados.update(avaliacao)
            lista_avaliacao_llm.append(_dados)
            max_modelos = max(max_modelos, len(avaliacao))
        # mantém apenas os que possuem todas as avaliações
        if max_modelos > 0:
            lista_avaliacao_llm = [a for a in lista_avaliacao_llm if len(a) == (1 + max_modelos)]  # +1 para campo ID

        # Agora adiciona as observabilidades mescladas com métricas de campos
        for id_peca, obs in obs_dict.items():
            # Cada obs já contém todas as chaves de todos os modelos
            # Ex: {'agentes_gpt5_SEG': 84, 'agentes_gpt5_IT': 1, ...}
            _dados = {nome_campo_id: id_peca}
            _dados.update({c:v for c,v in obs.items() if not c.startswith('_')})
            
            # Mescla métricas de campos se existirem para este id_peca
            if id_peca in campos_dict:
                _dados.update(campos_dict[id_peca])
            
            lista_observabilidade.append(_dados)
        
        # Adiciona IDs que estão APENAS em campos_dict (origem sem .obs.json)
        ids_apenas_campos = set(campos_dict.keys()) - set(obs_dict.keys())
        for id_peca in ids_apenas_campos:
            _dados = {nome_campo_id: id_peca}
            if id_peca in campos_dict:
                _dados.update(campos_dict[id_peca])
            lista_observabilidade.append(_dados)
        
        for id_peca, token_dict in tokens_dict.items():
            # Cada token_dict já contém todas as chaves de todos os modelos
            # Ex: {'GPT5_input': 100, 'GPT5_output': 50, 'agentes_gpt5_input': 120, ...}
            _dados = {nome_campo_id: id_peca}
            _dados.update(token_dict)
            resumo_tokens.append(_dados)
        
        # Atribui aos atributos públicos
        self.tokens = resumo_tokens
        self.avaliacao_llm = lista_avaliacao_llm
        self.observabilidade = lista_observabilidade

        print(f"\n📋 Status dos dados:")
        print(f"   ✅ Sucesso: {erros_count['sucesso']}")
        if len(self.tokens) > 0:
            print(f"   🪙 Tokens disponíveis para {len(self.tokens)} IDs")
        if len(self.avaliacao_llm) > 0:
            print(f"   ⭐ Avaliações LLM disponíveis para {len(self.avaliacao_llm)} IDs")
        if len(self.observabilidade) > 0:
            print(f"   📈 Observabilidade disponível para {len(self.observabilidade)} IDs")
        print(f"   🔍 Filtrados (origem vazia): {erros_count['filtrado']}")
        if erros_count['inexistente'] > 0:
            print(f"   ❌ Inexistente: {erros_count['inexistente']}")
        if erros_count['erro_extracao'] > 0:
            print(f"   ⚠️  Erro na extração: {erros_count['erro_extracao']}")
        print(f"   📊 Total a comparar: {len(self.dados)} (de {len(ids_origem)} IDs na origem)")
        
        # ANÁLISE: Mostra % de dados filtrados
        total_origem = len(ids_origem)
        if erros_count['filtrado'] > 0:
            pct_filtrado = (erros_count['filtrado'] / total_origem) * 100
            print(f"\n⚠️  ATENÇÃO: {pct_filtrado:.1f}% dos JSONs da origem estão completamente vazios!")
            print(f"   Isso explica SIM=0.0 nas estatísticas de estrutura quando True está vazio.")
            print(f"   Considere revisar os dados de origem (RAW) ou ajustar o filtro.")
        
        # Cria e retorna JsonAnaliseDados
        dados_analise = JsonAnaliseDados(
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
        
        return dados_analise
        
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