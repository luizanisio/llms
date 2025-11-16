# -*- coding: utf-8 -*-
"""
Classe container para dados de análise de JSONs.

Autor: Luiz Anísio
Data: 30/10/2025
Fonte: https://github.com/luizanisio/llms/tree/main/src

Esta classe serve como container genérico para dados que serão analisados
pela JsonAnaliseDataFrame. Permite desacoplar a carga de dados da análise,
facilitando o uso em diferentes projetos.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ConfigAnaliseDados:
    """
    Configurações para análise de dados JSON.
    
    Attributes:
        pasta_origem: Caminho da pasta de origem dos JSONs
        pastas_destinos: Lista de caminhos das pastas de destino dos JSONs
        campos_comparacao: Lista de campos a serem comparados
        rotulos_destinos: Lista de rótulos dos modelos destinos
        nome_campo_id: Nome do campo que identifica cada peça/documento (usado internamente)
        rotulo_campo_id: Rótulo/label do campo ID (usado em exibições, padrão=nome_campo_id)
        rotulo_origem: Rótulo do modelo de referência/ground truth
    """
    pasta_origem: str = None
    pastas_destinos: List[str] = None
    campos_comparacao: List[str] = None
    rotulos_destinos: List[str] = None
    nome_campo_id: str = 'id_peca'
    rotulo_campo_id: str = None  # Se None, usa nome_campo_id
    rotulo_origem: str = 'Origem'

class JsonAnaliseDados:
    """
    Container genérico para dados de análise de JSONs.
    
    Esta classe encapsula todos os dados necessários para análise comparativa
    de JSONs, permitindo que diferentes fontes de dados (CargaDadosComparacao,
    APIs, bancos de dados, etc.) alimentem a JsonAnaliseDataFrame de forma uniforme.
    
    Attributes:
        dados: Lista de dicts com estrutura {'id': ..., 'True': ..., 'Modelo1': ..., 'Modelo2': ...}
        rotulos: Lista de rótulos ['id', 'True', 'Modelo1', 'Modelo2', ...]
        tokens: Lista de dicts com contagem de tokens (opcional)
        avaliacao_llm: Lista de dicts com avaliações LLM (opcional)
        metadados: Dict com informações adicionais sobre os dados (opcional)
    
    Example:
        >>> dados_analise = JsonAnaliseDados(
        ...     dados=[
        ...         {'id': '123', 'True': {...}, 'Modelo1': {...}},
        ...         {'id': '456', 'True': {...}, 'Modelo1': {...}}
        ...     ],
        ...     rotulos=['id', 'True', 'Modelo1'],
        ...     tokens=[
        ...         {'id_peca': '123', 'Modelo1_input': 100, 'Modelo1_output': 50},
        ...         {'id_peca': '456', 'Modelo1_input': 120, 'Modelo1_output': 60}
        ...     ]
        ... )
        >>> analisador = JsonAnaliseDataFrame(dados_analise, config={...})
    """
    
    def __init__(self,
                 dados: List[Dict[str, Any]],
                 rotulos: List[str],
                 tokens: List[Dict[str, Any]] = None,
                 avaliacao_llm: List[Dict[str, Any]] = None,
                 observabilidade: List[Dict[str, Any]] = None,
                 pasta_origem: str = None,
                 pastas_destinos: List[str] = None,
                 campos_comparacao: List[str] = None,
                 rotulos_destinos: List[str] = None,
                 nome_campo_id: str = 'id_peca',
                 rotulo_campo_id: str = None,
                 rotulo_origem: str = 'Origem'):
        """
        Inicializa o container de dados para análise.

        Args:
            dados: Lista de dicts com JSONs a comparar
                   Estrutura: [{'id': val, 'True': {...}, 'Modelo1': {...}, ...}, ...]
            rotulos: Lista de rótulos correspondentes às chaves em dados
                     Estrutura: ['id', 'True', 'Modelo1', 'Modelo2', ...]
            tokens: Lista de dicts com contagem de tokens por modelo (opcional)
                    Estrutura: [{'id_peca': val, 'Modelo1_input': n, 'Modelo1_output': m, ...}, ...]
            avaliacao_llm: Lista de dicts com avaliações LLM (opcional)
                           Estrutura: [{'id_peca': val, 'Modelo1_P': p, 'Modelo1_R': r, ...}, ...]
            observabilidade: Lista de dicts com métricas de observabilidade (opcional)
                             Estrutura: [{'id_peca': val, 'metrica1': x, 'metrica2': y, ...}, ...]
            pasta_origem: Pasta de origem dos dados (opcional)
            pastas_destinos: Lista de pastas de destino (opcional)
            campos_comparacao: Lista de campos comparados (opcional)
            rotulos_destinos: Lista de rótulos dos modelos destino (opcional)
            nome_campo_id: Nome interno do campo de ID (usado em DataFrames)
            rotulo_campo_id: Rótulo de exibição do campo ID (se None, usa nome_campo_id)
            rotulo_origem: Rótulo do modelo de referência/ground truth
        
        Raises:
            AssertionError: Se validações básicas falharem
        """
        # Validações básicas
        assert dados is not None and len(dados) > 0, "dados não pode ser vazio"
        assert rotulos is not None and len(rotulos) >= 2, "rotulos deve ter ao menos 2 elementos: ['id', 'True']"
        assert len(rotulos) == len(dados[0]), f"Número de rótulos ({len(rotulos)}) deve corresponder ao número de chaves em dados ({len(dados[0])})"
        
        # Atributos principais
        self.dados = dados
        self.rotulos = rotulos
        self.tokens = tokens if tokens is not None else []
        self.avaliacao_llm = avaliacao_llm if avaliacao_llm is not None else []
        self.observabilidade = observabilidade if observabilidade is not None else []
        self.config = ConfigAnaliseDados(
            pasta_origem=pasta_origem,
            pastas_destinos=pastas_destinos,
            campos_comparacao=campos_comparacao,
            rotulos_destinos=rotulos_destinos,
            nome_campo_id=nome_campo_id,
            rotulo_campo_id=rotulo_campo_id if rotulo_campo_id is not None else nome_campo_id,
            rotulo_origem=rotulo_origem
        )
        
        # Propriedades derivadas (cache)
        self._n_documentos = len(dados)
        self._n_modelos = len(rotulos_destinos) if rotulos_destinos is not None else len(rotulos) - 2
        self._rotulo_id = rotulos[0]
        self._rotulo_true = rotulos[1]
        self._rotulos_modelos = rotulos[2:]

    @property
    def n_documentos(self) -> int:
        """Retorna o número de documentos a serem analisados"""
        return self._n_documentos
    
    @property
    def n_modelos(self) -> int:
        """Retorna o número de modelos a serem comparados"""
        return self._n_modelos
    
    @property
    def rotulo_id(self) -> str:
        """Retorna o rótulo da coluna de ID"""
        return self._rotulo_id
    
    @property
    def rotulo_true(self) -> str:
        """Retorna o rótulo da coluna de ground truth"""
        return self._rotulo_true
    
    @property
    def rotulos_modelos(self) -> List[str]:
        """Retorna a lista de rótulos dos modelos"""
        return self._rotulos_modelos
    
    @property
    def tem_tokens(self) -> bool:
        """Verifica se há dados de tokens disponíveis"""
        return self.tokens is not None and len(self.tokens) > 0
    
    @property
    def tem_avaliacao_llm(self) -> bool:
        """Verifica se há avaliações LLM disponíveis"""
        return self.avaliacao_llm is not None and len(self.avaliacao_llm) > 0
    
    def validar(self) -> bool:
        """
        Valida a consistência dos dados.
        
        Returns:
            True se todos os dados estão consistentes
        
        Raises:
            ValueError: Se houver inconsistências nos dados
        """
        # Valida estrutura dos dados
        for i, linha in enumerate(self.dados):
            if not isinstance(linha, dict):
                raise ValueError(f"Linha {i} deve ser um dicionário, encontrado: {type(linha)}")
            
            if len(linha) != len(self.rotulos):
                raise ValueError(f"Linha {i} tem {len(linha)} campos, esperado {len(self.rotulos)}")
            
            for rotulo in self.rotulos:
                if rotulo not in linha:
                    raise ValueError(f"Linha {i} não contém o rótulo '{rotulo}'")
        
        # Valida tokens (se existirem)
        if self.tem_tokens:
            for i, token_dict in enumerate(self.tokens):
                if self.config.nome_campo_id not in token_dict:
                    raise ValueError(f"Token {i} deve conter '{self.config.nome_campo_id}'")
        
        # Valida avaliações LLM (se existirem)
        if self.tem_avaliacao_llm:
            for i, aval_dict in enumerate(self.avaliacao_llm):
                if self.config.nome_campo_id not in aval_dict:
                    raise ValueError(f"Avaliação LLM {i} deve conter '{self.config.nome_campo_id}'")
        
        return True
    
    def resumo(self) -> str:
        """
        Retorna um resumo textual dos dados.
        
        Returns:
            String com resumo formatado
        """
        linhas = [
            "=" * 60,
            "RESUMO DOS DADOS DE ANÁLISE",
            "=" * 60,
            f"📊 Documentos: {self.n_documentos}",
            f"🤖 Modelos: {self.n_modelos}",
            f"   Rótulos: {', '.join(self.rotulos_modelos)}",
            f"🪙 Tokens: {'Sim' if self.tem_tokens else 'Não'} ({len(self.tokens)} docs)" if self.tem_tokens else "🪙 Tokens: Não disponível",
            f"⭐ Avaliações LLM: {'Sim' if self.tem_avaliacao_llm else 'Não'} ({len(self.avaliacao_llm)} docs)" if self.tem_avaliacao_llm else "⭐ Avaliações LLM: Não disponível",
        ]
        
        # Adiciona informações de configuração se disponíveis
        if self.config:
            if self.config.pasta_origem:
                linhas.append(f"\n� Pasta origem: {self.config.pasta_origem}")
            if self.config.campos_comparacao:
                linhas.append(f"📋 Campos comparados: {len(self.config.campos_comparacao)}")
        
        linhas.append("=" * 60)
        return "\n".join(linhas)
    
    def __repr__(self) -> str:
        """Representação string do objeto"""
        return f"JsonAnaliseDados(docs={self.n_documentos}, modelos={self.n_modelos}, tokens={self.tem_tokens}, llm={self.tem_avaliacao_llm})"
    
    def __str__(self) -> str:
        """String amigável do objeto"""
        return self.resumo()


if __name__ == '__main__':
    """Exemplo de uso da classe JsonAnaliseDados"""
    
    # Exemplo 1: Criação manual
    print("\n" + "=" * 60)
    print("EXEMPLO 1: Criação Manual de JsonAnaliseDados")
    print("=" * 60)
    
    dados_exemplo = [
        {
            'id': 'doc001',
            'True': {'campo1': 'texto verdadeiro', 'campo2': 'outro texto'},
            'ModeloA': {'campo1': 'texto predito A', 'campo2': 'outro texto A'},
            'ModeloB': {'campo1': 'texto predito B', 'campo2': 'outro texto B'}
        },
        {
            'id': 'doc002',
            'True': {'campo1': 'segundo documento', 'campo2': 'mais texto'},
            'ModeloA': {'campo1': 'predição A doc 2', 'campo2': 'texto A'},
            'ModeloB': {'campo1': 'predição B doc 2', 'campo2': 'texto B'}
        }
    ]
    
    tokens_exemplo = [
        {'id_peca': 'doc001', 'ModeloA_input': 100, 'ModeloA_output': 50, 'ModeloB_input': 120, 'ModeloB_output': 60},
        {'id_peca': 'doc002', 'ModeloA_input': 110, 'ModeloA_output': 55, 'ModeloB_input': 130, 'ModeloB_output': 65}
    ]
    
    dados_analise = JsonAnaliseDados(
        dados=dados_exemplo,
        rotulos=['id', 'True', 'ModeloA', 'ModeloB'],
        tokens=tokens_exemplo,
        pasta_origem='./teste',
        campos_comparacao=['campo1', 'campo2']
    )
    
    print(dados_analise)
    print(f"\n✅ Validação: {dados_analise.validar()}")
    print(f"📊 Propriedades:")
    print(f"   - Documentos: {dados_analise.n_documentos}")
    print(f"   - Modelos: {dados_analise.n_modelos}")
    print(f"   - Rótulo ID: {dados_analise.rotulo_id}")
    print(f"   - Rótulo True: {dados_analise.rotulo_true}")
    print(f"   - Modelos: {dados_analise.rotulos_modelos}")
    
    # Exemplo 2: Uso com CargaDadosComparacao
    print("\n" + "=" * 60)
    print("EXEMPLO 2: Integração com CargaDadosComparacao")
    print("=" * 60)
    print("""
    from util_json_carga import CargaDadosComparacao
    from util_json_dados import JsonAnaliseDados
    
    # 1. Carrega dados
    carga = CargaDadosComparacao(...)
    carga.carregar()
    
    # 2. Cria container de análise
    dados_analise = JsonAnaliseDados(
        dados=carga.dados,
        rotulos=carga.rotulos,
        tokens=carga.tokens,
        avaliacao_llm=carga.avaliacao_llm,
        pasta_origem=carga.pasta_origem,
        campos_comparacao=carga.campos_comparacao
    )
    
    # 3. Usa na análise
    analisador = JsonAnaliseDataFrame(dados_analise, config={...})
    """)
