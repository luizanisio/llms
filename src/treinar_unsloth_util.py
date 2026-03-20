#!/usr/bin/env python3

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Utilitários para o pacote treinar_unsloth.py
Inclui a classe YamlTreinamento para validação e carregamento de configurações YAML.

Classes:
    - YamlTreinamento: Carrega, valida e processa configurações YAML para treinamento
    - DatasetPastas: Carrega datasets a partir de estrutura de pastas com arquivos de texto/json
"""

import os
import sys
import json
import random
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import re

import util  # garante que a pasta src está no sys.path

try:
    import yaml
except ImportError:
    print("Erro: O pacote 'yaml' não está instalado.")
    print("Por favor, instale-o executando: pip install pyyaml")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Erro: O pacote 'pandas' não está instalado.")
    print("Por favor, instale-o executando: pip install pandas")
    sys.exit(1)

# Importa utilitários do projeto
from util import UtilTextos as Util  # UtilTextos tem mensagem_to_json


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TIPO_ENTRADA_PASTAS = "pastas"
TIPO_ENTRADA_DATASET = "dataset"
TIPO_ENTRADA_CURRICULUM = "curriculum"
TIPOS_ENTRADA_VALIDOS = {TIPO_ENTRADA_PASTAS, TIPO_ENTRADA_DATASET, TIPO_ENTRADA_CURRICULUM}
# Tipos que usam infraestrutura de pastas (entrada, dataset, predicao, validacao)
TIPOS_BASEADOS_EM_PASTAS = {TIPO_ENTRADA_PASTAS, TIPO_ENTRADA_CURRICULUM}

FORMATO_SAIDA_JSON = "json"
FORMATO_SAIDA_TEXTO = "texto"
FORMATOS_SAIDA_VALIDOS = {FORMATO_SAIDA_JSON, FORMATO_SAIDA_TEXTO}

# Valores válidos para coluna "alvo" no CSV de divisão
VALORES_TREINO = {"treino", "train"}
VALORES_TESTE = {"teste", "test"}
VALORES_VALIDACAO = {"validacao", "validação", "validation", "eval", "evaluation", "avaliacao", "avaliação"}

# Proporções padrão para divisão do dataset
PROPORCAO_PADRAO = [0.7, 0.1, 0.2]  # treino, validação, teste
SEED_PADRAO = 42


# ---------------------------------------------------------------------------
# Funções Utilitárias Gerais
# ---------------------------------------------------------------------------

def calcular_rouge_l(referencia: str, hipotese: str) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Calcula métricas Rouge-L entre dois textos.
    Retorna uma tupla (metricas, msg_erro).
    
    Returns:
        tuple: (dict_metricas, erro_msg)
            - dict_metricas: {'P': 0.0, 'R': 0.0, 'F1': 0.0} (ou None em erro)
            - erro_msg: Mensagem de erro/warning (ou None em sucesso)
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(referencia, hipotese)
        rouge = scores['rougeL']
        return {
            'P': round(rouge.precision, 4), 
            'R': round(rouge.recall, 4), 
            'F1': round(rouge.fmeasure, 4)
        }, None
    except ImportError:
        return None, "Biblioteca rouge-score não instalada. Instale com: pip install rouge-score"
    except Exception as e:
        return None, f"Erro ao calcular Rouge-L: {str(e)}"


# ---------------------------------------------------------------------------
# Dataclasses para estruturação da configuração
# ---------------------------------------------------------------------------

@dataclass
class ConfigModelo:
    """Configuração do modelo."""
    base: str = ""
    saida: str = ""  # output_dir
    
    def __post_init__(self):
        if not self.base:
            raise ValueError("modelo.base é obrigatório")
        if not self.saida:
            raise ValueError("modelo.saida (output_dir) é obrigatório")


@dataclass
class ConfigLora:
    """Configuração do LoRA."""
    r: int = 8
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    def __post_init__(self):
        if self.r < 0:
            raise ValueError(f"lora.r deve ser >= 0, recebido: {self.r}")


@dataclass
class ConfigTreinamento:
    """Configuração de parâmetros de treinamento."""
    eval_steps: Union[str, int] = "15%"
    batch_size: int = 2
    grad_batch_size: int = 5
    epochs: int = 1
    max_seq_length: int = 4096
    learning_rate: float = 2e-4
    save_checkpoints: bool = True
    resume_from_checkpoint: bool = True
    warmup_steps: int = 5
    nbits: int = 4
    seed: int = 3407
    train_on_responses_only: bool = True  # Treina apenas nas respostas do assistant (recomendado)
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    validar_max_seq_length: bool = True  # Se False e max_seq_length>0, pula recálculo de tokens
    
    def __post_init__(self):
        # Validações de valores
        if self.batch_size <= 0:
            raise ValueError(f"batch_size deve ser > 0, recebido: {self.batch_size}")
        if self.grad_batch_size <= 0:
            raise ValueError(f"grad_batch_size deve ser > 0, recebido: {self.grad_batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs deve ser > 0, recebido: {self.epochs}")
        if self.max_seq_length < 0:
            raise ValueError(f"max_seq_length deve ser >= 0 (0 = automático), recebido: {self.max_seq_length}")
        if self.nbits not in {0, 4, 8, None}:
            raise ValueError(f"nbits deve ser 0, 4, 8 ou None, recebido: {self.nbits}")


@dataclass
class ConfigFormatos:
    """Configuração de formatos de entrada/saída."""
    tipo_entrada: str = TIPO_ENTRADA_DATASET
    formato_saida: str = FORMATO_SAIDA_TEXTO
    
    def __post_init__(self):
        if self.tipo_entrada not in TIPOS_ENTRADA_VALIDOS:
            raise ValueError(f"tipo_entrada deve ser um de {TIPOS_ENTRADA_VALIDOS}, recebido: '{self.tipo_entrada}'")
        if self.formato_saida not in FORMATOS_SAIDA_VALIDOS:
            raise ValueError(f"formato_saida deve ser um de {FORMATOS_SAIDA_VALIDOS}, recebido: '{self.formato_saida}'")


@dataclass
class ConfigMisc:
    """Configurações diversas do projeto."""
    log_level: str = "INFO"  # Nível de log (DEBUG, INFO, WARNING, ERROR)
    env_chave_criptografia: str = ""  # Nome da var de ambiente com chave de criptografia
    
    def __post_init__(self):
        niveis_validos = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in niveis_validos:
            raise ValueError(f"log_level deve ser um de {niveis_validos}, recebido: '{self.log_level}'")
        self.log_level = self.log_level.upper()  # Normaliza para maiúsculas


@dataclass
class ConfigPredicao:
    """Configuração da pasta de predições (saída gerada pelo modelo para avaliação)."""
    pasta: str = ""
    mascara: str = "*.txt"  # Padrão glob para filtrar arquivos
    _validar_caminhos: bool = field(default=True, repr=False)
    
    def __post_init__(self):
        # Pasta de predição é saída para avaliação — cria se não existir
        if self._validar_caminhos and self.pasta and not os.path.isdir(self.pasta):
            os.makedirs(self.pasta, exist_ok=True)


@dataclass
class ConfigGold:
    """Configuração da pasta com o gold dataset (saídas esperadas para treino)."""
    pasta: str = ""
    mascara: str = "*.txt"  # Padrão glob para filtrar arquivos
    _validar_caminhos: bool = field(default=True, repr=False)
    
    def __post_init__(self):
        if self._validar_caminhos and self.pasta and not os.path.isdir(self.pasta):
            raise ValueError(f"Pasta do gold dataset não encontrada: '{self.pasta}'")


@dataclass
class ConfigEntrada:
    """Configuração da pasta de entrada (textos para prompt)."""
    pasta: str = ""  # Pasta com arquivos de texto
    mascara: str = "*.txt"  # Padrão glob para filtrar arquivos
    prompt_template: str = ""  # Arquivo com template do prompt
    tag_texto: str = ""  # Tag a ser substituída pelo conteúdo
    dataframe: str = ""  # Caminho para arquivo parquet com textos
    dataframe_col: str = ""  # Coluna do dataframe com o texto
    dataframe_id: str = ""  # Coluna do dataframe com o ID
    _validar_caminhos: bool = field(default=True, repr=False)
    
    def __post_init__(self):
        # Valida que tem pasta ou dataframe, mas não ambos
        tem_pasta = bool(self.pasta)
        tem_df = bool(self.dataframe)
        
        if self._validar_caminhos:
            if tem_pasta and not os.path.isdir(self.pasta):
                raise ValueError(f"Pasta de entrada não encontrada: '{self.pasta}'")
            if tem_df and not os.path.isfile(self.dataframe):
                raise ValueError(f"Arquivo de dataframe não encontrado: '{self.dataframe}'")
            if self.prompt_template and not os.path.isfile(self.prompt_template):
                raise ValueError(f"Arquivo de template não encontrado: '{self.prompt_template}'")
        
        # Se usa dataframe, precisa informar as colunas
        if tem_df and (not self.dataframe_col or not self.dataframe_id):
            raise ValueError("Se 'dataframe' for informado, 'dataframe_col' e 'dataframe_id' são obrigatórios")
        
        # Se tem template, precisa ter tag (validação estrutural)
        if self.prompt_template and not self.tag_texto:
            raise ValueError("Se 'prompt_template' for informado, 'tag_texto' é obrigatório")


@dataclass
class ConfigDivisao:
    """Configuração de divisão treino/teste/avaliação."""
    arquivo: str = ""
    proporcao: List[float] = field(default_factory=lambda: PROPORCAO_PADRAO.copy())
    seed: int = SEED_PADRAO
    validar_ids: bool = True
    proporcao_reais: Optional[List[float]] = None # Campo para armazenar distribuição real do arquivo
    
    def __post_init__(self):
        # Valida proporções
        if len(self.proporcao) != 3:
            raise ValueError(f"proporcao deve ter 3 valores [treino, validacao, teste], recebido: {self.proporcao}")
        soma = sum(self.proporcao)
        if abs(soma - 1.0) > 0.01:
            raise ValueError(f"proporcao deve somar 1.0, recebido: {soma}")
        if any(p < 0 for p in self.proporcao):
            raise ValueError(f"proporcao não pode ter valores negativos: {self.proporcao}")


@dataclass
class ConfigValidacao:
    """Configuração de validação de saídas."""
    exigir_json_valido: bool = True
    skip_invalidos: bool = False


@dataclass
class ConfigPastas:
    """Configuração completa para modo 'pastas'."""
    predicao: ConfigPredicao = field(default_factory=ConfigPredicao)
    dataset: ConfigGold = field(default_factory=ConfigGold)
    entrada: ConfigEntrada = field(default_factory=ConfigEntrada)
    divisao: ConfigDivisao = field(default_factory=ConfigDivisao)
    validacao: ConfigValidacao = field(default_factory=ConfigValidacao)


@dataclass
class ConfigDataset:
    """Configuração para modo 'dataset' (arquivos parquet)."""
    train_prompt_col: str = "messages"
    eval_prompt_col: str = ""
    train_file: str = ""
    eval_file: str = ""
    test_file: str = ""
    _validar_caminhos: bool = field(default=True, repr=False)
    
    def __post_init__(self):
        if self._validar_caminhos:
            # Valida existência de arquivos se especificados
            for nome, caminho in [("train_file", self.train_file), 
                                   ("eval_file", self.eval_file), 
                                   ("test_file", self.test_file)]:
                if caminho and not os.path.isfile(caminho):
                    raise ValueError(f"Arquivo de dataset não encontrado ({nome}): '{caminho}'")





# ---------------------------------------------------------------------------
# Classe Principal: YamlTreinamento
# ---------------------------------------------------------------------------

class YamlTreinamento:
    """
    Carrega, valida e processa configurações YAML para treinamento de LLMs.
    
    Suporta dois modos de entrada:
    - 'pastas': Carrega dados de pastas com arquivos de texto/JSON
    - 'dataset': Carrega dados de arquivos parquet
    
    Exemplo de uso:
        >>> config = YamlTreinamento("config.yaml")
        >>> print(config.tipo_entrada)
        'pastas'
        >>> datasets = config.carregar_datasets()
    """
    
    def __init__(self, yaml_path: str, validar_caminhos: bool = True):
        """
        Inicializa a configuração a partir de um arquivo YAML.
        
        Args:
            yaml_path: Caminho para o arquivo YAML de configuração
            validar_caminhos: Se True, valida existência de pastas/arquivos
        """
        self.yaml_path = yaml_path
        self.validar_caminhos = validar_caminhos
        self._yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
        self._max_seq_auto = False  # Flag: max_seq_length foi calculado automaticamente
        
        # Carrega YAML bruto
        self._raw_config = self._carregar_yaml()
        
        # Processa e valida configurações
        self.formatos: ConfigFormatos = self._processar_formatos()
        self.misc: ConfigMisc = self._processar_misc()
        self.pastas: Optional[ConfigPastas] = None
        self.dataset: Optional[ConfigDataset] = None
        
        if self.tipo_entrada in TIPOS_BASEADOS_EM_PASTAS:
            if self.tipo_entrada == TIPO_ENTRADA_CURRICULUM:
                self.pastas = self._processar_pastas(secao="curriculum")
            else:
                self.pastas = self._processar_pastas()
        else:
            self.dataset = self._processar_dataset()
            
        self.modelo: ConfigModelo = self._processar_modelo()
        self.treinamento: ConfigTreinamento = self._processar_treinamento()
        self.lora: ConfigLora = self._processar_lora()
        

        
        # Gerenciador de datasets
        from treinar_unsloth_dataset import DatasetTreinamento
        self.dataset_manager = DatasetTreinamento(
            config_pastas=self.pastas,
            config_dataset=self.dataset,
            config_formatos=self.formatos,
            config_misc=self.misc
        )
        
        # Pipeline Universal: normaliza configuração em lista de etapas
        from treinar_unsloth_pipeline import construir_etapas
        self._curriculum: list = construir_etapas(self)
        
        # Para curriculum: usa o arquivo da primeira etapa como divisao padrão
        if self.tipo_entrada == TIPO_ENTRADA_CURRICULUM and self.pastas and self._curriculum:
            primeira = self._curriculum[0]
            if primeira.arquivo and not self.pastas.divisao.arquivo:
                self.pastas.divisao.arquivo = self._resolver_caminho(primeira.arquivo)
    
    @property
    def curriculum(self) -> list:
        """Retorna lista de etapas do curriculum (sempre >= 1 elemento)."""
        return self._curriculum
    
    @property
    def tipo_entrada(self) -> str:
        """Retorna o tipo de entrada configurado."""
        return self.formatos.tipo_entrada
    
    @property
    def formato_saida(self) -> str:
        """Retorna o formato de saída configurado."""
        return self.formatos.formato_saida
    
    @property
    def output_dir(self) -> str:
        """Retorna o diretório de saída do modelo."""
        return self.modelo.saida
    
    def base_model_name(self) -> str:
        """Retorna o nome do modelo base."""
        return self.modelo.base
    
    # ---------------------------------------------------------------------------
    # Métodos de carregamento do YAML
    # ---------------------------------------------------------------------------
    
    def _carregar_yaml(self) -> Dict[str, Any]:
        """Carrega o arquivo YAML e retorna como dicionário."""
        if not os.path.isfile(self.yaml_path):
            raise FileNotFoundError(f"Arquivo YAML não encontrado: '{self.yaml_path}'")
        
        with open(self.yaml_path, "r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp) or {}
        
        if not isinstance(config, dict):
            raise ValueError(f"Arquivo YAML deve conter um dicionário, recebido: {type(config)}")
        
        return config
    
    def _resolver_caminho(self, caminho: str) -> str:
        """
        Resolve um caminho relativo em relação ao diretório do YAML.
        Caminhos absolutos são mantidos como estão.
        """
        if not caminho:
            return caminho
        if os.path.isabs(caminho):
            return caminho
        return os.path.normpath(os.path.join(self._yaml_dir, caminho))
    
    # ---------------------------------------------------------------------------
    # Processamento de seções do YAML
    # ---------------------------------------------------------------------------
    
    def _processar_formatos(self) -> ConfigFormatos:
        """Processa a seção 'formatos' do YAML."""
        formatos_raw = self._raw_config.get("formatos", {})
        if not isinstance(formatos_raw, dict):
            formatos_raw = {}
        
        return ConfigFormatos(
            tipo_entrada=formatos_raw.get("tipo_entrada", TIPO_ENTRADA_DATASET),
            formato_saida=formatos_raw.get("formato_saida", FORMATO_SAIDA_TEXTO),
        )
    
    def _processar_misc(self) -> ConfigMisc:
        """Processa a seção 'misc' do YAML."""
        misc_raw = self._raw_config.get("misc", {})
        if not isinstance(misc_raw, dict):
            misc_raw = {}
        
        return ConfigMisc(
            log_level=misc_raw.get("log_level", "INFO"),
            env_chave_criptografia=misc_raw.get("env_chave_criptografia", "")
        )
    
    def _processar_pastas(self, secao: str = "pastas") -> ConfigPastas:
        """Processa a seção 'pastas' (ou 'curriculum') do YAML.
        
        Para tipo_entrada='curriculum', a seção 'curriculum' tem a mesma
        estrutura que 'pastas', mas 'divisao' é uma lista de etapas.
        
        Args:
            secao: Nome da seção no YAML ('pastas' ou 'curriculum')
        """
        pastas_raw = self._raw_config.get(secao, {})
        if not isinstance(pastas_raw, dict):
            raise ValueError(f"Seção '{secao}' deve ser um dicionário")
        
        # Processa subseções
        predicao_raw = pastas_raw.get("predicao", {})
        dataset_gold_raw = pastas_raw.get("dataset", {})
        entrada_raw = pastas_raw.get("entrada", {})
        # Para curriculum, divisao_raw é uma lista de etapas — extrair apenas campos de divisão
        divisao_raw = pastas_raw.get("divisao", {})
        if isinstance(divisao_raw, list):
            # Modo curriculum: divisao é a lista de etapas, não há arquivo único
            divisao_raw = {}
        validacao_raw = pastas_raw.get("validacao", {})
        
        # Resolve caminhos (sempre resolve, validação é controlada pelo _validar_caminhos)
        pasta_predicao = self._resolver_caminho(predicao_raw.get("pasta", ""))
        pasta_gold = self._resolver_caminho(dataset_gold_raw.get("pasta", ""))
        # Resolve caminhos de entrada (pasta ou dataframe)
        pasta_entrada = self._resolver_caminho(entrada_raw.get("pasta", ""))
        dataframe_path = self._resolver_caminho(entrada_raw.get("dataframe", ""))
        prompt_template = self._resolver_caminho(entrada_raw.get("prompt_template", ""))
        arquivo_divisao = divisao_raw.get("arquivo", "")
        if arquivo_divisao:
            arquivo_divisao = self._resolver_caminho(arquivo_divisao)
        
        # Cria configurações passando _validar_caminhos
        predicao = ConfigPredicao(
            pasta=pasta_predicao,
            mascara=predicao_raw.get("mascara", r"^(.+)\.txt$"),
            _validar_caminhos=self.validar_caminhos
        )
        
        # Gold dataset (obrigatório)
        if not pasta_gold:
            raise ValueError(f"Seção '{secao}.dataset.pasta' é obrigatória (pasta com o gold dataset)")
        gold = ConfigGold(
            pasta=pasta_gold,
            mascara=dataset_gold_raw.get("mascara", "*.txt"),
            _validar_caminhos=self.validar_caminhos
        )
        
        entrada = ConfigEntrada(
            pasta=pasta_entrada,
            mascara=entrada_raw.get("mascara", r"^(.+)\.txt$"),
            prompt_template=prompt_template,
            tag_texto=entrada_raw.get("tag_texto", ""),
            dataframe=dataframe_path,
            dataframe_col=entrada_raw.get("dataframe_col", ""),
            dataframe_id=entrada_raw.get("dataframe_id", ""),
            _validar_caminhos=self.validar_caminhos
        )
        
        # Processa proporções
        proporcao_raw = divisao_raw.get("proporcao", PROPORCAO_PADRAO.copy())
        proporcao = [0.0, 0.0, 0.0]

        if isinstance(proporcao_raw, list) and all(isinstance(x, (int, float)) for x in proporcao_raw):
            proporcao = proporcao_raw # Lista simples [0.8, 0.1, 0.1]
        
        elif isinstance(proporcao_raw, dict):
            # Formato dict: {treino: 0.8, validacao: 0.1, teste: 0.1}
            proporcao[0] = float(proporcao_raw.get("treino", 0))
            proporcao[1] = float(proporcao_raw.get("validacao", 0) or proporcao_raw.get("validação", 0) or proporcao_raw.get("avaliacao", 0))
            proporcao[2] = float(proporcao_raw.get("teste", 0))
            
        elif isinstance(proporcao_raw, list) and all(isinstance(x, dict) for x in proporcao_raw):
             # Formato lista de dicts: [- treino: 0.8, - validacao: 0.1]
             for item in proporcao_raw:
                 for k, v in item.items():
                     k_norm = k.lower().strip()
                     if k_norm in ["treino", "train"]:
                         proporcao[0] = float(v)
                     elif k_norm in ["validacao", "validação", "validation", "avaliacao", "avaliação"]:
                         proporcao[1] = float(v)
                     elif k_norm in ["teste", "test"]:
                         proporcao[2] = float(v)

        elif isinstance(proporcao_raw, str):
            proporcao = [float(x.strip()) for x in proporcao_raw.split(",")]
        
        divisao = ConfigDivisao(
            arquivo=arquivo_divisao,
            proporcao=proporcao,
            seed=divisao_raw.get("seed", SEED_PADRAO),
            validar_ids=divisao_raw.get("validar_ids", True)
        )
        
        validacao = ConfigValidacao(
            exigir_json_valido=validacao_raw.get("exigir_json_valido", True),
            skip_invalidos=validacao_raw.get("skip_invalidos", False)
        )
        
        return ConfigPastas(
            predicao=predicao,
            dataset=gold,
            entrada=entrada,
            divisao=divisao,
            validacao=validacao
        )
    
    def _processar_dataset(self) -> ConfigDataset:
        """Processa a seção 'dataset' do YAML."""
        dataset_raw = self._raw_config.get("dataset", {})
        if not isinstance(dataset_raw, dict):
            dataset_raw = {}
        
        # Resolve caminhos (sempre resolve, validação é controlada pelo _validar_caminhos)
        train_file = dataset_raw.get("train_file", "")
        eval_file = dataset_raw.get("eval_file", "")
        test_file = dataset_raw.get("test_file", "")
        
        if train_file:
            train_file = self._resolver_caminho(train_file)
        if eval_file:
            eval_file = self._resolver_caminho(eval_file)
        if test_file:
            test_file = self._resolver_caminho(test_file)
        
        return ConfigDataset(
            train_prompt_col=dataset_raw.get("train_prompt_col", "messages"),
            eval_prompt_col=dataset_raw.get("eval_prompt_col", ""),
            train_file=train_file,
            eval_file=eval_file,
            test_file=test_file,
            _validar_caminhos=self.validar_caminhos
        )
    
    def _processar_modelo(self) -> ConfigModelo:
        """Processa a seção 'modelo' do YAML."""
        modelo_raw = self._raw_config.get("modelo", {})
        if not isinstance(modelo_raw, dict):
            raise ValueError("Seção 'modelo' é obrigatória e deve ser um dicionário")
        
        saida = modelo_raw.get("saida", "")
        if saida:
            saida = self._resolver_caminho(saida)
        
        return ConfigModelo(
            base=modelo_raw.get("base", "") or modelo_raw.get("base_model_name", ""),
            saida=saida
        )

    def _processar_treinamento(self) -> ConfigTreinamento:
        """Processa a seção 'treinamento' do YAML."""
        treino_raw = self._raw_config.get("treinamento", {})
        if not isinstance(treino_raw, dict):
            raise ValueError("Seção 'treinamento' é obrigatória e deve ser um dicionário")

        # Processa nbits (pode ser None)
        nbits = treino_raw.get("nbits", 4)
        if nbits is None or nbits == 0:
            nbits = 0
            
        return ConfigTreinamento(
            eval_steps=treino_raw.get("eval_steps", "15%"),
            batch_size=int(treino_raw.get("batch_size", 2)),
            grad_batch_size=int(treino_raw.get("grad_batch_size", 5)),
            epochs=int(treino_raw.get("epochs") or treino_raw.get("num_train_epochs") or 1),
            max_seq_length=int(treino_raw.get("max_seq_length") or 0),
            learning_rate=float(treino_raw.get("learning_rate", 2e-4)),
            save_checkpoints=treino_raw.get("save_checkpoints", True) in {True, "true", "True", 1, "1", "sim"},
            resume_from_checkpoint=treino_raw.get("resume_from_checkpoint", True) in {True, "true", "True", 1, "1", "sim"},
            warmup_steps=int(treino_raw.get("warmup_steps", 5)),
            nbits=nbits,
            seed=int(treino_raw.get("seed", 3407)),
            train_on_responses_only=treino_raw.get("train_on_responses_only", True) in {True, "true", "True", 1, "1", "sim"},
            validar_max_seq_length=treino_raw.get("validar_max_seq_length", True) in {True, "true", "True", 1, "1", "sim"}
        )

    def _processar_lora(self) -> ConfigLora:
        """Processa a seção 'lora' do YAML."""
        lora_raw = self._raw_config.get("lora", {})
        if not isinstance(lora_raw, dict):
            lora_raw = {}  # LoRA é opcional, mas se não declarado fica vazio, não busca na raiz
        
        target_modules = lora_raw.get("target_modules", None)
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                             "gate_proj", "up_proj", "down_proj"]
                             
        return ConfigLora(
            r=int(lora_raw.get("r", 8)),
            alpha=int(lora_raw.get("alpha", 32)),
        )
    
    
    
    # ---------------------------------------------------------------------------
    # Métodos Públicos
    # ---------------------------------------------------------------------------

    @staticmethod
    def _arredondar_seq_length(valor: int, margem_minima: int = 256) -> int:
        """
        Arredonda um valor de tokens para o próximo múltiplo de 256
        garantindo margem mínima entre o valor original e o resultado.
        
        Args:
            valor: Número máximo de tokens encontrado
            margem_minima: Folga mínima entre valor e resultado (padrão: 256)
        
        Returns:
            Próximo múltiplo de 256 com margem garantida
        """
        from treinar_unsloth_pipeline import arredondar_seq_length
        return arredondar_seq_length(valor, margem_minima)

    def calcular_max_seq_length(self, cache=None, alias: str = "Principal") -> int:
        """
        Calcula max_seq_length automaticamente com base nos tokens das mensagens do dataset.
        Usa cache em _dados_automaticos.json para evitar recálculos.
        Usa o tokenizer do modelo base para contagem precisa.
        
        Args:
            cache: Instância de CacheSeqLength (se None, cria automaticamente)
            alias: Alias da etapa do curriculum para cache por etapa
            
        Returns:
            Valor de max_seq_length arredondado com margem de segurança
        """
        from treinar_unsloth_pipeline import CacheSeqLength, arredondar_seq_length
        
        # Inicializa cache se necessário
        if cache is None:
            cache = CacheSeqLength(self.modelo.saida, self.yaml_path)
        
        # Tenta usar cache válido
        valor_cache = cache.obter_max_seq_length(alias)
        if valor_cache is not None:
            print(f"   💾 max_seq_length={valor_cache} (cache válido em {CacheSeqLength.NOME_ARQUIVO})")
            return valor_cache
        
        print("🔄 max_seq_length=0: calculando automaticamente a partir do dataset...")
        
        # Coleta textos completos (user + assistant) de todos os alvos
        todos_textos = []
        
        if self.tipo_entrada in TIPOS_BASEADOS_EM_PASTAS:
            for alvo in ("treino", "validacao", "teste"):
                try:
                    mensagens = self.dataset_manager.carregar_mensagens_de_pastas(alvo=alvo)
                    for msg in mensagens:
                        if isinstance(msg, dict) and "messages" in msg:
                            texto_completo = " ".join(
                                m.get("content", "") for m in msg["messages"]
                            )
                            todos_textos.append(texto_completo)
                except Exception:
                    continue
        else:
            # Modo dataset: não é possível calcular sem carregar o arquivo
            print("   ⚠️ Cálculo automático não disponível para tipo_entrada='dataset'. Usando 4096.")
            return 4096
        
        if not todos_textos:
            print("   ⚠️ Nenhuma mensagem encontrada. Usando 4096.")
            return 4096
        
        # Tenta usar tokenizer para contagem precisa
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.modelo.base, use_fast=True)
            comprimentos = [len(tokenizer.encode(t)) for t in todos_textos]
            metodo = "tokenizer"
        except Exception:
            # Fallback: estimativa por caracteres (1 token ≈ 4 chars)
            comprimentos = [len(t) // 4 for t in todos_textos]
            metodo = "estimativa (chars/4)"
        
        max_tokens = max(comprimentos)
        media_tokens = sum(comprimentos) / len(comprimentos)
        
        # Arredonda com margem de segurança (mín 256 tokens de folga, múltiplo de 256)
        valor_calculado = arredondar_seq_length(max_tokens)
        
        print(f"   📊 {len(comprimentos)} registros analisados ({metodo})")
        print(f"   📊 Tokens: max={max_tokens}, média={media_tokens:.0f}")
        print(f"   ✅ max_seq_length calculado: {max_tokens} → {valor_calculado} (margem: {valor_calculado - max_tokens})")
        
        # Salva no cache
        try:
            cache.salvar(
                max_seq_length=valor_calculado,
                alias=alias,
                max_tokens_encontrado=max_tokens,
                media_tokens=round(media_tokens, 1),
                total_registros=len(comprimentos),
                metodo=metodo
            )
        except Exception as e:
            print(f"   ⚠️ Erro ao salvar cache: {e}")
        
        return valor_calculado

    def resolver_max_seq_length(self) -> None:
        """
        Resolve max_seq_length respeitando a flag validar_max_seq_length e o cache.
        
        Lógica:
        - Se validar_max_seq_length=False e max_seq_length>0: usa o valor do YAML diretamente (bypass)
        - Se max_seq_length==0: calcula automaticamente (com cache)
        - Se validar_max_seq_length=True e max_seq_length>0: recalcula para validar
        
        Deve ser chamado antes de carregar o modelo.
        """
        msl = self.treinamento.max_seq_length
        validar = self.treinamento.validar_max_seq_length
        
        if not validar and msl > 0:
            # Bypass: confia no valor do YAML sem recalcular
            print(f"⏭️  max_seq_length={msl} (validar_max_seq_length=false, bypass)")
            return
        
        if msl == 0:
            # Precisa calcular
            self.treinamento.max_seq_length = self.calcular_max_seq_length()
            self._max_seq_auto = True
        elif validar:
            # Tem valor mas precisa validar — apenas verifica sem sobrescrever
            # Calcula e avisa se o valor configurado é insuficiente
            from treinar_unsloth_pipeline import CacheSeqLength
            cache = CacheSeqLength(self.modelo.saida, self.yaml_path)
            valor_cache = cache.obter_max_seq_length()
            if valor_cache is not None:
                if msl < valor_cache:
                    print(f"⚠️  max_seq_length={msl} pode ser insuficiente (cache sugere {valor_cache})")
                else:
                    print(f"✅ max_seq_length={msl} validado (cache: {valor_cache})")
    
    
    def info(self) -> str:
        """Retorna string com resumo da configuração.
        
        O resumo é adaptado ao tipo de entrada (dataset, pastas ou curriculum),
        omitindo seções irrelevantes para o formato selecionado.
        """
        lines = [
            "=" * 60,
            "📋 CONFIGURAÇÃO YAML",
            "=" * 60,
            f"Arquivo: {self.yaml_path}",
            f"Tipo de entrada: {self.tipo_entrada}",
            f"Formato de saída: {self.formato_saida}",
            "",
            "🤖 MODELO:",
            f"  Base: {self.modelo.base}",
            f"  Saída: {self.modelo.saida}",
            "",
            "⚙️ TREINAMENTO:",
            f"  Batch size: {self.treinamento.batch_size}",
            f"  Grad batch size: {self.treinamento.grad_batch_size}",
            f"  Épocas: {self.treinamento.epochs}",
            f"  Max seq length: {self.treinamento.max_seq_length}{' (automático)' if self._max_seq_auto else ''}",
            f"  Validar max seq length: {self.treinamento.validar_max_seq_length}",
            f"  LoRA r: {self.lora.r}",
            f"  Learning rate: {self.treinamento.learning_rate}",
            f"  Train on responses only: {self.treinamento.train_on_responses_only}",
        ]
        
        # --- Seção de dados: varia conforme tipo de entrada ---
        
        if self.tipo_entrada in TIPOS_BASEADOS_EM_PASTAS:
            lines.extend([
                "",
                "📁 PASTAS:",
                f"  Entrada: {self.pastas.entrada.pasta}",
                f"  Gold Dataset: {self.pastas.dataset.pasta}",
                f"  Predição: {self.pastas.predicao.pasta or '(será criado)'}",
            ])
            # Divisão aparece aqui somente para modo 'pastas' (etapa única)
            if self.tipo_entrada != TIPO_ENTRADA_CURRICULUM:
                lines.extend([
                    f"  Divisão: {self.pastas.divisao.arquivo or '(será criado)'}",
                    f"  Validar IDs: {self.pastas.divisao.validar_ids}",
                    f"  Proporções (yaml): treino={self.pastas.divisao.proporcao[0]}, validacao={self.pastas.divisao.proporcao[1]}, teste={self.pastas.divisao.proporcao[2]}",
                ])
                if self.pastas.divisao.proporcao_reais:
                    pr = self.pastas.divisao.proporcao_reais
                    lines.append(f"  Proporções (efetivas): treino={pr[0]:.2f}, validacao={pr[1]:.2f}, teste={pr[2]:.2f}")
        else:
            lines.extend([
                "",
                "📊 DATASET:",
                f"  Treino: {self.dataset.train_file}",
                f"  Avaliação: {self.dataset.eval_file or '(não configurado)'}",
                f"  Teste: {self.dataset.test_file or '(não configurado)'}",
            ])
        
        # --- Seção pipeline/curriculum ---
        
        is_curriculum = self.tipo_entrada == TIPO_ENTRADA_CURRICULUM
        lines.extend(["", f"🔄 PIPELINE{' CURRICULUM' if is_curriculum else ''}:"])
        lines.append(f"  Etapas: {len(self._curriculum)}")
        
        for i, etapa in enumerate(self._curriculum):
            # Linha principal com alias e tipo
            partes = [f"alias='{etapa.alias}'", f"tipo={etapa.tipo}"]
            # Inclui overrides apenas quando diferem do valor global (>0)
            if etapa.max_seq_length > 0:
                partes.append(f"max_seq_length={etapa.max_seq_length}")
            if etapa.pace_epochs > 0:
                partes.append(f"epochs={etapa.pace_epochs}")
            if etapa.learning_rate > 0:
                partes.append(f"lr={etapa.learning_rate}")
            if etapa.pace_loss > 0:
                partes.append(f"pace_loss={etapa.pace_loss}")
            lines.append(f"  [{i}] {', '.join(partes)}")
            
            # Sub-detalhes da etapa (divisão e contagens)
            if etapa.arquivo:
                lines.append(f"      - divisão: {os.path.basename(etapa.arquivo)}")
                contagens = self._contar_instancias_divisao(etapa.arquivo)
                if contagens:
                    parts = [f"{k}={v}" for k, v in contagens.items()]
                    lines.append(f"      - {', '.join(parts)}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _contar_instancias_divisao(self, arquivo_divisao: str) -> dict:
        """Conta instâncias por alvo (treino/validacao/teste) em um arquivo de divisão CSV.
        
        Args:
            arquivo_divisao: Caminho para o CSV com colunas id_arquivo e alvo.
            
        Returns:
            Dict ordenado {treino: N, validacao: N, teste: N} ou {} se erro/inexistente.
        """
        import pandas as pd
        if not arquivo_divisao or not os.path.isfile(arquivo_divisao):
            return {}
        try:
            df = pd.read_csv(arquivo_divisao, sep=None, engine='python')
            col_alvo = "alvo"
            if col_alvo not in df.columns:
                # Tenta nomes antigos
                for alt in ("divisão", "divisao", "grupo"):
                    if alt in df.columns:
                        col_alvo = alt
                        break
                else:
                    return {}
            contagem = df[col_alvo].value_counts().to_dict()
            # Ordena na ordem padrão: treino, validacao, teste, outros
            ordem = ["treino", "validacao", "teste"]
            resultado = {}
            for k in ordem:
                if k in contagem:
                    resultado[k] = contagem[k]
            # Inclui chaves extras não previstas
            for k, v in contagem.items():
                if k not in resultado:
                    resultado[k] = v
            return resultado
        except Exception:
            return {}
    
    def __repr__(self) -> str:
        return f"YamlTreinamento(yaml_path='{self.yaml_path}', tipo_entrada='{self.tipo_entrada}')"


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dicas e Comentários para os Templates YAML
# ---------------------------------------------------------------------------
# (Código movido para treinar_unsloth_dicas.py)

from treinar_unsloth_dicas import injetar_dicas_yaml, DICAS_YAML




def criar_yaml_exemplo_pastas(caminho: str) -> None:
    """Cria um arquivo YAML de exemplo para modo 'pastas'."""
    template = {
        "formatos": {
            "tipo_entrada": "pastas",
            "formato_saida": "json"
        },
        "pastas": {
            "predicao": {
                "pasta": "./saidas/predicoes",
                "mascara": r"^(.+)\.txt$"
            },
            "dataset": {
                "pasta": "./saidas/gold",
                "mascara": "*.txt"
            },
            "entrada": {
                "pasta": "./saidas/textos",
                "mascara": r"^(.+)\.txt$",
                "prompt_template": "./prompts/template.txt",
                "tag_texto": "<<TEXTO>>"
            },
            "divisao": {
                "arquivo": "./saidas/divisao.csv",
                "proporcao": [
                    {"treino": 0.7},
                    {"validacao": 0.1},
                    {"teste": 0.2}
                ],
                "seed": 42
            },
            "validacao": {
                "exigir_json_valido": True,
                "skip_invalidos": False
            }
        },
        "modelo": {
            "base_model_name": "unsloth/Qwen2.5-1.5B-Instruct",
            "saida": "./modelos/meu_modelo"
        },
        "treinamento": {
            "eval_steps": "15%",
            "batch_size": 2,
            "grad_batch_size": 5,
            "num_train_epochs": 1,
            "max_seq_length": 4096,
            "learning_rate": 0.0002,
            "save_checkpoints": True,
            "resume_from_checkpoint": True,
            "warmup_steps": 5,
            "nbits": 4,
            "train_on_responses_only": True
        },
        "lora": {
            "r": 8,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        }
    }
    
    yaml_str = yaml.safe_dump(template, sort_keys=False, allow_unicode=True, default_flow_style=False)
    yaml_com_dicas = _injetar_dicas_yaml(yaml_str, DICAS_YAML)
    
    with open(caminho, "w", encoding="utf-8") as fp:
        fp.write(yaml_com_dicas)


def criar_yaml_exemplo_dataset(caminho: str) -> None:
    """Cria um arquivo YAML de exemplo para modo 'dataset'."""
    template = {
        "formatos": {
            "tipo_entrada": "dataset",
            "formato_saida": "texto"
        },
        "dataset": {
            "train_prompt_col": "messages",
            "eval_prompt_col": "",
            "train_file": "./dados/treino.parquet",
            "eval_file": "./dados/avaliacao.parquet",
            "test_file": ""
        },
        "modelo": {
            "base_model_name": "unsloth/Qwen2.5-1.5B-Instruct",
            "saida": "./modelos/meu_modelo"
        },
        "treinamento": {
            "eval_steps": "15%",
            "batch_size": 2,
            "grad_batch_size": 5,
            "num_train_epochs": 1,
            "max_seq_length": 4096,
            "learning_rate": 0.0002,
            "save_checkpoints": True,
            "resume_from_checkpoint": True,
            "warmup_steps": 5,
            "nbits": 4,
            "train_on_responses_only": True
        },
        "lora": {
            "r": 8,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        }
    }
    
    yaml_str = yaml.safe_dump(template, sort_keys=False, allow_unicode=True, default_flow_style=False)
    yaml_com_dicas = _injetar_dicas_yaml(yaml_str, DICAS_YAML)
    
    with open(caminho, "w", encoding="utf-8") as fp:
        fp.write(yaml_com_dicas)


# ---------------------------------------------------------------------------
# Validação Interativa
# ---------------------------------------------------------------------------

# Lista de modelos base comuns para sugestão
MODELOS_BASE_COMUNS = [
    "unsloth/Qwen2.5-0.5B-Instruct",
    "unsloth/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-3B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/gemma-3-4b-it",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
    "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    "unsloth/DeepSeek-R1-Distill-Qwen-7B",
]


class ValidadorInterativo:
    """
    Valida configurações YAML interativamente, perguntando ao usuário
    sobre ajustes quando detecta problemas ou configurações faltantes.
    """
    
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self._yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
        self._modificado = False
        
        # Carrega YAML bruto
        with open(yaml_path, "r", encoding="utf-8") as fp:
            self._config = yaml.safe_load(fp) or {}
    
    def _resolver_caminho(self, caminho: str) -> str:
        """Resolve caminho relativo ao diretório do YAML."""
        if not caminho or os.path.isabs(caminho):
            return caminho
        return os.path.normpath(os.path.join(self._yaml_dir, caminho))
    
    def _perguntar_sim_nao(self, pergunta: str, padrao: bool = True) -> bool:
        """Pergunta sim/não ao usuário."""
        opcao = "[S/n]" if padrao else "[s/N]"
        try:
            resposta = input(f"{pergunta} {opcao}: ").strip().lower()
            if not resposta:
                return padrao
            return resposta in ("s", "sim", "y", "yes")
        except (KeyboardInterrupt, EOFError):
            print()
            return False
    
    def _perguntar_opcao(self, pergunta: str, opcoes: List[str], padrao: int = 0) -> int:
        """Pergunta ao usuário para escolher uma opção."""
        print(f"\n{pergunta}")
        for i, opcao in enumerate(opcoes):
            marcador = "*" if i == padrao else " "
            print(f"  [{i+1}]{marcador} {opcao}")
        
        try:
            resposta = input(f"\nEscolha [1-{len(opcoes)}] (Enter = {padrao+1}): ").strip()
            if not resposta:
                return padrao
            idx = int(resposta) - 1
            if 0 <= idx < len(opcoes):
                return idx
            return padrao
        except (ValueError, KeyboardInterrupt, EOFError):
            print()
            return padrao
    
    def _perguntar_texto(self, pergunta: str, padrao: str = "") -> str:
        """Pergunta texto ao usuário."""
        try:
            sufixo = f" [{padrao}]" if padrao else ""
            resposta = input(f"{pergunta}{sufixo}: ").strip()
            return resposta if resposta else padrao
        except (KeyboardInterrupt, EOFError):
            print()
            return padrao
    
    def _salvar_config(self):
        """Salva configuração modificada de volta ao arquivo."""
        with open(self.yaml_path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(self._config, fp, sort_keys=False, allow_unicode=True, default_flow_style=False)
        print(f"💾 Configuração salva em: {self.yaml_path}")
    
    def validar_modelo_base(self) -> bool:
        """Valida se o modelo base está configurado."""
        modelo = self._config.get("modelo", {})
        base_model = modelo.get("base_model_name", "")
        
        if not base_model:
            print("\n⚠️  Modelo base não configurado!")
            idx = self._perguntar_opcao(
                "Escolha um modelo base:",
                MODELOS_BASE_COMUNS + ["Outro (digitar manualmente)"]
            )
            
            if idx < len(MODELOS_BASE_COMUNS):
                base_model = MODELOS_BASE_COMUNS[idx]
            else:
                base_model = self._perguntar_texto("Digite o nome do modelo")
            
            if base_model:
                if "modelo" not in self._config:
                    self._config["modelo"] = {}
                self._config["modelo"]["base_model_name"] = base_model
                self._modificado = True
                print(f"✅ Modelo configurado: {base_model}")
                return True
            return False
        
        return True
    
    def validar_pasta_saida(self) -> bool:
        """Valida se o diretório de saída existe ou pode ser criado."""
        modelo = self._config.get("modelo", {})
        saida = modelo.get("saida", "")
        
        if not saida:
            saida = self._perguntar_texto(
                "\n⚠️  Diretório de saída não configurado. Informe o caminho",
                "./modelos/meu_modelo"
            )
            if saida:
                if "modelo" not in self._config:
                    self._config["modelo"] = {}
                self._config["modelo"]["saida"] = saida
                self._modificado = True
        
        saida_abs = self._resolver_caminho(saida)
        
        if saida_abs and os.path.isdir(saida_abs):
            # Verifica se já existe modelo treinado
            adapter_config = os.path.join(saida_abs, "adapter_config.json")
            if os.path.isfile(adapter_config):
                print(f"\n⚠️  Diretório de saída já contém modelo treinado: {saida_abs}")
                opcao = self._perguntar_opcao(
                    "O que deseja fazer?",
                    [
                        "Continuar treinamento (resume_from_checkpoint)",
                        "Criar novo diretório com sufixo numérico",
                        "Sobrescrever (apagar modelo existente)",
                        "Manter configuração atual"
                    ],
                    padrao=0
                )
                
                if opcao == 0:
                    self._config.setdefault("treinamento", {})["resume_from_checkpoint"] = True
                    self._modificado = True
                    print("✅ Configurado para continuar treinamento existente")
                elif opcao == 1:
                    # Encontra próximo número disponível
                    i = 2
                    while os.path.isdir(f"{saida_abs}_{i}"):
                        i += 1
                    novo_saida = f"{saida}_{i}"
                    self._config["modelo"]["saida"] = novo_saida
                    self._modificado = True
                    print(f"✅ Novo diretório configurado: {novo_saida}")
                elif opcao == 2:
                    if self._perguntar_sim_nao("⚠️  ATENÇÃO: Isso apagará o modelo. Confirma?", False):
                        import shutil
                        shutil.rmtree(saida_abs)
                        print(f"🗑️  Diretório removido: {saida_abs}")
        
        elif saida_abs and not os.path.exists(saida_abs):
            if self._perguntar_sim_nao(f"\n📁 Criar diretório de saída? ({saida_abs})"):
                os.makedirs(saida_abs, exist_ok=True)
                print(f"✅ Diretório criado: {saida_abs}")
        
        return bool(saida)
    
    def validar_pastas_entrada(self) -> bool:
        """Valida pastas de entrada e gold dataset para modo pastas/curriculum."""
        formatos = self._config.get("formatos", {})
        tipo = formatos.get("tipo_entrada")
        if tipo not in ("pastas", "curriculum"):
            return True
        
        # Para curriculum, lê da seção 'curriculum'; para pastas, lê de 'pastas'
        secao = "curriculum" if tipo == "curriculum" else "pastas"
        pastas = self._config.get(secao, {})
        problemas = []
        
        # Verifica pasta do gold dataset (saídas esperadas para treino)
        gold_pasta = pastas.get("dataset", {}).get("pasta", "")
        gold_abs = self._resolver_caminho(gold_pasta) if gold_pasta else ""
        if gold_pasta and not os.path.isdir(gold_abs):
            problemas.append(("gold dataset", gold_pasta, gold_abs, ["pastas", "dataset", "pasta"]))
        
        # Verifica pasta de entrada
        ent_pasta = pastas.get("entrada", {}).get("pasta", "")
        ent_abs = self._resolver_caminho(ent_pasta) if ent_pasta else ""
        if ent_pasta and not os.path.isdir(ent_abs):
            problemas.append(("entrada", ent_pasta, ent_abs, ["pastas", "entrada", "pasta"]))
        
        # Verifica template de prompt
        template = pastas.get("entrada", {}).get("prompt_template", "")
        template_abs = self._resolver_caminho(template) if template else ""
        if template and not os.path.isfile(template_abs):
            print(f"\n⚠️  Template de prompt não encontrado: {template_abs}")
            opcao = self._perguntar_opcao(
                "O que deseja fazer?",
                [
                    "Remover configuração de template (usar entrada direta)",
                    "Informar novo caminho",
                    "Ignorar (corrigir manualmente depois)"
                ]
            )
            if opcao == 0:
                self._config.setdefault("pastas", {}).setdefault("entrada", {})["prompt_template"] = ""
                self._config["pastas"]["entrada"]["tag_texto"] = ""
                self._modificado = True
                print("✅ Template removido. Arquivos de entrada serão usados diretamente como prompt.")
            elif opcao == 1:
                novo = self._perguntar_texto("Informe o caminho do template")
                if novo:
                    self._config.setdefault("pastas", {}).setdefault("entrada", {})["prompt_template"] = novo
                    self._modificado = True
        
        for nome, rel, absoluto, chaves in problemas:
            print(f"\n⚠️  Pasta de {nome} não encontrada: {absoluto}")
            opcao = self._perguntar_opcao(
                "O que deseja fazer?",
                [
                    "Informar novo caminho",
                    "Criar pasta vazia",
                    "Ignorar (corrigir manualmente depois)"
                ]
            )
            
            if opcao == 0:
                novo = self._perguntar_texto(f"Informe o caminho da pasta de {nome}")
                if novo:
                    # Navega até a chave correta
                    obj = self._config
                    for chave in chaves[:-1]:
                        obj = obj.setdefault(chave, {})
                    obj[chaves[-1]] = novo
                    self._modificado = True
            elif opcao == 1:
                os.makedirs(absoluto, exist_ok=True)
                print(f"✅ Pasta criada: {absoluto}")
        
        return True
    
    def validar_divisao(self) -> bool:
        """Valida arquivo de divisão para modo pastas/curriculum."""
        formatos = self._config.get("formatos", {})
        tipo = formatos.get("tipo_entrada")
        if tipo not in ("pastas", "curriculum"):
            return True
        
        # Para curriculum, divisao é uma lista de etapas — validação será feita no pipeline
        if tipo == "curriculum":
            return True
        
        pastas = self._config.get("pastas", {})
        divisao = pastas.get("divisao", {})
        arquivo = divisao.get("arquivo", "")
        
        if not arquivo:
            if self._perguntar_sim_nao("\n📊 Deseja configurar arquivo de divisão treino/teste/avaliação?"):
                arquivo = self._perguntar_texto(
                    "Informe o caminho do arquivo CSV",
                    "./divisao_dataset.csv"
                )
                if arquivo:
                    self._config.setdefault("pastas", {}).setdefault("divisao", {})["arquivo"] = arquivo
                    self._modificado = True
        
        arquivo_abs = self._resolver_caminho(arquivo) if arquivo else ""
        if arquivo and not os.path.isfile(arquivo_abs):
            print(f"\n📊 Arquivo de divisão não existe: {arquivo_abs}")
            print("   Será criado automaticamente na primeira execução do treinamento.")
            
            proporcao = divisao.get("proporcao", [0.7, 0.15, 0.15])
            print(f"   Proporções atuais: treino={proporcao[0]:.0%}, teste={proporcao[1]:.0%}, avaliação={proporcao[2]:.0%}")
            
            if self._perguntar_sim_nao("Deseja ajustar as proporções?", False):
                try:
                    treino = float(self._perguntar_texto("Proporção treino (0-1)", str(proporcao[0])))
                    teste = float(self._perguntar_texto("Proporção teste (0-1)", str(proporcao[1])))
                    avaliacao = 1.0 - treino - teste
                    if avaliacao < 0:
                        print("⚠️  Proporções inválidas. Usando padrão.")
                    else:
                        nova_prop = [treino, teste, round(avaliacao, 2)]
                        self._config.setdefault("pastas", {}).setdefault("divisao", {})["proporcao"] = nova_prop
                        self._modificado = True
                        print(f"✅ Proporções: treino={treino:.0%}, teste={teste:.0%}, avaliação={avaliacao:.0%}")
                except ValueError:
                    print("⚠️  Valor inválido. Mantendo proporções atuais.")
        
        return True
    
    def validar_parametros_treinamento(self) -> bool:
        """Valida parâmetros críticos de treinamento."""
        treinamento = self._config.get("treinamento", {})
        
        # Verifica parâmetros potencialmente problemáticos
        max_seq = treinamento.get("max_seq_length", 4096)
        if max_seq > 8192:
            print(f"\n⚠️  max_seq_length={max_seq} é muito alto e pode causar problemas de memória.")
            if self._perguntar_sim_nao("Deseja reduzir para 8192?"):
                self._config.setdefault("treinamento", {})["max_seq_length"] = 8192
                self._modificado = True
        
        batch_size = treinamento.get("batch_size", 2)
        if batch_size > 8:
            print(f"\n⚠️  batch_size={batch_size} pode causar problemas de memória em GPUs menores.")
            if self._perguntar_sim_nao("Deseja reduzir para 2?"):
                self._config.setdefault("treinamento", {})["batch_size"] = 2
                self._modificado = True
        
        return True
    
    def executar(self) -> bool:
        """Executa validação interativa completa."""
        print("\n" + "=" * 60)
        print("🔍 VALIDAÇÃO INTERATIVA")
        print("=" * 60)
        
        # Executa todas as validações
        self.validar_modelo_base()
        self.validar_pasta_saida()
        self.validar_pastas_entrada()
        self.validar_divisao()
        self.validar_parametros_treinamento()
        
        # Salva se houve modificações
        if self._modificado:
            print("\n" + "-" * 60)
            if self._perguntar_sim_nao("💾 Salvar alterações no arquivo YAML?"):
                self._salvar_config()
                return True
            else:
                print("Alterações descartadas.")
        else:
            print("\n✅ Nenhuma alteração necessária.")
        
        return not self._modificado


# ---------------------------------------------------------------------------
# CLI para testes
# ---------------------------------------------------------------------------

def _perguntar_criar_exemplo(yaml_path: str) -> bool:
    """Pergunta ao usuário se deseja criar um arquivo YAML de exemplo."""
    print(f"\n⚠️  Arquivo não encontrado: {yaml_path}")
    print("\nDeseja criar um arquivo YAML de exemplo?")
    print("  [1] Modo 'pastas' (arquivos de texto/JSON em diretórios)")
    print("  [2] Modo 'dataset' (arquivos parquet)")
    print("  [N] Não criar (sair)")
    
    try:
        resposta = input("\nEscolha [1/2/N]: ").strip().lower()
        
        if resposta == "1":
            criar_yaml_exemplo_pastas(yaml_path)
            print(f"\n✅ Arquivo de exemplo criado: {yaml_path}")
            print("   Edite o arquivo com suas configurações e execute novamente.")
            return True
        elif resposta == "2":
            criar_yaml_exemplo_dataset(yaml_path)
            print(f"\n✅ Arquivo de exemplo criado: {yaml_path}")
            print("   Edite o arquivo com suas configurações e execute novamente.")
            return True
        else:
            print("Operação cancelada.")
            return False
    except (KeyboardInterrupt, EOFError):
        print("\nOperação cancelada.")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Utilitários para treinar_unsloth.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  %(prog)s config.yaml                    Valida o arquivo YAML
  %(prog)s config.yaml --interativo       Valida e corrige interativamente
  %(prog)s config.yaml --listar-arquivos  Lista arquivos pareados (modo pastas)
  %(prog)s config.yaml --criar-divisao    Cria arquivo de divisão (modo pastas)
  %(prog)s --criar-exemplo-pastas novo.yaml
        """
    )
    parser.add_argument("yaml", nargs="?", help="Arquivo YAML para validar")
    parser.add_argument("--interativo", "-i", action="store_true", 
                        help="Modo interativo: valida e pergunta sobre ajustes")
    parser.add_argument("--criar-exemplo-pastas", type=str, metavar="ARQUIVO",
                        help="Cria YAML de exemplo para modo pastas")
    parser.add_argument("--criar-exemplo-dataset", type=str, metavar="ARQUIVO",
                        help="Cria YAML de exemplo para modo dataset")
    parser.add_argument("--listar-arquivos", action="store_true", 
                        help="Lista arquivos pareados (modo pastas)")
    parser.add_argument("--criar-divisao", action="store_true", 
                        help="Cria arquivo de divisão (modo pastas)")
    
    args = parser.parse_args()
    
    if args.criar_exemplo_pastas:
        criar_yaml_exemplo_pastas(args.criar_exemplo_pastas)
        print(f"✅ Exemplo criado: {args.criar_exemplo_pastas}")
        sys.exit(0)
    
    if args.criar_exemplo_dataset:
        criar_yaml_exemplo_dataset(args.criar_exemplo_dataset)
        print(f"✅ Exemplo criado: {args.criar_exemplo_dataset}")
        sys.exit(0)
    
    if not args.yaml:
        parser.print_help()
        sys.exit(1)
    
    # Verifica se o arquivo existe antes de tentar carregar
    if not os.path.isfile(args.yaml):
        if _perguntar_criar_exemplo(args.yaml):
            # Se criou o exemplo, pergunta se quer executar modo interativo
            try:
                resposta = input("\nDeseja configurar o arquivo agora? [S/n]: ").strip().lower()
                if resposta in ("", "s", "sim", "y", "yes"):
                    args.interativo = True
                else:
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                sys.exit(0)
        else:
            sys.exit(1)
    
    # Modo interativo
    if args.interativo:
        try:
            validador = ValidadorInterativo(args.yaml)
            validador.executar()
        except Exception as e:
            print(f"❌ Erro: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)
    
    # Modo padrão: apenas validação
    try:
        config = YamlTreinamento(args.yaml, validar_caminhos=True)
        print(config.info())
        
        if args.listar_arquivos and config.tipo_entrada == TIPO_ENTRADA_PASTAS:
            print("\n📁 ARQUIVOS PAREADOS:")
            pares = config.parear_arquivos()
            for par in pares[:10]:  # Mostra apenas os 10 primeiros
                print(f"  - {par['id']}")
            if len(pares) > 10:
                print(f"  ... e mais {len(pares) - 10} arquivos")
        
        if args.criar_divisao and config.tipo_entrada == TIPO_ENTRADA_PASTAS:
            config.carregar_ou_criar_divisao()
        
        print("\n✅ Configuração válida!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        # Se houver erro de validação, sugere modo interativo
        print("\n💡 Dica: Use --interativo para corrigir problemas de configuração")
        print(f"   python {sys.argv[0]} {args.yaml} --interativo")
        sys.exit(1)
