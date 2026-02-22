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

# Configuração de path para permitir execução de qualquer diretório
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

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
TIPOS_ENTRADA_VALIDOS = {TIPO_ENTRADA_PASTAS, TIPO_ENTRADA_DATASET}

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
    
    def __post_init__(self):
        # Validações de valores
        if self.batch_size <= 0:
            raise ValueError(f"batch_size deve ser > 0, recebido: {self.batch_size}")
        if self.grad_batch_size <= 0:
            raise ValueError(f"grad_batch_size deve ser > 0, recebido: {self.grad_batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs deve ser > 0, recebido: {self.epochs}")
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length deve ser > 0, recebido: {self.max_seq_length}")
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
    """Configuração da pasta de predição (saídas esperadas)."""
    pasta: str = ""
    mascara: str = "*.txt"  # Padrão glob para filtrar arquivos
    _validar_caminhos: bool = field(default=True, repr=False)
    
    def __post_init__(self):
        if self._validar_caminhos and self.pasta and not os.path.isdir(self.pasta):
            raise ValueError(f"Pasta de predição não encontrada: '{self.pasta}'")


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
        
        # Carrega YAML bruto
        self._raw_config = self._carregar_yaml()
        
        # Processa e valida configurações
        self.formatos: ConfigFormatos = self._processar_formatos()
        self.misc: ConfigMisc = self._processar_misc()
        self.pastas: Optional[ConfigPastas] = None
        self.dataset: Optional[ConfigDataset] = None
        
        if self.tipo_entrada == TIPO_ENTRADA_PASTAS:
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
    
    def _processar_pastas(self) -> ConfigPastas:
        """Processa a seção 'pastas' do YAML."""
        pastas_raw = self._raw_config.get("pastas", {})
        if not isinstance(pastas_raw, dict):
            raise ValueError("Seção 'pastas' deve ser um dicionário")
        
        # Processa subseções
        predicao_raw = pastas_raw.get("predicao", {})
        entrada_raw = pastas_raw.get("entrada", {})
        divisao_raw = pastas_raw.get("divisao", {})
        validacao_raw = pastas_raw.get("validacao", {})
        
        # Resolve caminhos (sempre resolve, validação é controlada pelo _validar_caminhos)
        pasta_predicao = self._resolver_caminho(predicao_raw.get("pasta", ""))
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
            max_seq_length=int(treino_raw.get("max_seq_length", 4096)),
            learning_rate=float(treino_raw.get("learning_rate", 2e-4)),
            save_checkpoints=treino_raw.get("save_checkpoints", True) in {True, "true", "True", 1, "1", "sim"},
            resume_from_checkpoint=treino_raw.get("resume_from_checkpoint", True) in {True, "true", "True", 1, "1", "sim"},
            warmup_steps=int(treino_raw.get("warmup_steps", 5)),
            nbits=nbits,
            seed=int(treino_raw.get("seed", 3407)),
            train_on_responses_only=treino_raw.get("train_on_responses_only", True) in {True, "true", "True", 1, "1", "sim"}
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

    
    
    def info(self) -> str:
        """Retorna string com resumo da configuração."""
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
            f"  Max seq length: {self.treinamento.max_seq_length}",
            f"  LoRA r: {self.lora.r}",
            f"  Learning rate: {self.treinamento.learning_rate}",
            f"  Train on responses only: {self.treinamento.train_on_responses_only}",
        ]
        
        if self.tipo_entrada == TIPO_ENTRADA_PASTAS:
            lines.extend([
                "",
                "📁 PASTAS:",
                f"  Entrada: {self.pastas.entrada.pasta}",
                f"  Predição: {self.pastas.predicao.pasta}",
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
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
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
        """Valida pastas de entrada/predição para modo pastas."""
        formatos = self._config.get("formatos", {})
        if formatos.get("tipo_entrada") != "pastas":
            return True
        
        pastas = self._config.get("pastas", {})
        problemas = []
        
        # Verifica pasta de predição
        pred_pasta = pastas.get("predicao", {}).get("pasta", "")
        pred_abs = self._resolver_caminho(pred_pasta) if pred_pasta else ""
        if pred_pasta and not os.path.isdir(pred_abs):
            problemas.append(("predição", pred_pasta, pred_abs, ["pastas", "predicao", "pasta"]))
        
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
        """Valida arquivo de divisão para modo pastas."""
        formatos = self._config.get("formatos", {})
        if formatos.get("tipo_entrada") != "pastas":
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
