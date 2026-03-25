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

FORMATO_SAIDA_JSON = "json"
FORMATO_SAIDA_TEXTO = "texto"

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
    ollama: str = ""  # Nome do modelo no Ollama (opcional)
    ollama_base: str = ""  # Nome do modelo base no Ollama (opcional, para usar_base=True)
    ollama_url: str = ""  # URL customizada da API Ollama (opcional)

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
            raise ValueError(
                f"max_seq_length é obrigatório e deve ser > 0, recebido: {self.max_seq_length}. "
                f"Consulte os dados de tokens no CSV de divisão (coluna token_total) "
                f"para definir um valor adequado."
            )
        if self.nbits not in {0, 4, 8, None}:
            raise ValueError(f"nbits deve ser 0, 4, 8 ou None, recebido: {self.nbits}")


@dataclass
class ConfigBatchSize:
    """Configuração de cálculo automático de batch size.
    
    Quando efetivo > 0, o sistema calcula automaticamente grad_batch_size
    para atingir o batch efetivo desejado, independente do número de GPUs.
    
    Fórmula: grad_batch_size = round(efetivo / (batch_size × n_gpus))
    
    Exemplo (YAML):
        curriculum:
          batch_size:
            efetivo: 16   # Batch efetivo desejado
            batch_size: 2 # Batch por GPU (fixo, determinado empiricamente)
    
    Com 2 GPUs: grad_batch_size = round(16 / (2 × 2)) = 4 → efetivo real = 16
    Com 3 GPUs: grad_batch_size = round(16 / (2 × 3)) = 3 → efetivo real = 18
    """
    efetivo: int = 0       # Batch efetivo desejado (0 = desativado, usa batch_size/grad_batch_size manuais)
    batch_size: int = 2    # Batch por GPU (determinado empiricamente para evitar OOM)
    
    def __post_init__(self):
        if self.efetivo < 0:
            raise ValueError(f"batch_size.efetivo deve ser >= 0, recebido: {self.efetivo}")
        if self.efetivo > 0 and self.batch_size <= 0:
            raise ValueError(f"batch_size.batch_size deve ser > 0 quando efetivo > 0, recebido: {self.batch_size}")







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
class ConfigSaida:
    """Configuração da fonte de saídas esperadas (gold dataset para treino).
    
    Suporta dois modos:
    - Pasta de arquivos: pasta + mascara (ID = nome sem extensão)
    - Dataframe: dataframe + dataframe_col + dataframe_id
    
    formato: 'json' ou 'texto' (qualquer valor != 'json' assume texto)
    texto_criptografado: Se True, saídas serão descriptografadas
    """
    pasta: str = ""
    mascara: str = "*.txt"  # Padrão glob para filtrar arquivos (obrigatório se pasta)
    dataframe: str = ""  # Caminho para arquivo parquet com saídas
    dataframe_col: str = "saida"  # Coluna do dataframe com a saída (padrão: saida)
    dataframe_id: str = "id_peca"  # Coluna do dataframe com o ID (padrão: id_peca)
    formato: str = FORMATO_SAIDA_TEXTO  # 'json' ou 'texto'
    texto_criptografado: bool = False  # Se True, saídas serão descriptografadas
    _validar_caminhos: bool = field(default=True, repr=False)
    
    def __post_init__(self):
        tem_pasta = bool(self.pasta)
        tem_df = bool(self.dataframe)
        
        if not tem_pasta and not tem_df:
            raise ValueError("Seção 'saida' deve ter 'pasta' ou 'dataframe' configurado")
        
        if self._validar_caminhos:
            if tem_pasta and not os.path.isdir(self.pasta):
                raise ValueError(f"Pasta do gold dataset não encontrada: '{self.pasta}'")
            if tem_df and not os.path.isfile(self.dataframe):
                raise ValueError(f"Arquivo de dataframe de saída não encontrado: '{self.dataframe}'")
        
        # Normaliza formato: qualquer valor != 'json' vira 'texto'
        if self.formato != FORMATO_SAIDA_JSON:
            self.formato = FORMATO_SAIDA_TEXTO


@dataclass
class ConfigEntrada:
    """Configuração da fonte de entrada (textos para prompt).
    
    Suporta dois modos:
    - Pasta de arquivos: pasta + mascara
    - Dataframe: dataframe + dataframe_col + dataframe_id
    """
    pasta: str = ""  # Pasta com arquivos de texto
    mascara: str = "*.txt"  # Padrão glob para filtrar arquivos
    prompt_template: str = ""  # Arquivo com template do prompt
    tag_texto: str = ""  # Tag a ser substituída pelo conteúdo
    dataframe: str = ""  # Caminho para arquivo parquet com textos
    dataframe_col: str = "texto"  # Coluna do dataframe com o texto (padrão: texto)
    dataframe_id: str = "id_peca"  # Coluna do dataframe com o ID (padrão: id_peca)
    texto_criptografado: bool = False  # Se True, texto de entrada será descriptografado
    prompt_criptografado: bool = False  # Se True, template de prompt será descriptografado
    _validar_caminhos: bool = field(default=True, repr=False)
    
    def __post_init__(self):
        # Valida que tem pasta ou dataframe
        tem_pasta = bool(self.pasta)
        tem_df = bool(self.dataframe)
        
        if self._validar_caminhos:
            if tem_pasta and not os.path.isdir(self.pasta):
                raise ValueError(f"Pasta de entrada não encontrada: '{self.pasta}'")
            if tem_df and not os.path.isfile(self.dataframe):
                raise ValueError(f"Arquivo de dataframe não encontrado: '{self.dataframe}'")
            if self.prompt_template and not os.path.isfile(self.prompt_template):
                raise ValueError(f"Arquivo de template não encontrado: '{self.prompt_template}'")
        
        # Se tem template, precisa ter tag (validação estrutural)
        if self.prompt_template and not self.tag_texto:
            raise ValueError("Se 'prompt_template' for informado, 'tag_texto' é obrigatório")


@dataclass
class ConfigDivisao:
    """Configuração de divisão treino/teste/avaliação."""
    arquivo: str = ""
    proporcao: List[float] = field(default_factory=lambda: PROPORCAO_PADRAO.copy())
    seed: int = SEED_PADRAO
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
    exigir_ids_pareados: bool = True


@dataclass
class ConfigCurriculum:
    """Configuração completa do curriculum (única forma de entrada de dados)."""
    predicao: ConfigPredicao = field(default_factory=ConfigPredicao)
    saida: ConfigSaida = field(default_factory=ConfigSaida)
    entrada: ConfigEntrada = field(default_factory=ConfigEntrada)
    divisao: ConfigDivisao = field(default_factory=ConfigDivisao)
    validacao: ConfigValidacao = field(default_factory=ConfigValidacao)








# ---------------------------------------------------------------------------
# Classe Principal: YamlTreinamento
# ---------------------------------------------------------------------------

class YamlTreinamento:
    """
    Carrega, valida e processa configurações YAML para treinamento de LLMs.
    
    Formato único de entrada: curriculum (seção 'curriculum' no YAML).
    Um treinamento de etapa única é configurado como um curriculum com uma divisão.
    
    Exemplo de uso:
        >>> config = YamlTreinamento("config.yaml")
        >>> print(config.formato_saida)
        'json'
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
        self.misc: ConfigMisc = self._processar_misc()
        self.curriculum_config: ConfigCurriculum = self._processar_curriculum()
        self.modelo: ConfigModelo = self._processar_modelo()
        self.treinamento: ConfigTreinamento = self._processar_treinamento()
        self.lora: ConfigLora = self._processar_lora()
        
        # Validação de criptografia: se alguma flag *_criptografado é True,
        # a chave de criptografia deve estar configurada e acessível
        self._validar_criptografia()
        
        # Batch automático (calcula grad_batch_size com base no nº de GPUs)
        self.batch_size_auto: Optional[ConfigBatchSize] = self._processar_batch_size_auto()
        self._aplicar_batch_size_auto()
        
        # Gerenciador de datasets
        from treinar_unsloth_dataset import DatasetTreinamento
        self.dataset_manager = DatasetTreinamento(
            config_curriculum=self.curriculum_config,
            config_misc=self.misc
        )
        
        # Pipeline Universal: normaliza configuração em lista de etapas
        from treinar_unsloth_pipeline import construir_etapas
        self._curriculum: list = construir_etapas(self)
        
        # Usa o arquivo da primeira etapa treinável como divisao padrão e aplica
        # overrides (max_seq_length, epochs, learning_rate) para que o código
        # downstream reflita os valores efetivos.
        primeira = next((e for e in self._curriculum if e.is_treinavel), None)
        if primeira:
            if primeira.arquivo and not self.curriculum_config.divisao.arquivo:
                self.curriculum_config.divisao.arquivo = self._resolver_caminho(primeira.arquivo)
            if primeira.pace_epochs > 0:
                self.treinamento.epochs = primeira.pace_epochs
            if primeira.learning_rate > 0:
                self.treinamento.learning_rate = primeira.learning_rate
            if primeira.max_seq_length > 0:
                self.treinamento.max_seq_length = primeira.max_seq_length
    
    @property
    def curriculum(self) -> list:
        """Retorna lista de TODAS as etapas do curriculum (inclui somente-predict)."""
        return self._curriculum

    @property
    def curriculum_treino(self) -> list:
        """Retorna apenas etapas treináveis (tipo não-vazio). Usada pelo loop de treino."""
        return [e for e in self._curriculum if e.is_treinavel]
    
    @property
    def formato_saida(self) -> str:
        """Retorna o formato de saída configurado ('json' ou 'texto')."""
        return self.curriculum_config.saida.formato
    
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
        """Resolve um caminho relativo em relação ao diretório do YAML."""
        if not caminho:
            return caminho
        if os.path.isabs(caminho):
            return caminho
        return os.path.normpath(os.path.join(self._yaml_dir, caminho))
    
    # ---------------------------------------------------------------------------
    # Processamento de seções do YAML
    # ---------------------------------------------------------------------------
    
    def _processar_misc(self) -> ConfigMisc:
        """Processa a seção 'misc' do YAML."""
        misc_raw = self._raw_config.get("misc", {})
        if not isinstance(misc_raw, dict):
            misc_raw = {}
        
        return ConfigMisc(
            log_level=misc_raw.get("log_level", "INFO"),
            env_chave_criptografia=misc_raw.get("env_chave_criptografia", "")
        )
    
    def _processar_curriculum(self) -> ConfigCurriculum:
        """Processa a seção 'curriculum' do YAML.
        
        A seção curriculum contém: predicao, saida, entrada, validacao, divisao.
        'divisao' é uma lista de etapas do pipeline.
        """
        curriculum_raw = self._raw_config.get("curriculum", {})
        if not isinstance(curriculum_raw, dict):
            raise ValueError("Seção 'curriculum' é obrigatória e deve ser um dicionário")
        
        # Processa subseções
        predicao_raw = curriculum_raw.get("predicao", {}) or {}
        saida_raw = curriculum_raw.get("saida", {}) or {}
        entrada_raw = curriculum_raw.get("entrada", {}) or {}
        validacao_raw = curriculum_raw.get("validacao", {}) or {}
        
        # divisao_raw é uma lista de etapas — não usada aqui (processada pelo pipeline)
        # Mas pode ter campos de divisão se for etapa única
        divisao_raw = curriculum_raw.get("divisao", {})
        if isinstance(divisao_raw, list):
            divisao_raw = {}  # Lista de etapas: sem arquivo único de divisão
        
        # --- Resolve caminhos ---
        pasta_predicao = self._resolver_caminho(predicao_raw.get("pasta", ""))
        
        # Saída (gold dataset): pasta ou dataframe
        pasta_saida = self._resolver_caminho(saida_raw.get("pasta", ""))
        dataframe_saida = self._resolver_caminho(saida_raw.get("dataframe", ""))
        
        # Entrada: pasta ou dataframe
        pasta_entrada = self._resolver_caminho(entrada_raw.get("pasta", ""))
        dataframe_entrada = self._resolver_caminho(entrada_raw.get("dataframe", ""))
        prompt_template = self._resolver_caminho(entrada_raw.get("prompt_template", ""))
        
        # Divisão
        arquivo_divisao = divisao_raw.get("arquivo", "")
        if arquivo_divisao:
            arquivo_divisao = self._resolver_caminho(arquivo_divisao)
        
        # --- Cria configurações ---
        predicao = ConfigPredicao(
            pasta=pasta_predicao,
            mascara=predicao_raw.get("mascara", r"^(.+)\.txt$"),
            _validar_caminhos=self.validar_caminhos
        )
        
        saida = ConfigSaida(
            pasta=pasta_saida,
            mascara=saida_raw.get("mascara", "*.txt"),
            dataframe=dataframe_saida,
            dataframe_col=saida_raw.get("dataframe_col", "saida"),
            dataframe_id=saida_raw.get("dataframe_id", "id_peca"),
            formato=saida_raw.get("formato", FORMATO_SAIDA_TEXTO),
            texto_criptografado=saida_raw.get("texto_criptografado", False) in {True, "true", "True", 1, "1", "sim"},
            _validar_caminhos=self.validar_caminhos
        )
        
        entrada = ConfigEntrada(
            pasta=pasta_entrada,
            mascara=entrada_raw.get("mascara", "*.txt"),
            prompt_template=prompt_template,
            tag_texto=entrada_raw.get("tag_texto", ""),
            dataframe=dataframe_entrada,
            dataframe_col=entrada_raw.get("dataframe_col", "texto"),
            dataframe_id=entrada_raw.get("dataframe_id", "id_peca"),
            texto_criptografado=entrada_raw.get("texto_criptografado", False) in {True, "true", "True", 1, "1", "sim"},
            prompt_criptografado=entrada_raw.get("prompt_criptografado", False) in {True, "true", "True", 1, "1", "sim"},
            _validar_caminhos=self.validar_caminhos
        )
        
        # Processa proporções
        proporcao_raw = divisao_raw.get("proporcao", PROPORCAO_PADRAO.copy())
        proporcao = self._processar_proporcao(proporcao_raw)
        
        divisao = ConfigDivisao(
            arquivo=arquivo_divisao,
            proporcao=proporcao,
            seed=divisao_raw.get("seed", SEED_PADRAO)
        )
        
        validacao = ConfigValidacao(
            exigir_json_valido=validacao_raw.get("exigir_json_valido", True),
            exigir_ids_pareados=validacao_raw.get("exigir_ids_pareados", True)
        )
        
        return ConfigCurriculum(
            predicao=predicao,
            saida=saida,
            entrada=entrada,
            divisao=divisao,
            validacao=validacao
        )
    
    def _processar_proporcao(self, proporcao_raw) -> List[float]:
        """Processa proporções de divisão em diversos formatos YAML."""
        proporcao = [0.0, 0.0, 0.0]
        
        if isinstance(proporcao_raw, list) and all(isinstance(x, (int, float)) for x in proporcao_raw):
            proporcao = proporcao_raw
        elif isinstance(proporcao_raw, dict):
            proporcao[0] = float(proporcao_raw.get("treino", 0))
            proporcao[1] = float(proporcao_raw.get("validacao", 0) or proporcao_raw.get("validação", 0) or proporcao_raw.get("avaliacao", 0))
            proporcao[2] = float(proporcao_raw.get("teste", 0))
        elif isinstance(proporcao_raw, list) and all(isinstance(x, dict) for x in proporcao_raw):
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
        
        return proporcao
    
    def _validar_criptografia(self) -> None:
        """Valida que a chave de criptografia existe se alguma flag *_criptografado é True.
        
        Verifica as flags:
        - entrada.texto_criptografado
        - entrada.prompt_criptografado
        - saida.texto_criptografado
        
        Se qualquer uma for True, misc.env_chave_criptografia deve apontar para
        uma variável de ambiente existente e não vazia.
        """
        entrada = self.curriculum_config.entrada
        saida = self.curriculum_config.saida
        
        precisa_criptografia = (
            entrada.texto_criptografado or
            entrada.prompt_criptografado or
            saida.texto_criptografado
        )
        
        if not precisa_criptografia:
            return
        
        nome_var = self.misc.env_chave_criptografia
        if not nome_var:
            flags_ativas = []
            if entrada.texto_criptografado:
                flags_ativas.append("entrada.texto_criptografado")
            if entrada.prompt_criptografado:
                flags_ativas.append("entrada.prompt_criptografado")
            if saida.texto_criptografado:
                flags_ativas.append("saida.texto_criptografado")
            raise ValueError(
                f"❌ Flags de criptografia ativas ({', '.join(flags_ativas)}) "
                f"mas 'misc.env_chave_criptografia' não está configurada no YAML."
            )
        
        valor = os.getenv(nome_var)
        if not valor or not valor.strip():
            raise EnvironmentError(
                f"\n{'='*70}\n"
                f"❌ ERRO CRÍTICO: Variável de ambiente '{nome_var}' "
                f"não está definida ou está vazia.\n\n"
                f"O YAML configurou 'misc.env_chave_criptografia: {nome_var}',\n"
                f"e dados criptografados foram marcados para descriptografia.\n\n"
                f"Sem a chave, o treinamento usará texto criptografado (ilegível).\n\n"
                f"Para corrigir, defina a variável antes de executar:\n"
                f"  export {nome_var}=\"sua_chave_fernet_aqui\"\n"
                f"{'='*70}"
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
            saida=saida,
            ollama=modelo_raw.get("ollama", ""),
            ollama_base=modelo_raw.get("ollama_base", ""),
            ollama_url=modelo_raw.get("ollama_url", "")
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
            
        # batch_size pode ser int (manual) ou dict (auto: {efetivo, batch_size})
        batch_size_raw = treino_raw.get("batch_size", 2)
        if isinstance(batch_size_raw, dict):
            # Modo automático — valores provisórios, sobrescritos por _aplicar_batch_size_auto()
            batch_size_val = int(batch_size_raw.get("batch_size", 2))
            grad_batch_size_val = int(treino_raw.get("grad_batch_size", 1))
        else:
            batch_size_val = int(batch_size_raw)
            grad_batch_size_val = int(treino_raw.get("grad_batch_size", 5))

        # max_seq_length é obrigatório — o pesquisador deve definir com base
        # nos dados de tokens do CSV de divisão (coluna token_total)
        msl_raw = treino_raw.get("max_seq_length")
        if msl_raw is None or int(msl_raw) <= 0:
            raise ValueError(
                "❌ 'treinamento.max_seq_length' é obrigatório e deve ser > 0.\n"
                "   Consulte a coluna 'token_total' no CSV de divisão para definir um valor adequado."
            )

        return ConfigTreinamento(
            eval_steps=treino_raw.get("eval_steps", "15%"),
            batch_size=batch_size_val,
            grad_batch_size=grad_batch_size_val,
            epochs=int(treino_raw.get("epochs") or treino_raw.get("num_train_epochs") or 1),
            max_seq_length=int(msl_raw),
            learning_rate=float(treino_raw.get("learning_rate", 2e-4)),
            save_checkpoints=treino_raw.get("save_checkpoints", True) in {True, "true", "True", 1, "1", "sim"},
            resume_from_checkpoint=treino_raw.get("resume_from_checkpoint", True) in {True, "true", "True", 1, "1", "sim"},
            warmup_steps=int(treino_raw.get("warmup_steps", 5)),
            nbits=nbits,
            seed=int(treino_raw.get("seed", 3407)),
            train_on_responses_only=treino_raw.get("train_on_responses_only", True) in {True, "true", "True", 1, "1", "sim"},
            weight_decay=float(treino_raw.get("weight_decay", 0.01)),
            optim=str(treino_raw.get("optim", "adamw_8bit")),
            lr_scheduler_type=str(treino_raw.get("lr_scheduler_type", "linear")),
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
            dropout=float(lora_raw.get("dropout", 0.05)),
            target_modules=target_modules,
        )
    
    def _processar_batch_size_auto(self) -> Optional[ConfigBatchSize]:
        """Processa configuração de batch automático do YAML (treinamento.batch_size como dict).
        
        Quando treinamento.batch_size é um dict com {efetivo, batch_size}, ativa o
        cálculo automático de grad_batch_size. Funciona para qualquer formato
        (pastas, dataset, curriculum).
        
        Retorna ConfigBatchSize se configurado, ou None se não aplicável.
        """
        treino_raw = self._raw_config.get("treinamento", {})
        if not isinstance(treino_raw, dict):
            return None
        
        batch_raw = treino_raw.get("batch_size", None)
        if not isinstance(batch_raw, dict) or not batch_raw:
            return None
        
        efetivo = int(batch_raw.get("efetivo", 0))
        if efetivo <= 0:
            return None
        
        return ConfigBatchSize(
            efetivo=efetivo,
            batch_size=int(batch_raw.get("batch_size", 2))
        )
    
    def _aplicar_batch_size_auto(self) -> None:
        """Calcula grad_batch_size automaticamente para atingir o batch efetivo desejado.
        
        Sobrescreve treinamento.batch_size e treinamento.grad_batch_size de forma
        transparente, para que todo o código downstream (SFTConfig, eval_steps,
        MetricsLoggerCallback) funcione sem alterações.
        """
        if not self.batch_size_auto or self.batch_size_auto.efetivo <= 0:
            return
        
        import torch
        n_gpus = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
        
        bs = self.batch_size_auto.batch_size
        efetivo_desejado = self.batch_size_auto.efetivo
        
        # grad_batch_size = efetivo / (batch_size × n_gpus), mínimo 1
        grad = max(1, round(efetivo_desejado / (bs * n_gpus)))
        efetivo_real = bs * grad * n_gpus
        
        # Sobrescreve configuração de treinamento (downstream usa estes valores)
        self.treinamento.batch_size = bs
        self.treinamento.grad_batch_size = grad
        
        # Log informativo
        from treinar_unsloth_logging import get_logger
        _logger = get_logger(__name__)
        if efetivo_real == efetivo_desejado:
            _logger.info(
                f"<verde>📊 Batch automático: batch_size={bs} × grad_accum={grad} × "
                f"{n_gpus} GPU(s) = {efetivo_real} (efetivo desejado: {efetivo_desejado})</verde>"
            )
        else:
            _logger.info(
                f"<amarelo>📊 Batch automático (arredondado): batch_size={bs} × grad_accum={grad} × "
                f"{n_gpus} GPU(s) = {efetivo_real} (efetivo desejado: {efetivo_desejado})</amarelo>"
            )
    
    # ---------------------------------------------------------------------------
    # Métodos Públicos
    # ---------------------------------------------------------------------------

    def validar_max_seq_length(self) -> None:
        """Valida que max_seq_length está definido e exibe informações de tokens por etapa.
        
        max_seq_length é obrigatório (já validado no __post_init__).
        Aqui apenas exibimos informações úteis baseadas nos dados de tokens
        disponíveis no CSV de divisão (coluna token_total), para que o pesquisador
        possa verificar se o valor escolhido é adequado.
        """
        msl = self.treinamento.max_seq_length
        print(f"✅ max_seq_length={msl} (definido no YAML)")
        
        # Exibe informações de tokens por etapa do curriculum (se disponível nos CSVs)
        for etapa in self._curriculum:
            msl_etapa = etapa.max_seq_length if etapa.max_seq_length > 0 else msl
            info_tokens = self._ler_info_tokens_divisao(etapa.arquivo)
            if info_tokens:
                max_tk = info_tokens.get("max", 0)
                media_tk = info_tokens.get("media", 0)
                total = info_tokens.get("total", 0)
                suficiente = "✅" if msl_etapa >= max_tk else "⚠️  INSUFICIENTE 🚩"
                print(f"   📊 Etapa '{etapa.alias}': {total} registros, "
                      f"tokens max={max_tk}, média={media_tk:.0f} → "
                      f"max_seq_length={msl_etapa} {suficiente}")
    
    def _ler_info_tokens_divisao(self, arquivo_divisao: str) -> dict:
        """Lê estatísticas de tokens do CSV de divisão (coluna token_total).
        
        O CSV de divisão gerado pelo pacote de comparação inclui colunas
        token_total e token_output com a contagem de tokens por instância.
        
        Args:
            arquivo_divisao: Caminho para o CSV de divisão.
            
        Returns:
            Dict com {max, media, min, total} ou {} se coluna não disponível.
        """
        import pandas as pd
        if not arquivo_divisao or not os.path.isfile(arquivo_divisao):
            return {}
        try:
            df = pd.read_csv(arquivo_divisao, sep=None, engine='python')
            if 'token_total' not in df.columns:
                return {}
            tokens = pd.to_numeric(df['token_total'], errors='coerce').dropna()
            if tokens.empty:
                return {}
            return {
                "max": int(tokens.max()),
                "media": float(tokens.mean()),
                "min": int(tokens.min()),
                "total": len(tokens),
            }
        except Exception:
            return {}

    def estimar_contexto_predict(self, margem: float = 1.1) -> dict:
        """Estima contexto ideal para predição com base nos dados reais de tokens.

        Lê ``token_total`` e ``token_output`` de todos os CSVs do curriculum
        e calcula:
        - ``max_input``: max(token_total − token_output) observado nos dados
        - ``max_output``: max(token_output) observado nos dados
        - ``contexto``: max_input + max_output × margem (margem para saídas
          um pouco maiores que o visto no treino)
        - ``max_new_tokens``: max_output × margem

        Quando os CSVs não têm as colunas de token, usa fallback ``2 × max_seq_length``.

        Args:
            margem: Fator de folga sobre max_output (padrão 1.1 = 10% extra).

        Returns:
            Dict ``{"contexto": int, "max_new_tokens": int, "fonte": str}``
        """
        import pandas as pd
        msl = self.treinamento.max_seq_length

        max_total_obs = 0
        max_output_obs = 0
        tem_colunas = False

        for etapa in self._curriculum:
            arq = etapa.arquivo
            if not arq or not os.path.isfile(arq):
                continue
            try:
                df = pd.read_csv(arq, sep=None, engine='python')
                if 'token_total' not in df.columns:
                    continue
                totais = pd.to_numeric(df['token_total'], errors='coerce').dropna()
                if totais.empty:
                    continue
                tem_colunas = True
                max_total_obs = max(max_total_obs, int(totais.max()))

                if 'token_output' in df.columns:
                    outputs = pd.to_numeric(df['token_output'], errors='coerce').dropna()
                    if not outputs.empty:
                        max_output_obs = max(max_output_obs, int(outputs.max()))
            except Exception:
                continue

        if tem_colunas and max_total_obs > 0:
            # Calcula input observado (total − output)
            max_input_obs = max_total_obs - max_output_obs if max_output_obs > 0 else max_total_obs
            # Aplica margem sobre o output para permitir respostas um pouco maiores
            max_new = int(max_output_obs * margem) if max_output_obs > 0 else msl
            contexto = max_input_obs + max_new
            # Garante mínimo do max_seq_length (treinamento pode ter limitado)
            contexto = max(contexto, msl)
            fonte = (f"dados reais: max_input≈{max_input_obs}, "
                     f"max_output={max_output_obs} × {margem:.0%} → {max_new}")
        else:
            contexto = msl * 2
            max_new = msl
            fonte = f"fallback 2× max_seq_length (CSVs sem coluna token_total)"

        # Arredonda para múltiplo de 128 (ceil) — alinhamento típico de hardware GPU
        def _ceil128(v: int) -> int:
            return ((v + 127) // 128) * 128

        contexto = _ceil128(contexto)
        max_new = _ceil128(max_new)

        # Informa se excede max_position_embeddings (vLLM aceita via
        # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1, que habilita RoPE scaling)
        max_pos = self._ler_max_position_embeddings()
        if max_pos and contexto > max_pos:
            fonte += f", acima de max_position_embeddings={max_pos} (RoPE scaling)"

        return {
            "contexto": contexto,
            "max_new_tokens": max_new,
            "fonte": fonte,
        }

    def _ler_max_position_embeddings(self) -> int:
        """Lê max_position_embeddings do config.json do modelo base.

        Returns:
            Valor de max_position_embeddings ou 0 se não encontrado.
        """
        modelo_base = self.modelo.base
        if not modelo_base:
            return 0
        config_path = os.path.join(modelo_base, "config.json")
        if not os.path.isfile(config_path):
            return 0
        try:
            import json as _json
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = _json.load(f)
            return int(cfg.get("max_position_embeddings", 0))
        except Exception:
            return 0

    
    def info(self) -> str:
        """Retorna string com resumo da configuração."""
        cc = self.curriculum_config
        lines = [
            "=" * 60,
            "📋 CONFIGURAÇÃO YAML",
            "=" * 60,
            f"Arquivo: {self.yaml_path}",
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
            "",
            "📁 DADOS:",
        ]
        
        # Entrada
        if cc.entrada.pasta:
            lines.append(f"  Entrada (pasta): {cc.entrada.pasta}")
        elif cc.entrada.dataframe:
            lines.append(f"  Entrada (dataframe): {cc.entrada.dataframe} [col={cc.entrada.dataframe_col}, id={cc.entrada.dataframe_id}]")
        
        # Saída (gold)
        if cc.saida.pasta:
            lines.append(f"  Saída/Gold (pasta): {cc.saida.pasta}")
        elif cc.saida.dataframe:
            lines.append(f"  Saída/Gold (dataframe): {cc.saida.dataframe} [col={cc.saida.dataframe_col}, id={cc.saida.dataframe_id}]")
        
        lines.append(f"  Predição: {cc.predicao.pasta or '(será criado)'}")
        
        # Flags de criptografia
        flags_crypto = []
        if cc.entrada.texto_criptografado:
            flags_crypto.append("entrada.texto")
        if cc.entrada.prompt_criptografado:
            flags_crypto.append("entrada.prompt")
        if cc.saida.texto_criptografado:
            flags_crypto.append("saida.texto")
        if flags_crypto:
            lines.append(f"  🔐 Criptografado: {', '.join(flags_crypto)}")
        
        # --- Seção pipeline/curriculum ---
        
        treinaveis = [e for e in self._curriculum if e.is_treinavel]
        somente_predict = len(self._curriculum) - len(treinaveis)
        is_curriculum = len(treinaveis) > 1
        lines.extend(["", f"🔄 PIPELINE{' CURRICULUM' if is_curriculum else ''}:"])
        lines.append(f"  Etapas treináveis: {len(treinaveis)}" + (f" (+{somente_predict} somente predict)" if somente_predict else ""))
        
        for i, etapa in enumerate(self._curriculum):
            # Linha principal com alias e tipo
            partes = [f"alias='{etapa.alias}'", f"tipo={etapa.tipo or '(somente predict)'}"]
            if not etapa.is_treinavel:
                partes = [f"alias='{etapa.alias}'", "📋 somente predict (não treina)"]
            else:
                # Inclui overrides apenas quando diferem do valor global (>0)
                if etapa.max_seq_length > 0:
                    partes.append(f"max_seq_length={etapa.max_seq_length}")
                if etapa.pace_epochs > 0:
                    partes.append(f"epochs={etapa.pace_epochs}")
                if etapa.learning_rate > 0:
                    partes.append(f"lr={etapa.learning_rate}")
                if etapa.pace_loss > 0:
                    partes.append(f"pace_loss={etapa.pace_loss}")
            msl_etapa = etapa.max_seq_length if etapa.max_seq_length > 0 else self.treinamento.max_seq_length
            lines.append(f"  [{i}] {', '.join(partes)}")
            
            # Sub-detalhes da etapa (divisão e contagens)
            if etapa.arquivo:
                lines.append(f"      - divisão: {os.path.basename(etapa.arquivo)}")
                contagens = self._contar_instancias_divisao(etapa.arquivo)
                if contagens:
                    parts = [f"{k}={v}" for k, v in contagens.items()]
                    lines.append(f"      - {', '.join(parts)}")
                # Informações de tokens do CSV de divisão
                info_tokens = self._ler_info_tokens_divisao(etapa.arquivo)
                if info_tokens:
                    max_tk = info_tokens["max"]
                    media_tk = info_tokens["media"]
                    suficiente = "✅" if msl_etapa >= max_tk else "⚠️ INSUFICIENTE 🚩"
                    lines.append(f"      - tokens: max={max_tk}, média={media_tk:.0f}, "
                                 f"max_seq_length={msl_etapa} {suficiente}")
        
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
        return f"YamlTreinamento(yaml_path='{self.yaml_path}')"


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dicas e Comentários para os Templates YAML
# ---------------------------------------------------------------------------
# (Código movido para treinar_unsloth_dicas.py)

from treinar_unsloth_dicas import injetar_dicas_yaml, DICAS_YAML




def criar_yaml_exemplo_curriculum(caminho: str) -> None:
    """Cria um arquivo YAML de exemplo para modo curriculum."""
    template = {
        "curriculum": {
            "predicao": {
                "pasta": "./saidas/predicoes",
                "mascara": r"^(.+)\.txt$"
            },
            "saida": {
                "pasta": "./saidas/gold",
                "mascara": "*.txt",
                "formato": "json"
            },
            "entrada": {
                "pasta": "./saidas/textos",
                "mascara": r"^(.+)\.txt$",
                "prompt_template": "./prompts/template.txt",
                "tag_texto": "<<TEXTO>>"
            },
            "divisao": [
                {
                    "arquivo": "./divisao_facil.csv",
                    "alias": "facil",
                    "tipo": "lora",
                    "pace_epochs": 2,
                    "max_seq_length": 1024,
                    "proporcao": [
                        {"treino": 0.70},
                        {"validacao": 0.10},
                        {"teste": 0.20}
                    ]
                }
            ],
            "validacao": {
                "exigir_json_valido": True,
                "exigir_ids_pareados": True
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
    yaml_com_dicas = injetar_dicas_yaml(yaml_str, DICAS_YAML)
    
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
        """Valida pastas de entrada e gold dataset para modo curriculum."""
        curriculum = self._config.get("curriculum", {})
        if not curriculum:
            return True
        
        problemas = []
        
        # Verifica pasta/dataframe da saída (gold dataset)
        saida_cfg = curriculum.get("saida", {})
        gold_pasta = saida_cfg.get("pasta", "")
        gold_abs = self._resolver_caminho(gold_pasta) if gold_pasta else ""
        if gold_pasta and not os.path.isdir(gold_abs):
            problemas.append(("gold/saída", gold_pasta, gold_abs, ["curriculum", "saida", "pasta"]))
        
        # Verifica pasta de entrada
        ent_cfg = curriculum.get("entrada", {})
        ent_pasta = ent_cfg.get("pasta", "")
        ent_abs = self._resolver_caminho(ent_pasta) if ent_pasta else ""
        if ent_pasta and not os.path.isdir(ent_abs):
            problemas.append(("entrada", ent_pasta, ent_abs, ["curriculum", "entrada", "pasta"]))
        
        # Verifica template de prompt
        template = ent_cfg.get("prompt_template", "")
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
                self._config.setdefault("curriculum", {}).setdefault("entrada", {})["prompt_template"] = ""
                self._config["curriculum"]["entrada"]["tag_texto"] = ""
                self._modificado = True
                print("✅ Template removido. Arquivos de entrada serão usados diretamente como prompt.")
            elif opcao == 1:
                novo = self._perguntar_texto("Informe o caminho do template")
                if novo:
                    self._config.setdefault("curriculum", {}).setdefault("entrada", {})["prompt_template"] = novo
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
        """Valida divisão para modo curriculum (lista de etapas)."""
        curriculum = self._config.get("curriculum", {})
        if not curriculum:
            return True
        
        divisao = curriculum.get("divisao", [])
        if isinstance(divisao, list):
            # Modo curriculum com etapas: validação feita pelo pipeline
            if not divisao:
                print("\n⚠️  curriculum.divisao está vazio (nenhuma etapa configurada)")
            return True
        
        # divisao como dict (etapa única): valida arquivo
        arquivo = divisao.get("arquivo", "") if isinstance(divisao, dict) else ""
        arquivo_abs = self._resolver_caminho(arquivo) if arquivo else ""
        
        if arquivo and not os.path.isfile(arquivo_abs):
            print(f"\n📊 Arquivo de divisão não existe: {arquivo_abs}")
            print("   Será criado automaticamente na primeira execução do treinamento.")
        
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
    print("\nDeseja criar um arquivo YAML de exemplo (curriculum)?")
    print("  [S] Sim — criar exemplo")
    print("  [N] Não — sair")
    
    try:
        resposta = input("\nEscolha [S/N]: ").strip().lower()
        
        if resposta in ("s", "sim", "y", "yes", ""):
            criar_yaml_exemplo_curriculum(yaml_path)
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
  %(prog)s --criar-exemplo novo.yaml      Cria arquivo YAML de exemplo
        """
    )
    parser.add_argument("yaml", nargs="?", help="Arquivo YAML para validar")
    parser.add_argument("--interativo", "-i", action="store_true", 
                        help="Modo interativo: valida e pergunta sobre ajustes")
    parser.add_argument("--criar-exemplo", type=str, metavar="ARQUIVO",
                        help="Cria YAML de exemplo (curriculum)")
    
    args = parser.parse_args()
    
    if args.criar_exemplo:
        criar_yaml_exemplo_curriculum(args.criar_exemplo)
        print(f"✅ Exemplo criado: {args.criar_exemplo}")
        sys.exit(0)
    
    if not args.yaml:
        parser.print_help()
        sys.exit(1)
    
    # Verifica se o arquivo existe antes de tentar carregar
    if not os.path.isfile(args.yaml):
        if _perguntar_criar_exemplo(args.yaml):
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
        print("\n✅ Configuração válida!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("\n💡 Dica: Use --interativo para corrigir problemas de configuração")
        print(f"   python {sys.argv[0]} {args.yaml} --interativo")
        sys.exit(1)
