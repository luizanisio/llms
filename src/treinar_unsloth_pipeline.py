#!/usr/bin/env python3
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Pipeline Universal para treinamento com suporte a Curriculum Learning.
Normaliza a configuração YAML em uma lista universal de etapas
e gerencia rastreamento unificado de estado e métricas.

Classes:
    - CurriculumTracker: Rastreamento de estado e métricas do pipeline
"""

import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from treinar_unsloth_logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclass: Etapa do Curriculum
# ---------------------------------------------------------------------------

@dataclass
class EtapaCurriculum:
    """Representa uma etapa do pipeline de treinamento (curriculum learning).
    
    Para entradas simples (dataset/pastas), o sistema gera automaticamente
    uma lista com uma única etapa com alias='Principal'.
    """
    alias: str = "Principal"
    arquivo: str = ""          # Arquivo de divisão ou dataset específico da etapa
    tipo: str = "lora"         # "lora" ou "full"
    pace_epochs: int = 0       # 0 = usa epochs global do YAML
    pace_loss: float = 0.0     # 0 = sem early stopping por loss
    max_seq_length: int = 0    # 0 = usa valor global
    learning_rate: float = 0.0 # 0 = usa valor global


# ---------------------------------------------------------------------------
# CurriculumTracker: Estado e Métricas Unificados
# ---------------------------------------------------------------------------

class CurriculumTracker:
    """Rastreamento unificado do estado do pipeline de treinamento.
    
    Gerencia dois arquivos:
    - curriculum_state.json: Estado corrente do pipeline (etapa atual, status)
    - curriculum_metrics.jsonl: Métricas registradas etapa a etapa (append-only)
    
    Exemplo de curriculum_state.json:
        {"current_step": 0, "status": "running", "alias": "Principal", "updated_at": "..."}
    
    Exemplo de linha em curriculum_metrics.jsonl:
        {"alias": "Principal", "event": "etapa_fim", "train_loss": 0.5, ...}
    """
    
    # Status válidos para o pipeline
    STATUS_PENDENTE = "pendente"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.state_file = os.path.join(output_dir, "curriculum_state.json")
        self.metrics_file = os.path.join(output_dir, "curriculum_metrics.jsonl")
    
    # --- Estado ---
    
    def carregar_estado(self) -> Dict[str, Any]:
        """Carrega estado do pipeline ou retorna estado inicial."""
        if os.path.isfile(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Erro ao ler curriculum_state.json: {e}. Recriando.")
        return {"current_step": 0, "status": self.STATUS_PENDENTE}
    
    def salvar_estado(self, current_step: int, status: str, alias: str = "", **extras) -> None:
        """Salva estado corrente do pipeline."""
        estado = {
            "current_step": current_step,
            "status": status,
            "alias": alias,
            "updated_at": datetime.now().isoformat(),
            **extras
        }
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(estado, f, ensure_ascii=False, indent=2)
    
    # --- Métricas ---
    
    def registrar_metrica(self, alias: str, event: str, **metricas) -> None:
        """Acrescenta uma linha de métricas ao arquivo JSONL."""
        registro = {
            "timestamp": datetime.now().isoformat(),
            "alias": alias,
            "event": event,
            **{k: v for k, v in metricas.items() if v is not None}
        }
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    
    def iniciar_etapa(self, step_index: int, etapa: EtapaCurriculum) -> None:
        """Marca início de uma etapa do curriculum."""
        self.salvar_estado(
            current_step=step_index,
            status=self.STATUS_RUNNING,
            alias=etapa.alias,
            tipo=etapa.tipo
        )
        self.registrar_metrica(
            alias=etapa.alias,
            event="etapa_inicio",
            step_index=step_index,
            tipo=etapa.tipo
        )
        logger.info(f"<azul>▶ Etapa {step_index} iniciada: alias='{etapa.alias}', tipo={etapa.tipo}</azul>")
    
    def finalizar_etapa(self, step_index: int, alias: str, **metricas) -> None:
        """Marca fim de uma etapa e registra métricas finais."""
        self.salvar_estado(
            current_step=step_index,
            status=self.STATUS_COMPLETED,
            alias=alias
        )
        self.registrar_metrica(alias=alias, event="etapa_fim", **metricas)
        logger.info(f"<verde>✅ Etapa {step_index} finalizada: alias='{alias}'</verde>")
    
    def marcar_falha(self, step_index: int, alias: str, erro: str) -> None:
        """Registra falha em uma etapa."""
        self.salvar_estado(
            current_step=step_index,
            status=self.STATUS_FAILED,
            alias=alias,
            erro=erro
        )
        self.registrar_metrica(alias=alias, event="etapa_falha", erro=erro)
        logger.error(f"<vermelho>❌ Etapa {step_index} falhou: alias='{alias}' - {erro}</vermelho>")

    def marcar_conclusao(self, total_etapas: int, target_epochs: float = 0.0) -> None:
        """Marca o pipeline inteiro como concluído."""
        self.salvar_estado(
            current_step=total_etapas,
            status="finished",
            alias="COMPLETO",
            target_epochs=target_epochs,
            mensagem="Treinamento atingiu seu objetivo final."
        )
        self.registrar_metrica(alias="COMPLETO", event="treinamento_concluido", total_etapas=total_etapas, target_epochs=target_epochs)
        logger.info(f"<verde>✅ TREINAMENTO COMPLETO — {total_etapas} etapas de curriculum finalizadas</verde>")

    def is_concluido(self) -> bool:
        """Verifica se o treinamento já foi dado como concluído."""
        estado = self.carregar_estado()
        return estado.get("status") == "finished"


# ---------------------------------------------------------------------------
# Funções utilitárias do pipeline
# ---------------------------------------------------------------------------

def arredondar_seq_length(max_tokens: int, margem_minima: int = 256) -> int:
    """Arredonda max_tokens para o próximo múltiplo de 256 com margem mínima.
    
    Args:
        max_tokens: Número máximo de tokens encontrado no dataset
        margem_minima: Folga mínima entre max_tokens e o valor final (padrão: 256)
    
    Returns:
        Valor arredondado para cima (múltiplo de 256) com margem garantida
    
    Exemplos:
        arredondar_seq_length(3900) -> 4352  (3900+256=4156, ceil→4352)
        arredondar_seq_length(3840) -> 4352  (3840+256=4096, ceil→4352 pois 4096-3840=256)
        arredondar_seq_length(256)  -> 512   (256+256=512, ceil→512)
    """
    if max_tokens <= 0:
        return 256
    valor_com_margem = max_tokens + margem_minima
    return ((valor_com_margem + 255) // 256) * 256


def construir_etapas(yaml_config) -> List[EtapaCurriculum]:
    """Normaliza a configuração YAML em uma lista de etapas do curriculum.
    
    Interpreta a seção 'curriculum.divisao' do YAML. Se for uma lista,
    cada item vira uma etapa. Se for um dict (etapa única), gera uma etapa com alias='Principal'.
    
    Args:
        yaml_config: Instância de YamlTreinamento
    
    Returns:
        Lista de EtapaCurriculum (sempre >= 1 elemento)
    """
    raw = yaml_config._raw_config
    treinamento = yaml_config.treinamento
    lora = yaml_config.lora
    tipo_padrao = "lora" if lora.r not in (0, None, False) else "full"
    
    curriculum_raw = raw.get("curriculum", {})
    if not isinstance(curriculum_raw, dict):
        raise ValueError("Seção 'curriculum' é obrigatória e deve ser um dicionário")
    
    divisao_raw = curriculum_raw.get("divisao", [])
    
    # Se divisao é um dict (etapa única), converte em lista com um item
    if isinstance(divisao_raw, dict):
        divisao_list = [divisao_raw]
    elif isinstance(divisao_raw, list):
        divisao_list = divisao_raw
    else:
        divisao_list = []
    
    if not divisao_list:
        # Sem divisao explícita: gera etapa padrão única
        etapa = EtapaCurriculum(
            alias="Principal",
            tipo=tipo_padrao,
            pace_epochs=treinamento.epochs,
            max_seq_length=treinamento.max_seq_length,
            learning_rate=treinamento.learning_rate,
        )
        return [etapa]
    
    etapas = []
    for i, item in enumerate(divisao_list):
        if not isinstance(item, dict):
            raise ValueError(f"Etapa {i} do curriculum deve ser um dicionário, recebido: {type(item)}")
        
        alias = item.get("alias", f"etapa_{i}" if len(divisao_list) > 1 else "Principal")
        arquivo = item.get("arquivo", "")
        if arquivo:
            arquivo = yaml_config._resolver_caminho(arquivo)
        
        etapa = EtapaCurriculum(
            alias=alias,
            arquivo=arquivo,
            tipo=item.get("tipo", tipo_padrao),
            pace_epochs=int(item.get("pace_epochs", treinamento.epochs)),
            pace_loss=float(item.get("pace_loss", 0.0)),
            max_seq_length=int(item.get("max_seq_length", 0)),
            learning_rate=float(item.get("learning_rate", 0.0)),
        )
        etapas.append(etapa)
    
    logger.info(f"<azul>📋 Curriculum: {len(etapas)} etapa(s) configurada(s)</azul>")
    for i, e in enumerate(etapas):
        logger.info(f"<cinza>   [{i}] alias='{e.alias}', tipo={e.tipo}, arquivo={os.path.basename(e.arquivo) if e.arquivo else '(vazio)'}</cinza>")
    
    return etapas
