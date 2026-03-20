# -*- coding: utf-8 -*-

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Módulo de gráficos para treinamento Unsloth.

Contém classes especializadas para geração de gráficos de:
- Estatísticas de tokens (boxplots)
- Métricas de treinamento (loss, learning rate)
"""

import os
import json
from typing import Dict, List, Optional, Any
from treinar_unsloth_logging import get_logger

logger = get_logger(__name__)


class GraficoTokens:
    """Gera gráficos relacionados a estatísticas de tokens."""
    
    @staticmethod
    def boxplot_comparativo(
        dados: Dict[str, List[int]], 
        output_path: str,
        titulo: str = "Distribuição de Tokens por Subset"
    ) -> bool:
        """
        Gera boxplot comparativo de tokens por subset.
        
        Args:
            dados: Dict com chave=nome_categoria, valor=lista de valores
                   Ex: {"Treino (In)": [100, 200, ...], "Treino (Out)": [...]}
            output_path: Caminho para salvar o gráfico
            titulo: Título do gráfico
            
        Returns:
            True se gerou com sucesso, False caso contrário
        """
        if not dados:
            logger.warning("Nenhum dado para gerar boxplot de tokens.")
            return False
            
        try:
            from util_graficos import UtilGraficos
            
            UtilGraficos.gerar_boxplot(
                dados=dados,
                titulo=titulo,
                ylabel="Quantidade de Tokens",
                arquivo_saida=output_path
            )
            return True
        except Exception as e:
            logger.error(f"Erro ao gerar boxplot de tokens: {e}")
            return False


# ---------------------------------------------------------------------------
# Funções reutilizáveis para marcadores de época e etapas do curriculum
# Usadas por GraficoTreinamento.evolucao_loss() e GraficoHardware.evolucao_memoria()
# ---------------------------------------------------------------------------

def construir_marcadores_epocas(train_data: List[Dict]) -> List[Dict]:
    """Constrói marcadores de época a partir dos dados de treino.
    
    Detecta transições de época inteira usando epoch_global (contínuo entre
    etapas do curriculum) e retorna marcadores verticais verdes.
    
    Args:
        train_data: Lista de dicts com {step, epoch, epoch_global?, ...}
        
    Returns:
        Lista de dicts para marcadores_epoca do UtilGraficos.gerar_grafico_linhas()
    """
    marcadores = []
    epochs_seen = set()
    for t in train_data:
        eg = t.get("epoch_global", t.get("epoch", 0))
        epoch_int = int(eg)
        if epoch_int > 0 and epoch_int not in epochs_seen and eg == epoch_int:
            marcadores.append({
                'x': t["step"],
                'label': f'Época {epoch_int}',
                'cor': 'green',
                'y_frac': 0.05,
                'va': 'bottom',
            })
            epochs_seen.add(epoch_int)
    return marcadores


def construir_marcadores_etapas(
    marcadores_epoca: List[Dict],
    etapas_curriculum: List[Dict],
    all_steps: List = None,
) -> None:
    """Acrescenta marcadores de transição de etapa do curriculum (in-place).
    
    Adiciona linhas verticais violeta para cada transição entre etapas do curriculum
    (a partir da 2ª etapa). Os steps das etapas são também adicionados a all_steps
    para que o eixo x do gráfico inclua esses pontos.
    
    Args:
        marcadores_epoca: Lista de marcadores já existentes (será modificada in-place)
        etapas_curriculum: Lista de {step_global, alias, index} (pode ser None)
        all_steps: Lista de steps para ampliar limites do eixo x (pode ser None)
    """
    if not etapas_curriculum or len(etapas_curriculum) <= 1:
        return
    for i, et in enumerate(etapas_curriculum[1:]):
        y_frac = 0.92 - (i % 3) * 0.12
        marcadores_epoca.append({
            'x': et["step_global"],
            'label': f'Etapa: {et["alias"]}',
            'cor': 'darkviolet',
            'y_frac': y_frac,
            'va': 'top',
        })
        if all_steps is not None:
            all_steps.append(et["step_global"])


class GraficoTreinamento:
    """Gera gráficos de métricas de treinamento.
    
    Suporta duas fontes de dados:
    - training_metrics.jsonl (preferida): contém etapa do curriculum e instâncias acumuladas
    - trainer_state.json (fallback): dados básicos de loss por step
    """
    
    @staticmethod
    def carregar_training_metrics(treinamento_dir: str) -> Optional[Dict[str, Any]]:
        """
        Carrega métricas enriquecidas de training_metrics.jsonl.
        
        Esta é a fonte preferida pois inclui:
        - etapa_alias/etapa_index: identifica transições de curriculum
        - step_global: step contínuo somando todas as etapas
        - instancias_acumuladas: total de instâncias processadas
        
        Args:
            treinamento_dir: Diretório 'treinamento/' contendo training_metrics.jsonl
            
        Returns:
            Dict com chaves:
            - train_data: [{step_global, epoch, loss, lr, etapa_alias, etapa_index, instancias_acumuladas}]
            - eval_data: [{step_global, epoch, eval_loss, etapa_alias, etapa_index}]
            - etapas: [{step_global, alias, index}] — transições entre etapas do curriculum
            Ou None se arquivo não existir ou estiver vazio
        """
        arquivo = os.path.join(treinamento_dir, "training_metrics.jsonl")
        if not os.path.exists(arquivo):
            return None
        
        registros = []
        try:
            with open(arquivo, "r", encoding="utf-8") as f:
                for linha in f:
                    linha = linha.strip()
                    if linha:
                        try:
                            registros.append(json.loads(linha))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Erro ao carregar training_metrics.jsonl: {e}")
            return None
        
        if not registros:
            return None
        
        train_data = []
        eval_data = []
        etapas = []  # Transições de etapa detectadas pelo evento train_begin
        
        for r in registros:
            evento = r.get("event", "")
            # step_global garante eixo x contínuo entre etapas do curriculum
            step_global = r.get("step_global", r.get("step", 0))
            epoch = round(r.get("epoch", 0), 4)
            etapa_alias = r.get("etapa_alias", "Principal")
            etapa_index = r.get("etapa_index", 0)
            
            if evento == "train_begin":
                # Marca início de uma nova etapa do curriculum
                # train_begin grava step_offset (posição de início da etapa no eixo global)
                etapa_step = r.get("step_offset", step_global)
                etapas.append({
                    "step_global": etapa_step,
                    "alias": etapa_alias,
                    "index": etapa_index,
                })
            
            elif evento == "log":
                entry = {
                    "step": step_global,
                    "epoch": epoch,
                    "epoch_global": r.get("epoch_global", epoch),
                    "etapa_alias": etapa_alias,
                    "etapa_index": etapa_index,
                    "instancias_acumuladas": r.get("instancias_acumuladas", 0),
                    "tokens_acumulados": r.get("tokens_acumulados", 0),
                    "elapsed_seconds": r.get("elapsed_seconds", 0),
                }
                if "train_loss" in r:
                    entry["loss"] = round(r["train_loss"], 4)
                    entry["lr"] = r.get("learning_rate", 0)
                    train_data.append(entry)
                if "eval_loss" in r:
                    eval_entry = {
                        "step": step_global,
                        "epoch": epoch,
                        "epoch_global": r.get("epoch_global", epoch),
                        "eval_loss": round(r["eval_loss"], 4),
                        "etapa_alias": etapa_alias,
                        "etapa_index": etapa_index,
                    }
                    eval_data.append(eval_entry)
            
            elif evento == "evaluate":
                eval_loss = r.get("eval_loss")
                if eval_loss is not None:
                    eval_data.append({
                        "step": step_global,
                        "epoch": epoch,
                        "epoch_global": r.get("epoch_global", epoch),
                        "eval_loss": round(eval_loss, 4),
                        "etapa_alias": etapa_alias,
                        "etapa_index": etapa_index,
                    })
        
        return {
            "train_data": train_data,
            "eval_data": eval_data,
            "etapas": etapas,
        }

    @staticmethod
    def carregar_trainer_state(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
        """
        Carrega trainer_state.json do checkpoint mais recente (fallback).
        
        Args:
            checkpoint_dir: Diretório contendo os checkpoints (ex: modelo/chkpt)
            
        Returns:
            Dict com dados do trainer_state ou None se não encontrar
        """
        if not os.path.exists(checkpoint_dir):
            return None
            
        # Lista checkpoints e ordena
        checkpoints = []
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, item)):
                # Extrai número(s) do nome
                parts = item.replace("checkpoint-", "").split("-")
                try:
                    # Suporta formato antigo (checkpoint-8) e novo (checkpoint-00008)
                    step = int(parts[-1])  # Último número é sempre o step
                    checkpoints.append((step, item))
                except ValueError:
                    continue
        
        if not checkpoints:
            return None
            
        # Ordena pelo step
        checkpoints.sort(key=lambda x: x[0])
        ultimo_chkpt = checkpoints[-1][1]
        
        trainer_state_path = os.path.join(checkpoint_dir, ultimo_chkpt, "trainer_state.json")
        if not os.path.exists(trainer_state_path):
            return None
            
        try:
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar trainer_state.json: {e}")
            return None
    
    @staticmethod
    def extrair_metricas(trainer_state: Dict[str, Any]) -> tuple:
        """
        Extrai métricas de treino e validação do trainer_state.
        
        Args:
            trainer_state: Dict carregado do trainer_state.json
            
        Returns:
            Tuple (train_data, eval_data) onde cada um é uma lista de dicts
        """
        log_history = trainer_state.get("log_history", [])
        
        train_data = []
        eval_data = []
        
        for entry in log_history:
            step = entry.get("step", 0)
            epoch = entry.get("epoch", 0)
            
            if "loss" in entry and "eval_loss" not in entry:
                train_data.append({
                    "step": step,
                    "epoch": round(epoch, 2),
                    "loss": round(entry["loss"], 4),
                    "lr": entry.get("learning_rate", 0)
                })
            elif "eval_loss" in entry:
                eval_data.append({
                    "step": step,
                    "epoch": round(epoch, 2),
                    "eval_loss": round(entry["eval_loss"], 4)
                })
        
        return train_data, eval_data
    
    @staticmethod
    def listar_checkpoints(checkpoint_dir: str) -> List[str]:
        """
        Lista checkpoints ordenados por step.
        
        Args:
            checkpoint_dir: Diretório de checkpoints
            
        Returns:
            Lista de nomes de checkpoints ordenados
        """
        if not os.path.exists(checkpoint_dir):
            return []
            
        checkpoints = []
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, item)):
                parts = item.replace("checkpoint-", "").split("-")
                try:
                    step = int(parts[-1])
                    checkpoints.append((step, item))
                except ValueError:
                    continue
        
        checkpoints.sort(key=lambda x: x[0])
        return [c[1] for c in checkpoints]
    
    @staticmethod
    def evolucao_loss(
        train_data: List[Dict],
        eval_data: List[Dict],
        checkpoints: List[str],
        output_path: str,
        titulo: str = "Evolução do Loss durante Treinamento",
        etapas_curriculum: List[Dict] = None
    ) -> bool:
        """
        Gera gráfico de evolução do loss com marcações de época, checkpoints e etapas.
        
        Se os dados contiverem 'instancias_acumuladas', um eixo secundário (topo)
        exibirá a escala de instâncias treinadas.
        Se houver etapas de curriculum, linhas verticais coloridas marcam as transições.
        
        Args:
            train_data: Lista de dicts com {step, epoch, loss} (step = step_global se disponível)
            eval_data: Lista de dicts com {step, epoch, eval_loss}
            checkpoints: Lista de nomes de checkpoints
            output_path: Caminho para salvar o gráfico
            titulo: Título do gráfico
            etapas_curriculum: Lista de {step_global, alias, index} para marcar transições
            
        Returns:
            True se gerou com sucesso, False caso contrário
        """
        if not train_data and not eval_data:
            logger.warning("Nenhum dado de loss para gerar gráfico.")
            return False
            
        try:
            from util_graficos import UtilGraficos
            import math
            
            def _is_valid(v):
                return v is not None and isinstance(v, (int, float)) and not math.isnan(v)
            
            # Filtra NaN: séries só incluem pontos com loss válido
            series = {}
            
            if train_data:
                valid = [(t["step"], t["loss"]) for t in train_data if "loss" in t and _is_valid(t["loss"])]
                if valid:
                    series['Train Loss'] = {
                        'x': [v[0] for v in valid],
                        'y': [v[1] for v in valid],
                        'cor': 'blue',
                        'marcador': 'o',
                        'tamanho_marcador': 4
                    }
            
            if eval_data:
                valid = [(e["step"], e["eval_loss"]) for e in eval_data if "eval_loss" in e and _is_valid(e["eval_loss"])]
                if valid:
                    series['Eval Loss'] = {
                        'x': [v[0] for v in valid],
                        'y': [v[1] for v in valid],
                        'cor': 'red',
                        'marcador': 's',
                        'tamanho_marcador': 4
                    }
            
            # Coleta todos os step values para definir limites do eixo x
            all_steps = [t["step"] for t in train_data] + [e["step"] for e in eval_data]
            
            # Marcadores reutilizáveis de época e etapas do curriculum
            marcadores_epoca = construir_marcadores_epocas(train_data)
            construir_marcadores_etapas(marcadores_epoca, etapas_curriculum, all_steps)
            
            # Marcadores de checkpoints
            marcadores_verticais = []
            for chkpt in checkpoints:
                parts = chkpt.replace("checkpoint-", "").split("-")
                try:
                    step_num = int(parts[-1])
                    marcadores_verticais.append({
                        'x': step_num,
                        'cor': 'gray',
                        'estilo': ':',
                        'alpha': 0.4
                    })
                except ValueError:
                    continue
            
            # Info text com instâncias acumuladas (se disponível)
            info_text = None
            ultima_instancia = 0
            if train_data and "instancias_acumuladas" in train_data[-1]:
                ultima_instancia = train_data[-1]["instancias_acumuladas"]
                info_text = f"Instâncias treinadas: {ultima_instancia:,}".replace(",", ".")
            
            # Adiciona info de etapas curriculum no texto informativo
            if etapas_curriculum and len(etapas_curriculum) > 1:
                nomes = [et["alias"] for et in etapas_curriculum]
                prefixo = info_text + " | " if info_text else ""
                info_text = prefixo + f"Etapas: {' → '.join(nomes)}"
            
            # Nota quando sem dados válidos de loss
            if not series:
                sufixo = "\n⚠ Sem dados de loss válidos para plotar"
                info_text = (info_text + sufixo) if info_text else sufixo.strip()
            
            # Calcula limites do eixo x baseado em todos os steps conhecidos
            xlim = None
            if all_steps:
                x_min, x_max = min(all_steps), max(all_steps)
                margem = max(0.5, (x_max - x_min) * 0.05)
                xlim = (x_min - margem, x_max + margem)
            
            # Gera gráfico usando util_graficos
            resultado = UtilGraficos.gerar_grafico_linhas(
                series=series,
                titulo=titulo,
                ylabel='Loss',
                xlabel='Step (global)',
                arquivo_saida=output_path,
                marcadores_verticais=marcadores_verticais,
                marcadores_epoca=marcadores_epoca,
                info_text=info_text,
                xlim=xlim,
            )
            
            return resultado is not None
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de loss: {e}")
            return False
    
    @staticmethod
    def tabela_loss_markdown(train_data: List[Dict], eval_data: List[Dict]) -> List[str]:
        """
        Gera tabela markdown com evolução do loss.
        
        Se os dados contiverem 'etapa_alias' e 'instancias_acumuladas',
        inclui colunas adicionais; caso contrário exibe tabela simplificada.
        
        Args:
            train_data: Lista de dicts com {step, epoch, loss, [etapa_alias, instancias_acumuladas]}
            eval_data: Lista de dicts com {step, epoch, eval_loss}
            
        Returns:
            Lista de linhas markdown para a tabela
        """
        lines = []
        
        # Detecta se há dados enriquecidos (curriculum/instâncias)
        tem_etapa = any("etapa_alias" in t for t in train_data) if train_data else False
        tem_instancias = any("instancias_acumuladas" in t for t in train_data) if train_data else False
        
        # Cabeçalho adaptado ao formato dos dados
        if tem_etapa and tem_instancias:
            lines.append("| Step | Época | Etapa | Train Loss | Eval Loss | Instâncias |")
            lines.append("|------|-------|-------|------------|-----------|------------|")
        elif tem_etapa:
            lines.append("| Step | Época | Etapa | Train Loss | Eval Loss |")
            lines.append("|------|-------|-------|------------|-----------|")
        else:
            lines.append("| Step | Época | Train Loss | Eval Loss |")
            lines.append("|------|-------|------------|-----------|")
        
        # Combina por step
        steps_data = {}
        for t in train_data:
            steps_data[t["step"]] = {
                "epoch": t.get("epoch_global", t["epoch"]),
                "train": t["loss"],
                "eval": "-",
                "etapa": t.get("etapa_alias", ""),
                "instancias": t.get("instancias_acumuladas", ""),
            }
        for e in eval_data:
            if e["step"] in steps_data:
                steps_data[e["step"]]["eval"] = e["eval_loss"]
            else:
                steps_data[e["step"]] = {
                    "epoch": e.get("epoch_global", e["epoch"]),
                    "train": "-",
                    "eval": e["eval_loss"],
                    "etapa": e.get("etapa_alias", ""),
                    "instancias": "",
                }
        
        for step in sorted(steps_data.keys()):
            d = steps_data[step]
            inst_fmt = f"{d['instancias']:,}".replace(",", ".") if isinstance(d['instancias'], int) and d['instancias'] > 0 else "-"
            if tem_etapa and tem_instancias:
                lines.append(f"| {step} | {d['epoch']} | {d['etapa']} | {d['train']} | {d['eval']} | {inst_fmt} |")
            elif tem_etapa:
                lines.append(f"| {step} | {d['epoch']} | {d['etapa']} | {d['train']} | {d['eval']} |")
            else:
                lines.append(f"| {step} | {d['epoch']} | {d['train']} | {d['eval']} |")
        
        return lines


class GraficoMonitor:
    """Gera gráficos de monitoramento de recursos (RAM, GPU)."""
    
    @staticmethod
    def uso_memoria(
        tempos: List[float],
        ram_usadas: List[float],
        gpu_usadas: List[float],
        output_path: str,
        titulo: str = "Uso de Memória Durante Predições",
        num_gpus: int = 0
    ) -> bool:
        """
        Gera gráfico de uso de memória RAM e GPU ao longo do tempo.
        
        Args:
            tempos: Lista de tempos em segundos (relativo ao início)
            ram_usadas: Lista de uso de RAM em GB
            gpu_usadas: Lista de uso de GPU em GB
            output_path: Caminho para salvar o gráfico
            titulo: Título do gráfico
            num_gpus: Número de GPUs (para info text)
            
        Returns:
            True se gerou com sucesso, False caso contrário
        """
        if not tempos or not ram_usadas:
            logger.warning("Nenhum dado para gerar gráfico de memória.")
            return False
            
        try:
            from util_graficos import UtilGraficos
            
            # Prepara séries
            series = {
                'RAM (GB)': {
                    'x': tempos,
                    'y': ram_usadas,
                    'cor': 'blue',
                    'largura': 2,
                    'alpha': 0.8
                }
            }
            
            # Adiciona GPU se tiver dados
            if gpu_usadas and any(g > 0 for g in gpu_usadas):
                series['GPU (GB)'] = {
                    'x': tempos,
                    'y': gpu_usadas,
                    'cor': 'red',
                    'largura': 2,
                    'alpha': 0.8
                }
            
            # Info text
            ram_max = max(ram_usadas)
            gpu_max = max(gpu_usadas) if gpu_usadas else 0
            info_text = f"RAM máx: {ram_max:.1f} GB"
            if num_gpus > 0:
                info_text += f" | GPU máx: {gpu_max:.1f} GB ({num_gpus} GPU{'s' if num_gpus > 1 else ''})"
            
            # Gera gráfico
            resultado = UtilGraficos.gerar_grafico_linhas(
                series=series,
                titulo=titulo,
                ylabel='Memória (GB)',
                xlabel='Tempo (segundos)',
                arquivo_saida=output_path,
                preencher_area=True,
                info_text=info_text
            )
            
            return resultado is not None
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de memória: {e}")
            return False


class GraficoHardware:
    """Gera gráficos e tabelas de métricas de hardware durante treinamento."""
    
    @staticmethod
    def carregar_metricas(treinamento_dir: str) -> List[Dict[str, Any]]:
        """
        Carrega métricas de hardware do arquivo unificado training_metrics.jsonl.
        
        Extrai registros relevantes para o gráfico de memória:
        - event=log com train_loss → fase "train" (GPU durante treino)
        - event=evaluate → fase "eval" (GPU durante avaliação)
        - Ignora train_begin/train_end (bookend) e on_log com só eval_loss
          (redundante com on_evaluate)
        
        Compatibilidade: se training_metrics.jsonl não existir ou não contiver
        dados de hardware, tenta o legado hardware_metrics.jsonl.
        
        Args:
            treinamento_dir: Diretório 'treinamento/' contendo os arquivos JSONL
            
        Returns:
            Lista de dicts com métricas de hardware por step
        """
        # Fonte principal: training_metrics.jsonl (unificado)
        arquivo = os.path.join(treinamento_dir, "training_metrics.jsonl")
        metricas = []
        
        if os.path.exists(arquivo):
            try:
                with open(arquivo, "r", encoding="utf-8") as f:
                    for linha in f:
                        try:
                            r = json.loads(linha.strip())
                        except json.JSONDecodeError:
                            continue
                        # Só inclui registros com dados de hardware
                        if "ram_usada_gb" not in r:
                            continue
                        
                        event = r.get("event", "log")
                        
                        # Ignora eventos de bookend (sem step relevante)
                        if event in ("train_begin", "train_end"):
                            continue
                        
                        # on_log com train_loss → GPU reflete estado pós-treino
                        if event == "log" and "train_loss" in r:
                            r["fase"] = "train"
                        # on_evaluate → GPU reflete estado durante avaliação
                        elif event == "evaluate":
                            r["fase"] = "eval"
                        else:
                            # on_log sem train_loss (ex: só eval_loss) — redundante
                            # com on_evaluate, pular para evitar duplicatas
                            continue
                        
                        # Normaliza campos
                        r["step"] = r.get("step_global", r.get("step", 0))
                        r["epoch"] = r.get("epoch_global", r.get("epoch", 0))
                        metricas.append(r)
            except Exception as e:
                logger.error(f"Erro ao carregar métricas de hardware de training_metrics.jsonl: {e}")
        
        # Fallback: hardware_metrics.jsonl legado
        if not metricas:
            arquivo_legado = os.path.join(treinamento_dir, "hardware_metrics.jsonl")
            if os.path.exists(arquivo_legado):
                try:
                    with open(arquivo_legado, "r", encoding="utf-8") as f:
                        for linha in f:
                            try:
                                metricas.append(json.loads(linha.strip()))
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.error(f"Erro ao carregar hardware_metrics.jsonl: {e}")
            
        return metricas
    
    @staticmethod
    def evolucao_memoria(
        metricas: List[Dict[str, Any]],
        output_path: str,
        titulo: str = "Uso de Memória durante Treinamento",
        train_data: List[Dict] = None,
        etapas_curriculum: List[Dict] = None,
    ) -> bool:
        """
        Gera gráfico de evolução de memória RAM e GPU por step.
        
        Args:
            metricas: Lista de dicts com métricas por step
            output_path: Caminho para salvar o gráfico
            titulo: Título do gráfico
            train_data: Dados de treino (para marcadores de época). Se None, usa epoch das próprias métricas.
            etapas_curriculum: Lista de {step_global, alias, index} para marcar transições
            output_path: Caminho para salvar o gráfico
            titulo: Título do gráfico
            
        Returns:
            True se gerou com sucesso, False caso contrário
        """
        if not metricas:
            logger.warning("Nenhuma métrica de hardware para gerar gráfico.")
            return False
            
        try:
            from util_graficos import UtilGraficos
            
            # Prepara dados gerais
            steps = [m.get("step", i) for i, m in enumerate(metricas)]
            ram_usadas = [m.get("ram_usada_gb", 0) for m in metricas]

            # Separa dados de GPU por fase
            gpu_train_x, gpu_train_y = [], []
            gpu_eval_x, gpu_eval_y = [], []
            
            for m in metricas:
                step = m.get("step", 0)
                fase = m.get("fase", "")
                
                # Calcula GPU total
                gpu_total = 0
                for key in m.keys():
                    if key.startswith("gpu") and "reservada_gb" in key and "max" not in key:
                        gpu_total += m.get(key, 0)
                
                if "eval" in fase:
                    gpu_eval_x.append(step)
                    gpu_eval_y.append(gpu_total)
                else:
                    # Train ou Train Begin
                    gpu_train_x.append(step)
                    gpu_train_y.append(gpu_total)

            # Prepara séries
            series = {
                'RAM (GB)': {
                    'x': steps,
                    'y': ram_usadas,
                    'cor': 'blue',
                    'marcador': 'o',
                    'tamanho_marcador': 3
                }
            }
            
            if gpu_train_x:
                series['GPU (Treino)'] = {
                    'x': gpu_train_x,
                    'y': gpu_train_y,
                    'cor': 'red',
                    'marcador': '^',
                    'tamanho_marcador': 5,
                    'estilo': '-',
                    'alpha': 0.7,
                    'preencher': False,
                }
                
            if gpu_eval_x:
                series['GPU (Avaliação)'] = {
                    'x': gpu_eval_x,
                    'y': gpu_eval_y,
                    'cor': 'darkorange',
                    'marcador': 's',
                    'tamanho_marcador': 5,
                    'estilo': '--',
                    'alpha': 0.7,
                    'preencher': False,
                }
            
            # Info text
            ram_max = max(ram_usadas) if ram_usadas else 0
            
            # Combina GPU data para achar o máximo
            gpu_all = gpu_train_y + gpu_eval_y
            gpu_max = max(gpu_all) if gpu_all else 0
            n_evals = len([m for m in metricas if m.get("fase") == "eval"])
            info_text = f"RAM máx: {ram_max:.1f} GB"
            if gpu_max > 0:
                info_text += f" | GPU máx: {gpu_max:.1f} GB"
            if n_evals > 0:
                info_text += f" | {n_evals} avaliações"
            
            # Marcadores de época e etapas do curriculum (reutilizáveis)
            # Se train_data foi fornecido, usa-o; caso contrário, constrói a
            # partir das próprias métricas de hardware (que contêm step e epoch)
            fonte_epocas = train_data if train_data else [
                {"step": m.get("step", 0), "epoch": m.get("epoch", 0), "epoch_global": m.get("epoch", 0)}
                for m in metricas
            ]
            marcadores_epoca = construir_marcadores_epocas(fonte_epocas)
            all_steps = list(steps)
            construir_marcadores_etapas(marcadores_epoca, etapas_curriculum, all_steps)
            
            # Calcula limites do eixo x para incluir etapas
            xlim = None
            if all_steps:
                x_min, x_max = min(all_steps), max(all_steps)
                margem = max(0.5, (x_max - x_min) * 0.05)
                xlim = (x_min - margem, x_max + margem)
            
            # Gera gráfico
            resultado = UtilGraficos.gerar_grafico_linhas(
                series=series,
                titulo=titulo,
                ylabel='Memória (GB)',
                xlabel='Step',
                arquivo_saida=output_path,
                preencher_area=True,
                info_text=info_text,
                marcadores_epoca=marcadores_epoca,
                xlim=xlim,
            )
            
            return resultado is not None
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de hardware: {e}")
            return False
    
    @staticmethod
    def tabela_resumo_markdown(metricas: List[Dict[str, Any]]) -> List[str]:
        """
        Gera tabela markdown com resumo das métricas de hardware.
        
        Args:
            metricas: Lista de dicts com métricas por step
            
        Returns:
            Lista de linhas markdown
        """
        if not metricas:
            return ["*Nenhuma métrica de hardware disponível*"]
            
        # Extrai valores
        ram_usadas = [m.get("ram_usada_gb", 0) for m in metricas]
        cpu_usos = [m.get("cpu_uso_%", 0) for m in metricas]
        
        # GPU
        gpu_reservada = []
        gpu_max_reservada = []
        for m in metricas:
            gpu_total = 0
            gpu_max_total = 0
            for key in m.keys():
                if key.startswith("gpu") and "reservada_gb" in key and "max" not in key:
                    gpu_total += m.get(key, 0)
                if key.startswith("gpu") and "max_reservada_gb" in key:
                    gpu_max_total += m.get(key, 0)
            gpu_reservada.append(gpu_total)
            gpu_max_reservada.append(gpu_max_total)
        
        lines = []
        lines.append("| Métrica | Mínimo | Máximo | Média |")
        lines.append("|---------|--------|--------|-------|")
        
        # RAM
        if ram_usadas:
            lines.append(f"| RAM Usada (GB) | {min(ram_usadas):.2f} | {max(ram_usadas):.2f} | {sum(ram_usadas)/len(ram_usadas):.2f} |")
        
        # CPU
        if cpu_usos and any(c > 0 for c in cpu_usos):
            lines.append(f"| CPU (%) | {min(cpu_usos):.1f} | {max(cpu_usos):.1f} | {sum(cpu_usos)/len(cpu_usos):.1f} |")
        
        # GPU
        if gpu_reservada and any(g > 0 for g in gpu_reservada):
            lines.append(f"| GPU Reservada (GB) | {min(gpu_reservada):.2f} | {max(gpu_reservada):.2f} | {sum(gpu_reservada)/len(gpu_reservada):.2f} |")
        
        if gpu_max_reservada and any(g > 0 for g in gpu_max_reservada):
            lines.append(f"| GPU Pico (GB) | - | {max(gpu_max_reservada):.2f} | - |")
        
        return lines


# ---------------------------------------------------------------------------
# Gráficos de eficiência computacional (tokens, instâncias)
# ---------------------------------------------------------------------------

def _formatar_numero(valor: int) -> str:
    """Formata número grande de forma legível: 1.2M, 45.3K, etc."""
    if valor >= 1_000_000:
        return f"{valor / 1_000_000:.1f}M"
    if valor >= 1_000:
        return f"{valor / 1_000:.1f}K"
    return str(valor)


class GraficoEficiencia:
    """Gera gráficos de eficiência computacional do treinamento.
    
    Tokens acumulados e instâncias acumuladas por step, com marcadores de
    época e etapa do curriculum para comparação entre pipelines.
    """
    
    @staticmethod
    def evolucao_tokens(
        train_data: List[Dict],
        output_path: str,
        titulo: str = "Custo Computacional do Treinamento",
        etapas_curriculum: List[Dict] = None,
    ) -> bool:
        """
        Gera gráfico de tokens e instâncias acumulados ao longo dos steps.
        
        Eixo Y esquerdo: tokens acumulados (azul, área preenchida)
        Eixo Y direito: instâncias acumuladas (laranja, linha)
        Inclui marcadores de época e etapa, e info_text com eficiência.
        
        Args:
            train_data: Lista de dicts com {step, tokens_acumulados, instancias_acumuladas,
                        epoch_global?, loss?}
            output_path: Caminho para salvar o gráfico
            titulo: Título do gráfico
            etapas_curriculum: Lista de {step_global, alias, index} para transições
            
        Returns:
            True se gerou com sucesso, False caso contrário
        """
        if not train_data:
            logger.warning("Nenhum dado de treino para gerar gráfico de tokens.")
            return False
        
        # Filtra apenas pontos que têm tokens_acumulados > 0
        dados_validos = [t for t in train_data if t.get("tokens_acumulados", 0) > 0]
        if not dados_validos:
            logger.warning("Nenhum dado de tokens_acumulados disponível (treinamento anterior à v2?).")
            return False
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            steps = [t["step"] for t in dados_validos]
            tokens = [t["tokens_acumulados"] for t in dados_validos]
            instancias = [t.get("instancias_acumuladas", 0) for t in dados_validos]
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # --- Eixo esquerdo: Tokens acumulados ---
            cor_tokens = '#2563EB'  # azul
            ax1.plot(steps, tokens, color=cor_tokens, linewidth=2, marker='o',
                     markersize=4, alpha=0.9, label='Tokens Acumulados')
            ax1.fill_between(steps, tokens, alpha=0.15, color=cor_tokens)
            ax1.set_xlabel('Step (global)', fontsize=12)
            ax1.set_ylabel('Tokens Acumulados', fontsize=12, color=cor_tokens)
            ax1.tick_params(axis='y', labelcolor=cor_tokens)
            
            # Formata eixo Y esquerdo com K/M
            from matplotlib.ticker import FuncFormatter
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _formatar_numero(int(v))))
            
            # --- Eixo direito: Instâncias acumuladas ---
            cor_inst = '#EA580C'  # laranja escuro
            ax2 = ax1.twinx()
            ax2.plot(steps, instancias, color=cor_inst, linewidth=2, marker='s',
                     markersize=3, alpha=0.7, linestyle='--', label='Instâncias Acumuladas')
            ax2.set_ylabel('Instâncias Acumuladas', fontsize=12, color=cor_inst)
            ax2.tick_params(axis='y', labelcolor=cor_inst)
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _formatar_numero(int(v))))
            
            # --- Marcadores de época e etapas ---
            marcadores_epoca = construir_marcadores_epocas(train_data)
            all_steps = list(steps)
            construir_marcadores_etapas(marcadores_epoca, etapas_curriculum, all_steps)
            
            for m in marcadores_epoca:
                x_pos = m.get('x', 0)
                label = m.get('label', '')
                cor = m.get('cor', 'green')
                alpha = m.get('alpha', 0.7)
                ax1.axvline(x=x_pos, color=cor, linestyle='--', alpha=alpha, linewidth=1.5)
                if label:
                    y_frac = m.get('y_frac', 0.05)
                    va = m.get('va', 'bottom')
                    y_min, y_max = ax1.get_ylim()
                    y_pos = y_min + (y_max - y_min) * y_frac
                    ha = 'right' if va == 'top' else 'left'
                    ax1.text(x_pos, y_pos, label,
                             rotation=90, va=va, ha=ha, fontsize=9, color=cor)
            
            # --- Limites do eixo x ---
            if all_steps:
                x_min, x_max = min(all_steps), max(all_steps)
                margem = max(0.5, (x_max - x_min) * 0.05)
                ax1.set_xlim(x_min - margem, x_max + margem)
            
            # --- Info text: resumo de eficiência ---
            tokens_total = tokens[-1] if tokens else 0
            inst_total = instancias[-1] if instancias else 0
            info_parts = [f"Tokens: {_formatar_numero(tokens_total)}"]
            info_parts.append(f"Instâncias: {_formatar_numero(inst_total)}")
            
            # Eficiência: Δloss / Mtokens
            perdas = [t.get("loss") for t in dados_validos if t.get("loss") is not None]
            if len(perdas) >= 2 and tokens_total > 0:
                delta_loss = perdas[0] - perdas[-1]
                mtokens = tokens_total / 1_000_000
                if mtokens > 0:
                    eficiencia = delta_loss / mtokens
                    info_parts.append(f"Eficiência: {eficiencia:.2f} Δloss/Mtok")
            
            # Etapas do curriculum
            if etapas_curriculum and len(etapas_curriculum) > 1:
                nomes = [et["alias"] for et in etapas_curriculum]
                info_parts.append(f"Etapas: {' → '.join(nomes)}")
            
            info_text = " | ".join(info_parts)
            ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # --- Estilo ---
            ax1.set_title(titulo, fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Legenda combinada dos dois eixos
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de tokens: {e}")
            return False
