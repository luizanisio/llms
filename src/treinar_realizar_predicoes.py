#!/usr/bin/env python3

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Classes de predição e inferência interativa para modelos LLM.

Hierarquia:
    UtilPredicao (ABC)
      ├── UtilPredicaoHF        — HuggingFace Transformers (LLMsTrainer)
      ├── UtilPredicaoVLLM      — vLLM (batch nativo com PagedAttention)
      ├── UtilPredicaoUnsloth   — Unsloth (FastLanguageModel.for_inference)
      └── UtilPredicaoOllama    — Ollama (API local)

Saída padronizada (compatível com get_resposta de util_openai.py):
    {
        "resposta": str | dict,
        "usage": {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int
        },
        "model": str,
        "tempo": float,
        "erro": str           # apenas quando ocorre falha
    }

Uso:
    preditor = UtilPredicaoOllama(yaml_path)
    preditor.executar_predict(subsets=['teste'])
    preditor.executar_modelo(n_exemplos=3)

    # Reuso de modelo já carregado (otimiza memória):
    preditor = UtilPredicaoHF(yaml_path)
    preditor.carregar_modelo()
    preditor.executar_predict(subsets=['teste'])
    preditor.executar_modelo(n_exemplos=5)
    preditor.liberar_modelo()
"""

import os
import sys
import gc
import json
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime

# garante que a pasta src está no sys.path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from treinar_unsloth_logging import get_logger, log_separador
from treinar_unsloth_util import YamlTreinamento, FORMATO_SAIDA_JSON
from treinar_unsloth_actions import (
    _exibir_cabecalho_modelo,
    _verificar_modelo_treinado,
    _perguntar_confirmacao,
)
from treinar_unsloth_export import (
    _construir_mapa_etapas,
    _copiar_para_pastas_etapas,
    _registro_ja_exportado,
    _todas_predicoes_exportadas,
    gerar_estatisticas_predicoes,
    gerar_estatisticas_predicoes_etapas,
)
from util import UtilEnv

logger = get_logger(__name__)

# Intervalo mínimo entre logs de progresso (segundos)
_INTERVALO_PROGRESSO_S = 120  # 2 minutos
# Frequência padrão de log de progresso (a cada N registros)
_FREQUENCIA_PROGRESSO = 10


def _formatar_eta(inicio: float, feitos: int, total: int) -> str:
    """Retorna string com ETA estimado, ex: '⏱️ ~1h32min restantes'."""
    if feitos <= 0:
        return ""
    elapsed = time.time() - inicio
    restantes = total - feitos
    eta_s = (elapsed / feitos) * restantes
    if eta_s < 60:
        return f"⏱️ ~{int(eta_s)}s restantes"
    elif eta_s < 3600:
        return f"⏱️ ~{int(eta_s // 60)}min restantes"
    else:
        h = int(eta_s // 3600)
        m = int((eta_s % 3600) // 60)
        return f"⏱️ ~{h}h{m:02d}min restantes"


# ============================================================================
# Classe base — UtilPredicao
# ============================================================================

class UtilPredicao(ABC):
    """Classe base abstrata para predição e inferência interativa de modelos LLM.

    Implementa o fluxo completo de predict (exportação em lote por subset) e
    inferência interativa (N exemplos do treino), delegando às subclasses
    apenas o carregamento de modelo e a chamada de predição unitária.

    Retorno padronizado de ``predizer()``::

        {
            'resposta': str | dict,
            'usage': {
                'prompt_tokens': int,
                'completion_tokens': int,
                'total_tokens': int,
            },
            'model': str,
            'tempo': float,
            'erro': str,           # apenas quando ocorre falha
        }
    """

    NOME_ENGINE: str = "base"
    ICONE: str = "🤖"
    SUBSETS_PADRAO: list = ['teste']
    REQUER_MODELO_LOCAL: bool = True

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    def __init__(self, yaml_path: str, usar_base: bool = False):
        self.yaml_path = yaml_path
        self.usar_base = usar_base

        self._modelo = None
        self._modelo_carregado = False
        self._monitor = None
        self.metricas_memoria: Dict[str, Any] = {}

        # Carrega configuração YAML
        validar = self.REQUER_MODELO_LOCAL
        self.yaml_config = YamlTreinamento(yaml_path, validar_caminhos=validar)
        self.max_seq_length: int = self.yaml_config.treinamento.max_seq_length
        self.formato_json: bool = self.yaml_config.formato_saida == FORMATO_SAIDA_JSON

    # ------------------------------------------------------------------
    # Métodos abstratos (cada engine implementa)
    # ------------------------------------------------------------------

    @abstractmethod
    def carregar_modelo(self) -> None:
        """Carrega o modelo na memória e armazena em ``self._modelo``."""

    @abstractmethod
    def predizer(self, messages: list, prompt_texto: str) -> Dict[str, Any]:
        """Executa predição para uma única entrada.

        Args:
            messages: Lista de mensagens no formato ``[{role, content}, ...]``
            prompt_texto: Texto extraído do role ``user``

        Returns:
            Dict padronizado com chaves ``resposta``, ``usage``, ``model``, ``tempo``
            (e opcionalmente ``erro``).
        """

    # ------------------------------------------------------------------
    # Métodos com implementação padrão (override opcional)
    # ------------------------------------------------------------------

    def predizer_lote(self, registros: List[dict], on_resultado=None) -> List[Dict[str, Any]]:
        """Executa predição em lote. Padrão: loop sequencial de ``predizer()``.

        A subclasse vLLM sobrescreve com batch nativo.

        Args:
            registros: Lista de dicts ``{messages, prompt_texto, registro_id, idx}``
            on_resultado: Callback opcional ``(reg, resultado)`` chamado
                imediatamente após cada predição, permitindo gravação
                contínua em disco.

        Returns:
            Lista de dicts de resultado na mesma ordem de ``registros``.
        """
        resultados = []
        ultimo_log = time.time()
        ini_lote = time.time()
        total = len(registros)
        for i, reg in enumerate(registros):
            resultado = self.predizer(reg['messages'], reg['prompt_texto'])
            resultados.append(resultado)
            if on_resultado:
                on_resultado(reg, resultado)
            # Log de progresso: a cada N registros OU se >= 2 min desde último log
            agora = time.time()
            if (i + 1) % _FREQUENCIA_PROGRESSO == 0 or (i + 1) == total or (agora - ultimo_log) >= _INTERVALO_PROGRESSO_S:
                eta = _formatar_eta(ini_lote, i + 1, total)
                logger.info(f"   Progresso: {i + 1}/{total} ({100 * (i + 1) // total}%) {eta}")
                ultimo_log = agora
        return resultados

    def liberar_modelo(self) -> None:
        """Libera modelo da memória e coleta métricas do monitor."""
        if self._monitor:
            self.metricas_memoria = self._monitor.parar()
            self._monitor = None
        if self._modelo is not None:
            del self._modelo
            self._modelo = None
        self._modelo_carregado = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def nome_pasta_predict(self) -> str:
        """Retorna nome da pasta de predições (ex: ``predict_vllm``, ``predict_base_ollama``)."""
        sufixo = f"_{self.NOME_ENGINE}" if self.NOME_ENGINE != "hf" else ""
        prefixo = "predict_base" if self.usar_base else "predict"
        return f"{prefixo}{sufixo}"

    def nome_modelo_usado(self) -> str:
        """Retorna identificação legível do modelo usado."""
        if self.usar_base:
            return self.yaml_config.modelo.base
        return self.yaml_config.modelo.saida

    # ------------------------------------------------------------------
    # Monitoramento de memória
    # ------------------------------------------------------------------

    def _iniciar_monitor(self, nome_arquivo: str = "memoria_predicao") -> None:
        """Inicia monitoramento de RAM/GPU em thread separada."""
        try:
            from treinar_unsloth_monitor import MonitorRecursos
            output_dir = self.yaml_config.modelo.saida
            os.makedirs(output_dir, exist_ok=True)
            self._monitor = MonitorRecursos(
                output_dir=output_dir,
                intervalo_segundos=0.5,
                nome_arquivo=nome_arquivo,
            )
            self._monitor.iniciar()
        except Exception as e:
            logger.warning(f"<amarelo>⚠️ Não foi possível iniciar monitor de memória: {e}</amarelo>")
            self._monitor = None

    # ------------------------------------------------------------------
    # Helpers comuns
    # ------------------------------------------------------------------

    @staticmethod
    def _extrair_prompt(messages: list) -> str:
        """Extrai conteúdo do role ``user`` da lista de mensagens."""
        for m in messages:
            if m.get('role') == 'user':
                return m.get('content', '')
        return ''

    @staticmethod
    def _extrair_resposta_esperada(messages: list) -> str:
        """Extrai conteúdo do role ``assistant`` da lista de mensagens."""
        for m in messages:
            if m.get('role') == 'assistant':
                return m.get('content', '')
        return ''

    def _formatar_resposta(self, resposta) -> str:
        """Converte resposta para string, tentando JSON se formato_json=True."""
        if isinstance(resposta, dict):
            return json.dumps(resposta, ensure_ascii=False, indent=2)
        if not isinstance(resposta, str):
            resposta = str(resposta)
        if self.formato_json and resposta.strip():
            try:
                from util import UtilTextos
                json_obj = UtilTextos.mensagem_to_json(resposta)
                if json_obj:
                    return json.dumps(json_obj, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return resposta

    def _salvar_resultado(self, subset_dir: str, registro_id: str,
                          resposta_texto: str, resultado: Dict[str, Any]) -> None:
        """Persiste resultado de predição em .txt + .json."""
        output_txt = os.path.join(subset_dir, f"{registro_id}.txt")
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(resposta_texto)

        usage = resultado.get('usage', {})
        usage_data = {
            'id': registro_id,
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0),
            'time_s': round(resultado.get('tempo', 0), 3),
            'model': resultado.get('model', ''),
        }
        output_json = os.path.join(subset_dir, f"{registro_id}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(usage_data, f, ensure_ascii=False, indent=2)

    def _preview(self, texto: str, max_chars: int = 500) -> str:
        """Retorna preview de texto longo."""
        if len(texto) > max_chars:
            metade = max_chars // 2
            return f"{texto[:metade]} [...] {texto[-metade:]}"
        return texto

    def _verificar_modelo_e_decidir_base(self) -> bool:
        """Verifica modelo treinado e ajusta ``self.usar_base`` se necessário.

        Returns:
            True se pode continuar, False se o usuário cancelou.
        """
        if self.usar_base:
            logger.info(f"<cinza>ℹ️  Opção base ativada: usando modelo base.</cinza>")
            return True
        if not _verificar_modelo_treinado(self.yaml_config):
            logger.warning("<amarelo>\n⚠️  Não foi encontrado modelo LoRA treinado na pasta de saída.</amarelo>")
            if not _perguntar_confirmacao("Deseja continuar com o modelo base?", padrao=False):
                logger.info("Operação cancelada.")
                return False
            self.usar_base = True
            logger.info("Continuando com modelo base (sem fine-tuning)...\n")
        else:
            logger.info(f"<verde>✅ Modelo treinado encontrado em: {self.yaml_config.modelo.saida}</verde>")
        return True

    # ------------------------------------------------------------------
    # executar_predict — exportação em lote
    # ------------------------------------------------------------------

    def executar_predict(self, subsets: list = None) -> None:
        """Exporta predições para os subsets especificados.

        Itera registros de cada subset, gera predições via ``predizer_lote()``,
        salva .txt + .json por registro, e gera resumo.json + resumo_geral.json.

        Registros já exportados são automaticamente ignorados (permite continuação).

        Args:
            subsets: Lista de subsets (ex: ``['teste']``). Se None, usa ``SUBSETS_PADRAO``.
        """
        logger.info("\n")
        log_separador(caractere="=", largura=80)
        logger.info(f"<azul>>> MODO PREDICT - {self.NOME_ENGINE.upper()} {self.ICONE}</azul>")
        log_separador(caractere="=", largura=80)

        _exibir_cabecalho_modelo(self.yaml_config)
        self.yaml_config.validar_max_seq_length()

        logger.info(f"<cinza>\n📋 Formato de saída: {self.yaml_config.formato_saida}</cinza>")

        if not self._preparar_para_execucao():
            return

        if subsets is None:
            subsets = list(self.SUBSETS_PADRAO)

        logger.info(f"<cinza>📋 Subsets a processar: {', '.join(subsets)}</cinza>")

        # Diretório de predições
        predict_dir = os.path.join(self.yaml_config.modelo.saida, self.nome_pasta_predict())
        os.makedirs(predict_dir, exist_ok=True)

        # Divisão unificada: dicionário {id: {alvo, divisoes, etapas}}
        divisao_dict = self.yaml_config.dataset_manager.carregar_divisao_completa(self.yaml_config.curriculum)
        mapa_etapas = _construir_mapa_etapas(divisao_dict)

        # Pré-check: se todas as predições já existem, sai sem carregar o modelo
        if _todas_predicoes_exportadas(predict_dir, subsets, divisao_dict, mapa_etapas):
            return

        # Garante modelo carregado
        modelo_proprio = not self._modelo_carregado
        if modelo_proprio:
            try:
                self.carregar_modelo()
            except Exception as e:
                logger.error(f"<vermelho>❌ Erro ao carregar modelo: {e}</vermelho>")
                return

        uso_total = {
            'input_tokens': 0, 'output_tokens': 0,
            'total_registros': 0, 'tempo_total_s': 0,
            'por_subset': {},
        }

        try:
            for subset in subsets:
                logger.info(f"<azul>\n📂 Processando subset: {subset}</azul>")
                log_separador(caractere="-", largura=60)

                try:
                    mensagens = self.yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=subset, divisao=divisao_dict)
                    if not mensagens:
                        logger.warning(f"<amarelo>   ⚠️ Nenhum dado encontrado para {subset}</amarelo>")
                        continue
                    logger.info(f"<cinza>   📊 {len(mensagens)} registros encontrados</cinza>")
                except Exception as e:
                    logger.error(f"<vermelho>   ❌ Erro ao carregar {subset}: {e}</vermelho>")
                    continue

                subset_dir = os.path.join(predict_dir, subset)
                os.makedirs(subset_dir, exist_ok=True)

                subset_stats = {
                    'input_tokens': 0, 'output_tokens': 0,
                    'registros_ok': 0, 'registros_erro': 0, 'registros_skip': 0,
                }
                ini_subset = time.time()

                # Filtra pendentes e prepara registros
                registros_pendentes = []
                for idx, msg in enumerate(mensagens):
                    if not (isinstance(msg, dict) and 'messages' in msg):
                        subset_stats['registros_erro'] += 1
                        continue
                    messages = msg['messages']
                    prompt_texto = self._extrair_prompt(messages)
                    if not prompt_texto:
                        subset_stats['registros_erro'] += 1
                        continue
                    registro_id = msg.get('id', f'{subset}_{idx:04d}')
                    if _registro_ja_exportado(subset_dir, registro_id):
                        subset_stats['registros_skip'] += 1
                        _copiar_para_pastas_etapas(predict_dir, subset, registro_id, mapa_etapas)
                        continue
                    registros_pendentes.append({
                        'messages': messages,
                        'prompt_texto': prompt_texto,
                        'registro_id': registro_id,
                        'idx': idx,
                    })

                if not registros_pendentes:
                    skip_msg = f" ({subset_stats['registros_skip']} já exportados)" if subset_stats['registros_skip'] else ""
                    logger.info(f"<cinza>   Nenhum registro pendente{skip_msg}</cinza>")
                    continue

                logger.info(f"<cinza>   📝 {len(registros_pendentes)} registros pendentes</cinza>")

                # Callback para gravação contínua: salva cada resultado
                # imediatamente após a predição, permitindo retomada em
                # caso de erro (registros já exportados são ignorados).
                def _on_resultado(reg, resultado):
                    if 'erro' in resultado:
                        logger.error(f"<vermelho>   ❌ Erro no registro {reg['idx']}: {resultado['erro']}</vermelho>")
                        subset_stats['registros_erro'] += 1
                        return

                    resposta_texto = self._formatar_resposta(resultado.get('resposta', ''))
                    self._salvar_resultado(subset_dir, reg['registro_id'], resposta_texto, resultado)
                    _copiar_para_pastas_etapas(predict_dir, subset, reg['registro_id'], mapa_etapas)

                    usage = resultado.get('usage', {})
                    subset_stats['input_tokens'] += usage.get('prompt_tokens', 0)
                    subset_stats['output_tokens'] += usage.get('completion_tokens', 0)
                    subset_stats['registros_ok'] += 1

                # Predição em lote com gravação contínua
                # (vLLM faz batch nativo, outros fazem loop sequencial)
                self.predizer_lote(registros_pendentes, on_resultado=_on_resultado)

                tempo_subset = time.time() - ini_subset

                # Resumo do subset
                resumo_subset = {
                    'subset': subset,
                    'total_registros': len(mensagens),
                    'registros_ok': subset_stats['registros_ok'],
                    'registros_erro': subset_stats['registros_erro'],
                    'registros_skip': subset_stats['registros_skip'],
                    'input_tokens_total': subset_stats['input_tokens'],
                    'output_tokens_total': subset_stats['output_tokens'],
                    'tempo_processamento_s': round(tempo_subset, 2),
                    'data_geracao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'modelo': self.nome_modelo_usado(),
                    'engine': self.NOME_ENGINE,
                    'formato_saida': self.yaml_config.formato_saida,
                }
                resumo_file = os.path.join(subset_dir, "resumo.json")
                with open(resumo_file, 'w', encoding='utf-8') as f:
                    json.dump(resumo_subset, f, ensure_ascii=False, indent=2)

                skip_msg = f", {subset_stats['registros_skip']} já exportados" if subset_stats['registros_skip'] else ""
                logger.info(f"<verde>   ✅ {subset_stats['registros_ok']} predições salvas em: {subset_dir}</verde>")
                logger.info(f"<cinza>   📊 Tokens: {subset_stats['input_tokens']} entrada, "
                            f"{subset_stats['output_tokens']} saída ({tempo_subset:.1f}s){skip_msg}</cinza>")
                gerar_estatisticas_predicoes(subset_dir)
                gerar_estatisticas_predicoes_etapas(predict_dir, subset)

                uso_total['input_tokens'] += subset_stats['input_tokens']
                uso_total['output_tokens'] += subset_stats['output_tokens']
                uso_total['total_registros'] += subset_stats['registros_ok']
                uso_total['tempo_total_s'] += tempo_subset
                uso_total['por_subset'][subset] = resumo_subset

        finally:
            if modelo_proprio:
                self.liberar_modelo()

        # Resumo geral
        resumo_geral = {
            'data_geracao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'modelo_base': self.yaml_config.modelo.base,
            'modelo_saida': self.yaml_config.modelo.saida,
            'modelo_usado': self.nome_modelo_usado(),
            'engine': self.NOME_ENGINE,
            'formato_saida': self.yaml_config.formato_saida,
            'total_registros': uso_total['total_registros'],
            'input_tokens_total': uso_total['input_tokens'],
            'output_tokens_total': uso_total['output_tokens'],
            'tempo_total_s': round(uso_total['tempo_total_s'], 2),
            'subsets': uso_total['por_subset'],
        }
        resumo_file = os.path.join(predict_dir, "resumo_geral.json")
        with open(resumo_file, 'w', encoding='utf-8') as f:
            json.dump(resumo_geral, f, ensure_ascii=False, indent=2)

        log_separador(caractere="=", largura=80)
        logger.info(f"<verde>✅ PREDICT {self.NOME_ENGINE.upper()} COMPLETO - Resultados em: {predict_dir}</verde>")
        logger.info(f"<cinza>📊 Total: {uso_total['total_registros']} registros, "
                    f"{uso_total['input_tokens']} + {uso_total['output_tokens']} tokens "
                    f"({uso_total['tempo_total_s']:.1f}s)</cinza>")
        log_separador(caractere="=", largura=80)

    # ------------------------------------------------------------------
    # executar_predict_dataset — exportação do dataset completo
    # ------------------------------------------------------------------

    def executar_predict_dataset(self) -> None:
        """Exporta predições para TODOS os itens do dataset, sem filtragem por subset.

        Carrega todos os itens de entrada (dataframe com ``dataset_filtro``
        aplicado ou pasta de arquivos), gera predições e salva em
        ``{output_dir}/predict_dataset_{engine}/``.

        Registros já exportados são automaticamente ignorados (permite continuação).
        """
        logger.info("\n")
        log_separador(caractere="=", largura=80)
        logger.info(f"<azul>>> MODO PREDICT DATASET - {self.NOME_ENGINE.upper()} {self.ICONE}</azul>")
        log_separador(caractere="=", largura=80)

        _exibir_cabecalho_modelo(self.yaml_config)
        self.yaml_config.validar_max_seq_length()

        logger.info(f"<cinza>\n📋 Formato de saída: {self.yaml_config.formato_saida}</cinza>")

        filtro = getattr(self.yaml_config.curriculum_config.entrada, 'dataset_filtro', None)
        if filtro:
            logger.info(f"<cinza>🔍 dataset_filtro: {filtro}</cinza>")

        if not self._preparar_para_execucao():
            return

        # Diretório de predições do dataset
        sufixo = f"_{self.NOME_ENGINE}" if self.NOME_ENGINE != "hf" else ""
        predict_dir = os.path.join(self.yaml_config.modelo.saida, f"predict_dataset{sufixo}")
        os.makedirs(predict_dir, exist_ok=True)

        # Carrega TODOS os itens do dataset (sem filtragem por subset)
        logger.info("<azul>\n📂 Carregando dataset completo...</azul>")
        try:
            mensagens = self.yaml_config.dataset_manager.carregar_mensagens_dataset_completo()
            if not mensagens:
                logger.error("<vermelho>❌ Nenhum dado encontrado no dataset.</vermelho>")
                return
            logger.info(f"<cinza>   📊 {len(mensagens)} registros encontrados</cinza>")
        except Exception as e:
            logger.error(f"<vermelho>❌ Erro ao carregar dataset: {e}</vermelho>")
            return

        # Pré-check: verifica quantos já foram exportados
        total_skip = sum(1 for msg in mensagens
                         if _registro_ja_exportado(predict_dir, msg.get('id', '')))
        if total_skip == len(mensagens):
            logger.info(f"<verde>✅ Todas as {total_skip} predições já exportadas em: {predict_dir}</verde>")
            gerar_estatisticas_predicoes(predict_dir)
            return
        if total_skip > 0:
            logger.info(f"<cinza>   ⏭️ {total_skip} já exportados, {len(mensagens) - total_skip} pendentes</cinza>")

        # Garante modelo carregado
        modelo_proprio = not self._modelo_carregado
        if modelo_proprio:
            try:
                self.carregar_modelo()
            except Exception as e:
                logger.error(f"<vermelho>❌ Erro ao carregar modelo: {e}</vermelho>")
                return

        stats = {
            'input_tokens': 0, 'output_tokens': 0,
            'registros_ok': 0, 'registros_erro': 0, 'registros_skip': 0,
        }
        ini_total = time.time()

        try:
            # Filtra pendentes e prepara registros
            registros_pendentes = []
            for idx, msg in enumerate(mensagens):
                if not (isinstance(msg, dict) and 'messages' in msg):
                    stats['registros_erro'] += 1
                    continue
                messages = msg['messages']
                prompt_texto = self._extrair_prompt(messages)
                if not prompt_texto:
                    stats['registros_erro'] += 1
                    continue
                registro_id = msg.get('id', f'dataset_{idx:04d}')
                if _registro_ja_exportado(predict_dir, registro_id):
                    stats['registros_skip'] += 1
                    continue
                registros_pendentes.append({
                    'messages': messages,
                    'prompt_texto': prompt_texto,
                    'registro_id': registro_id,
                    'idx': idx,
                })

            if not registros_pendentes:
                skip_msg = f" ({stats['registros_skip']} já exportados)" if stats['registros_skip'] else ""
                logger.info(f"<cinza>   Nenhum registro pendente{skip_msg}</cinza>")
            else:
                total_pendentes = len(registros_pendentes)
                logger.info(f"<cinza>   📝 {total_pendentes} registros pendentes</cinza>")

                # Predição incremental — salva cada resultado imediatamente
                ultimo_log = time.time()
                ini_lote = time.time()
                for i, reg in enumerate(registros_pendentes):
                    resultado = self.predizer(reg['messages'], reg['prompt_texto'])

                    if 'erro' in resultado:
                        logger.error(f"<vermelho>   ❌ Erro no registro {reg['registro_id']}: {resultado['erro']}</vermelho>")
                        stats['registros_erro'] += 1
                    else:
                        resposta_texto = self._formatar_resposta(resultado.get('resposta', ''))
                        self._salvar_resultado(predict_dir, reg['registro_id'], resposta_texto, resultado)

                        usage = resultado.get('usage', {})
                        stats['input_tokens'] += usage.get('prompt_tokens', 0)
                        stats['output_tokens'] += usage.get('completion_tokens', 0)
                        stats['registros_ok'] += 1

                    # Log de progresso
                    agora = time.time()
                    if (i + 1) % _FREQUENCIA_PROGRESSO == 0 or (i + 1) == total_pendentes or (agora - ultimo_log) >= _INTERVALO_PROGRESSO_S:
                        eta = _formatar_eta(ini_lote, i + 1, total_pendentes)
                        logger.info(f"   Progresso: {i + 1}/{total_pendentes} ({100 * (i + 1) // total_pendentes}%) {eta}")
                        ultimo_log = agora

        finally:
            if modelo_proprio:
                self.liberar_modelo()

        tempo_total = time.time() - ini_total

        # Resumo
        resumo = {
            'data_geracao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'modelo_base': self.yaml_config.modelo.base,
            'modelo_saida': self.yaml_config.modelo.saida,
            'modelo_usado': self.nome_modelo_usado(),
            'engine': self.NOME_ENGINE,
            'formato_saida': self.yaml_config.formato_saida,
            'dataset_filtro': filtro,
            'total_registros': stats['registros_ok'],
            'registros_skip': stats['registros_skip'],
            'registros_erro': stats['registros_erro'],
            'input_tokens_total': stats['input_tokens'],
            'output_tokens_total': stats['output_tokens'],
            'tempo_total_s': round(tempo_total, 2),
        }
        resumo_file = os.path.join(predict_dir, "resumo_geral.json")
        with open(resumo_file, 'w', encoding='utf-8') as f:
            json.dump(resumo, f, ensure_ascii=False, indent=2)

        skip_msg = f", {stats['registros_skip']} já exportados" if stats['registros_skip'] else ""
        logger.info(f"<verde>   ✅ {stats['registros_ok']} predições salvas</verde>")
        gerar_estatisticas_predicoes(predict_dir)

        log_separador(caractere="=", largura=80)
        logger.info(f"<verde>✅ PREDICT DATASET {self.NOME_ENGINE.upper()} COMPLETO - Resultados em: {predict_dir}</verde>")
        logger.info(f"<cinza>📊 Total: {stats['registros_ok']} registros, "
                    f"{stats['input_tokens']} + {stats['output_tokens']} tokens "
                    f"({tempo_total:.1f}s){skip_msg}</cinza>")
        log_separador(caractere="=", largura=80)

    # ------------------------------------------------------------------
    # executar_modelo — inferência interativa
    # ------------------------------------------------------------------

    def executar_modelo(self, n_exemplos: int = 1) -> None:
        """Testa inferência interativa com N exemplos do dataset de treino.

        Carrega exemplos, exibe prompt / resposta esperada / resposta do modelo
        com estatísticas de tokens e tempo.

        Args:
            n_exemplos: Número de exemplos para testar.
        """
        logger.info("\n")
        log_separador(caractere="=", largura=80)
        logger.info(f"<azul>>> MODO MODELO - {self.NOME_ENGINE.upper()} {self.ICONE} "
                    f"TESTANDO INFERÊNCIA ({n_exemplos} exemplo(s))</azul>")
        log_separador(caractere="=", largura=80)

        _exibir_cabecalho_modelo(self.yaml_config)

        if not self._preparar_para_execucao():
            return

        # Carrega exemplos do dataset de treino
        logger.info("<azul>\n📂 Carregando exemplos do dataset de treino...</azul>")
        try:
            mensagens = self.yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo="treino")
            if not mensagens:
                logger.error("<vermelho>❌ Nenhum dado de treino encontrado.</vermelho>")
                return
        except Exception as e:
            logger.error(f"<vermelho>❌ Erro ao carregar dados de treino: {e}</vermelho>")
            return

        n_exemplos = min(n_exemplos, len(mensagens))
        logger.info(f"<cinza>   📊 {len(mensagens)} registros disponíveis, testando {n_exemplos}</cinza>")

        # Garante modelo carregado
        modelo_proprio = not self._modelo_carregado
        if modelo_proprio:
            try:
                self.carregar_modelo()
            except Exception as e:
                logger.error(f"<vermelho>❌ Erro ao carregar modelo: {e}</vermelho>")
                return

        resultados = []

        try:
            for i in range(n_exemplos):
                log_separador(caractere="-", largura=60)
                logger.info(f">> EXEMPLO {i + 1}/{n_exemplos}")
                log_separador(caractere="-", largura=60)

                msg = mensagens[i]
                if not (isinstance(msg, dict) and 'messages' in msg):
                    logger.warning(f"   ⚠️ Formato não reconhecido no registro {i}")
                    continue

                messages = msg['messages']
                prompt_texto = self._extrair_prompt(messages)
                resposta_esperada = self._extrair_resposta_esperada(messages)

                if not prompt_texto:
                    logger.warning(f"   ⚠️ Prompt vazio no registro {i}")
                    continue

                logger.info(f">> PROMPT:")
                logger.info(f"   {self._preview(prompt_texto)}")
                logger.info(f"\n>> RESPOSTA ESPERADA:")
                logger.info(f"   {self._preview(resposta_esperada)}")

                # Predição
                resultado = self.predizer(messages, prompt_texto)

                if 'erro' in resultado:
                    logger.error(f"<vermelho>   ❌ Erro: {resultado['erro']}</vermelho>")
                    continue

                resposta = resultado.get('resposta', '')
                if isinstance(resposta, dict):
                    resposta = json.dumps(resposta, ensure_ascii=False, indent=2)
                elif not isinstance(resposta, str):
                    resposta = str(resposta)

                usage = resultado.get('usage', {})
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                tempo_pred = resultado.get('tempo', 0)

                logger.info(f"\n>> RESPOSTA DO MODELO ({self.NOME_ENGINE} {self.ICONE}):")
                logger.info(f"   {self._preview(resposta)}")

                logger.info(f"\n>> ESTATÍSTICAS:")
                logger.info(f"   - Tokens do prompt: {input_tokens}")
                logger.info(f"   - Tokens da resposta: {output_tokens}")
                logger.info(f"   - Tempo de predição: {tempo_pred:.2f}s")
                if output_tokens > 0 and tempo_pred > 0:
                    logger.info(f"   - Velocidade: {output_tokens / tempo_pred:.1f} tokens/s")

                resultados.append({
                    'exemplo': i + 1,
                    'prompt_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'tempo_segundos': round(tempo_pred, 2),
                })

        finally:
            if modelo_proprio:
                self.liberar_modelo()

        # Resumo final
        if resultados:
            total_tokens = sum(r['output_tokens'] for r in resultados)
            total_tempo = sum(r['tempo_segundos'] for r in resultados)
            logger.info(f"\n📊 RESUMO {self.NOME_ENGINE.upper()}:")
            logger.info(f"   Exemplos processados: {len(resultados)}/{n_exemplos}")
            logger.info(f"   Tokens gerados: {total_tokens}")
            logger.info(f"   Tempo total: {total_tempo:.2f}s")
            if total_tempo > 0:
                logger.info(f"   Throughput médio: {total_tokens / total_tempo:.1f} tokens/s")

        # Métricas de memória
        if self.metricas_memoria:
            logger.info("\n📊 RESUMO DE USO DE MEMÓRIA:")
            ram = self.metricas_memoria.get('ram', {})
            gpu = self.metricas_memoria.get('gpu', {})
            if ram:
                logger.info(f"   RAM: máx={ram.get('max_gb', 0):.1f} GB, média={ram.get('media_gb', 0):.1f} GB")
            if gpu and gpu.get('num_gpus', 0) > 0:
                logger.info(f"   GPU: máx={gpu.get('max_gb', 0):.1f} GB, média={gpu.get('media_gb', 0):.1f} GB "
                            f"({gpu.get('num_gpus', 0)} GPU(s))")

        log_separador(caractere="=", largura=80)
        logger.info(f"<verde>✅ TESTE DE INFERÊNCIA {self.NOME_ENGINE.upper()} COMPLETO</verde>")
        log_separador(caractere="=", largura=80)

    # ------------------------------------------------------------------
    # Hook de preparação (override por engine)
    # ------------------------------------------------------------------

    def _preparar_para_execucao(self) -> bool:
        """Prepara o ambiente para execução (verificações, logs específicos).

        Retorna True se pode continuar, False se deve abortar.
        Subclasses podem sobrescrever para adicionar verificações extras.
        """
        return True


# ============================================================================
# UtilPredicaoHF — HuggingFace Transformers
# ============================================================================

class UtilPredicaoHF(UtilPredicao):
    """Predição via HuggingFace Transformers (LLMsTrainer)."""

    NOME_ENGINE = "hf"
    ICONE = "🤗"

    def _preparar_para_execucao(self) -> bool:
        return self._verificar_modelo_e_decidir_base()

    def carregar_modelo(self) -> None:
        from treinar_unsloth import LLMsTrainer

        logger.info("<azul>\n🔄 Carregando modelo via LLMsTrainer...</azul>")
        self._iniciar_monitor("memoria_predicao_hf")
        ini = time.time()
        self._modelo = LLMsTrainer(self.yaml_path, force_base=self.usar_base)
        logger.info(f"<verde>   ✅ Modelo carregado em {time.time() - ini:.1f}s</verde>")
        self._modelo_carregado = True

    def predizer(self, messages: list, prompt_texto: str) -> Dict[str, Any]:
        tempo_inicio = time.time()
        try:
            resultado = self._modelo.prompt(
                prompt_texto,
                temperatura=0.01,
                max_new_tokens=self.max_seq_length,
            )
            tempo = time.time() - tempo_inicio
            return {
                'resposta': resultado.get('texto', ''),
                'usage': {
                    'prompt_tokens': resultado.get('prompt_tokens', 0),
                    'completion_tokens': resultado.get('completion_tokens', 0),
                    'total_tokens': resultado.get('prompt_tokens', 0) + resultado.get('completion_tokens', 0),
                },
                'model': self.nome_modelo_usado(),
                'tempo': tempo,
            }
        except Exception as e:
            return {
                'erro': f'{type(e).__name__}: {e}',
                'model': self.nome_modelo_usado(),
                'tempo': time.time() - tempo_inicio,
            }


# ============================================================================
# UtilPredicaoVLLM — vLLM com batch nativo
# ============================================================================

class UtilPredicaoVLLM(UtilPredicao):
    """Predição via vLLM (inferência rápida com PagedAttention e batching)."""

    NOME_ENGINE = "vllm"
    ICONE = "🚀"

    def __init__(self, yaml_path: str, usar_base: bool = False):
        super().__init__(yaml_path, usar_base)
        self._engine = None
        self._vllm_tokenizer = None
        self._config = None

    def _preparar_para_execucao(self) -> bool:
        try:
            from treinar_vllm_inference import VLLM_AVAILABLE
            if not VLLM_AVAILABLE:
                logger.error("❌ vLLM não está instalado! Instale com: pip install vllm")
                return False
        except ImportError:
            logger.error("❌ Módulo treinar_vllm_inference não encontrado!")
            return False
        return self._verificar_modelo_e_decidir_base()

    def carregar_modelo(self) -> None:
        import torch
        from treinar_vllm_inference import VLLMInferenceEngine, get_recommended_config

        # Libera memória GPU de processos anteriores (ex: modelo HF/Unsloth
        # ainda carregado, ou caches CUDA residuais) antes de inicializar
        # o vLLM, que precisa de um bloco contíguo de VRAM.
        self._liberar_memoria_gpu()

        self._iniciar_monitor("memoria_predicao_vllm")

        modelo_base_path = self.yaml_config.modelo.base
        lora_adapter_path = None if self.usar_base else self.yaml_config.modelo.saida

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        logger.info(f"\n🎮 GPUs disponíveis: {num_gpus}")

        self._config = get_recommended_config(num_gpus=num_gpus, model_size="7B")
        # Contexto estimado a partir dos dados reais de tokens (CSVs do curriculum)
        ctx_info = self.yaml_config.estimar_contexto_predict()
        self._config.max_model_len = ctx_info["contexto"]
        self._vllm_max_new_tokens = ctx_info["max_new_tokens"]
        logger.info(f"⚙️  Tensor Parallel: {self._config.tensor_parallel_size} GPU(s)")
        logger.info(f"⚙️  GPU Memory Utilization: {self._config.gpu_memory_utilization * 100:.0f}%")
        logger.info(f"⚙️  Max Model Len: {self._config.max_model_len} ({ctx_info['fonte']})")
        logger.info(f"⚙️  Max New Tokens: {self._vllm_max_new_tokens}")

        logger.info("<azul>\n🚀 Inicializando vLLM...</azul>")
        ini = time.time()
        self._engine = VLLMInferenceEngine(
            model_path=modelo_base_path,
            config=self._config,
            lora_path=lora_adapter_path,
        )
        self._vllm_tokenizer = self._engine.llm.get_tokenizer()
        if getattr(self._vllm_tokenizer, "pad_token", None) is None:
            self._vllm_tokenizer.pad_token = self._vllm_tokenizer.eos_token
        logger.info(f"<verde>   ✅ vLLM inicializado em {time.time() - ini:.1f}s</verde>")
        self._modelo = self._engine  # para liberar_modelo
        self._modelo_carregado = True

    @staticmethod
    def _liberar_memoria_gpu():
        """Libera memória GPU antes de iniciar o vLLM.

        Executa gc.collect() + torch.cuda.empty_cache() + synchronize()
        para devolver ao sistema operacional a VRAM mantida em cache pelo
        PyTorch.  Loga a memória livre em cada GPU para diagnóstico.
        """
        try:
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("<cinza>🧹 Memória GPU liberada antes de inicializar vLLM:</cinza>")
                for i in range(torch.cuda.device_count()):
                    livre_gb = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)
                    total_gb = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
                    logger.info(
                        f"<cinza>   GPU {i}: {livre_gb:.1f} GB livres / {total_gb:.1f} GB total</cinza>"
                    )
        except Exception as e:
            logger.debug(f"Erro ao liberar memória GPU: {e}")

    def predizer(self, messages: list, prompt_texto: str) -> Dict[str, Any]:
        """Predição unitária via vLLM (usado pela inferência interativa)."""
        tempo_inicio = time.time()
        try:
            # Formata com chat template (exclui assistant = resposta esperada)
            chat_msgs = [m for m in messages if m.get('role') != 'assistant']
            formatted_prompt = self._vllm_tokenizer.apply_chat_template(
                chat_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self._vllm_tokenizer.encode(formatted_prompt)
            max_input_len = self._config.max_model_len - 256
            if len(prompt_ids) > max_input_len:
                prompt_ids = prompt_ids[:max_input_len]
                formatted_prompt = self._vllm_tokenizer.decode(prompt_ids, skip_special_tokens=False)

            tokens_para_gerar = min(self._vllm_max_new_tokens, self._config.max_model_len - len(prompt_ids))

            resultado = self._engine.generate_batch(
                prompts=[formatted_prompt],
                max_tokens=tokens_para_gerar,
                temperature=0.01,
                top_k=2,
                n=1,
            )
            tempo = time.time() - tempo_inicio

            if resultado:
                return {
                    'resposta': resultado[0]["output"],
                    'usage': {
                        'prompt_tokens': len(prompt_ids),
                        'completion_tokens': resultado[0]["tokens"],
                        'total_tokens': len(prompt_ids) + resultado[0]["tokens"],
                    },
                    'model': self.nome_modelo_usado(),
                    'tempo': tempo,
                }
            return {'erro': 'Sem resposta do vLLM', 'model': self.nome_modelo_usado(), 'tempo': tempo}
        except Exception as e:
            return {'erro': f'{type(e).__name__}: {e}', 'model': self.nome_modelo_usado(), 'tempo': time.time() - tempo_inicio}

    def predizer_lote(self, registros: List[dict], on_resultado=None) -> List[Dict[str, Any]]:
        """Predição em lote real via vLLM (batch nativo)."""
        prompts_batch = []
        metas_batch = []

        for reg in registros:
            # Formata com chat template (exclui assistant = resposta esperada)
            chat_msgs = [m for m in reg['messages'] if m.get('role') != 'assistant']
            formatted_prompt = self._vllm_tokenizer.apply_chat_template(
                chat_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self._vllm_tokenizer.encode(formatted_prompt)
            max_input_len = self._config.max_model_len - 256
            if len(prompt_ids) > max_input_len:
                prompt_ids = prompt_ids[:max_input_len]
                formatted_prompt = self._vllm_tokenizer.decode(prompt_ids, skip_special_tokens=False)
            prompts_batch.append(formatted_prompt)
            metas_batch.append({'input_tokens': len(prompt_ids)})

        if not prompts_batch:
            return []

        max_len_batch = max(m['input_tokens'] for m in metas_batch)
        tokens_para_gerar = min(self._vllm_max_new_tokens, self._config.max_model_len - max_len_batch)

        logger.info(f"<cinza>   🚀 vLLM batch: {len(prompts_batch)} prompts, max_tokens={tokens_para_gerar}</cinza>")

        tempo_inicio = time.time()
        try:
            resultados_vllm = self._engine.generate_batch(
                prompts=prompts_batch,
                max_tokens=tokens_para_gerar,
                temperature=0.01,
                top_k=2,
                n=1,
            )
        except Exception as e:
            erros = [{'erro': f'{type(e).__name__}: {e}', 'model': self.nome_modelo_usado(),
                      'tempo': 0} for _ in registros]
            if on_resultado:
                for reg, err in zip(registros, erros):
                    on_resultado(reg, err)
            return erros
        tempo_total = time.time() - tempo_inicio
        tempo_por_item = tempo_total / len(prompts_batch) if prompts_batch else 0

        resultados = []
        for i, res in enumerate(resultados_vllm):
            resultado = {
                'resposta': res["output"],
                'usage': {
                    'prompt_tokens': metas_batch[i]['input_tokens'],
                    'completion_tokens': res["tokens"],
                    'total_tokens': metas_batch[i]['input_tokens'] + res["tokens"],
                },
                'model': self.nome_modelo_usado(),
                'tempo': round(tempo_por_item, 3),
            }
            resultados.append(resultado)
            if on_resultado:
                on_resultado(registros[i], resultado)

        logger.info(f"<cinza>   ✅ Batch concluído em {tempo_total:.1f}s "
                    f"({sum(r['usage']['completion_tokens'] for r in resultados)} tokens gerados)</cinza>")
        return resultados


# ============================================================================
# UtilPredicaoUnsloth — Unsloth (FastLanguageModel.for_inference)
# ============================================================================

class UtilPredicaoUnsloth(UtilPredicao):
    """Predição via Unsloth com FastLanguageModel.for_inference (2x mais rápido)."""

    NOME_ENGINE = "unsloth"
    ICONE = "⚡"

    def __init__(self, yaml_path: str, usar_base: bool = False):
        super().__init__(yaml_path, usar_base)
        self._tokenizer = None
        # Contexto estimado a partir dos dados reais de tokens (CSVs do curriculum)
        ctx_info = self.yaml_config.estimar_contexto_predict()
        self._unsloth_context = ctx_info["contexto"]
        self._unsloth_max_new_tokens = ctx_info["max_new_tokens"]
        self._ctx_fonte = ctx_info["fonte"]

    def _preparar_para_execucao(self) -> bool:
        try:
            from unsloth import FastLanguageModel  # noqa: F401
        except ImportError:
            logger.error("❌ Módulo unsloth não encontrado! Instale com: pip install unsloth")
            return False
        return self._verificar_modelo_e_decidir_base()

    def carregar_modelo(self) -> None:
        import shutil as _shutil
        import torch
        from unsloth import FastLanguageModel

        # Garante compilador C para triton
        if not os.environ.get('CC'):
            cc_path = _shutil.which('gcc') or _shutil.which('cc')
            if cc_path:
                os.environ['CC'] = cc_path

        self._iniciar_monitor("memoria_predicao_unsloth")

        model_name = self.yaml_config.modelo.base if self.usar_base else self.yaml_config.modelo.saida
        logger.info(f"<azul>\n⚡ Carregando modelo com unsloth...</azul>")
        logger.info(f"<cinza>   Contexto Unsloth: {self._unsloth_context} ({self._ctx_fonte})</cinza>")

        ini = time.time()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self._unsloth_context,
            dtype=None,
            load_in_4bit=self.yaml_config.treinamento.nbits == 4,
        )
        FastLanguageModel.for_inference(model)
        logger.info(f"<verde>   ✅ Modelo carregado com unsloth em {time.time() - ini:.1f}s</verde>")
        logger.info(f"<cinza>   ⚡ FastLanguageModel.for_inference() ativado</cinza>")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._modelo = model
        self._tokenizer = tokenizer
        self._modelo_carregado = True

    def predizer(self, messages: list, prompt_texto: str) -> Dict[str, Any]:
        import torch

        tempo_inicio = time.time()
        try:
            chat_msgs = [{"role": "user", "content": prompt_texto}]
            inputs = self._tokenizer.apply_chat_template(
                chat_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self._modelo.device)

            # Trunca se exceder contexto (reservando max_new_tokens estimados para geração)
            max_input_len = max(128, self._unsloth_context - self._unsloth_max_new_tokens)
            if inputs.shape[1] > max_input_len:
                inputs = inputs[:, :max_input_len]

            input_length = inputs.shape[1]
            attention_mask = torch.ones_like(inputs)
            max_new_tokens = max(256, self._unsloth_context - input_length)

            with torch.inference_mode():
                outputs = self._modelo.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=0.01,
                    top_k=2,
                    do_sample=False,
                )
            tempo = time.time() - tempo_inicio

            resposta = self._tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            output_tokens = len(outputs[0]) - input_length

            return {
                'resposta': resposta,
                'usage': {
                    'prompt_tokens': input_length,
                    'completion_tokens': output_tokens,
                    'total_tokens': input_length + output_tokens,
                },
                'model': self.nome_modelo_usado(),
                'tempo': tempo,
            }
        except Exception as e:
            return {'erro': f'{type(e).__name__}: {e}', 'model': self.nome_modelo_usado(), 'tempo': time.time() - tempo_inicio}


# ============================================================================
# UtilPredicaoOllama — Ollama API local
# ============================================================================

class UtilPredicaoOllama(UtilPredicao):
    """Predição via Ollama (API local). Não requer modelo local em disco."""

    NOME_ENGINE = "ollama"
    ICONE = "🦙"
    REQUER_MODELO_LOCAL = False

    def __init__(self, yaml_path: str, usar_base: bool = False):
        super().__init__(yaml_path, usar_base)
        self._ollama_url = getattr(self.yaml_config.modelo, 'ollama_url', None) or None
        self._modelo_ollama = self._resolver_nome_modelo()
        self._timeout = UtilEnv.get_int('OLLAMA_TIMEOUT', 600)
        # Contexto estimado a partir dos dados reais de tokens
        ctx_info = self.yaml_config.estimar_contexto_predict()
        self._ollama_context = ctx_info["contexto"]
        self._ollama_max_tokens = ctx_info["max_new_tokens"]
        self._ctx_fonte = ctx_info["fonte"]

    def _resolver_nome_modelo(self) -> str:
        """Resolve nome do modelo Ollama com base em usar_base."""
        modelo_cfg = self.yaml_config.modelo
        if self.usar_base:
            nome = getattr(modelo_cfg, 'ollama_base', '') or ''
            if not nome:
                logger.warning("<amarelo>⚠️ modelo.ollama_base não configurado, usando modelo.ollama</amarelo>")
                nome = getattr(modelo_cfg, 'ollama', '') or ''
            return nome
        return getattr(modelo_cfg, 'ollama', '') or ''

    def _preparar_para_execucao(self) -> bool:
        if not self._modelo_ollama:
            chave = 'modelo.ollama_base' if self.usar_base else 'modelo.ollama'
            logger.error(f"<vermelho>❌ Chave '{chave}' não configurada no YAML</vermelho>")
            return False
        logger.info(f"<cinza>🦙 Modelo Ollama: {self._modelo_ollama}</cinza>")
        if self._ollama_url:
            logger.info(f"<cinza>🌐 API URL: {self._ollama_url}</cinza>")
        return True

    def carregar_modelo(self) -> None:
        """Verifica disponibilidade da API e do modelo no Ollama."""
        from util_openai import UtilOllama

        try:
            status = UtilOllama.status(api_url=self._ollama_url)
            if not status.get('api'):
                raise ConnectionError(
                    f"Ollama API não disponível em {self._ollama_url or 'http://localhost:11434/api'}")
            logger.info(f"<verde>✅ Ollama versão: {status.get('versao', '?')}</verde>")

            if self._modelo_ollama not in status.get('modelos', []):
                logger.warning(f"<amarelo>⚠️ Modelo '{self._modelo_ollama}' não encontrado localmente</amarelo>")
                logger.info(f"   Modelos disponíveis: {', '.join(status.get('modelos', []))}")
                if not _perguntar_confirmacao("Deseja continuar mesmo assim?", padrao=False):
                    raise RuntimeError("Operação cancelada pelo usuário")
        except (ConnectionError, RuntimeError):
            raise
        except Exception as e:
            raise ConnectionError(f"Erro ao verificar status do Ollama: {e}") from e

        self._modelo = True  # marca como "carregado" (API-driven)
        self._modelo_carregado = True

    def predizer(self, messages: list, prompt_texto: str) -> Dict[str, Any]:
        from util_openai import UtilOllama

        tempo_inicio = time.time()
        try:
            resultado = UtilOllama.chat_completion_padronizado(
                messages=messages,
                modelo=self._modelo_ollama,
                temperature=0.01,
                max_tokens=self._ollama_max_tokens,
                num_ctx=self._ollama_context,
                as_json=self.formato_json,
                raw=False,
                timeout=self._timeout,
                api_url=self._ollama_url,
            )
            tempo = time.time() - tempo_inicio

            if 'erro' in resultado:
                return {
                    'erro': resultado['erro'],
                    'model': resultado.get('model', self._modelo_ollama),
                    'tempo': tempo,
                }

            usage = resultado.get('usage', {})
            return {
                'resposta': resultado.get('resposta', ''),
                'usage': {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0),
                },
                'model': resultado.get('model', self._modelo_ollama),
                'tempo': tempo,
            }
        except Exception as e:
            return {
                'erro': f'{type(e).__name__}: {e}',
                'model': self._modelo_ollama,
                'tempo': time.time() - tempo_inicio,
            }

    def liberar_modelo(self) -> None:
        """Ollama é API — libera apenas referências."""
        if self._monitor:
            self.metricas_memoria = self._monitor.parar()
            self._monitor = None
        self._modelo = None
        self._modelo_carregado = False

    def nome_modelo_usado(self) -> str:
        return self._modelo_ollama


# ============================================================================
# CLI de teste
# ============================================================================

if __name__ == "__main__":
    import argparse
    from util import UtilEnv
    UtilEnv.carregar_env(pastas=['../../src', '../src'])  # Carrega variáveis de ambiente do .env

    parser = argparse.ArgumentParser(description="Testar predições com diferentes engines")
    parser.add_argument("yaml_path", help="Caminho para o arquivo YAML de configuração")
    parser.add_argument("--engine", choices=["hf", "vllm", "unsloth", "ollama"], default="hf",
                        help="Engine de predição (default: hf)")
    parser.add_argument("--predict", action="store_true", help="Executar predições (exportação em lote)")
    parser.add_argument("--modelo", type=int, metavar="N", help="Inferência interativa com N exemplos")
    parser.add_argument("--base", action="store_true", help="Usar modelo base (sem fine-tuning)")
    parser.add_argument("--subsets", nargs="+", help="Subsets para predict (ex: teste validacao treino)")
    args = parser.parse_args()

    ENGINES = {
        'hf': UtilPredicaoHF,
        'vllm': UtilPredicaoVLLM,
        'unsloth': UtilPredicaoUnsloth,
        'ollama': UtilPredicaoOllama,
    }

    engine_cls = ENGINES[args.engine]
    preditor = engine_cls(args.yaml_path, usar_base=args.base)

    if args.predict:
        preditor.executar_predict(subsets=args.subsets)
    elif args.modelo:
        preditor.executar_modelo(n_exemplos=args.modelo)
    else:
        parser.print_help()
        
    # Exemplos de uso:
    # python treinar_realizar_predicoes.py config.yaml --engine hf --predict
    # python ../../src/treinar_realizar_predicoes.py treinar_3060_curriculo.yaml --engine unsloth --modelo 1