#!/usr/bin/env python3

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Módulo de histórico de treinamento para o pacote treinar_unsloth.

Centraliza a criação e atualização dos arquivos de histórico na pasta
'treinamento' dentro do diretório de saída do modelo:

Arquivos (criados/recriados quando treinamento inicia pela primeira vez ou é resetado):
    - treinamento_exemplos.md  : Exemplo de treino e de validação
    - modelo_info.md           : Dados do modelo, arquitetura, template, origem
    - treinamento_config/      : Pasta com cópias versionadas do YAML de treinamento
    - treinamento_eventos.md   : Eventos de alto nível (arquivo atualizado ao longo das execuções)

Lógica de versionamento do YAML:
    - Cada cópia é nomeada como "nome_arquivo (v001).yaml", "nome_arquivo (v002).yaml", etc.
    - Uma nova cópia é criada somente se a data de modificação do YAML atual for 
      maior que a data da última cópia.
    - Se o treinamento reiniciar (reset), as cópias anteriores são apagadas.
"""

import os
import json
import shutil
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from treinar_unsloth_logging import get_logger

logger = get_logger(__name__)


class HistoricoTreinamento:
    """Gerencia os arquivos de histórico na pasta 'treinamento' do modelo."""

    def __init__(self, output_dir: str, yaml_path: str):
        """
        Args:
            output_dir: Diretório de saída do modelo (modelo.saida)
            yaml_path: Caminho absoluto para o YAML de treinamento
        """
        self.output_dir = output_dir
        self.yaml_path = os.path.abspath(yaml_path)
        self.treino_dir = os.path.join(output_dir, "treinamento")
        self.config_dir = os.path.join(self.treino_dir, "treinamento_config")

        # Caminhos dos arquivos
        self.exemplos_path = os.path.join(self.treino_dir, "treinamento_exemplos.md")
        self.modelo_info_path = os.path.join(self.treino_dir, "modelo_info.md")
        self.eventos_path = os.path.join(self.treino_dir, "treinamento_eventos.md")

    # -----------------------------------------------------------------------
    # Métodos Públicos
    # -----------------------------------------------------------------------

    def inicializar_novo_treinamento(
        self,
        yaml_config,
        model,
        tokenizer,
        train_ds,
        eval_ds=None,
    ) -> None:
        """Chamado quando o treinamento inicia pela primeira vez ou após reset.

        Apaga e recria todos os arquivos de histórico (incluindo cópias do YAML).

        Args:
            yaml_config: Instância de YamlTreinamento
            model: Modelo carregado (para extrair arquitetura)
            tokenizer: Tokenizer do modelo
            train_ds: Dataset de treino
            eval_ds: Dataset de validação (pode ser None)
        """
        logger.info("📁 Inicializando histórico de treinamento...")

        # Cria diretórios
        os.makedirs(self.treino_dir, exist_ok=True)

        # Limpa cópias anteriores do YAML (apenas arquivos .yaml/.yml)
        if os.path.isdir(self.config_dir):
            for arq in os.listdir(self.config_dir):
                if arq.lower().endswith(('.yaml', '.yml')):
                    os.remove(os.path.join(self.config_dir, arq))
        os.makedirs(self.config_dir, exist_ok=True)

        # Gera arquivos
        self._gerar_exemplos(yaml_config, train_ds, eval_ds)
        self._gerar_modelo_info(yaml_config, model, tokenizer)
        self._copiar_yaml_versionado(force_new=True)
        self._gerar_eventos_inicial()

        logger.info(f"   ✅ Histórico criado em: {self.treino_dir}")

    def registrar_evento(self, titulo: str, detalhes: str = "") -> None:
        """Adiciona um evento ao arquivo treinamento_eventos.md.

        Cada nova execução adiciona uma seção separada com data/hora e informações.

        Args:
            titulo: Título do evento (ex: "Treinamento iniciado", "Checkpoint encontrado")
            detalhes: Detalhes adicionais em texto livre
        """
        os.makedirs(self.treino_dir, exist_ok=True)

        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bloco = f"\n### [{agora}] {titulo}\n"
        if detalhes:
            bloco += f"\n{detalhes}\n"

        # Cria o arquivo se não existir
        if not os.path.isfile(self.eventos_path):
            self._gerar_eventos_inicial()

        with open(self.eventos_path, "a", encoding="utf-8") as f:
            f.write(bloco)

    def atualizar_yaml_se_necessario(self) -> Optional[str]:
        """Cria nova cópia versionada do YAML se ele foi modificado desde a última cópia.

        Returns:
            Caminho da nova cópia ou None se não houve mudança
        """
        return self._copiar_yaml_versionado(force_new=False)

    # -----------------------------------------------------------------------
    # Geração de arquivos
    # -----------------------------------------------------------------------

    def _gerar_exemplos(self, yaml_config, train_ds, eval_ds) -> None:
        """Gera treinamento_exemplos.md com um exemplo de treino e um de validação."""
        linhas = [
            "# Exemplos de Treinamento",
            f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Exemplo de treino
        linhas.append("## Exemplo de Treino")
        linhas.append("")
        if train_ds and len(train_ds) > 0:
            exemplo_treino = self._extrair_exemplo(train_ds[0])
            linhas.append("```json")
            linhas.append(json.dumps(exemplo_treino, indent=2, ensure_ascii=False))
            linhas.append("```")
        else:
            linhas.append("*Nenhum dado de treino disponível*")
        linhas.append("")

        # Exemplo de validação
        linhas.append("## Exemplo de Validação")
        linhas.append("")
        if eval_ds and len(eval_ds) > 0:
            exemplo_val = self._extrair_exemplo(eval_ds[0])
            linhas.append("```json")
            linhas.append(json.dumps(exemplo_val, indent=2, ensure_ascii=False))
            linhas.append("```")
        else:
            linhas.append("*Nenhum dado de validação disponível*")

        with open(self.exemplos_path, "w", encoding="utf-8") as f:
            f.write("\n".join(linhas))

        logger.info(f"   📄 {os.path.basename(self.exemplos_path)}")

    def _extrair_exemplo(self, registro) -> dict:
        """Extrai um exemplo do dataset no formato legível.

        Remove campos internos como input_ids, attention_mask, labels
        e mantém apenas as mensagens/prompt/completion.
        """
        if isinstance(registro, dict):
            # Filtra campos internos de tokenização
            campos_internos = {"input_ids", "attention_mask", "labels", "token_type_ids"}
            resultado = {}
            for k, v in registro.items():
                if k in campos_internos:
                    continue
                # Converte tipos numpy/tensor para tipos nativos Python
                try:
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                    elif hasattr(v, "item"):
                        v = v.item()
                except Exception:
                    v = str(v)
                resultado[k] = v
            return resultado
        return {"dados": str(registro)[:2000]}

    def _gerar_modelo_info(self, yaml_config, model, tokenizer) -> None:
        """Gera modelo_info.md com dados do modelo, arquitetura, template e origem."""
        linhas = [
            "# Informações do Modelo",
            f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Origem
        linhas.append("## Origem")
        linhas.append(f"- **Modelo Base:** `{yaml_config.modelo.base}`")
        linhas.append(f"- **Diretório de Saída:** `{yaml_config.modelo.saida}`")
        linhas.append(f"- **Modo:** curriculum")
        linhas.append("")

        # Tipo e arquitetura
        model_type = type(model).__name__
        is_peft = hasattr(model, "peft_config") or hasattr(model, "base_model")
        linhas.append("## Tipo do Modelo")
        linhas.append(f"- **Classe:** `{model_type}`")
        linhas.append(f"- **É modelo PEFT:** {is_peft}")
        linhas.append("")

        # LoRA
        if is_peft and hasattr(model, "peft_config"):
            linhas.append("## Configuração LoRA")
            for adapter_name, config in model.peft_config.items():
                linhas.append(f"### Adaptador: {adapter_name}")
                linhas.append(f"- **r:** {getattr(config, 'r', 'N/A')}")
                linhas.append(f"- **alpha:** {getattr(config, 'lora_alpha', 'N/A')}")
                linhas.append(f"- **dropout:** {getattr(config, 'lora_dropout', 'N/A')}")
                modules = getattr(config, "target_modules", [])
                if isinstance(modules, str) and modules.startswith("(?:"):
                    modules_str = "Unsloth Default (Todos os módulos lineares)"
                else:
                    modules_str = str(modules)
                linhas.append(f"- **Target Modules:** {modules_str}")
            linhas.append("")

        # Parâmetros (estado inicial — LoRA por padrão)
        try:
            import torch
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            quantizados = sum(p.numel() for p in model.parameters()
                              if p.dtype not in (torch.float32, torch.float16, torch.bfloat16))
            pct = (trainable / total * 100) if total > 0 else 0
            linhas.append("## Parâmetros (estado inicial)")
            linhas.append(f"- **Treináveis:** {trainable:,}")
            linhas.append(f"- **Totais:** {total:,}")
            if quantizados > 0:
                nbits = yaml_config.treinamento.nbits
                linhas.append(f"- **Quantizados ({nbits}-bit, congelados):** {quantizados:,}")
                linhas.append(f"- **Float (desbloqueáveis):** {total - quantizados:,}")
            linhas.append(f"- **Percentual treinável:** {pct:.4f}%")
            linhas.append("")
        except Exception:
            pass

        # Etapas do curriculum (full vs lora)
        etapas = yaml_config.curriculum
        if etapas:
            linhas.append("## Etapas do Curriculum")
            linhas.append("")
            linhas.append("| # | Alias | Tipo | Epochs | Max Seq |")
            linhas.append("|---|-------|------|--------|---------|")
            for i, etapa in enumerate(etapas):
                ep = etapa.pace_epochs if etapa.pace_epochs > 0 else yaml_config.treinamento.epochs
                msl = etapa.max_seq_length if etapa.max_seq_length > 0 else yaml_config.treinamento.max_seq_length
                linhas.append(f"| {i} | {etapa.alias} | {etapa.tipo or '(predict)'} | {ep} | {msl} |")
            linhas.append("")

        # Template do tokenizer
        linhas.append("## Chat Template")
        linhas.append("")
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            # Limita o template para não ficar imenso
            template_resumido = chat_template
            if len(template_resumido) > 2000:
                template_resumido = template_resumido[:2000] + "\n... (truncado)"
            linhas.append("```jinja2")
            linhas.append(template_resumido)
            linhas.append("```")
        else:
            linhas.append("*Chat template não disponível*")
        linhas.append("")

        # Arquitetura do modelo
        linhas.append("## Arquitetura")
        linhas.append("")
        try:
            model_str = str(model)
            linhas.append("```")
            linhas.append(model_str)
            linhas.append("```")
        except Exception:
            linhas.append("*Não foi possível obter a arquitetura do modelo*")

        with open(self.modelo_info_path, "w", encoding="utf-8") as f:
            f.write("\n".join(linhas))

        logger.info(f"   📄 {os.path.basename(self.modelo_info_path)}")

    def _gerar_eventos_inicial(self) -> None:
        """Cria o arquivo treinamento_eventos.md com cabeçalho inicial."""
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conteudo = [
            "# Eventos de Treinamento",
            "",
            f"---",
            f"## Sessão iniciada em {agora}",
            "",
            f"### [{agora}] TREINAMENTO INICIADO",
            "",
            f"Início de novo treinamento (reset ou primeira execução)",
            "",
        ]

        os.makedirs(self.treino_dir, exist_ok=True)
        with open(self.eventos_path, "w", encoding="utf-8") as f:
            f.write("\n".join(conteudo))

        logger.info(f"   📄 {os.path.basename(self.eventos_path)}")

    # -----------------------------------------------------------------------
    # Versionamento do YAML
    # -----------------------------------------------------------------------

    def _copiar_yaml_versionado(self, force_new: bool = False) -> Optional[str]:
        """Cria cópia versionada do YAML na pasta treinamento_config/.

        Regras:
            - Nomeação: nome_arquivo (v001).yaml, (v002).yaml, etc.
            - Nova cópia apenas se data de modificação do YAML > data da última cópia
            - Se force_new=True, sempre cria (usado no reset/início)

        Args:
            force_new: Se True, ignora verificação de data e força criação

        Returns:
            Caminho da cópia criada ou None se não precisou
        """
        if not os.path.isfile(self.yaml_path):
            logger.warning(f"YAML não encontrado: {self.yaml_path}")
            return None

        os.makedirs(self.config_dir, exist_ok=True)

        yaml_basename = os.path.splitext(os.path.basename(self.yaml_path))[0]
        yaml_ext = os.path.splitext(os.path.basename(self.yaml_path))[1] or ".yaml"

        # Encontra cópias existentes e determina próxima versão
        copias_existentes = self._listar_copias_yaml(yaml_basename, yaml_ext)

        if not force_new and copias_existentes:
            # Compara conteúdo do YAML atual com a última cópia
            ultima_copia = copias_existentes[-1]
            ultima_copia_path = os.path.join(self.config_dir, ultima_copia["arquivo"])

            try:
                with open(self.yaml_path, "r", encoding="utf-8") as f:
                    conteudo_atual = f.read()
                with open(ultima_copia_path, "r", encoding="utf-8") as f:
                    conteudo_copia = f.read()
                if conteudo_atual == conteudo_copia:
                    return None
            except Exception:
                pass  # em caso de erro de leitura, cria nova cópia

        # Determina próxima versão
        if copias_existentes:
            proxima_versao = copias_existentes[-1]["versao"] + 1
        else:
            proxima_versao = 1

        # Cria a cópia
        nome_copia = f"{yaml_basename} (v{proxima_versao:03d}){yaml_ext}"
        destino = os.path.join(self.config_dir, nome_copia)
        shutil.copy2(self.yaml_path, destino)

        logger.info(f"   📋 YAML copiado: {nome_copia}")
        return destino

    def _listar_copias_yaml(self, basename: str, ext: str) -> List[Dict[str, Any]]:
        """Lista cópias versionadas do YAML, ordenadas por versão.

        Args:
            basename: Nome base do arquivo YAML (sem extensão)
            ext: Extensão do arquivo (ex: .yaml)

        Returns:
            Lista de dicts com {arquivo, versao}, ordenada por versão
        """
        if not os.path.isdir(self.config_dir):
            return []

        # Padrão para encontrar cópias: nome_arquivo (v001).yaml
        # Escapamos o basename pois pode conter caracteres especiais em regex
        pattern = re.compile(
            re.escape(basename) + r"\s*\(v(\d{3,})\)" + re.escape(ext) + r"$"
        )

        copias = []
        for arquivo in os.listdir(self.config_dir):
            match = pattern.match(arquivo)
            if match:
                versao = int(match.group(1))
                copias.append({"arquivo": arquivo, "versao": versao})

        copias.sort(key=lambda x: x["versao"])
        return copias

    # -----------------------------------------------------------------------
    # Métodos auxiliares para eventos comuns
    # -----------------------------------------------------------------------

    def evento_checkpoint_encontrado(self, checkpoint_path: str, step: int) -> None:
        """Registra evento de checkpoint encontrado."""
        self.registrar_evento(
            "CHECKPOINT ENCONTRADO",
            f"- **Caminho:** `{checkpoint_path}`\n- **Step:** {step}",
        )

    def evento_checkpoint_retomado(self, sucesso: bool, erro: str = "") -> None:
        """Registra evento de retomada de checkpoint."""
        if sucesso:
            self.registrar_evento("CHECKPOINT RETOMADO COM SUCESSO")
        else:
            self.registrar_evento(
                "FALHA AO RETOMAR CHECKPOINT",
                f"- **Erro:** {erro}\n- Treinamento reiniciado do início",
            )

    def evento_etapa_curriculum(
        self, step_index: int, alias: str, tipo: str, **extras
    ) -> None:
        """Registra início de uma etapa do curriculum."""
        detalhes = f"- **Etapa:** {step_index}\n- **Alias:** {alias}\n- **Tipo:** {tipo}"
        for k, v in extras.items():
            detalhes += f"\n- **{k}:** {v}"
        self.registrar_evento(f"ETAPA CURRICULUM: {alias}", detalhes)

    def evento_treinamento_concluido(self, stats: dict = None, alias: str = "", tipo: str = "",
                                     instancias_acumuladas: int = 0, tokens_acumulados: int = 0,
                                     etapa_config=None) -> None:
        """Registra conclusão do treinamento com motivo de parada e acumulados."""
        detalhes = ""
        if alias:
            detalhes += f"- **Etapa:** {alias}\n"
        if tipo:
            detalhes += f"- **Modo:** {tipo}\n"

        # Motivo da conclusão
        if stats and etapa_config:
            motivo = self._motivo_conclusao(stats, etapa_config)
            if motivo:
                detalhes += f"- **Conclusão:** {motivo}\n"

        if stats:
            if "training_loss" in stats:
                detalhes += f"- **Training Loss:** {stats['training_loss']:.4f}\n"
            if "eval_loss" in stats:
                detalhes += f"- **Eval Loss:** {stats['eval_loss']:.4f}\n"
            if "train_runtime" in stats:
                runtime = stats["train_runtime"]
                h, m, s = int(runtime // 3600), int((runtime % 3600) // 60), int(runtime % 60)
                detalhes += f"- **Tempo:** {h}h {m}m {s}s\n"
            if "global_step" in stats:
                detalhes += f"- **Steps:** {stats['global_step']}\n"

        # Acumulados (tokens e instâncias)
        if tokens_acumulados > 0:
            if tokens_acumulados >= 1_000_000:
                tok_fmt = f"{tokens_acumulados/1_000_000:.1f}M"
            elif tokens_acumulados >= 1_000:
                tok_fmt = f"{tokens_acumulados/1_000:.1f}K"
            else:
                tok_fmt = str(tokens_acumulados)
            detalhes += f"- **Tokens acumulados:** {tok_fmt}\n"
        if instancias_acumuladas > 0:
            detalhes += f"- **Instâncias acumuladas:** {instancias_acumuladas:,}\n".replace(",", ".")

        self.registrar_evento("TREINAMENTO CONCLUÍDO", detalhes)

    @staticmethod
    def _motivo_conclusao(stats: dict, etapa_config) -> str:
        """Determina motivo de conclusão da etapa baseado nos stats e config."""
        pace_loss = getattr(etapa_config, 'pace_loss', 0)
        pace_epochs = getattr(etapa_config, 'pace_epochs', 0)
        pace_epochs_max = getattr(etapa_config, 'pace_epochs_max', 0)

        if stats.get("pace_loss_atingido"):
            epoch = stats.get("pace_loss_epoch", "?")
            loss_val = stats.get("pace_loss_valor", 0)
            return f"🎯 pace_loss atingido (eval_loss={loss_val:.4f} < {pace_loss}) na época {epoch}"

        if pace_loss > 0 and not stats.get("pace_loss_atingido", True):
            if pace_epochs_max > 0:
                return f"⏱️ máximo de épocas ({pace_epochs_max}) — pace_loss={pace_loss} não atingido"
            return f"⏱️ épocas completadas ({pace_epochs}) — pace_loss={pace_loss} não atingido"

        if pace_epochs > 0:
            return f"✅ {pace_epochs} épocas completadas"

        return ""

    def evento_modelo_salvo(self, output_dir: str, arquivos: list = None) -> None:
        """Registra salvamento do modelo."""
        detalhes = f"- **Diretório:** `{output_dir}`"
        if arquivos:
            detalhes += f"\n- **Arquivos:** {', '.join(arquivos)}"
        self.registrar_evento("MODELO SALVO", detalhes)

    def evento_geracao_estatisticas(self, detalhes_extra: str = "") -> None:
        """Registra geração de estatísticas/relatório."""
        self.registrar_evento("ESTATÍSTICAS GERADAS", detalhes_extra)

    def evento_reinicio(self, motivo: str = "") -> None:
        """Registra reinício de sessão de treinamento (continuação sem reset).

        Adiciona um novo separador de sessão ao arquivo de eventos.
        """
        os.makedirs(self.treino_dir, exist_ok=True)
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        bloco = f"\n---\n## Sessão retomada em {agora}\n"
        if motivo:
            bloco += f"\n{motivo}\n"

        # Cria o arquivo se não existir
        if not os.path.isfile(self.eventos_path):
            self._gerar_eventos_inicial()
        else:
            with open(self.eventos_path, "a", encoding="utf-8") as f:
                f.write(bloco)

        # Verifica se YAML mudou e faz cópia versionada
        self.atualizar_yaml_se_necessario()
