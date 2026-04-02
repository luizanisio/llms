# Autor: Luiz Anísio
# Fonte: https://github.com/luizanisio/llms/tree/main/src

import os
import json
from typing import Dict, Any, Optional
from treinar_unsloth_util import YamlTreinamento

# ---------------------------------------------------------------------------
# Helpers para formatação do relatório
# ---------------------------------------------------------------------------

def _formatar_mem_gpu(before: dict, after: dict) -> list:
    """Formata a seção de memória GPU comparando antes/depois.
    
    Dados fixos aparecem uma vez; dados que mudaram aparecem no formato 'antes → depois'.
    """
    linhas = []

    def _flatten(d, prefix=""):
        """Achata dict aninhado em pares (chave, valor)."""
        items = []
        for k, v in d.items():
            chave = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                items.extend(_flatten(v, chave))
            elif isinstance(v, list):
                for i, elem in enumerate(v):
                    if isinstance(elem, dict):
                        items.extend(_flatten(elem, f"{chave}[{i}]"))
                    else:
                        items.append((f"{chave}[{i}]", elem))
            else:
                items.append((chave, v))
        return items

    flat_before = dict(_flatten(before))
    flat_after = dict(_flatten(after)) if after else {}

    todas_chaves = list(dict.fromkeys(list(flat_before.keys()) + list(flat_after.keys())))

    fixos = []
    mudaram = []

    for chave in todas_chaves:
        val_b = flat_before.get(chave)
        val_a = flat_after.get(chave)
        if flat_after and val_b != val_a and val_a is not None and val_b is not None:
            mudaram.append((chave, val_b, val_a))
        else:
            fixos.append((chave, val_b if val_b is not None else val_a))

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:g}"
        if isinstance(v, bool):
            return "✅" if v else "❌"
        return str(v)

    if fixos:
        linhas.append("**Dados fixos:**")
        linhas.append("")
        linhas.append("| Recurso | Valor |")
        linhas.append("|---|---|")
        for chave, val in fixos:
            linhas.append(f"| {chave} | {_fmt(val)} |")

    if mudaram:
        linhas.append("")
        linhas.append("**Variação (antes → depois):**")
        linhas.append("")
        linhas.append("| Recurso | Antes | Depois | Δ |")
        linhas.append("|---|---|---|---|")
        for chave, val_b, val_a in mudaram:
            delta = ""
            if isinstance(val_b, (int, float)) and isinstance(val_a, (int, float)):
                diff = val_a - val_b
                sinal = "+" if diff > 0 else ""
                if isinstance(diff, float):
                    delta = f"{sinal}{diff:.3f}"
                else:
                    delta = f"{sinal}{diff}"
            linhas.append(f"| {chave} | {_fmt(val_b)} | {_fmt(val_a)} | {delta} |")

    return linhas


# ---------------------------------------------------------------------------
# Classe GeradorRelatorio
# ---------------------------------------------------------------------------

class GeradorRelatorio:
    """
    Gera relatórios em Markdown sobre o treinamento, salvando na pasta 
    'treinamento' dentro do diretório de saída do modelo.
    """
    
    
    def __init__(self, yaml_config: YamlTreinamento):
        self.yaml_config = yaml_config
        self.output_dir = yaml_config.modelo.saida
        self.report_dir = os.path.join(self.output_dir, "treinamento")
        self.report_file = os.path.join(self.report_dir, "relatorio_treinamento.md")
        

        """
        Gera e salva o relatório.
        
        Args:
            dataset_stats: Estatísticas do dataset (contagem, tokens, etc)
            train_stats: Métricas finais do treinamento (loss, tempo, etc)
            hardware_info: Informações da máquina (CPUs, Memória, GPU)
            print_only: Se True, apenas imprime no console e não salva arquivo.
        """
        # ... implementation ...


    def collect_token_stats(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coleta estatísticas de tokens para os datasets fornecidos usando UtilTikToken.
        
        Args:
            datasets: Dict com keys 'train', 'eval', 'test' contendo os datasets.
            
        Returns:
            Dict com estatísticas de tokens para cada split.
        """
        from util_tiktoken import UtilTikToken
        
        try:
            tokenizer = UtilTikToken()
        except ImportError:
            print("⚠️ TikToken não disponível, estatísticas de token serão ignoradas.")
            return {}
            
        stats = {}
        
        # Nome do modelo para o tiktoken (usa o base model name)
        model_name = self.yaml_config.modelo.base
        
        # Carrega template se existir para estatísticas diferenciadas
        template_tokens_base = 0
        has_template = False
        try:
            cfg_entrada = self.yaml_config.curriculum_config.entrada
            if cfg_entrada.prompt_template and os.path.isfile(cfg_entrada.prompt_template):
                with open(cfg_entrada.prompt_template, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                
                # Remove a tag para contar os tokens "fixos" do template
                # Isso é uma aproximação, pois a tokenização pode mudar nas fronteiras,
                # mas serve bem para estatística.
                tag = cfg_entrada.tag_texto or '<<--TEXTO-->>'
                template_base = template_content.replace(tag, '')
                template_tokens_base = tokenizer.contar_tokens(template_base, modelo=model_name)['qtd_tokens']
                has_template = True
                print(f"   📄 Template base carregado: {template_tokens_base} tokens (aprox)")
        except Exception as e:
            print(f"⚠️ Erro ao processar template para estatísticas: {e}")

        # Mapeamento de nomes de split para nomes de exibição
        splits = {
            'train': 'Treino',
            'eval': 'Validação',
            'test': 'Teste'
        }
        
        print(f"\n📊 Calculando estatísticas de tokens (usando tiktoken)...")
        
        for split_key, display_name in splits.items():
            if split_key not in datasets or datasets[split_key] is None:
                continue
                
            dataset = datasets[split_key]
            if len(dataset) == 0:
                continue
                
            print(f"   Processando {display_name} ({len(dataset)} registros)...")
            
            total_tokens = 0
            total_unique_tokens = 0
            tokens_list = []
            
            # Listas para estatísticas do CONTEÚDO (texto injetado)
            doc_tokens_list = []
            
            # Itera sobre os exemplos para contar tokens
            for item in dataset:
                texto_full = ""
                user_content = ""
                
                if 'messages' in item:
                    # Formato chat: messages list maps to string for total stats
                    # Tenta identificar o user prompt para estatística do documento
                    msgs = item['messages']
                    texto_full = json.dumps(msgs, ensure_ascii=False)
                    
                    # Assume primeira mensagem ou busca role='user'
                    for m in msgs:
                        if m.get('role') == 'user':
                            user_content = m.get('content', '')
                            break
                            
                elif 'prompt' in item and 'completion' in item:
                    # Formato legacy
                    user_content = item['prompt']
                    texto_full = f"{item['prompt']}\n{item['completion']}"
                elif 'text' in item:
                    texto_full = item['text']
                    # Difícil extrair o prompt base apenas do text formatado sem parser complexo
                else:
                    texto_full = str(item)
                    
                # 1. Estatística Geral (Exemplo completo)
                count = tokenizer.contar_tokens(texto_full, modelo=model_name)
                t_count = count['qtd_tokens']
                t_unique = count['qtd_tokens_unicos']
                
                tokens_list.append(t_count)
                total_tokens += t_count
                total_unique_tokens += t_unique
                
                # 2. Estatística do Texto Injetado (Doc Tokens)
                if has_template and user_content:
                    # Conta tokens do prompt do usuário
                    user_count = tokenizer.contar_tokens(user_content, modelo=model_name)['qtd_tokens']
                    # Subtrai o template base
                    doc_tokens = max(0, user_count - template_tokens_base)
                    doc_tokens_list.append(doc_tokens)
                
            if tokens_list:
                stats[split_key] = {
                    'count': len(dataset),
                    'total_tokens': total_tokens,
                    'avg_tokens': round(total_tokens / len(dataset), 1),
                    'total_unique_tokens': total_unique_tokens,
                    'min_tokens': min(tokens_list),
                    'max_tokens': max(tokens_list)
                }
                
                # Adiciona estatísticas do texto injetado se disponível
                if doc_tokens_list:
                    stats[split_key]['doc_stats'] = {
                         'template_tokens': template_tokens_base,
                         'avg_doc_tokens': round(sum(doc_tokens_list) / len(doc_tokens_list), 1),
                         'min_doc_tokens': min(doc_tokens_list),
                         'max_doc_tokens': max(doc_tokens_list)
                    }
                
        return stats
    
    def gerar_relatorio(self, 
                        dataset_stats: Dict[str, Any] = None, 
                        train_stats: Dict[str, Any] = None,
                        hardware_info: Dict[str, Any] = None,
                        print_only: bool = False) -> str:
        """
        Gera e salva o relatório.
        
        Args:
            dataset_stats: Estatísticas do dataset (contagem, tokens, etc)
            train_stats: Métricas finais do treinamento (loss, tempo, etc)
            hardware_info: Informações da máquina (CPUs, Memória, GPU)
            print_only: Se True, apenas imprime no console e não salva arquivo.
        """
        from datetime import datetime
        
        cfg = self.yaml_config
        conteudo = []
        conteudo.append(f"# Relatório de Treinamento LLM")
        conteudo.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        conteudo.append(f"**Modelo Base:** `{cfg.modelo.base}`")
        conteudo.append(f"**Diretório de Saída:** `{self.output_dir}`")
        conteudo.append(f"**Modo:** curriculum")
        
        # --- 1. Configuração ---
        # Valores originais do YAML (antes de overrides de curriculum/batch_auto)
        treino_raw = cfg._raw_config.get("treinamento", {})
        yaml_epochs = int(treino_raw.get("epochs") or treino_raw.get("num_train_epochs") or 1)
        yaml_lr = float(treino_raw.get("learning_rate", 2e-4))
        yaml_msl = int(treino_raw.get("max_seq_length", 4096))
        # batch_size pode ser int (manual) ou dict (auto: {efetivo, batch_size})
        yaml_bs_raw = treino_raw.get("batch_size", 2)
        if isinstance(yaml_bs_raw, dict):
            yaml_bs = int(yaml_bs_raw.get("batch_size", 2))
        else:
            yaml_bs = int(yaml_bs_raw)
        yaml_grad = int(treino_raw.get("grad_batch_size", 5))
        
        # Valores efetivos (após batch_size_auto e overrides da 1ª etapa curriculum)
        bs_efetivo = cfg.treinamento.batch_size
        grad_efetivo = cfg.treinamento.grad_batch_size
        
        # Cálculo do batch efetivo total
        import torch
        n_gpus = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
        batch_total = bs_efetivo * grad_efetivo * n_gpus
        
        conteudo.append("\n## 1. Configuração do Treinamento")
        
        # Detecta tipos de treinamento presentes no curriculum
        etapas_todas = cfg.curriculum
        tipos_presentes = set(e.tipo for e in etapas_todas if e.tipo)
        tem_full = "full" in tipos_presentes
        tem_lora = "lora" in tipos_presentes
        
        # --- 1.1 Parâmetros Gerais (compartilhados entre todas as etapas) ---
        conteudo.append("\n### Parâmetros Gerais")
        conteudo.append("")
        conteudo.append("| Parâmetro | Valor |")
        conteudo.append("|---|---|")
        
        # Batch size: mostra efetivo e, se auto-calculado, indica
        batch_auto = getattr(cfg, 'batch_size_auto', None)
        if batch_auto and batch_auto.efetivo > 0:
            conteudo.append(f"| Batch size (por GPU) | {bs_efetivo} |")
            conteudo.append(f"| Grad accumulation | {grad_efetivo} (auto: efetivo_alvo={batch_auto.efetivo}) |")
            conteudo.append(f"| **Batch efetivo total** | **{batch_total}** ({bs_efetivo} × {grad_efetivo} × {n_gpus} GPU) |")
            if batch_total != batch_auto.efetivo:
                conteudo.append(f"| ⚠️ Arredondamento | alvo={batch_auto.efetivo}, real={batch_total} |")
        else:
            conteudo.append(f"| Batch size (por GPU) | {bs_efetivo} |")
            conteudo.append(f"| Grad accumulation | {grad_efetivo} |")
            conteudo.append(f"| **Batch efetivo total** | **{batch_total}** ({bs_efetivo} × {grad_efetivo} × {n_gpus} GPU) |")
        
        conteudo.append(f"| Learning rate | {cfg.treinamento.learning_rate} |")
        conteudo.append(f"| Train on responses only | {cfg.treinamento.train_on_responses_only} |")
        conteudo.append(f"| Warmup steps | {cfg.treinamento.warmup_steps} |")
        conteudo.append(f"| Weight decay | {cfg.treinamento.weight_decay} |")
        conteudo.append(f"| Otimizador | {cfg.treinamento.optim} |")
        conteudo.append(f"| LR scheduler | {cfg.treinamento.lr_scheduler_type} |")
        conteudo.append(f"| Quantização (nbits) | {cfg.treinamento.nbits} |")
        conteudo.append(f"| Seed | {cfg.treinamento.seed} |")
        
        # --- Otimizações de memória GPU ---
        conteudo.append(f"| **Flash Attention 2** | {'✅ ativo' if cfg.treinamento.flash_attention_2 else '❌ desativado (usando SDPA)'} |")
        conteudo.append(f"| **Liger Kernel** | {'✅ ativo (fused CE + RoPE + RMSNorm)' if cfg.treinamento.liger_kernel else '❌ desativado'} |")
        
        # --- Modo de treinamento usado ---
        if tem_full and tem_lora:
            conteudo.append(f"| **Modos de treinamento** | Full + LoRA (por etapa — ver curriculum abaixo) |")
        elif tem_full:
            conteudo.append(f"| **Modo de treinamento** | Full (todos os parâmetros float treináveis) |")
        elif tem_lora:
            conteudo.append(f"| **Modo de treinamento** | LoRA (apenas adaptadores treináveis) |")
        
        # --- 1.2 Configuração LoRA (se ao menos uma etapa usa LoRA) ---
        if tem_lora:
            conteudo.append("\n### Configuração LoRA")
            conteudo.append(f"Aplicada nas etapas: {', '.join(e.alias for e in etapas_todas if e.tipo == 'lora')}")
            conteudo.append("")
            conteudo.append("| Parâmetro | Valor |")
            conteudo.append("|---|---|")
            conteudo.append(f"| r (rank) | {cfg.lora.r} |")
            conteudo.append(f"| alpha | {cfg.lora.alpha} |")
            conteudo.append(f"| dropout | {cfg.lora.dropout} |")
            conteudo.append(f"| Target modules | {', '.join(cfg.lora.target_modules)} |")
            conteudo.append(f"| Ratio (alpha/r) | {cfg.lora.alpha / cfg.lora.r:.1f} |")
        
        # --- 1.3 Modo Full (se ao menos uma etapa usa Full) ---
        if tem_full:
            conteudo.append("\n### Modo Full Fine-Tuning")
            conteudo.append(f"Aplicado nas etapas: {', '.join(e.alias for e in etapas_todas if e.tipo == 'full')}")
            conteudo.append("")
            nbits = cfg.treinamento.nbits
            if nbits < 16:
                conteudo.append(
                    f"> ⚠️ Com quantização {nbits}-bit, apenas parâmetros float (embeddings, "
                    f"layer norms, adaptadores LoRA) são treináveis no modo Full. "
                    f"Pesos quantizados permanecem congelados. "
                    f"Para full fine-tuning de 100% dos pesos, use `nbits: 16`."
                )
            else:
                conteudo.append("> Todos os parâmetros do modelo são treináveis (sem quantização).")
        
        # --- Seção de dados: curriculum ---
        conteudo.append("\n### Dados (Curriculum)")
        cc = cfg.curriculum_config
        if cc.entrada.pasta:
            conteudo.append(f"- **Entrada (pasta):** `{cc.entrada.pasta}`")
        elif cc.entrada.dataframe:
            conteudo.append(f"- **Entrada (dataframe):** `{cc.entrada.dataframe}` col=`{cc.entrada.dataframe_col}`")
        if cc.saida.pasta:
            conteudo.append(f"- **Saída (pasta):** `{cc.saida.pasta}`")
        elif cc.saida.dataframe:
            conteudo.append(f"- **Saída (dataframe):** `{cc.saida.dataframe}` col=`{cc.saida.dataframe_col}`")
        conteudo.append(f"- **Predição:** `{cc.predicao.pasta or '(será criado)'}`")
        
        # --- Pipeline / Curriculum ---
        etapas = cfg.curriculum
        conteudo.append(f"\n### Pipeline Curriculum")
        conteudo.append(f"**Etapas:** {len(etapas)}")
        
        if len(etapas) > 1:
            # Tabela detalhada das etapas do curriculum
            # Usa valores originais do YAML como fallback (não os mutados por _aplicar_etapa_curriculum)
            conteudo.append("")
            
            # Detecta colunas opcionais (só aparecem se alguma etapa as define)
            tem_pace_loss = any(e.pace_loss > 0 for e in etapas)
            tem_pace_max = any(e.pace_epochs_max > 0 for e in etapas)
            tem_batch = any(e.batch_size > 0 for e in etapas)
            
            # Cabeçalho
            header = "| # | Alias | Tipo | Epochs"
            sep    = "|---|-------|------|-------"
            if tem_pace_max:
                header += " | Max Ep"
                sep    += "|-------"
            if tem_pace_loss:
                header += " | Pace Loss"
                sep    += "|----------"
            header += " | LR | Max Seq"
            sep    += "|----|--------"
            if tem_batch:
                header += " | Batch"
                sep    += "|------"
            header += " | Divisão | Treino | Valid. | Teste |"
            sep    += "|---------|--------|--------|-------|"
            conteudo.append(header)
            conteudo.append(sep)
            
            for i, etapa in enumerate(etapas):
                ep = etapa.pace_epochs if etapa.pace_epochs > 0 else yaml_epochs
                lr = etapa.learning_rate if etapa.learning_rate > 0 else yaml_lr
                msl = etapa.max_seq_length if etapa.max_seq_length > 0 else yaml_msl
                divisao_nome = os.path.basename(etapa.arquivo) if etapa.arquivo else "-"
                
                # Conta instâncias por alvo no CSV da etapa
                contagens = cfg._contar_instancias_divisao(etapa.arquivo)
                n_treino = contagens.get("treino", "-")
                n_valid = contagens.get("validacao", "-")
                n_teste = contagens.get("teste", "-")
                
                linha = f"| {i} | {etapa.alias} | {etapa.tipo} | {ep}"
                if tem_pace_max:
                    linha += f" | {etapa.pace_epochs_max if etapa.pace_epochs_max > 0 else '-'}"
                if tem_pace_loss:
                    linha += f" | {etapa.pace_loss if etapa.pace_loss > 0 else '-'}"
                linha += f" | {lr} | {msl}"
                if tem_batch:
                    linha += f" | {etapa.batch_size if etapa.batch_size > 0 else '-'}"
                linha += f" | {divisao_nome} | {n_treino} | {n_valid} | {n_teste} |"
                
                conteudo.append(linha)
        else:
            # Etapa única: resumo em texto
            etapa = etapas[0]
            conteudo.append(f"- **Alias:** {etapa.alias}")
            conteudo.append(f"- **Tipo:** {etapa.tipo}")
        
        # 2. Hardware
        if hardware_info:
            conteudo.append("\n## 2. Hardware")
            
            # Seção CPU
            conteudo.append("\n### CPU")
            conteudo.append(f"- **CPUs Físicas:** {hardware_info.get('cpus_fisicas', 'N/A')}")
            conteudo.append(f"- **CPUs Lógicas:** {hardware_info.get('cpus_logicas', 'N/A')}")
            conteudo.append(f"- **Uso CPU (sistema):** {hardware_info.get('cpu_uso_%', 'N/A')}%")
            conteudo.append(f"- **Uso CPU (processo):** {hardware_info.get('cpu_uso_processo_%', 'N/A')}%")
            
            # Seção RAM
            conteudo.append("\n### Memória RAM")
            conteudo.append(f"- **Total:** {hardware_info.get('mem_total_gb', 'N/A')} GB")
            conteudo.append(f"- **Disponível:** {hardware_info.get('mem_disponivel_gb', 'N/A')} GB")
            conteudo.append(f"- **Em Uso:** {hardware_info.get('mem_usada_gb', 'N/A')} GB ({hardware_info.get('mem_uso_%', 'N/A')}%)")
            
            # Seção Disco
            conteudo.append("\n### Disco")
            conteudo.append(f"- **Uso:** {hardware_info.get('disco_uso_%', 'N/A')}%")
            
            # Seção GPU (nova estrutura)
            gpu_info = hardware_info.get('gpu', {})
            if gpu_info:
                conteudo.append("\n### GPU")
                if gpu_info.get('disponivel', False):
                    conteudo.append(f"- **Total GPUs:** {gpu_info.get('total_gpus', 0)}")
                    gpus = gpu_info.get('gpus', [])
                    for gpu in gpus:
                        if 'erro' in gpu:
                            conteudo.append(f"- **GPU[{gpu['idx']}]:** Erro: {gpu['erro']}")
                        else:
                            conteudo.append(f"- **GPU[{gpu.get('idx', '?')}]:** {gpu.get('nome', 'N/A')}")
                            conteudo.append(f"  - Memória Total: {gpu.get('mem_total_gb', 'N/A')} GB")
                            conteudo.append(f"  - Memória Reservada: {gpu.get('mem_reservada_gb', 'N/A')} GB")
                            conteudo.append(f"  - Memória Alocada: {gpu.get('mem_alocada_gb', 'N/A')} GB")
                            conteudo.append(f"  - Pico Reservado: {gpu.get('mem_max_reservada_gb', 'N/A')} GB")
                            conteudo.append(f"  - Compute Capability: {gpu.get('compute_capability', 'N/A')}")
                else:
                    motivo = gpu_info.get('motivo', 'Não disponível')
                    conteudo.append(f"- **Status:** {motivo}")
        
        # 3. Dataset
        if dataset_stats:
            conteudo.append("\n## 3. Estatísticas do Dataset")
            conteudo.append(f"- **Total Registros Treino:** {dataset_stats.get('treino_len', 'N/A')}")
            conteudo.append(f"- **Total Registros Validação:** {dataset_stats.get('validacao_len', 'N/A')}")
            
            if 'doc_stats' in dataset_stats:
                ds = dataset_stats['doc_stats']
                conteudo.append("\n### Estatísticas do Documento (Sem Prompt)")
                conteudo.append(f"- **Template (Base):** {ds.get('template_tokens', 0)} tokens")
                conteudo.append(f"- **Documento (Média):** {ds.get('avg_doc_tokens', 0)} tokens")
                conteudo.append(f"- **Documento (Max):** {ds.get('max_doc_tokens', 0)} tokens")

            if 'token_stats' in dataset_stats:
                # Caso antigo (compatibilidade) ou se não tiver estrutura detalhada
                # (Se o token_stats for o dicionário principal, o código anterior já pega)
                # O código atual de collect_returns retorna um dict {split: stats}
                # O dataset_stats passado aqui pode ser apenas o 'train' stats se extraído
                # Mas geralmente dataset_stats contém {'treino_len', ...} e talvez 'token_stats'.
                # Vamos ajustar para iterar splits se existirem
                pass # Tratado acima se implementado na chamada, mas aqui assume estrutura plana
                
            # Exibe stats por split se available (formato novo)
            for split in ['train', 'eval', 'test']:
                if split in dataset_stats and isinstance(dataset_stats[split], dict):
                    s = dataset_stats[split]
                    conteudo.append(f"\n### Distribuição de Tokens - {split.upper()}")
                    conteudo.append(f"- **Média:** {s.get('avg_tokens', 0)}")
                    conteudo.append(f"- **Mín/Máx:** {s.get('min_tokens', 0)} / {s.get('max_tokens', 0)}")
                    
                    if 'doc_stats' in s:
                         ds = s['doc_stats']
                         conteudo.append(f"- **Input Texto (Média):** {ds.get('avg_doc_tokens', 0)} (excl. {ds.get('template_tokens', 0)} do template)")
                         conteudo.append(f"- **Input Texto (Max):** {ds.get('max_doc_tokens', 0)}")

        # 4. Treinamento
        if train_stats:
            conteudo.append("\n## 4. Resultados do Treinamento")
            
            # Tabela de métricas principais
            conteudo.append("| Métrica | Valor |")
            conteudo.append("|---|---|")
            
            # Métricas comuns do TRL/Transformers
            metrics_map = {
                "train_runtime": "Tempo Total (s)",
                "train_samples_per_second": "Amostras/seg",
                "total_flos": "Total FLOS",
                "train_loss": "Loss Final (Treino)",
                "epoch": "Épocas Concluídas"
            }
            
            for k, v in train_stats.items():
                if k in metrics_map:
                    val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    conteudo.append(f"| {metrics_map[k]} | {val} |")
                elif k.startswith("train_") or k == "loss":
                     # Outras métricas de treino
                     pass
            
            # Adiciona outras métricas importantes se presentes
            if 'global_step' in train_stats:
                conteudo.append(f"| Steps Totais | {train_stats['global_step']} |")
                
            # Memória GPU se disponível (no train_stats customizado)
            if 'mem_gpu_before' in train_stats:
                conteudo.append("\n### Memória GPU")
                before = train_stats['mem_gpu_before']
                after = train_stats.get('mem_gpu_after', {})
                conteudo.extend(_formatar_mem_gpu(before, after))

        texto_final = "\n".join(conteudo)

        if print_only:
             print("\n" + "="*80)
             print("📄 PRÉVIA DO RELATÓRIO DE TREINAMENTO")
             print("="*80)
             print(texto_final)
             print("="*80 + "\n")
             return ""
        
        # Modo de gravação
        if not os.path.isdir(self.output_dir):
            raise FileNotFoundError(f"Erro: Pasta do modelo treinado não encontrada: '{self.output_dir}'. Não é possível salvar o relatório.")

        # Cria diretório de relatório (agora garantido que o pai existe)
        os.makedirs(self.report_dir, exist_ok=True)
            
        # Salva arquivo
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write(texto_final)
            
        print(f"\n📝 Relatório gerado em: {self.report_file}")
        return self.report_file
