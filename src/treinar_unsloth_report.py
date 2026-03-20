# Autor: Luiz Anísio
# Fonte: https://github.com/luizanisio/llms/tree/main/src

import os
import json
from typing import Dict, Any, Optional
from treinar_unsloth_util import YamlTreinamento

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
            cfg_entrada = self.yaml_config.pastas.entrada
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
        conteudo.append(f"**Tipo de entrada:** {cfg.tipo_entrada}")
        
        # --- 1. Configuração ---
        conteudo.append("\n## 1. Configuração do Treinamento")
        conteudo.append("")
        conteudo.append("| Parâmetro | Valor |")
        conteudo.append("|---|---|")
        conteudo.append(f"| Batch size | {cfg.treinamento.batch_size} |")
        conteudo.append(f"| Grad accumulation | {cfg.treinamento.grad_batch_size} |")
        conteudo.append(f"| Épocas | {cfg.treinamento.epochs} |")
        auto = ""
        conteudo.append(f"| Max seq length | {cfg.treinamento.max_seq_length}{auto} |")
        conteudo.append(f"| LoRA r | {cfg.lora.r} |")
        conteudo.append(f"| Learning rate | {cfg.treinamento.learning_rate} |")
        conteudo.append(f"| Train on responses only | {cfg.treinamento.train_on_responses_only} |")
        conteudo.append(f"| Warmup steps | {cfg.treinamento.warmup_steps} |")
        conteudo.append(f"| Weight decay | {cfg.treinamento.weight_decay} |")
        conteudo.append(f"| Otimizador | {cfg.treinamento.optim} |")
        conteudo.append(f"| LR scheduler | {cfg.treinamento.lr_scheduler_type} |")
        conteudo.append(f"| Quantização (nbits) | {cfg.treinamento.nbits} |")
        
        # --- Seção de dados: adaptada ao tipo de entrada ---
        from treinar_unsloth_util import TIPOS_BASEADOS_EM_PASTAS, TIPO_ENTRADA_CURRICULUM
        
        is_curriculum = cfg.tipo_entrada == TIPO_ENTRADA_CURRICULUM
        is_pastas = cfg.tipo_entrada in TIPOS_BASEADOS_EM_PASTAS
        
        if is_pastas:
            conteudo.append("\n### Pastas de Dados")
            conteudo.append(f"- **Gold Dataset:** `{cfg.pastas.dataset.pasta}`")
            if cfg.pastas.entrada.pasta:
                conteudo.append(f"- **Entrada (textos):** `{cfg.pastas.entrada.pasta}`")
            conteudo.append(f"- **Predição:** `{cfg.pastas.predicao.pasta or '(será criado)'}`")
            
            if not is_curriculum:
                # Modo pastas simples: mostra divisão e proporções
                conteudo.append(f"- **Divisão:** `{cfg.pastas.divisao.arquivo or '(será criado)'}`")
                prop = cfg.pastas.divisao.proporcao
                conteudo.append(f"- **Proporções (yaml):** treino={prop[0]}, validação={prop[1]}, teste={prop[2]}")
                if cfg.pastas.divisao.proporcao_reais:
                    pr = cfg.pastas.divisao.proporcao_reais
                    conteudo.append(f"- **Proporções (efetivas):** treino={pr[0]:.2f}, validação={pr[1]:.2f}, teste={pr[2]:.2f}")
        else:
            # Modo dataset (parquet)
            conteudo.append("\n### Arquivos de Dataset")
            conteudo.append(f"- **Treino:** `{cfg.dataset.train_file}`")
            if cfg.dataset.eval_file:
                conteudo.append(f"- **Avaliação:** `{cfg.dataset.eval_file}`")
            if cfg.dataset.test_file:
                conteudo.append(f"- **Teste:** `{cfg.dataset.test_file}`")
        
        # --- Pipeline / Curriculum ---
        etapas = cfg.curriculum
        conteudo.append(f"\n### Pipeline{' Curriculum' if is_curriculum else ''}")
        conteudo.append(f"**Etapas:** {len(etapas)}")
        
        if is_curriculum and len(etapas) > 1:
            # Tabela detalhada das etapas do curriculum
            conteudo.append("")
            conteudo.append("| # | Alias | Tipo | Epochs | LR | Max Seq | Divisão | Treino | Valid. | Teste |")
            conteudo.append("|---|-------|------|--------|----|---------|---------|--------|--------|-------|")
            
            for i, etapa in enumerate(etapas):
                ep = etapa.pace_epochs if etapa.pace_epochs > 0 else cfg.treinamento.epochs
                lr = etapa.learning_rate if etapa.learning_rate > 0 else cfg.treinamento.learning_rate
                msl = etapa.max_seq_length if etapa.max_seq_length > 0 else cfg.treinamento.max_seq_length
                divisao_nome = os.path.basename(etapa.arquivo) if etapa.arquivo else "-"
                
                # Conta instâncias por alvo no CSV da etapa
                contagens = cfg._contar_instancias_divisao(etapa.arquivo)
                n_treino = contagens.get("treino", "-")
                n_valid = contagens.get("validacao", "-")
                n_teste = contagens.get("teste", "-")
                
                conteudo.append(
                    f"| {i} | {etapa.alias} | {etapa.tipo} | {ep} | {lr} "
                    f"| {msl} | {divisao_nome} | {n_treino} | {n_valid} | {n_teste} |"
                )
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
                conteudo.append("**Antes:**")
                conteudo.append("```json")
                conteudo.append(json.dumps(train_stats['mem_gpu_before'], indent=2, ensure_ascii=False))
                conteudo.append("```")
                
            if 'mem_gpu_after' in train_stats:
                conteudo.append("**Depois:**")
                conteudo.append("```json")
                conteudo.append(json.dumps(train_stats['mem_gpu_after'], indent=2, ensure_ascii=False))
                conteudo.append("```")

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
