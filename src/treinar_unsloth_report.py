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
        
        conteudo = []
        conteudo.append(f"# Relatório de Treinamento LLM")
        conteudo.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        conteudo.append(f"**Modelo Base:** `{self.yaml_config.modelo.base}`")
        conteudo.append(f"**Diretório de Saída:** `{self.output_dir}`")
        
        # 1. Configuração
        conteudo.append("\n## 1. Configuração Utilizada")
        conteudo.append("```yaml")
        conteudo.append(self.yaml_config.info())
        conteudo.append("```")
        
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
            
            if 'token_stats' in dataset_stats:
                ts = dataset_stats['token_stats']
                conteudo.append("\n### Distribuição de Tokens (Treino)")
                conteudo.append(f"- **Mínimo:** {ts.get('min', 0)}")
                conteudo.append(f"- **Máximo:** {ts.get('max', 0)}")
                conteudo.append(f"- **Média:** {ts.get('avg', 0)}")
                conteudo.append(f"- **Excedente (> max_seq):** {ts.get('exceed_max_seq', 0)}")

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
