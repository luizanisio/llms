#!/usr/bin/env python3
"""
Módulo de monitoramento de recursos para teste de modelos.

Fornece:
- Monitoramento contínuo de RAM e GPU durante predições
- Geração de gráfico de linha com uso de memória ao longo do tempo
- Suporte para múltiplas GPUs (soma das memórias)

Uso:
    from treinar_unsloth_monitor import MonitorRecursos
    
    monitor = MonitorRecursos(output_dir="./saida")
    
    # Inicia monitoramento em thread separada
    monitor.iniciar()
    
    # ... executa predições ...
    
    # Para monitoramento e gera gráfico
    monitor.parar()
    grafico_path = monitor.gerar_grafico()
"""

import os
import sys
import time
import json
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field

# Configuração de path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# Dataclass para armazenar métricas
# ---------------------------------------------------------------------------

@dataclass
class MetricaInstante:
    """Representa métricas de um instante de tempo."""
    timestamp: float
    ram_usada_gb: float
    ram_total_gb: float
    gpu_usada_gb: float  # Soma de todas as GPUs
    gpu_total_gb: float  # Soma de todas as GPUs
    num_gpus: int = 0
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3],
            "ram_usada_gb": round(self.ram_usada_gb, 2),
            "ram_total_gb": round(self.ram_total_gb, 2),
            "gpu_usada_gb": round(self.gpu_usada_gb, 2),
            "gpu_total_gb": round(self.gpu_total_gb, 2),
            "num_gpus": self.num_gpus,
        }


# ---------------------------------------------------------------------------
# Classe principal de monitoramento
# ---------------------------------------------------------------------------

class MonitorRecursos:
    """
    Monitora uso de memória RAM e GPU durante execução de predições.
    
    Salva métricas em arquivo JSONL e gera gráfico de linha ao final.
    """
    
    def __init__(
        self,
        output_dir: str,
        intervalo_segundos: float = 0.5,
        nome_arquivo: str = "memoria_predicao"
    ):
        """
        Args:
            output_dir: Diretório para salvar métricas e gráfico
            intervalo_segundos: Intervalo entre coletas de métricas
            nome_arquivo: Nome base dos arquivos de saída
        """
        self.output_dir = output_dir
        self.intervalo = intervalo_segundos
        self.nome_arquivo = nome_arquivo
        
        # Estado do monitoramento
        self._ativo = False
        self._thread: Optional[threading.Thread] = None
        self._metricas: List[MetricaInstante] = []
        self._tempo_inicio: float = 0
        
        # Cria diretório de saída
        os.makedirs(os.path.join(output_dir, "treinamento"), exist_ok=True)
        
        # Arquivos de saída
        self._arquivo_jsonl = os.path.join(output_dir, "treinamento", f"{nome_arquivo}.jsonl")
        self._arquivo_grafico = os.path.join(output_dir, "treinamento", f"{nome_arquivo}.png")
    
    def coletar_metricas(self) -> MetricaInstante:
        """Coleta métricas atuais de RAM e GPU."""
        # Coleta RAM via psutil (mais confiável e direto)
        ram_usada = 0.0
        ram_total = 0.0
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            ram_usada = mem.used / (1024**3)
            ram_total = mem.total / (1024**3)
        except ImportError:
            # Fallback se psutil não disponível
            try:
                from util import Util
                hardware = Util.dados_hardware(incluir_gpu=False)
                ram_usada = hardware.get("mem_usada_gb", 0)
                ram_total = hardware.get("mem_total_gb", 0)
            except:
                pass
        
        # Coleta GPU via torch.cuda (direto, sem dependência do util)
        gpu_usada = 0.0
        gpu_total = 0.0
        num_gpus = 0
        
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                for i in range(num_gpus):
                    try:
                        # Memória reservada é mais representativa do uso real
                        gpu_usada += torch.cuda.memory_reserved(i) / (1024**3)
                        props = torch.cuda.get_device_properties(i)
                        gpu_total += props.total_memory / (1024**3)
                    except:
                        pass
        except ImportError:
            pass
        except Exception:
            pass
        
        return MetricaInstante(
            timestamp=time.time(),
            ram_usada_gb=ram_usada,
            ram_total_gb=ram_total,
            gpu_usada_gb=gpu_usada,
            gpu_total_gb=gpu_total,
            num_gpus=num_gpus,
        )
    
    def _loop_monitoramento(self) -> None:
        """Loop de monitoramento executado em thread separada."""
        # Abre arquivo JSONL para escrita
        with open(self._arquivo_jsonl, "w", encoding="utf-8") as fp:
            while self._ativo:
                try:
                    metrica = self.coletar_metricas()
                    self._metricas.append(metrica)
                    
                    # Escreve no arquivo
                    fp.write(json.dumps(metrica.to_dict(), ensure_ascii=False) + "\n")
                    fp.flush()
                    
                except Exception as e:
                    # Não interrompe por erro de coleta
                    pass
                
                time.sleep(self.intervalo)
    
    def iniciar(self) -> None:
        """Inicia o monitoramento em thread separada."""
        if self._ativo:
            return
        
        self._ativo = True
        self._tempo_inicio = time.time()
        self._metricas.clear()
        
        self._thread = threading.Thread(target=self._loop_monitoramento, daemon=True)
        self._thread.start()
        
        print(f"📊 Monitoramento de memória iniciado (gravando em {self._arquivo_jsonl})")
    
    def parar(self) -> Dict[str, Any]:
        """
        Para o monitoramento e retorna resumo das métricas.
        
        Returns:
            Dict com resumo das métricas coletadas
        """
        if not self._ativo:
            return {}
        
        self._ativo = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        tempo_total = time.time() - self._tempo_inicio
        
        # Calcula resumo
        resumo = self._calcular_resumo(tempo_total)
        
        print(f"📊 Monitoramento encerrado após {tempo_total:.1f}s ({len(self._metricas)} amostras)")
        
        return resumo
    
    def _calcular_resumo(self, tempo_total: float) -> Dict[str, Any]:
        """Calcula resumo estatístico das métricas coletadas."""
        if not self._metricas:
            return {"erro": "Nenhuma métrica coletada"}
        
        ram_usadas = [m.ram_usada_gb for m in self._metricas]
        gpu_usadas = [m.gpu_usada_gb for m in self._metricas]
        
        return {
            "tempo_total_segundos": round(tempo_total, 2),
            "amostras_coletadas": len(self._metricas),
            "intervalo_segundos": self.intervalo,
            "ram": {
                "min_gb": round(min(ram_usadas), 2),
                "max_gb": round(max(ram_usadas), 2),
                "media_gb": round(sum(ram_usadas) / len(ram_usadas), 2),
                "total_gb": round(self._metricas[-1].ram_total_gb, 2),
            },
            "gpu": {
                "min_gb": round(min(gpu_usadas), 2) if gpu_usadas else 0,
                "max_gb": round(max(gpu_usadas), 2) if gpu_usadas else 0,
                "media_gb": round(sum(gpu_usadas) / len(gpu_usadas), 2) if gpu_usadas else 0,
                "total_gb": round(self._metricas[-1].gpu_total_gb, 2) if self._metricas else 0,
                "num_gpus": self._metricas[-1].num_gpus if self._metricas else 0,
            },
            "arquivo_jsonl": self._arquivo_jsonl,
        }
    
    def gerar_grafico(self) -> str:
        """
        Gera gráfico de linha com uso de memória RAM e GPU.
        
        Returns:
            Caminho do arquivo PNG gerado
        """
        if not self._metricas:
            print("⚠️  Nenhuma métrica para gerar gráfico")
            return ""
        
        try:
            from treinar_unsloth_graficos import GraficoMonitor
            
            # Prepara dados
            tempos = [(m.timestamp - self._tempo_inicio) for m in self._metricas]
            ram_usadas = [m.ram_usada_gb for m in self._metricas]
            gpu_usadas = [m.gpu_usada_gb for m in self._metricas]
            num_gpus = self._metricas[-1].num_gpus if self._metricas else 0
            
            # Gera gráfico usando classe centralizada
            sucesso = GraficoMonitor.uso_memoria(
                tempos=tempos,
                ram_usadas=ram_usadas,
                gpu_usadas=gpu_usadas,
                output_path=self._arquivo_grafico,
                titulo='Uso de Memória Durante Predições',
                num_gpus=num_gpus
            )
            
            if sucesso:
                print(f"📈 Gráfico de memória salvo em: {self._arquivo_grafico}")
                return self._arquivo_grafico
            else:
                print("⚠️  Erro ao gerar gráfico de memória")
                return ""
                
        except ImportError as e:
            print(f"⚠️  Módulo de gráficos não disponível: {e}")
            return ""
        except Exception as e:
            print(f"⚠️  Erro ao gerar gráfico: {e}")
            return ""
    
    @property
    def metricas(self) -> List[MetricaInstante]:
        """Retorna lista de métricas coletadas."""
        return self._metricas.copy()


# ---------------------------------------------------------------------------
# Função utilitária para uso rápido
# ---------------------------------------------------------------------------

def monitorar_predicao(output_dir: str, funcao_predicao, *args, **kwargs) -> Dict[str, Any]:
    """
    Wrapper para monitorar uma função de predição.
    
    Args:
        output_dir: Diretório para salvar métricas e gráfico
        funcao_predicao: Função a ser executada
        *args, **kwargs: Argumentos para a função
    
    Returns:
        Dict com resultado da função e resumo de métricas
    
    Exemplo:
        resultado = monitorar_predicao(
            "./saida",
            trainer.testar_predicoes,
            n_exemplos=5
        )
    """
    monitor = MonitorRecursos(output_dir)
    monitor.iniciar()
    
    try:
        resultado_funcao = funcao_predicao(*args, **kwargs)
    finally:
        resumo = monitor.parar()
        monitor.gerar_grafico()
    
    return {
        "resultado": resultado_funcao,
        "metricas_memoria": resumo,
    }


# ---------------------------------------------------------------------------
# Teste do módulo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Testa o monitor de recursos")
    parser.add_argument("--duracao", type=float, default=5.0, help="Duração do teste em segundos")
    parser.add_argument("--output", type=str, default="./teste_monitor", help="Diretório de saída")
    args = parser.parse_args()
    
    print(f"🧪 Testando monitor por {args.duracao}s...")
    
    monitor = MonitorRecursos(args.output, intervalo_segundos=0.25)
    monitor.iniciar()
    
    time.sleep(args.duracao)
    
    resumo = monitor.parar()
    print(f"\n📊 Resumo:\n{json.dumps(resumo, indent=2, ensure_ascii=False)}")
    
    grafico = monitor.gerar_grafico()
    if grafico:
        print(f"✅ Gráfico salvo em: {grafico}")
