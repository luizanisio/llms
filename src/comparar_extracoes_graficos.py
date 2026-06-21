# -*- coding: utf-8 -*-
"""
Módulo auxiliar para geração de gráficos das comparações de extrações.
"""

import os
import json
import pandas as pd
import seaborn as sns
from util_graficos import UtilGraficos, Cores
from util_json_graficos import traduzir_rotulos
import util

class CompararExtracoesGraficos:
    
    @staticmethod
    def gerar_grafico_erros(dados_analise, pasta_saida, lang='pt'):
        """Gera gráfico de barras empilhadas com status dos documentos por modelo."""
        # Usa rotulos[1:] para incluir também o modelo de origem (base) mantendo a mesma ordem do YAML
        rotulos_modelos = dados_analise.rotulos[1:] if len(dados_analise.rotulos) > 1 else []
        # Chaves internas em português para processamento
        stats = {m: {'Sucesso': 0, 'Erro': 0, 'Inexistente': 0} for m in rotulos_modelos}
        
        for linha in dados_analise.dados_completos:
            for modelo in rotulos_modelos:
                val = linha.get(modelo)
                # Verifica status baseado no valor
                if val is None:
                    stats[modelo]['Inexistente'] += 1
                elif isinstance(val, dict) and 'erro' in val:
                    erro_msg = str(val['erro'])
                    if 'Inexistente' in erro_msg:
                         stats[modelo]['Inexistente'] += 1
                    else:
                         stats[modelo]['Erro'] += 1
                else:
                    stats[modelo]['Sucesso'] += 1
                    
        df_stats = pd.DataFrame(stats).transpose()
        
        # Ordena colunas para mapear na paleta RdYlGn (Red -> Green)
        # 1. Inexistente (Red)
        # 2. Erro (Yellow/Orange)
        # 3. Sucesso (Green)
        cols_ordem = ['Inexistente', 'Erro', 'Sucesso']
    
        # Garante que todas as colunas existem
        for c in cols_ordem:
            if c not in df_stats.columns:
                df_stats[c] = 0
                
        pasta_graficos = os.path.join(pasta_saida, 'graficos')
        os.makedirs(pasta_graficos, exist_ok=True)
                
        df_stats = df_stats[cols_ordem]
        
        # Traduz nomes de colunas para exibição no gráfico
        cols_traduzidas = {
            'Inexistente': traduzir_rotulos('status_inexistente', lang),
            'Erro': traduzir_rotulos('status_erro', lang),
            'Sucesso': traduzir_rotulos('status_sucesso', lang)
        }
        df_stats = df_stats.rename(columns=cols_traduzidas)
        
        arquivo = os.path.join(pasta_graficos, 'status_modelos.png')
        UtilGraficos.gerar_grafico_empilhado(
            df_stats, 
            titulo=traduzir_rotulos('status_titulo', lang),
            ylabel=traduzir_rotulos('status_ylabel', lang),
            xlabel=traduzir_rotulos('modelo_xlabel', lang),
            arquivo_saida=arquivo,
            paleta_cores=Cores.RdYlGn
        )
        print(f"   ✓ Gráfico de status gerado: {os.path.basename(arquivo)}")

    @staticmethod
    def gerar_graficos_treinamento(config, base_dir_yaml, pasta_base_ativa, dados_analise, pasta_saida, lang='pt'):
        """Gera gráficos de treinamento (loss vs tokens/instâncias) se configurado."""
        modelos_comp_all = config.get('modelos_comparacao', [])
        modelos_ativos = [m for m in modelos_comp_all if m.get('ativo', True)]
        modelo_base = config.get('modelo_base', {})
    
        mapa_modelos = {modelo_base.get('rotulo'): modelo_base}
        for m in modelos_ativos:
            mapa_modelos[m.get('rotulo')] = m
    
        try:
            paleta = sns.color_palette(Cores.Tab10.value, len(dados_analise.rotulos))
        except Exception:
            paleta = sns.color_palette("tab10", len(dados_analise.rotulos))
    
        series_eval_loss_tokens = {}
        series_eval_loss_inst = {}
        series_eval_loss_global_tokens = {}
        series_eval_loss_global_inst = {}
        
        algum_modelo_tem_metricas = False
    
        for idx, rotulo in enumerate(dados_analise.rotulos):
            m = mapa_modelos.get(rotulo)
            if not m:
                continue
            
            pasta_treinamento_raw = m.get('pasta_treinamento')
            if not pasta_treinamento_raw:
                continue
                
            pasta_treinamento = util.Util.resolver_caminho(pasta_treinamento_raw, base_dir_yaml, pasta_base_ativa)
            arquivo_metricas = os.path.join(pasta_treinamento, 'training_metrics.jsonl')
            
            if not os.path.isfile(arquivo_metricas):
                continue
                
            algum_modelo_tem_metricas = True
            
            eval_loss_by_step = {}
            eval_loss_global_by_step = {}
            
            current_tokens = None
            current_instances = None
            
            try:
                with open(arquivo_metricas, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            obj = json.loads(line)
                            current_tokens = obj.get("tokens_acumulados", current_tokens)
                            current_instances = obj.get("instancias_acumuladas", current_instances)
                            
                            step = obj.get("step_global", obj.get("step"))
                            if step is not None:
                                if "eval_loss" in obj:
                                    if step not in eval_loss_by_step:
                                        eval_loss_by_step[step] = {'loss': obj["eval_loss"]}
                                    eval_loss_by_step[step]['tokens'] = current_tokens
                                    eval_loss_by_step[step]['instancias'] = current_instances
                                    
                                if "eval_loss_global" in obj:
                                    if step not in eval_loss_global_by_step:
                                        eval_loss_global_by_step[step] = {'loss': obj["eval_loss_global"]}
                                    eval_loss_global_by_step[step]['tokens'] = current_tokens
                                    eval_loss_global_by_step[step]['instancias'] = current_instances
                        except Exception:
                            pass
            except Exception as e:
                print(f"   ⚠️ Erro ao ler {arquivo_metricas}: {e}")
                continue
    
            def extrair_series(dict_steps):
                steps = sorted(dict_steps.keys())
                x_tok, x_inst, y = [], [], []
                for s in steps:
                    t = dict_steps[s].get('tokens')
                    i = dict_steps[s].get('instancias')
                    l = dict_steps[s].get('loss')
                    if t is not None and l is not None:
                        x_tok.append(t)
                    if i is not None and l is not None:
                        x_inst.append(i)
                    if l is not None:
                        y.append(l)
                return (x_tok, y) if len(x_tok) == len(y) else ([], []), (x_inst, y) if len(x_inst) == len(y) else ([], [])
    
            (x_tok_eval, y_eval), (x_inst_eval, _) = extrair_series(eval_loss_by_step)
            (x_tok_eval_g, y_eval_g), (x_inst_eval_g, _) = extrair_series(eval_loss_global_by_step)
            
            cor = paleta[idx]
            
            if x_tok_eval and y_eval:
                series_eval_loss_tokens[rotulo] = {'x': x_tok_eval, 'y': y_eval, 'cor': cor, 'estilo': '-'}
            if x_inst_eval and y_eval:
                series_eval_loss_inst[rotulo] = {'x': x_inst_eval, 'y': y_eval, 'cor': cor, 'estilo': '-'}
                
            if x_tok_eval_g and y_eval_g:
                series_eval_loss_global_tokens[rotulo] = {'x': x_tok_eval_g, 'y': y_eval_g, 'cor': cor, 'estilo': '-'}
            if x_inst_eval_g and y_eval_g:
                series_eval_loss_global_inst[rotulo] = {'x': x_inst_eval_g, 'y': y_eval_g, 'cor': cor, 'estilo': '-'}
    
        if not algum_modelo_tem_metricas:
            return
    
        pasta_graficos = os.path.join(pasta_saida, 'graficos')
        os.makedirs(pasta_graficos, exist_ok=True)
    
        print("\n📉 Gerando gráficos de métricas de treinamento...")
    
        if series_eval_loss_tokens:
            arquivo = os.path.join(pasta_graficos, 'treinamento_eval_loss_vs_tokens.png')
            UtilGraficos.gerar_grafico_linhas(series_eval_loss_tokens,
                titulo="Eval Loss vs Tokens Acumulados",
                ylabel="Eval Loss", xlabel="Tokens Acumulados",
                arquivo_saida=arquivo)
            print(f"   ✓ Gráfico salvo: {os.path.basename(arquivo)}")
    
        if series_eval_loss_global_tokens:
            arquivo = os.path.join(pasta_graficos, 'treinamento_eval_loss_global_vs_tokens.png')
            UtilGraficos.gerar_grafico_linhas(series_eval_loss_global_tokens,
                titulo="Eval Loss Global vs Tokens Acumulados",
                ylabel="Eval Loss Global", xlabel="Tokens Acumulados",
                arquivo_saida=arquivo)
            print(f"   ✓ Gráfico salvo: {os.path.basename(arquivo)}")
    
        if series_eval_loss_inst:
            arquivo = os.path.join(pasta_graficos, 'treinamento_eval_loss_vs_instancias.png')
            UtilGraficos.gerar_grafico_linhas(series_eval_loss_inst,
                titulo="Eval Loss vs Instâncias Acumuladas",
                ylabel="Eval Loss", xlabel="Instâncias Acumuladas",
                arquivo_saida=arquivo)
            print(f"   ✓ Gráfico salvo: {os.path.basename(arquivo)}")
    
        if series_eval_loss_global_inst:
            arquivo = os.path.join(pasta_graficos, 'treinamento_eval_loss_global_vs_instancias.png')
            UtilGraficos.gerar_grafico_linhas(series_eval_loss_global_inst,
                titulo="Eval Loss Global vs Instâncias Acumuladas",
                ylabel="Eval Loss Global", xlabel="Instâncias Acumuladas",
                arquivo_saida=arquivo)
            print(f"   ✓ Gráfico salvo: {os.path.basename(arquivo)}")
