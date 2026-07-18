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
            
            # Fallback: verifica subpasta "treinamento" dentro da pasta do modelo
            if not os.path.isfile(arquivo_metricas):
                arquivo_metricas_alt = os.path.join(pasta_treinamento, 'treinamento', 'training_metrics.jsonl')
                if os.path.isfile(arquivo_metricas_alt):
                    arquivo_metricas = arquivo_metricas_alt
            
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
            
            # Fallback: se o modelo não tem eval_loss_global (sem divisões de currículo),
            # usa eval_loss como eval_loss_global, pois sem currículo o eval loss já é global
            if not eval_loss_global_by_step and eval_loss_by_step:
                eval_loss_global_by_step = dict(eval_loss_by_step)
    
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
    
        def _adicionar_melhor_loss(series_dict):
            """Adiciona marcador de diamante no melhor (menor) loss e renomeia legenda com o valor."""
            series_final = {}
            for nome, config in series_dict.items():
                x_vals = config['x']
                y_vals = config['y']
                cor = config['cor']
                
                # Encontra índice do menor loss
                melhor_idx = min(range(len(y_vals)), key=lambda i: y_vals[i])
                melhor_loss = y_vals[melhor_idx]
                melhor_x = x_vals[melhor_idx]
                
                # Renomeia legenda com o valor do melhor loss
                nome_com_loss = f"{nome} ({melhor_loss:.4f})"
                series_final[nome_com_loss] = {
                    'x': x_vals, 'y': y_vals, 'cor': cor, 'estilo': config.get('estilo', '-')
                }
                
                # Adiciona série invisível apenas com o marcador diamante no melhor ponto
                series_final[f"_best_{nome}"] = {
                    'x': [melhor_x], 'y': [melhor_loss], 'cor': cor,
                    'estilo': 'None', 'marcador': 'D', 'tamanho_marcador': 10,
                    'alpha': 1.0, 'largura': 0
                }
            return series_final
    
        # Adiciona marcadores de melhor loss em todas as séries
        series_eval_loss_tokens = _adicionar_melhor_loss(series_eval_loss_tokens) if series_eval_loss_tokens else {}
        series_eval_loss_inst = _adicionar_melhor_loss(series_eval_loss_inst) if series_eval_loss_inst else {}
        series_eval_loss_global_tokens = _adicionar_melhor_loss(series_eval_loss_global_tokens) if series_eval_loss_global_tokens else {}
        series_eval_loss_global_inst = _adicionar_melhor_loss(series_eval_loss_global_inst) if series_eval_loss_global_inst else {}
    
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

    @staticmethod
    def _extrair_custo_melhor_modelo(arquivo_metricas):
        """
        Lê training_metrics.jsonl e retorna o custo (tokens, instâncias) e eval_loss
        no ponto do último is_best_eval_global (ou is_best_eval como fallback).
        
        Prioridade: is_best_eval_global > is_best_eval.
        O fallback para is_best_eval só é usado se nenhum is_best_eval_global for encontrado.
        
        Returns:
            dict: {'tokens': int, 'instancias': int, 'eval_loss': float} ou None se não encontrado
        """
        best_global = None
        best_local = None
        last_tokens = 0
        last_inst = 0
        
        try:
            with open(arquivo_metricas, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        # Atualiza tokens/instâncias acumulados
                        if obj.get('tokens_acumulados'):
                            last_tokens = obj['tokens_acumulados']
                        if obj.get('instancias_acumuladas'):
                            last_inst = obj['instancias_acumuladas']
                        # Registra o melhor ponto global (prioridade)
                        if obj.get('is_best_eval_global'):
                            loss = obj.get('eval_loss_global', obj.get('eval_loss'))
                            best_global = {
                                'tokens': last_tokens,
                                'instancias': last_inst,
                                'eval_loss': loss
                            }
                        # Registra o melhor ponto local (fallback)
                        elif obj.get('is_best_eval'):
                            loss = obj.get('eval_loss', obj.get('eval_loss_global'))
                            best_local = {
                                'tokens': last_tokens,
                                'instancias': last_inst,
                                'eval_loss': loss
                            }
                    except Exception:
                        pass
        except Exception as e:
            print(f"   ⚠️ Erro ao ler métricas de treinamento: {e}")
            return None
        
        return best_global if best_global is not None else best_local

    @staticmethod
    def gerar_graficos_custo_eficiencia(config, base_dir_yaml, pasta_base_ativa, 
                                         dados_analise, pasta_saida, arquivo_excel=None, lang='pt'):
        """
        Gera gráficos scatter de custo-eficiência: tokens/instâncias (até melhor eval loss) vs Score.
        
        Para cada combinação campo × técnica × sufixo, gera 2 gráficos:
        - custo_tk_f1_<técnica>_<campo>.png (eixo X = tokens, Y = F1)
        - custo_tk_sim_<técnica>_<campo>.png (eixo X = tokens, Y = SIM)
        - custo_inst_f1_<técnica>_<campo>.png (eixo X = instâncias, Y = F1)
        - custo_inst_sim_<técnica>_<campo>.png (eixo X = instâncias, Y = SIM)
        
        Modelos com 'baseline: true' são plotados como linhas horizontais.
        O modelo com melhor (menor) eval_loss recebe marcador estrela (★).
        O modelo com pior (maior) eval_loss recebe marcador triângulo (▲).
        
        Suporta métricas _F1 (BERTScore, ROUGE, SBERT) e _SIM (Levenshtein).
        SBERT é diferenciado por tamanho: SBERTp (pequeno), SBERTm (médio), SBERTg (grande).
        """
        import util
        
        modelos_comp_all = config.get('modelos_comparacao', [])
        modelos_ativos = [m for m in modelos_comp_all if m.get('ativo', True)]
        modelo_base = config.get('modelo_base', {})
        
        # Mapa rotulo -> config do modelo
        mapa_modelos = {modelo_base.get('rotulo'): modelo_base}
        for m in modelos_ativos:
            mapa_modelos[m.get('rotulo')] = m
        
        # Paleta de cores consistente com os demais gráficos
        try:
            paleta = sns.color_palette(Cores.Tab10.value, len(dados_analise.rotulos))
        except Exception:
            paleta = sns.color_palette("tab10", len(dados_analise.rotulos))
        
        # --- PASSO 1: Extrair custo de treinamento (tokens/inst até melhor eval loss) ---
        dados_custo = {}  # rotulo -> {tokens, instancias, eval_loss}
        dados_baseline = {}  # rotulo -> {alias, cor, idx}
        
        for idx, rotulo in enumerate(dados_analise.rotulos):
            m = mapa_modelos.get(rotulo)
            if not m:
                continue
            
            cor = paleta[idx]
            alias = m.get('alias', rotulo)
            
            # Baseline: sem custo de treinamento, gera linha horizontal
            if m.get('baseline', False):
                dados_baseline[rotulo] = {'alias': alias, 'cor': cor, 'idx': idx}
                continue
            
            # Modelo treinado: lê métricas
            pasta_treinamento_raw = m.get('pasta_treinamento')
            if not pasta_treinamento_raw:
                continue
            
            pasta_treinamento = util.Util.resolver_caminho(pasta_treinamento_raw, base_dir_yaml, pasta_base_ativa)
            arquivo_metricas = os.path.join(pasta_treinamento, 'training_metrics.jsonl')
            
            # Fallback: subpasta "treinamento"
            if not os.path.isfile(arquivo_metricas):
                arquivo_metricas_alt = os.path.join(pasta_treinamento, 'treinamento', 'training_metrics.jsonl')
                if os.path.isfile(arquivo_metricas_alt):
                    arquivo_metricas = arquivo_metricas_alt
            
            if not os.path.isfile(arquivo_metricas):
                continue
            
            custo = CompararExtracoesGraficos._extrair_custo_melhor_modelo(arquivo_metricas)
            if custo and custo['tokens'] > 0:
                dados_custo[rotulo] = {
                    **custo,
                    'alias': alias,
                    'cor': cor,
                    'idx': idx
                }
        
        if not dados_custo:
            return
        
        # --- PASSO 2: Ler F1 scores do Excel ---
        if not arquivo_excel or not os.path.isfile(arquivo_excel):
            print("   ⚠️ Arquivo Excel não encontrado para gráficos de custo-eficiência")
            return
        
        try:
            xl_file = pd.ExcelFile(arquivo_excel)
        except Exception as e:
            print("   ⚠️ Erro ao abrir Excel: {e}")
            return
        
        abas_resultados = [aba for aba in xl_file.sheet_names if aba.startswith('Resultados')]
        if not abas_resultados:
            return
        
        # Consolida todas as abas num DataFrame com colunas formato: modelo_campo_tecnica_F1
        df_consolidado = None
        col_id_nome = None
        
        for aba in abas_resultados:
            try:
                df_aba = pd.read_excel(arquivo_excel, sheet_name=aba)
            except Exception:
                continue
            
            if col_id_nome is None:
                col_id_nome = df_aba.columns[0]
            
            # Extrai técnica do nome da aba
            if '_' in aba:
                tecnica_nome = aba.split('_', 1)[1].lower()
                if tecnica_nome.startswith('rouge-'):
                    tecnica_nome = tecnica_nome.replace('-', '')
                else:
                    tecnica_nome = tecnica_nome.replace('-', '_')
            else:
                tecnica_nome = 'geral'
            
            # Renomeia colunas: modelo_campo_F1 -> modelo_campo_tecnica_F1
            colunas_renomeadas = {col_id_nome: col_id_nome}
            for col in df_aba.columns:
                if col != col_id_nome:
                    partes = col.split('_')
                    metrica = partes[-1]
                    if metrica in ['F1', 'P', 'R', 'LS', 'SIM']:
                        novo_nome = '_'.join(partes[:-1]) + f'_{tecnica_nome}_{metrica}'
                        colunas_renomeadas[col] = novo_nome
                    else:
                        colunas_renomeadas[col] = col
            
            df_aba_renomeada = df_aba.rename(columns=colunas_renomeadas)
            
            if df_consolidado is None:
                df_consolidado = df_aba_renomeada
            else:
                df_consolidado = pd.merge(df_consolidado, df_aba_renomeada, on=col_id_nome, how='outer')
        
        if df_consolidado is None or df_consolidado.empty:
            return
        
        # --- PASSO 3: Identificar pares (campo, técnica, sufixo) com colunas _F1 e _SIM ---
        # Formato: modelo_campo_tecnica_F1 ou modelo_campo_tecnica_SIM
        # Precisamos agrupar por (campo, técnica, sufixo) e mapear para modelos
        
        # Todos os rótulos conhecidos (para parsing das colunas)
        known_models = list(dados_analise.rotulos)
        
        # Mapeamento de técnica para nome curto (para nomes de arquivo)
        _tecnica_nome_arquivo = {
            'sbert': 'SBERTp',
            'sbert_pequeno': 'SBERTp',
            'sbert_medio': 'SBERTm',
            'sbert_grande': 'SBERTg',
        }
        
        # Identifica colunas F1 e SIM e agrupa por (campo, técnica, sufixo)
        sufixos_alvo = ['_F1', '_SIM']
        
        # Estrutura: {(campo, técnica, sufixo): {modelo: media_score}}
        campo_tecnica_dados = {}
        
        colunas_alvo = [c for c in df_consolidado.columns 
                        if any(c.endswith(s) for s in sufixos_alvo) and c != col_id_nome]
        
        for col in colunas_alvo:
            # Identifica o sufixo
            sufixo = next((s for s in sufixos_alvo if col.endswith(s)), None)
            if not sufixo:
                continue
            
            base = col[:-len(sufixo)]  # modelo_campo_tecnica
            
            # Identifica o modelo pelo prefixo mais longo que casa
            modelo_match = None
            for m in sorted(known_models, key=len, reverse=True):
                if base.startswith(m + '_'):
                    modelo_match = m
                    break
            
            if not modelo_match:
                continue
            
            resto = base[len(modelo_match)+1:]  # campo_tecnica
            
            # Extrai técnica (última parte) e campo (restante)
            # Técnicas conhecidas (ordem importa: sbert_grande antes de sbert)
            tecnicas_conhecidas = ['bertscore', 'rouge1', 'rouge2', 'rougel', 'levenshtein',
                                   'sbert_grande', 'sbert_medio', 'sbert_pequeno', 'sbert']
            
            tecnica_match = None
            for t in tecnicas_conhecidas:
                if resto.endswith('_' + t):
                    tecnica_match = t
                    break
            
            if not tecnica_match:
                continue
            
            campo = resto[:-len('_' + tecnica_match)]
            if not campo:
                continue
            
            chave = (campo, tecnica_match, sufixo)
            if chave not in campo_tecnica_dados:
                campo_tecnica_dados[chave] = {}
            
            media_score = df_consolidado[col].mean()
            campo_tecnica_dados[chave][modelo_match] = media_score
        
        if not campo_tecnica_dados:
            return
        
        # --- PASSO 4: Determinar melhor e pior eval_loss (para marcadores ★/▲) ---
        eval_losses = {rotulo: info['eval_loss'] for rotulo, info in dados_custo.items() 
                      if info.get('eval_loss') is not None}
        
        melhor_modelo = min(eval_losses, key=eval_losses.get) if eval_losses else None
        pior_modelo = max(eval_losses, key=eval_losses.get) if eval_losses else None
        # Se só tem 1 modelo treinado, não marca pior
        if melhor_modelo == pior_modelo:
            pior_modelo = None
        
        # Lê aliases de técnicas (para título do gráfico)
        modelos_aliases = {}
        try:
            if 'Config' in xl_file.sheet_names:
                df_config = pd.read_excel(arquivo_excel, sheet_name='Config')
                for _, row in df_config.iterrows():
                    p = row.get('parametro')
                    v = row.get('valor')
                    if pd.notna(p) and p in ['sbert_pequeno', 'sbert_medio', 'sbert_grande', 'bertscore']:
                        modelos_aliases[p] = str(v) if pd.notna(v) else ''
        except Exception:
            pass
        
        # --- PASSO 5: Gerar gráficos ---
        pasta_graficos = os.path.join(pasta_saida, 'graficos')
        os.makedirs(pasta_graficos, exist_ok=True)
        
        print("\n📊 Gerando gráficos de custo-eficiência (tokens/instâncias vs Score)...")
        
        total_gerados = 0
        
        for (campo, tecnica, sufixo), modelos_score in sorted(campo_tecnica_dados.items()):
            # Determina label Y e sufixo para título/arquivo baseado na métrica
            sufixo_limpo = sufixo.lstrip('_')  # 'F1' ou 'SIM'
            if sufixo == '_SIM':
                ylabel = 'Similarity (SIM)' if lang == 'en' else 'Similaridade (SIM)'
            else:
                ylabel = 'F1 Score'
            
            # Nome da técnica para título (com alias se disponível)
            tecnica_display = tecnica.upper().replace('_', ' ')
            alias_tecnica = modelos_aliases.get(tecnica, '')
            if alias_tecnica:
                tecnica_display = f"{tecnica_display} ({alias_tecnica})"
            
            # Nome da técnica para arquivo (usa SBERTp/m/g)
            tecnica_arquivo = _tecnica_nome_arquivo.get(tecnica, tecnica)
            
            # Gera para tokens e instâncias
            for eixo, campo_custo, label_eixo, prefixo in [
                ('tokens', 'tokens', 'Accumulated Tokens' if lang == 'en' else 'Tokens Acumulados', 'custo_tk'),
                ('instancias', 'instancias', 'Accumulated Instances' if lang == 'en' else 'Instâncias Acumuladas', 'custo_inst'),
            ]:
                pontos = []
                baselines_plot = []
                
                # Modelos treinados (pontos no scatter)
                for rotulo, info in dados_custo.items():
                    if rotulo not in modelos_score:
                        continue
                    
                    score_val = modelos_score[rotulo]
                    x_val = info[campo_custo]
                    
                    if x_val is None or x_val <= 0:
                        continue
                    
                    # Marcador especial
                    if rotulo == melhor_modelo:
                        marcador = '*'
                        tamanho = 300
                    elif rotulo == pior_modelo:
                        marcador = '^'
                        tamanho = 180
                    else:
                        marcador = 'o'
                        tamanho = 120
                    
                    pontos.append({
                        'label': info['alias'],
                        'x': x_val,
                        'y': score_val,
                        'cor': info['cor'],
                        'marcador': marcador,
                        'tamanho': tamanho
                    })
                
                # Modelos baseline (linhas horizontais)
                for rotulo, info in dados_baseline.items():
                    if rotulo not in modelos_score:
                        continue
                    
                    baselines_plot.append({
                        'label': info['alias'],
                        'y': modelos_score[rotulo],
                        'cor': info['cor']
                    })
                
                if not pontos:
                    continue
                
                titulo = f"Training Cost-Efficiency — {tecnica_display} {sufixo_limpo} — {campo}" if lang == 'en' else \
                         f"Custo-Eficiência de Treinamento — {tecnica_display} {sufixo_limpo} — {campo}"
                
                nome_arquivo = f"{prefixo}_{sufixo_limpo.lower()}_{tecnica_arquivo}_{campo}.png"
                arquivo = os.path.join(pasta_graficos, nome_arquivo)
                
                UtilGraficos.gerar_scatter_custo(
                    pontos=pontos,
                    baselines=baselines_plot,
                    titulo=titulo,
                    ylabel=ylabel,
                    xlabel=label_eixo,
                    arquivo_saida=arquivo,
                    lang=lang
                )
                total_gerados += 1
        
        if total_gerados > 0:
            print(f"   ✓ {total_gerados} gráficos de custo-eficiência gerados em: {pasta_graficos}")
        else:
            print("   ⚠️ Nenhum gráfico de custo-eficiência gerado (sem dados suficientes)")
