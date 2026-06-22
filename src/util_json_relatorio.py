# -*- coding: utf-8 -*-

'''
Utilitário para geração de relatórios Markdown de análises JSON.

Autor: Luiz Anísio
Data: 31/12/2025
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Gera relatórios markdown estruturados das análises de comparação JSON,
incluindo configurações, métricas, campos analisados e resultados.
Permite atualizações pontuais de seções específicas.
'''

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any


class JsonAnaliseRelatorio:
    """
    Gera e atualiza relatórios Markdown de análises JSON.
    
    O relatório é dividido em seções marcadas com códigos especiais que permitem
    atualizações pontuais sem reescrever o arquivo inteiro.
    
    Seções:
    - HEADER: Cabeçalho com título e data
    - OVERVIEW: Visão geral do experimento
    - CONFIG: Configuração da análise (métricas, campos)
    - RESULTS: Resultados principais (estatísticas globais)
    - GRAPHICS: Lista de gráficos gerados
    - LLM_EVAL: Avaliação LLM (se disponível)
    - FOOTER: Informações finais
    
    Example:
        >>> relatorio = JsonAnaliseRelatorio(pasta_saida='./resultados')
        >>> relatorio.set_overview(titulo='Comparação RAW vs Base', descricao='...')
        >>> relatorio.set_config(config_dict, campos_comparacao)
        >>> relatorio.adicionar_grafico('boxplot_f1.png', 'F1-Score por modelo')
        >>> relatorio.salvar('relatorio.md')
    """
    
    # Códigos de marcação para identificação de seções
    MARKER_START = "<!-- SECTION:"
    MARKER_END = "<!-- /SECTION:"
    
    def __init__(self, pasta_saida: str = '.', nome_arquivo: str = 'relatorio_analise.md'):
        """
        Inicializa o relatório.
        
        Args:
            pasta_saida: pasta onde o relatório será salvo
            nome_arquivo: nome do arquivo markdown
        """
        self.pasta_saida = pasta_saida
        self.nome_arquivo = nome_arquivo
        self.caminho_completo = os.path.join(pasta_saida, nome_arquivo)
        
        # Dados do relatório (seções)
        self.secoes: Dict[str, str] = {
            'HEADER': '',
            'OVERVIEW': '',
            'CONFIG': '',
            'RESULTS': '',
            'GRAPHICS': '',
            'LLM_EVAL': '',
            'OBSERVABILIDADE': '',
            'FOOTER': ''
        }
        
        # Metadados
        self.data_criacao = datetime.now()
        self.data_atualizacao = None
        
    def _wrap_section(self, nome: str, conteudo: str) -> str:
        """Envolve conteúdo com marcadores de seção."""
        if not conteudo.strip():
            return ''
        return f"{self.MARKER_START} {nome} -->\n{conteudo}\n{self.MARKER_END} {nome} -->\n"
    
    def _gerar_header(self, titulo: str = 'Relatório de Análise JSON') -> str:
        """Gera cabeçalho do relatório."""
        data_fmt = self.data_criacao.strftime('%d/%m/%Y %H:%M')
        return f"""# {titulo}

**Data de geração:** {data_fmt}

---
"""
    
    def set_overview(self, titulo: str, descricao: str, rotulos: List[str], 
                     total_documentos: int, campos_comparacao: List[str]):
        """
        Define visão geral do experimento.
        
        Args:
            titulo: título do experimento
            descricao: descrição do objetivo
            rotulos: lista de rótulos (origem + destinos)
            total_documentos: total de documentos analisados
            campos_comparacao: lista de campos comparados
        """
        rotulo_origem = rotulos[0] if rotulos else 'Origem'
        rotulos_destinos = rotulos[1:] if len(rotulos) > 1 else []
        
        # Formata listas de destinos e campos
        destinos_lista = '\n'.join([f"  - `{r}`" for r in rotulos_destinos])
        campos_lista = '\n'.join([f"  - `{c}`" for c in campos_comparacao[:10]])
        if len(campos_comparacao) > 10:
            campos_lista += f"\n  - ... e mais {len(campos_comparacao) - 10}"
        
        conteudo = f"""## 📋 Visão Geral

**Experimento:** {titulo}

**Objetivo:** {descricao}

**Escopo da análise:**
- **Origem:** `{rotulo_origem}`
- **Destinos:** {len(rotulos_destinos)} modelos/abordagens
{destinos_lista}
- **Documentos analisados:** {total_documentos}
- **Campos comparados:** {len(campos_comparacao)}
{campos_lista}

---
"""
        self.secoes['OVERVIEW'] = conteudo
    
    def set_config(self, config: Dict[str, Any], campos_comparacao: List[str]):
        """
        Define configuração da análise.
        
        Args:
            config: dicionário de configuração (do JsonAnalise)
            campos_comparacao: lista de campos analisados
        """
        # Extrai métricas configuradas
        campos_bertscore = config.get('campos_bertscore', [])
        campos_rouge = config.get('campos_rouge', [])
        campos_rouge1 = config.get('campos_rouge1', [])
        campos_rouge2 = config.get('campos_rouge2', [])
        campos_levenshtein = config.get('campos_levenshtein', [])
        
        padronizar = config.get('padronizar_simbolos', True)
        stemmer = config.get('rouge_stemmer', True)
        
        # Monta tabela de métricas por campo
        metricas_por_campo = self._mapear_metricas_por_campo(
            campos_comparacao, campos_bertscore, campos_rouge, 
            campos_rouge1, campos_rouge2, campos_levenshtein
        )
        
        modelos_aliases = config.get('modelos_aliases', {})
        tabela_modelos = ""
        if modelos_aliases:
            tabela_modelos = "\n### Modelos Utilizados nas Métricas\n\n| Métrica | Alias | Modelo |\n|---|---|---|\n"
            for param, (alias, caminho) in modelos_aliases.items():
                tabela_modelos += f"| `{param}` | `{alias}` | `{caminho}` |\n"
        
        conteudo = f"""## ⚙️ Configuração da Análise

### Configurações Ativas
- **Padronizar Símbolos:** {padronizar}
- **Stemmer ROUGE:** {stemmer}

### Métricas Utilizadas

**Filosofia de seleção:**
1. **BERTScore** → Similaridade semântica profunda (textos longos)
2. **ROUGE-L** → Sequências estruturadas (ordem importa)
3. **ROUGE-2** → Precisão de bigramas (fraseamento técnico)
4. **ROUGE-1** → Termos individuais (palavras-chave)
5. **Levenshtein** → Distância de edição (textos curtos exatos)

### Distribuição de Métricas por Campo

{self._formatar_tabela_metricas(metricas_por_campo)}
{tabela_modelos}
**Campos especiais:**
- `(global)`: Visão geral do documento completo
- `(estrutura)`: Acurácia estrutural (campos presentes/ausentes)

---
"""
        self.secoes['CONFIG'] = conteudo
    
    def set_results(self, estatisticas: 'pd.DataFrame', melhor_modelo: Optional[Dict[str, Any]] = None):
        """
        Define resultados principais.
        
        Args:
            estatisticas: DataFrame de estatísticas globais
            melhor_modelo: dict com info do melhor modelo {'nome', 'metrica', 'f1', 'tecnica'}
        """
        # Agrupa por técnica para (global)
        tecnicas_global = {}
        for _, row in estatisticas.iterrows():
            metrica = row['metrica']
            if '(global)' in metrica and '_F1' in metrica:
                tecnica = row['tecnica']
                if tecnica not in tecnicas_global:
                    tecnicas_global[tecnica] = []
                tecnicas_global[tecnica].append({
                    'modelo': row['modelo'],
                    'mean': row['mean'],
                    'median': row['median'],
                    'std': row['std']
                })
        
        # Monta seção de resultados
        conteudo = "## 📊 Resultados Principais\n\n"
        
        # Melhor modelo
        if melhor_modelo:
            conteudo += f"""### 🏆 Melhor Modelo

- **Modelo:** `{melhor_modelo['nome']}`
- **Métrica:** {melhor_modelo['metrica']} ({melhor_modelo['tecnica']})
- **F1-Score:** {melhor_modelo['f1']:.4f}

"""
        
        # Estatísticas por técnica
        conteudo += "### F1-Score Global por Técnica\n\n"
        
        for tecnica in sorted(tecnicas_global.keys()):
            conteudo += f"**{tecnica}:**\n\n"
            conteudo += "| Modelo | Mean | Median | Std |\n"
            conteudo += "|--------|------|--------|-----|\n"
            
            modelos = sorted(tecnicas_global[tecnica], key=lambda x: x['mean'], reverse=True)
            for m in modelos:
                conteudo += f"| {m['modelo']} | {m['mean']:.4f} | {m['median']:.4f} | {m['std']:.4f} |\n"
            conteudo += "\n"
        
        conteudo += "---\n"
        self.secoes['RESULTS'] = conteudo
    
    def adicionar_grafico(self, arquivo: str, descricao: str, categoria: str = 'Métricas'):
        """
        Adiciona gráfico ao relatório.
        
        Args:
            arquivo: nome do arquivo de gráfico
            descricao: descrição do gráfico
            categoria: categoria do gráfico (Métricas, Tokens, Avaliação LLM, etc)
        """
        # Inicializa seção de gráficos se vazia
        if not self.secoes['GRAPHICS']:
            self.secoes['GRAPHICS'] = "## 📈 Gráficos Gerados\n\n"
        
        # Adiciona entrada (agrupa por categoria internamente)
        # Formato: [categoria] arquivo - descrição
        self.secoes['GRAPHICS'] += f"- **[{categoria}]** `{arquivo}` - {descricao}\n"
    
    def set_graficos_completo(self, graficos: List[Dict[str, str]]):
        """
        Define lista completa de gráficos (reescreve seção).
        Agrupa por categoria e resume em formato de tabela.
        
        Args:
            graficos: lista de dicts {'arquivo', 'descricao', 'categoria'}
        """
        if not graficos:
            self.secoes['GRAPHICS'] = ''
            return
        
        # Agrupa por categoria
        por_categoria = {}
        for g in graficos:
            cat = g.get('categoria', 'Outros')
            if cat not in por_categoria:
                por_categoria[cat] = []
            por_categoria[cat].append(g)
        
        conteudo = "## 📈 Gráficos Gerados\n\n"
        
        for categoria in sorted(por_categoria.keys()):
            conteudo += f"### {categoria}\n\n"
            
            # Agrupa gráficos da mesma métrica/campo
            graficos_cat = por_categoria[categoria]
            
            # Extrai padrões comuns para agrupar
            grupos = self._agrupar_graficos_por_padrao(graficos_cat)
            
            if len(grupos) > 5:
                # Se há muitos grupos, usa formato de tabela compacta
                conteudo += "| Campo/Métrica | Gráficos |\n"
                conteudo += "|---------------|----------|\n"
                
                for nome_grupo, arquivos in sorted(grupos.items()):
                    # Lista apenas os tipos de métricas (P, R, F1, etc)
                    metricas = []
                    for arq in arquivos:
                        # Extrai métrica do nome do arquivo
                        partes = arq.split('_')
                        if partes:
                            metrica = partes[-1].replace('.png', '').upper()
                            if metrica not in metricas:
                                metricas.append(metrica)
                    
                    metricas_str = ', '.join(sorted(metricas))
                    conteudo += f"| {nome_grupo} | {len(arquivos)} gráfico(s): {metricas_str} |\n"
            else:
                # Se são poucos, lista individualmente
                for g in graficos_cat:
                    conteudo += f"- `{g['arquivo']}` - {g['descricao']}\n"
            
            conteudo += f"\n**Total:** {len(graficos_cat)} gráfico(s)\n\n"
        
        conteudo += "---\n"
        self.secoes['GRAPHICS'] = conteudo
    
    def _agrupar_graficos_por_padrao(self, graficos: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """
        Agrupa gráficos por padrão comum (campo/métrica).
        
        Returns:
            Dict com {nome_grupo: [lista de arquivos]}
        """
        grupos = {}
        
        for g in graficos:
            arquivo = g['arquivo']
            
            # Remove prefixos comuns e extensão
            nome_base = arquivo.replace('grafico_bp_', '').replace('grafico_', '').replace('boxplot_', '').replace('.png', '')
            
            # Identifica o grupo (tudo exceto a última parte que geralmente é a métrica)
            partes = nome_base.split('_')
            
            if len(partes) > 1:
                # Remove a métrica final (P, R, F1, SIM, etc)
                possiveis_metricas = ['p', 'r', 'f1', 'sim', 'ls', 'input', 'output', 'total', 'cache', 'seg', 'rev', 'it', 'agt', 'qtd', 'bytes', 'ok']
                if partes[-1].lower() in possiveis_metricas:
                    grupo = '_'.join(partes[:-1])
                else:
                    grupo = nome_base
            else:
                grupo = nome_base
            
            # Formata nome do grupo
            grupo_formatado = grupo.replace('_', ' ').title()
            
            if grupo_formatado not in grupos:
                grupos[grupo_formatado] = []
            grupos[grupo_formatado].append(arquivo)
        
        return grupos
    
    def set_avaliacao_llm(self, tem_global: bool = False, tem_campos: bool = False,
                         num_graficos: int = 0, metricas_disponiveis: List[str] = None):
        """
        Define seção de avaliação LLM.
        
        Args:
            tem_global: se tem métricas globais
            tem_campos: se tem métricas por campo
            num_graficos: número de gráficos gerados
            metricas_disponiveis: lista de métricas disponíveis (P, R, F1, nota, etc)
        """
        if not tem_global and not tem_campos:
            self.secoes['LLM_EVAL'] = ''
            return
        
        metricas_disponiveis = metricas_disponiveis or ['P', 'R', 'F1', 'nota', 'explicacao']
        
        conteudo = f"""## 🤖 Avaliação LLM (LLM as a Judge)

**Escopo:**
- Avaliação global: {'✅ Sim' if tem_global else '❌ Não'}
- Avaliação por campo: {'✅ Sim' if tem_campos else '❌ Não'}

**Métricas calculadas:** {', '.join(metricas_disponiveis)}

**Gráficos gerados:** {num_graficos} boxplots

**Abas no Excel:**
- `Avaliação LLM`: Métricas globais por modelo
- `Avaliação LLM Campos`: Métricas detalhadas por campo

---
"""
        self.secoes['LLM_EVAL'] = conteudo
    
    def set_observabilidade(self, tem_dados: bool = False, num_graficos: int = 0):
        """
        Define seção de observabilidade.
        
        Args:
            tem_dados: se tem dados de observabilidade
            num_graficos: número de gráficos gerados
        """
        if not tem_dados:
            self.secoes['OBSERVABILIDADE'] = ''
            return
        
        conteudo = f"""## 📊 Observabilidade

**Métricas de execução:**
- **SEG** - Tempo de execução em segundos
- **REV** - Número de revisões/tentativas realizadas
- **IT** - Iterações executadas no processamento
- **AGT** - Número de agentes utilizados
- **QTD** - Quantidade de campos preenchidos (somente origem)
- **BYTES** - Tamanho dos dados por campo em bytes (somente origem)
- **OK** - Status de sucesso da execução (0=erro, 1=sucesso)

**Gráficos gerados:** {num_graficos} boxplots

**Aba no Excel:**
- `Observabilidade`: Métricas de execução por modelo/agente

---
"""
        self.secoes['OBSERVABILIDADE'] = conteudo
    
    def set_footer(self, tempo_processamento: Optional[float] = None,
                   arquivos_gerados: Optional[List[str]] = None):
        """
        Define rodapé do relatório.
        
        Args:
            tempo_processamento: tempo total em segundos
            arquivos_gerados: lista de arquivos gerados
        """
        data_fmt = (self.data_atualizacao or self.data_criacao).strftime('%d/%m/%Y %H:%M')
        
        conteudo = f"""## 📁 Arquivos Gerados

"""
        if arquivos_gerados:
            for arq in arquivos_gerados:
                nome_arq = os.path.basename(arq)
                conteudo += f"- `{nome_arq}`\n"
        else:
            conteudo += "_Nenhum arquivo listado_\n"
        
        conteudo += f"\n---\n\n"
        
        if tempo_processamento:
            conteudo += f"**Tempo de processamento:** {tempo_processamento:.2f}s\n\n"
        
        conteudo += f"**Última atualização:** {data_fmt}\n"
        
        self.secoes['FOOTER'] = conteudo
    
    def salvar(self, arquivo: Optional[str] = None) -> str:
        """
        Salva relatório em arquivo markdown.
        
        Args:
            arquivo: caminho do arquivo (usa self.caminho_completo se None)
        
        Returns:
            Caminho do arquivo salvo
        """
        caminho = arquivo or self.caminho_completo
        
        # Garante que pasta existe
        pasta = os.path.dirname(caminho)
        if pasta and not os.path.exists(pasta):
            os.makedirs(pasta, exist_ok=True)
        
        # Monta conteúdo completo
        conteudo_completo = ""
        
        # Adiciona header sem marcadores (sempre reescrito)
        titulo = self._extrair_titulo_overview() or 'Relatório de Análise JSON'
        conteudo_completo += self._gerar_header(titulo)
        
        # Adiciona seções na ordem
        ordem = ['OVERVIEW', 'CONFIG', 'RESULTS', 'GRAPHICS', 'OBSERVABILIDADE', 'LLM_EVAL', 'FOOTER']
        for secao in ordem:
            if self.secoes[secao]:
                conteudo_completo += self._wrap_section(secao, self.secoes[secao])
        
        # Escreve arquivo
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(conteudo_completo)
        
        self.data_atualizacao = datetime.now()
        return caminho
    
    def atualizar_secao(self, nome_secao: str, conteudo: str):
        """
        Atualiza uma seção específica do relatório existente.
        
        Args:
            nome_secao: nome da seção (OVERVIEW, CONFIG, RESULTS, etc)
            conteudo: novo conteúdo da seção
        """
        if nome_secao not in self.secoes:
            raise ValueError(f"Seção '{nome_secao}' inválida. Use: {list(self.secoes.keys())}")
        
        self.secoes[nome_secao] = conteudo
        self.data_atualizacao = datetime.now()
    
    def carregar_existente(self, arquivo: Optional[str] = None) -> bool:
        """
        Carrega relatório existente para atualização parcial.
        
        Args:
            arquivo: caminho do arquivo (usa self.caminho_completo se None)
        
        Returns:
            True se carregou com sucesso, False se arquivo não existe
        """
        caminho = arquivo or self.caminho_completo
        
        if not os.path.exists(caminho):
            return False
        
        with open(caminho, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        
        # Extrai seções existentes
        for nome_secao in self.secoes.keys():
            inicio = f"{self.MARKER_START} {nome_secao} -->"
            fim = f"{self.MARKER_END} {nome_secao} -->"
            
            if inicio in conteudo and fim in conteudo:
                idx_inicio = conteudo.index(inicio) + len(inicio)
                idx_fim = conteudo.index(fim)
                secao_conteudo = conteudo[idx_inicio:idx_fim].strip()
                self.secoes[nome_secao] = secao_conteudo + "\n"
        
        return True
    
    # ═════════════════════════════════════════════════════════════════════
    # Métodos auxiliares de formatação
    # ═════════════════════════════════════════════════════════════════════
    
    def _formatar_lista(self, items: List[str], prefixo: str = '', max_items: int = None) -> str:
        """Formata lista como bullets markdown."""
        if not items:
            return f"{prefixo}- _(nenhum)_"
        
        items_exibir = items[:max_items] if max_items else items
        # Garante prefixo consistente para todos os itens
        linhas = [f"{prefixo}  - `{item}`" for item in items_exibir]
        
        if max_items and len(items) > max_items:
            linhas.append(f"{prefixo}  - ... e mais {len(items) - max_items}")
        
        return '\n'.join(linhas)
    
    def _mapear_metricas_por_campo(self, campos: List[str], 
                                   ber: List[str], rl: List[str], 
                                   r1: List[str], r2: List[str], lev: List[str]) -> Dict[str, List[str]]:
        """Mapeia quais métricas cada campo usa."""
        mapa = {}
        for campo in campos:
            metricas = []
            if campo in ber:
                metricas.append('BERTScore')
            if campo in rl:
                metricas.append('ROUGE-L')
            if campo in r1:
                metricas.append('ROUGE-1')
            if campo in r2:
                metricas.append('ROUGE-2')
            if campo in lev:
                metricas.append('Levenshtein')
            mapa[campo] = metricas if metricas else ['(padrão)']
        return mapa
    
    def _formatar_tabela_metricas(self, metricas_por_campo: Dict[str, List[str]]) -> str:
        """Formata tabela de métricas por campo."""
        if not metricas_por_campo:
            return "_Nenhum campo configurado_"
        
        linhas = ["| Campo | Métricas |", "|-------|----------|"]
        
        for campo, metricas in sorted(metricas_por_campo.items()):
            metricas_str = ', '.join(metricas)
            linhas.append(f"| `{campo}` | {metricas_str} |")
        
        return '\n'.join(linhas)
    
    def _extrair_titulo_overview(self) -> Optional[str]:
        """Extrai título do experimento da seção OVERVIEW."""
        overview = self.secoes.get('OVERVIEW', '')
        if '**Experimento:**' in overview:
            linha = [l for l in overview.split('\n') if '**Experimento:**' in l][0]
            return linha.split('**Experimento:**')[1].strip()
        return None
