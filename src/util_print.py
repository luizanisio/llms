# -*- coding: utf-8 -*-

"""
Utilitários para exportação de DataFrames para Excel com formatação.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Utilitário para impressão de logs com identificação de processo.

Fornece um context manager que agrupa prints e identifica automaticamente
o processo/thread que está executando, facilitando debug em ambientes
multi-processo/multi-thread.

Exemplo de uso:
    from util_print import print_process
    
    with print_process(titulo='Processamento da Fila', pid=True) as p:
        p('Carregando dados...')
        p('Processando item 1')
        p('Status: <verde>OK</verde>')  # Com cores
        p('Erro: <vermelho>Falha crítica')  # Auto-fecha
        p('✅ Concluído')
    
    # Saída:
    ============== [PID:12713] Processamento da Fila ===============
    | Carregando dados...
    | Processando item 1
    | Status: OK (em verde)
    | Erro: Falha (em vermelho)
    | ✅ Concluído
    ================================================================

Sintaxe de cores:
    - <cor>texto</cor>  # Com fechamento explícito
    - <cor>texto        # Auto-fecha no fim da linha
    
    Cores disponíveis: azul, vermelho, amarelo, laranja, verde
    
Dicas de uso:
    - Use o context manager para agrupar mensagens relacionadas e evitar entrelaçamento quando trabalhando com múltiplos processos ou threads.
    - Combine com timestamps para rastreamento detalhado.
    - Use cores automáticas para destacar status sem precisar marcar manualmente.
    - Em ambientes multi-thread, use include_thread=True para diferenciar threads.
"""

from asyncio import sleep
import os
import regex as re
import threading
from contextlib import contextmanager
from typing import Optional, List, Dict
from datetime import datetime, time

# Códigos ANSI para cores no terminal
CORES_ANSI: Dict[str, str] = {
    'reset': '\033[0m',
    'azul': '\033[94m',
    'vermelho': '\033[91m',
    'amarelo': '\033[93m',
    'laranja': '\033[38;5;208m',
    'verde': '\033[92m',
    'cinza': '\033[38;5;243m',
    'branco': '\033[97m',
    'chumbo': '\033[38;5;253m',
    'navy': '\033[96m',  # Ciano claro para processos/peças (melhor visibilidade)
    # Gradações de cinza para branco (escala de progresso)
    '25%': '\033[38;5;243m',   # Cinza escuro
    '50%': '\033[38;5;248m',   # Cinza médio
    '75%': '\033[38;5;253m',   # Cinza claro
    '100%': '\033[38;5;255m',  # Branco puro
    # Aliases comuns
    'blue': '\033[94m',
    'red': '\033[91m',
    'yellow': '\033[93m',
    'orange': '\033[38;5;208m',
    'green': '\033[92m',
    'grey': '\033[38;5;243m',
    'white': '\033[97m',
    'darkgray': '\033[38;5;253m',
}


def aplicar_cores(texto: str, enabled: bool = True) -> str:
    """
    Aplica formatação de cores ANSI usando tags HTML-like.
    Args:
        texto: Texto com marcadores de cor
        enabled: Se False, remove os marcadores mas não aplica cores
    Returns:
        Texto com códigos ANSI de cores aplicados ou sem marcadores
    Sintaxe suportada:
        - Tags HTML-like: <cor>texto</cor> ou <cor>texto
        - Auto-fecha tags não fechadas no final da linha
    Exemplos:
        >>> aplicar_cores('Status: <verde>OK</verde>')
        >>> aplicar_cores('Status: <verde>OK')  # Auto-fecha no fim
        >>> aplicar_cores('<azul>Azul</azul> e <vermelho>vermelho')
    """
    if not enabled:
        # Remove marcadores sem aplicar cores
        texto = re.sub(r'</?[a-z]+>', '', texto)
        return texto
    
    resultado = texto
    
    # Processar tags fechadas PRIMEIRO: <cor>texto</cor>
    def substituir_fechada(match):
        cor = match.group(1).lower()
        conteudo = match.group(2)
        if cor in CORES_ANSI:
            return f"{CORES_ANSI[cor]}{conteudo}{CORES_ANSI['reset']}"
        return conteudo
    
    resultado = re.sub(r'<([a-z]+)>(.*?)</\1>', substituir_fechada, resultado)
    
    # Processar tags abertas (sem fechamento): <cor>...
    # Pega cada tag aberta e colore até a próxima tag de abertura ou fim
    def substituir_aberta(match):
        cor = match.group(1).lower()
        if cor not in CORES_ANSI:
            return match.group(0)
        
        # Pega todo o texto após a tag até encontrar outra tag de abertura ou fim
        return f"{CORES_ANSI[cor]}"
    
    # Substitui tags de abertura por códigos ANSI
    resultado = re.sub(r'<([a-z]+)>', substituir_aberta, resultado)
    
    # Remove tags de fechamento órfãs (que não têm abertura correspondente)
    resultado = re.sub(r'</[a-z]+>', CORES_ANSI['reset'], resultado)
    
    # Adiciona reset no final se houver alguma cor ativa
    if any(cod in resultado for cod in CORES_ANSI.values() if cod != CORES_ANSI['reset']):
        resultado += CORES_ANSI['reset']
    
    return resultado


def aplicar_cores_auto(msg: str) -> str:
    """
    Aplica formatação de cores automática baseada em palavras-chave e padrões.
    
    Cores aplicadas:
        - Sucesso/OK → verde
        - Erro/Falha → vermelho  
        - Atenção/Aviso → amarelo
        - Info → azul
        - Números positivos → verde, negativos → vermelho
        - Datas/Horas → laranja
        - Processos/Peças → navy (prioridade máxima)
    """
    # Sets de palavras para cada cor
    PALAVRAS_VERDE = {
        'ok', 'sucesso', 'success', 'concluído', 'concluido', 'completo', 'finalizado',
        'ativo', 'online', 'conectado', 'aprovado', 'válido', 'valido', 'correto',
        'passou', 'aceito', 'disponível', 'disponivel', '✓', '✅', 'sim', 'yes'
    }
    
    PALAVRAS_VERMELHO = {
        'erro', 'error', 'falha', 'fail', 'failed', 'crítico', 'critico', 'fatal',
        'inválido', 'invalido', 'negado', 'rejeitado', 'offline', 'desconectado',
        'timeout', 'exception', 'falhou', 'incorreto', 'bloqueado', '✗', '❌',
        'não', 'nao', 'negativo','ignorado', 'ignorado', 'unavailable', 'indisponível', 'indisponivel',
        'falha', 'falhou', 'erro', 'error', 'negado', 'rejeitado', 'offline', 'desconectado','crítica', 
        'critico', 'fatal', 'inválido', 'invalido', 'bloqueado', 'timeout', 'exception', 'incorreto','alerta', 'alert','pendente', 
    }
    
    PALAVRAS_AMARELO = {
        'atenção', 'atencao', 'aviso', 'warning', 'warn', 'alerta', 'alert',
        'cuidado', 'pendente', 'aguardando', 'em andamento',
        'parcial', 'limitado', 'lento', '⚠', '⚠️', 'importante',
        'produção', 'producao', 'homologação', 'homologacao', 'desenvolvimento',
    }
    
    PALAVRAS_AZUL = {
        'info', 'informação', 'informacao', 'dados', 'debug', 'log',
        'iniciando', 'carregando', 'verificando', 'analisando', 'lendo',
        'executando', 'iniciado', 'ℹ', 'ℹ️'
    }
    
    PALAVRAS_LARANJA = {
        'manutenção', 'manutencao', 'atualização', 'atualizacao', 'migração',
        'migracao', 'instalando', 'configurando', 'preparando'
    }
    
    # Mapa de posições coloridas (True = já colorido, False = ainda não)
    colorido = [False] * len(msg)
    
    # Lista de substituições: (start, end, cor, texto_original)
    substituicoes = []
    
    # 1. PRIORIDADE MÁXIMA: Peças processuais
    # Padrão: 20NNNNNNNNNN.N+.N* (sem word boundary no final para capturar peças que terminam com ponto)
    for match in re.finditer(r'\b(20\d{10}\.\d+\.\d*)', msg):
        start, end = match.span()
        if not any(colorido[start:end]):
            substituicoes.append((start, end, 'navy', match.group(1)))
            colorido[start:end] = [True] * (end - start)
    
    # 2. PRIORIDADE MÁXIMA: Números de processos
    for match in re.finditer(r'\b(20\d{10})\b', msg):
        start, end = match.span()
        if not any(colorido[start:end]):
            substituicoes.append((start, end, 'navy', match.group(1)))
            colorido[start:end] = [True] * (end - start)
    
    # 3. Datas e horas
    for match in re.finditer(r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b', msg):
        start, end = match.span()
        if not any(colorido[start:end]):
            substituicoes.append((start, end, 'laranja', match.group(1)))
            colorido[start:end] = [True] * (end - start)
    
    for match in re.finditer(r'\b(\d{4}-\d{2}-\d{2})\b', msg):
        start, end = match.span()
        if not any(colorido[start:end]):
            substituicoes.append((start, end, 'laranja', match.group(1)))
            colorido[start:end] = [True] * (end - start)
    
    for match in re.finditer(r'\b(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\b', msg):
        start, end = match.span()
        if not any(colorido[start:end]):
            substituicoes.append((start, end, 'laranja', match.group(1)))
            colorido[start:end] = [True] * (end - start)
    
    # 4. Palavras-chave
    for palavras, cor in [
        (PALAVRAS_VERDE, 'verde'),
        (PALAVRAS_VERMELHO, 'vermelho'),
        (PALAVRAS_AMARELO, 'amarelo'),
        (PALAVRAS_AZUL, 'azul'),
        (PALAVRAS_LARANJA, 'laranja')
    ]:
        for palavra in palavras:
            for match in re.finditer(r'\b(' + re.escape(palavra) + r')\b', msg, flags=re.IGNORECASE):
                start, end = match.span()
                if not any(colorido[start:end]):
                    substituicoes.append((start, end, cor, match.group(1)))
                    colorido[start:end] = [True] * (end - start)
    
    # 5. Números (menor prioridade)
    # Unidades comuns em logs: s, ms, h, m, d (tempo), b, kb, mb, gb, tb (tamanho), % (percentual)
    unidades = r'(?:ms|[smhd]|[kmgt]b|%)?'
    
    # Números negativos (não marca se tiver letras/números/hífen antes ou letras/números depois, exceto unidades)
    for match in re.finditer(rf'(?<![a-zA-Z0-9-])(-\d+(?:[.,]\d+)?){unidades}(?![a-zA-Z0-9])', msg):
        start, end = match.span()
        if not any(colorido[start:end]):
            substituicoes.append((start, end, 'vermelho', match.group(0)))
            colorido[start:end] = [True] * (end - start)
    
    # Números positivos com + (não marca se tiver letras/números adjacentes, permite unidades)
    for match in re.finditer(rf'(?<![a-zA-Z0-9])\+(\d+(?:[.,]\d+)?){unidades}(?![a-zA-Z0-9])', msg):
        start, end = match.span()
        if not any(colorido[start:end]):
            substituicoes.append((start, end, 'verde', match.group(0)))
            colorido[start:end] = [True] * (end - start)
    
    # Números simples (não marca se tiver letras/números/hífen adjacentes, permite unidades)
    # Ex: AUTUA-01, ABC123, 123ABC não marcam | AUTUA 01, 123s, 456kb marcam
    for match in re.finditer(rf'(?<![a-zA-Z0-9-])(\d+(?:[.,]\d+)?){unidades}(?![a-zA-Z0-9])', msg):
        start, end = match.span()
        if not any(colorido[start:end]):
            substituicoes.append((start, end, 'verde', match.group(0)))
            colorido[start:end] = [True] * (end - start)
    
    # Reconstruir texto com substituições em ordem inversa de posição
    substituicoes.sort(reverse=True)
    resultado = msg
    for start, end, cor, texto in substituicoes:
        resultado = resultado[:start] + f'<{cor}>{texto}</{cor}>' + resultado[end:]
    
    # Aplicar cores ANSI
    return aplicar_cores(resultado)


def get_process_id(include_thread: bool = False) -> str:
    """
    Retorna um identificador único do processo atual.
    Args:
        include_thread: Se True, inclui o ID da thread além do PID
    Returns:
        String no formato 'PID:12345' ou 'PID:12345/T:67890'
    """
    pid = os.getpid()
    if include_thread:
        tid = threading.get_ident()
        return f"PID:{pid}/T:{tid}"
    return f"PID:{pid}"


class PrintProcess:
    """
    Context manager para agrupar prints com identificação de processo.
    Acumula mensagens e imprime tudo de uma vez de forma atômica.
    """
    
    def __init__(
        self, 
        titulo: Optional[str] = None,
        pid: bool = True,
        include_thread: bool = False,
        separador: str = '=',
        largura: int = 70,
        timestamp: bool = False,
        color_auto: bool = True
    ):
        """Inicializa o context manager."""
        self._titulo = titulo
        self._separador = separador
        self._largura = largura
        self._timestamp = timestamp
        self._color_auto = color_auto
        self._linhas: List[str] = []
        self._process_id = get_process_id(include_thread) if pid else None
    
    def __enter__(self):
        """Inicia o contexto."""
        if self._titulo:
            if self._process_id:
                # PID em cinza, título normal
                cinza = CORES_ANSI['cinza']
                reset = CORES_ANSI['reset']
                _titulo = aplicar_cores_auto(self._titulo) if self._color_auto else aplicar_cores(self._titulo, enabled=False)
                titulo_com_pid = f"{cinza}[{self._process_id}]{reset} {_titulo}"
            else:
                titulo_com_pid = self._titulo
            self._linhas.append(self._criar_separador(titulo_com_pid))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finaliza o contexto e imprime tudo."""
        if self._titulo:
            self._linhas.append(self._criar_separador())
        
        # Imprime com | nas linhas internas
        if self._linhas:
            for i, linha in enumerate(self._linhas):
                if self._titulo and (i == 0 or i == len(self._linhas) - 1):
                    print(linha, flush=True)  # Separadores sem |
                else:
                    print(f"| {linha}", flush=True)
        
        return False
    
    def __call__(self, mensagem: str):
        """Adiciona mensagem formatada ao buffer."""
        # Aplica cores: automáticas ou manuais
        msg_formatada = aplicar_cores_auto(mensagem) if self._color_auto else aplicar_cores(mensagem)
        
        if self._timestamp:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            msg_formatada = f"{CORES_ANSI['cinza']}[{ts}]{CORES_ANSI['reset']} {msg_formatada}"
        
        self._linhas.append(msg_formatada)
    
    def _criar_separador(self, texto: Optional[str] = None) -> str:
        """Cria linha de separação centralizada."""
        cinza = CORES_ANSI['cinza']
        reset = CORES_ANSI['reset']
        
        if texto:
            # Remove códigos ANSI temporariamente para calcular largura visível
            texto_sem_cores = re.sub(r'\033\[[0-9;]+m', '', texto)
            texto_fmt = f" {texto} "
            texto_fmt_sem_cores = f" {texto_sem_cores} "
            
            espacos = self._largura - len(texto_fmt_sem_cores)
            esq = espacos // 2
            dir = espacos - esq
            return f"{cinza}{self._separador * esq}{reset}{texto_fmt}{cinza}{self._separador * dir}{reset}"
        return f"{cinza}{self._separador * self._largura}{reset}"
    
    @property
    def pid(self) -> Optional[str]:
        """Retorna o identificador do processo."""
        return self._process_id


@contextmanager
def print_process(
    titulo: Optional[str] = None,
    pid: bool = True,
    include_thread: bool = False,
    separador: str = '=',
    largura: int = 70,
    timestamp: bool = False,
    color_auto: bool = True
):
    """
    Context manager para impressão com identificação de processo.
    
    Acumula todas as mensagens durante o contexto e imprime tudo de uma vez
    ao sair, garantindo impressão atômica sem entrelaçamento em ambientes
    multi-processo/multi-thread.
    
    Args:
        titulo: Título do bloco de impressão (opcional)
        pid: Se True, inclui o PID no título
        include_thread: Se True, inclui também o ID da thread
        separador: Caractere usado nos separadores (padrão: '=')
        largura: Largura total da linha de separação (padrão: 70)
        timestamp: Se True, inclui timestamp em cada linha
        color_auto: Se True, aplica cores automaticamente. Se False, usa tags manuais <cor>texto
    
    Yields:
        Objeto PrintProcess que pode ser chamado como função e tem property .pid
    
    Exemplo com cores manuais:
        with print_process(titulo='Processamento', pid=True) as p:
            p('Status: <verde>OK</verde>')  # Com fechamento
            p('Status: <verde>OK')  # Auto-fecha no fim
            p('Erro: <vermelho>Falha crítica</vermelho>')
    
    Exemplo com cores automáticas:
        with print_process(titulo='Processamento', pid=True, color_auto=True) as p:
            p('Status: OK')  # OK detectado automaticamente e colorido de verde
            p('Erro crítico')  # Erro detectado e colorido de vermelho
        
        # Cores disponíveis: azul, vermelho, amarelo, laranja, verde
    """
    pp = PrintProcess(
        titulo=titulo,
        pid=pid,
        include_thread=include_thread,
        separador=separador,
        largura=largura,
        timestamp=timestamp,
        color_auto=color_auto
    )
    
    with pp as print_func:
        yield print_func

def print_cores(*args, color_auto: bool = True, sep: str = ' ', end: str = '\n', file=None, flush: bool = False):
    """
    Imprime mensagens no console processando cores e mantendo compatibilidade com print() nativo.
    Se color_auto=True, colore palavras e padrões (ex: 'sucesso', datas, números) automaticamente,
    além de interpretar marcadores explícitos (ex: <verde>OK</verde>).
    """
    mensagem = sep.join(str(a) for a in args)
    if color_auto:
        mensagem = aplicar_cores_auto(mensagem)
    else:
        mensagem = aplicar_cores(mensagem)
    print(mensagem, end=end, file=file, flush=flush)

def print_linha_simples(mensagem: str, pid: bool = True, include_thread: bool = False, timestamp: bool = False):
    """
    Impressão simples de uma linha com identificação de processo.
    Usa tags de cor manuais como <verde>texto</verde> ou <verde>texto.
    
    Exemplo:
        print_linha_simples('Status: <verde>OK</verde>', pid=True)
        # Saída: [PID:12345] Status: OK (em verde)
    """
    # Formata mensagem com cores manuais
    msg = aplicar_cores(mensagem)
    
    # Adiciona timestamp
    if timestamp:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        msg = f"{CORES_ANSI['cinza']}[{ts}]{CORES_ANSI['reset']} {msg}"
    
    # Adiciona PID
    if pid:
        process_id = get_process_id(include_thread)
        msg = f"{CORES_ANSI['cinza']}[{process_id}]{CORES_ANSI['reset']} {msg}"
    
    print(msg, flush=True)

def is_debug(grupo = None) -> bool:
    """Retorna True se o ambiente estiver em modo debug (variável de ambiente DEBUG=1 ou grupo 'debug')."""
    debug = os.getenv('DEBUG', '0') or ''
    return debug.lower() in ('1', 'true', 'yes') or (grupo and str(grupo).lower() == 'debug')
    
def pausa_se_debug(segundos: float = 0, grupo: str = ''):
    """Pausa a execução por um tempo se o ambiente estiver em modo debug."""
    if is_debug(grupo):
        if isinstance(segundos, (int, float)) and segundos > 0:
            sleep(segundos)

def print_log(msg: str, incluir_hora: bool = True, incluir_pid: bool = False, grupo: str = '', pausa_debug: float = 0, somente_debug: bool = False):
    """Impressão de log com timestamp, PID e grupo opcionais."""
    cinza = CORES_ANSI['cinza']
    reset = CORES_ANSI['reset']
    prefixo = ''
    _debug = is_debug(grupo)
    if somente_debug and not _debug:
        return
    
    if grupo:
        prefixo = f"{cinza}{grupo}{reset} | "
    
    if incluir_hora:
        hora = datetime.now().strftime('%H:%M:%S')
        prefixo += f"{cinza}{hora}{reset} |"
    
    if incluir_pid:
        prefixo += f" {cinza}PID#{os.getpid()}{reset} |"
    
    msg_formatada = aplicar_cores(msg, enabled=True)
    print(f"{prefixo}>> {msg_formatada}", flush=True)
    
    if pausa_debug:
        pausa_se_debug(segundos=pausa_debug, grupo=grupo)

def print_log_cores(msg: str, grupo: str = '', pausa_debug: float = 0, somente_debug = False):
    """
    Impressão de log com formatação automática de cores baseada em palavras-chave e padrões.
    
    Args:
        msg: Mensagem a ser impressa
        grupo: Grupo/categoria do log (opcional)
        pausa_debug: Tempo de pausa após impressão em segundos (opcional)
    
    Cores aplicadas automaticamente:
        - Sucesso/OK → verde
        - Erro/Falha → vermelho
        - Atenção/Aviso → amarelo
        - Info → azul
        - Números positivos → verde
        - Números negativos → vermelho
        - Datas/Horas → laranja
    
    Exemplo:
        print_log_cores('Processamento OK: 150 registros, 3 erros encontrados')
        # OK em verde, 150 em verde, erros em vermelho
    """
    _debug = UtilEnv.debug() or str(grupo).lower()=='debug'
    if somente_debug and not _debug:
        return
    # Aplicar colorização automática
    resultado = aplicar_cores_auto(msg)
    
    # Montar prefixo em cinza
    cinza = CORES_ANSI['cinza']
    reset = CORES_ANSI['reset']
    prefixo = f'{cinza}{grupo}{reset} | ' if grupo else ''
    hora = datetime.now().strftime('%H:%M:%S')
    prefixo += f"{cinza}{hora}{reset} |"
    
    print(f"{prefixo}>> {resultado}", flush=True)
    
    if pausa_debug:
        UtilEnv.pausa_debug(pausa_debug)

def print_progress_cores(msg: str, grupo: str = '', valor: int = None, total: int = None, pausa_debug: float = 0, barra_progresso: bool = True, inline: bool = None):
    """
    Impressão de progresso com formatação automática de cores baseada em palavras-chave e padrões.
    Depois de imprimir, move o cursor para o início da linha para sobrescrever na próxima chamada.
    Se a mensagem for vazia, passa para a próxima linha.
    Se receber valor e total, calcula e exibe a porcentagem de progresso além de mostrar os valores valor/total nn%.
    
    Args:
        msg: Mensagem a ser impressa
        grupo: Grupo/categoria do log (opcional)
        valor: Valor atual do progresso (opcional)
        total: Valor total do progresso (opcional)
        pausa_debug: Tempo de pausa após impressão em segundos (opcional)
        barra_progresso: Exibir barra de progresso (opcional, padrão: True)
        inline: Impressão inline (sobrescreve linha). Se None, é True quando barra_progresso é True (opcional)
    Exemplo:
        print_progress_cores('Processando arquivos...', valor=50, total=100)
        # Saída: 11:30:45 |>> Processando arquivos... [50/100 50%]
    """
    # Se mensagem vazia, pula linha
    if not msg:
        print()
        return
    
    # Definir inline: se None, usa o valor de barra_progresso
    if inline is None:
        inline = barra_progresso
    
    # Aplicar colorização automática
    resultado = aplicar_cores_auto(msg)
    
    # Adicionar progresso se valor e total fornecidos
    finalizar_linha = False
    if valor is not None and total is not None:
        percentual = (valor / total * 100) if total > 0 else 0
        
        # Cor baseada no percentual usando gradações de cinza para branco
        if percentual < 25:
            cor_prog = '25%'
        elif percentual < 70:
            cor_prog = '50%'
        elif percentual < 95:
            cor_prog = '75%'
        else:
            cor_prog = '100%'
        
        # Criar barra de progresso com 20 caracteres
        largura_barra = 20
        preenchido = int((valor / total) * largura_barra) if total > 0 else 0
        vazio = largura_barra - preenchido
        barra = f"{'█' * preenchido}{'░' * vazio}" if barra_progresso else ''
        
        barra_texto = f"{CORES_ANSI[cor_prog]}[{barra}]{CORES_ANSI['reset']}" if barra_progresso else ''
        prog_texto = f"{CORES_ANSI[cor_prog]}[{valor}/{total} {percentual:.0f}%]{CORES_ANSI['reset']}"
        resultado = f"{resultado} {barra_texto} {prog_texto}"
        
        # Se chegou a 100%, finalizar a linha
        if percentual >= 100:
            finalizar_linha = True
    
    # Montar prefixo em cinza
    cinza = CORES_ANSI['cinza']
    reset = CORES_ANSI['reset']
    prefixo = f'{cinza}{grupo}{reset} | ' if grupo else ''
    hora = datetime.now().strftime('%H:%M:%S')
    prefixo += f"{cinza}{hora}{reset} |"
    
    # Imprimir com \r para sobrescrever na próxima chamada (inline), ou com \n se finalizar ou não-inline
    if finalizar_linha or not inline:
        prefixo_linha = "\r" if inline else ""
        print(f"{prefixo_linha}{prefixo}>> {resultado}", flush=True)
    else:
        print(f"\r{prefixo}>> {resultado}", end='', flush=True)
    
    if pausa_debug:
        UtilEnv.pausa_debug(pausa_debug)

def testes():
    """Função para executar testes simples do tipo passa e raise para serem chamados na compilação do pacote."""
    # Teste 1: Números com letras adjacentes NÃO devem ser coloridos
    nao_colorir_numeros = ['AUTUA-01', 'DOC-99', 'ABC123', '456DEF', 'PROC-789']
    for texto in nao_colorir_numeros:
        resultado = aplicar_cores_auto(texto)
        assert texto in resultado or CORES_ANSI['reset'] not in resultado, f"Erro: {texto} teve números coloridos incorretamente"
    
    # Teste 2: Processos devem ser coloridos em navy
    assert CORES_ANSI['navy'] in aplicar_cores_auto('Processo 202304042888'), "Processo não colorido"
    assert CORES_ANSI['navy'] in aplicar_cores_auto('Peça 202304042888.1.2'), "Peça não colorida"
    
    # Teste 3: Palavras-chave coloridas
    assert CORES_ANSI['verde'] in aplicar_cores_auto('Status: OK'), "OK não colorido"
    assert CORES_ANSI['vermelho'] in aplicar_cores_auto('Erro crítico'), "Erro não colorido"
    assert CORES_ANSI['amarelo'] in aplicar_cores_auto('Atenção necessária'), "Atenção não colorida"
    
    # Teste 4: Números simples, positivos e negativos
    assert CORES_ANSI['verde'] in aplicar_cores_auto('Esse teste tem 123 colorido'), "123 não colorido"
    assert CORES_ANSI['verde'] in aplicar_cores_auto('Esse teste tem +123 colorido'), "+123 não colorido verde"
    assert CORES_ANSI['vermelho'] in aplicar_cores_auto('Esse teste tem -123 colorido'), "-123 não colorido vermelho"
    
    # Teste 5: Datas com separadores / e -
    assert CORES_ANSI['laranja'] in aplicar_cores_auto('Data: 2024-02-14'), "Data yyyy-mm-dd não colorida"
    assert CORES_ANSI['laranja'] in aplicar_cores_auto('Data: 14/02/2024'), "Data dd/mm/yyyy não colorida"
    
    # Teste 6: Números com unidades (tempo, tamanho, etc)
    assert CORES_ANSI['verde'] in aplicar_cores_auto('Tempo: 123s'), "123s não colorido"
    assert CORES_ANSI['verde'] in aplicar_cores_auto('Espera: 500ms'), "500ms não colorido"
    assert CORES_ANSI['verde'] in aplicar_cores_auto('Duração: 2h'), "2h não colorido"
    assert CORES_ANSI['verde'] in aplicar_cores_auto('Tamanho: 1024kb'), "1024kb não colorido"
    assert CORES_ANSI['verde'] in aplicar_cores_auto('Arquivo: 5mb'), "5mb não colorido"
    
    # Teste 7: Peças processuais COMPLETAS devem ser coloridas (CRÍTICO)
    # Este teste é crítico pois costuma falhar com modificações no pacote
    pecas_criticas = [
        ('201234567890.1.2', 'Peça normal'),
        ('203333333333.1.455', 'Peça com segundo nível maior (455)'),
        ('204444444444.10.', 'Peça terminando com ponto'),
        ('200876543210.5.10', 'Peça com primeiro nível maior'),
        ('201111111111.99.12345', 'Peça com ambos níveis grandes'),
    ]
    for peca, descricao in pecas_criticas:
        resultado = aplicar_cores_auto(f'Peça: {peca}')
        # Verificar que a peça INTEIRA está no resultado colorido
        assert peca in re.sub(r'\033\[[0-9;]+m', '', resultado), f"Peça {peca} não encontrada completa ({descricao})"
        assert CORES_ANSI['navy'] in resultado, f"Peça {peca} não colorida em navy ({descricao})"
        # Verificar que não há quebra na colorização (ex: parte colorida, parte não)
        # Se a peça está presente e tem navy, significa que foi colorida corretamente
    
    print("✅ Todos os testes passaram com sucesso!")

if __name__ == '__main__':
    # Testes
    import sys
    testes()
    print("\n" + "="*70)
    print("TESTES DO UTILITÁRIO DE PRINT COM IDENTIFICAÇÃO DE PROCESSO")
    print("="*70 + "\n")
    
    # Teste 1: Print simples com PID
    print("\n[Teste 1] Print simples com PID:")
    print_linha_simples("Esta é uma mensagem simples", pid=True)
    
    # Teste 2: Context manager básico
    print("\n[Teste 2] Context manager básico:")
    with print_process(titulo='Processamento da Fila', pid=True) as p:
        p('Carregando dados...')
        p('Processando item 1')
        p('Processando item 2')
        p('✅ Concluído')
    
    # Teste 3: Com thread ID
    print("\n[Teste 3] Com PID e Thread ID:")
    with print_process(titulo='Processamento Multi-Thread', pid=True, include_thread=True) as p:
        p('Thread principal executando')
        p('Processando dados...')
    
    # Teste 4: Com timestamp
    print("\n[Teste 4] Com timestamp:")
    with print_process(titulo='Processamento com Timestamp', pid=True, timestamp=True) as p:
        p('Início do processamento')
        p('Processando...')
        p('Fim do processamento')
    
    # Teste 5: Auto flush (imprime linha por linha)
    print("\n[Teste 5] Auto flush (tempo real):")
    with print_process(titulo='Processamento em Tempo Real', pid=True) as p:
        p('Linha 1 - impressa imediatamente')
        p('Linha 2 - impressa imediatamente')
        p('Linha 3 - impressa imediatamente')
    
    # Teste 6: Sem título
    print("\n[Teste 6] Sem título:")
    with print_process(pid=True) as p:
        p('Mensagem 1 sem título')
        p('Mensagem 2 sem título')
    
    # Teste 7: Separador customizado
    print("\n[Teste 7] Separador customizado:")
    with print_process(titulo='Processamento Especial', pid=True, separador='*', largura=60) as p:
        p('Usando separador customizado')
        p('Largura reduzida para 60 caracteres')
    
    # Teste 8: Sem PID
    print("\n[Teste 8] Sem identificação de processo:")
    with print_process(titulo='Processamento sem PID', pid=False) as p:
        p('Esta linha não tem PID')
        p('Nem esta')
    
    # Teste 9: Formatação com cores
    print("\n[Teste 9] Formatação com cores:")
    with print_process(titulo='Teste de Cores', pid=True) as p:
        p('Texto normal sem cor')
        p('')
        p('--- Com fechamento explícito ---')
        p('Status: <verde>OK</verde> - Conexão estabelecida')
        p('Erro: <vermelho>Falha crítica</vermelho> no sistema')
        p('Info: <azul>Processando dados...</azul> aguarde')
        p('')
        p('--- Auto-fechamento no fim da linha ---')
        p('Aviso: <amarelo>Atenção necessária')
        p('Alerta: <laranja>Recurso limitado')
        p('Status: <verde>Sistema online')
        p('')
        p('--- Múltiplas cores na mesma linha ---')
        p('Mix: <azul>azul</azul> <verde>verde</verde> <vermelho>vermelho</vermelho>')
        p('Status: <verde>OK</verde> | Avisos: <amarelo>3</amarelo> | Erros: <vermelho>0</vermelho>')
        p('Dashboard: <azul>Usuários: 150</azul> | <verde>Ativo | <laranja>CPU: 75%')
    
    # Teste 10: Cores automáticas no print_process
    print("\n[Teste 10] Cores automáticas no print_process:")
    with print_process(titulo='Cores Automáticas', pid=True, color_auto=True) as p:
        p('Sistema iniciado com sucesso')
        p('Processamento OK: 50 registros')
        p('Erro crítico encontrado')
        p('Aviso: memória em 75%')
        p('Info: carregando dados')
    
    # Teste 11: Print linha simples com cor
    print("\n[Teste 11] Print linha simples com cor:")
    print_log('Status do sistema: <verde>ONLINE', incluir_pid=True)
    print_log('Alerta: <laranja>Manutenção em 5 minutos</laranja>', incluir_pid=True)
    print_log('Erro: <vermelho>Falha na conexão', incluir_pid=True)
    
    # Teste 12: Print com cores automáticas
    print("\n[Teste 12] Print com cores automáticas (print_log_cores):")
    print_log_cores('Sistema iniciado com sucesso')
    print_log_cores('Processamento OK: 150 registros processados')
    print_log_cores('Erro crítico: falha na conexão com banco de dados')
    print_log_cores('Aviso: memória em 85%, atenção necessária')
    print_log_cores('Info: carregando 1500 usuários ativos')
    print_log_cores('Números: +350 positivos, -25 negativos, total: 325')
    print_log_cores('Data: 14/02/2026 às 10:45:30 - sistema online')
    print_log_cores('Status: concluído | Erros: 3 | Avisos: 12 | Info: processando')
    print_log_cores('Manutenção agendada para 2026-02-15 às 22:00')
    
    print("\n[Teste 13] Print com cores automáticas e grupo:")
    print_log_cores('OK - 100 registros validados', grupo='VALIDAÇÃO')
    print_log_cores('Erro ao conectar | Timeout após 30 segundos', grupo='CONEXÃO')
    print_log_cores('Atenção: recurso limitado, aguardando liberação', grupo='SISTEMA')
    
    print("\n[Teste 14] Print com progresso (print_progress_cores):")
    import time
    
    # Demonstrar gradações de cores
    print("  Gradações de cores na barra de progresso:")
    print_progress_cores('Progresso 10%', valor=10, total=100)
    time.sleep(0.5)
    print_progress_cores('Progresso 30%', valor=30, total=100)
    time.sleep(0.5)
    print_progress_cores('Progresso 60%', valor=60, total=100)
    time.sleep(0.5)
    print_progress_cores('Progresso 85%', valor=85, total=100)
    time.sleep(0.5)
    print_progress_cores('Progresso 100% - Concluído', valor=100, total=100)
    
    print()  # Espaçamento
    print("  Progresso contínuo:")
    for i in range(0, 101, 20):
        print_progress_cores('Processando arquivos', valor=i, total=100)
        time.sleep(0.3)
    # Linha finalizada automaticamente ao atingir 100%
    
    print()  # Espaçamento
    print_progress_cores('Baixando dados...', grupo='DOWNLOAD', valor=25, total=100)
    time.sleep(0.3)
    print_progress_cores('Baixando dados...', grupo='DOWNLOAD', valor=60, total=100)
    time.sleep(0.3)
    print_progress_cores('Baixando dados...', grupo='DOWNLOAD', valor=90, total=100)
    time.sleep(0.3)
    print_progress_cores('Download concluído com sucesso', grupo='DOWNLOAD', valor=100, total=100)
    # Linha finalizada automaticamente
    
    print("\n[Teste 15] Colorização de processos e peças processuais:")
    print_log_cores('Processo 201234567890 analisado com sucesso')
    print_log_cores('Peça processual 201234567890.1.2 vinculada ao processo 201234567890')
    print_log_cores('Erro ao processar peça 209876543210.5.10 do processo 209876543210')
    print_log_cores('Peça com segundo nível maior: 203333333333.1.455')
    print_log_cores('Peça sem segundo nível: 204444444444.10.')
    print_log_cores('Total: 3 processos (201111111111, 202222222222, 203333333333) processados')
    print_log_cores('IMPORTANTE: Estrutura fixa de peça é sempre 20nnnnnnnnnn.n+.n*')
    
    print("\n" + "="*70)
    print("FIM DOS TESTES")
    print("="*70 + "\n")

    with print_process(f'INFO: Python') as p:
            p(f'Versão atual: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}. ')
            p(f'É só uma informação que pode ser ignorada.')
