# -*- coding: utf-8 -*-
'''
Menus interativos para seleção de arquivos, YAMLs e pastas.

Autor: Luiz Anísio

Métodos públicos:
    escolher_yaml(pasta, chave_obrigatoria, titulo, padrao_recente, limite, opcoes_extras)
        Lista arquivos YAML que contêm as chaves obrigatórias e apresenta menu de seleção.

    escolher_arquivo(pasta, mascara, titulo, padrao_recente, limite, opcoes_extras)
        Lista arquivos compatíveis com a máscara e apresenta menu de seleção.

    escolher_pasta(pasta, arquivo_obrigatorio, titulo, padrao_recente, limite, opcoes_extras)
        Lista subpastas que contêm o arquivo obrigatório e apresenta menu de seleção.

Todos os métodos retornam o caminho absoluto do item escolhido, o valor de uma
opcao_extra escolhida (qualquer tipo), ou None se o usuário sair/cancelar.

opcoes_extras é uma lista de tuplas (label, valor) adicionadas ao final do menu,
útil para ações como "Criar novo" ou "Sair":
    opcoes_extras=[
        ("Criar um novo arquivo de configuração", "CRIAR_NOVO"),
        ("Sair sem escolher", None)
    ]
'''

import os
import glob
import datetime

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ============================================================================
# Cores ANSI (auto-desativadas quando não é um terminal interativo)
# ============================================================================

def _suporta_cores() -> bool:
    import sys
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


class _C:
    '''Códigos ANSI usados no menu. Strings vazias quando não há terminal interativo.'''
    _on = None  # resolvido na primeira chamada

    @classmethod
    def _ativo(cls) -> bool:
        if cls._on is None:
            cls._on = _suporta_cores()
        return cls._on

    @classmethod
    def _a(cls, codigo: str) -> str:
        return codigo if cls._ativo() else ''

    # --- paleta ---
    @classmethod
    def reset(cls):   return cls._a('\033[0m')
    @classmethod
    def bold(cls):    return cls._a('\033[1m')
    @classmethod
    def dim(cls):     return cls._a('\033[2m')
    @classmethod
    def cyan(cls):    return cls._a('\033[96m')
    @classmethod
    def yellow(cls):  return cls._a('\033[93m')

    @classmethod
    def gray(cls):    return cls._a('\033[38;5;245m')
    @classmethod
    def green(cls):   return cls._a('\033[92m')
    @classmethod
    def red(cls):     return cls._a('\033[91m')


# ============================================================================
# Núcleo interno
# ============================================================================

def _preparar_lista(itens_todos, limite, padrao_recente):
    '''Ordena alfabeticamente, aplica limite e retorna (lista, idx_padrao base-0).'''
    itens = sorted(itens_todos)[:limite]
    idx_padrao = -1
    if padrao_recente and itens:
        idx_padrao = max(range(len(itens)), key=lambda i: os.path.getmtime(itens[i]))
    return itens, idx_padrao


def _exibir_menu(titulo, itens, idx_padrao=-1, opcoes_extras=None):
    '''
    Exibe menu numerado e retorna o valor do item escolhido.

    Parâmetros:
        titulo: str — cabeçalho exibido no menu
        itens: list[(label, valor)] — opções principais (arquivos/pastas)
        idx_padrao: int — índice base-0 do item padrão em `itens` (-1 = sem padrão)
        opcoes_extras: list[(label, valor)] — opções adicionais após os itens

    Retorna:
        O `valor` do item ou opção extra escolhida. Pode ser None se a opção
        extra de saída tiver valor None ou se o usuário interromper (Ctrl+C).
    '''
    R = _C.reset(); G = _C.gray(); Y = _C.yellow()
    sep = f"{G}{'─' * 60}{R}"

    print(f"\n{sep}")
    print(f"  {_C.bold()}{titulo}{R}")
    print(sep)

    num_padrao = None
    for i, (label, _) in enumerate(itens):
        num = i + 1
        # Separa nome e data: formato é "nome  (data)" ou "pasta/  (data)"
        if '  (' in label:
            nome, resto = label.split('  (', 1)
            data_str = f"  {G}({resto}{R}"
        else:
            nome, data_str = label, ''

        num_str = f"{G}[{_C.reset()}{_C.cyan()}{num}{R}{G}]{R}"

        if i == idx_padrao:
            num_padrao = num
            sufixo = f"  {Y}◀ mais recente{R}"
            linha_nome = f"{_C.bold()}{nome}{R}"
        else:
            sufixo = ''
            linha_nome = f"{nome}"

        print(f"  {num_str} {linha_nome}{data_str}{sufixo}")

    extras = list(opcoes_extras or [])
    extras_map = {}
    if extras:
        print(f"  {G}{'·' * 60}{R}")
        offset = len(itens)
        for j, (label, valor) in enumerate(extras):
            num = offset + j + 1
            extras_map[num] = valor
            num_str = f"{G}[{R}{G}{num}{R}{G}]{R}"
            if valor is None:
                # opção de saída/cancelar → cinza
                print(f"  {num_str} {G}{label}{R}")
            else:
                # opção de ação → ciano
                print(f"  {num_str} {_C.cyan()}{label}{R}")

    print(sep)

    padrao_str = f" {Y}(padrão {num_padrao}){R}" if num_padrao else ''
    prompt = f"  {_C.bold()}Escolha uma opção{R}{padrao_str}: "

    while True:
        try:
            entrada = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return None

        if not entrada:
            if num_padrao is not None:
                return itens[num_padrao - 1][1]
            print(f"  {Y}⚠️  Nenhum padrão definido. Digite um número.{R}")
            continue

        try:
            opcao = int(entrada)
        except ValueError:
            print(f"  {Y}⚠️  Entrada inválida. Digite um número.{R}")
            continue

        if 1 <= opcao <= len(itens):
            return itens[opcao - 1][1]
        if opcao in extras_map:
            return extras_map[opcao]
        print(f"  {Y}⚠️  Opção inválida.{R}")


def _listar_arquivos(pasta, mascaras):
    '''Lista arquivos em `pasta` que correspondem a qualquer das `mascaras` glob.'''
    pasta = pasta or '.'
    encontrados = []
    for m in mascaras:
        encontrados.extend(glob.glob(os.path.join(pasta, m)))
    # Deduplica e filtra apenas arquivos reais
    return list({os.path.normpath(f) for f in encontrados if os.path.isfile(f)})


def _yaml_contem_chaves(caminho, chaves):
    '''Verifica se o YAML contém todas as `chaves` no nível raiz.'''
    if not _HAS_YAML:
        return True  # sem PyYAML instalado, aceita todos
    try:
        with open(caminho, 'r', encoding='utf-8') as fh:
            data = _yaml.safe_load(fh)
        if not isinstance(data, dict):
            return False
        return all(k in data for k in chaves)
    except Exception:
        return False


def _label_arquivo(caminho):
    '''Gera rótulo "nome  (data)" para exibição no menu.'''
    ts = datetime.datetime.fromtimestamp(os.path.getmtime(caminho)).strftime('%Y-%m-%d %H:%M:%S')
    return f'{os.path.basename(caminho)}  ({ts})'


# ============================================================================
# API pública
# ============================================================================

def escolher_yaml(pasta='./', chave_obrigatoria=None,
                  titulo='Escolha um arquivo de configuração',
                  padrao_recente=True, limite=10, opcoes_extras=None):
    '''
    Lista arquivos YAML/YML em `pasta` que contêm todas as chaves obrigatórias
    e apresenta menu interativo de seleção.

    Parâmetros:
        pasta: diretório onde buscar (padrão: diretório atual)
        chave_obrigatoria: str ou list[str] — chave(s) exigidas no nível raiz do YAML;
                           None ou [] aceita qualquer YAML válido
        titulo: cabeçalho do menu
        padrao_recente: se True, o arquivo modificado mais recentemente é o padrão
        limite: número máximo de arquivos listados
        opcoes_extras: list[(label, valor)] — opções adicionais (ex: criar novo, sair)

    Retorna:
        Caminho absoluto do arquivo escolhido, valor de uma opção extra, ou None.
    '''
    chaves = ([chave_obrigatoria] if isinstance(chave_obrigatoria, str)
              else list(chave_obrigatoria or []))
    todos = _listar_arquivos(pasta, ['*.yaml', '*.yml'])
    if chaves:
        todos = [f for f in todos if _yaml_contem_chaves(f, chaves)]

    arquivos, idx_padrao = _preparar_lista(todos, limite, padrao_recente)

    if not arquivos and not opcoes_extras:
        aviso = f" com chave(s) {chaves}" if chaves else ""
        print(f"\n  {_C.yellow()}⚠️  Nenhum arquivo YAML encontrado em '{pasta}'{aviso}.{_C.reset()}")
        return None

    itens = [(_label_arquivo(f), os.path.abspath(f)) for f in arquivos]
    return _exibir_menu(titulo, itens, idx_padrao=idx_padrao, opcoes_extras=opcoes_extras)


def escolher_arquivo(pasta='./', mascara='*.txt',
                     titulo='Escolha o arquivo',
                     padrao_recente=True, limite=10, opcoes_extras=None):
    '''
    Lista arquivos em `pasta` compatíveis com `mascara` e apresenta menu de seleção.

    Parâmetros:
        pasta: diretório onde buscar
        mascara: padrão glob (str) ou lista de padrões (list[str]); ex: "*.json" ou ["*.yaml","*.yml"]
        titulo: cabeçalho do menu
        padrao_recente: se True, o arquivo modificado mais recentemente é o padrão
        limite: número máximo de arquivos listados
        opcoes_extras: list[(label, valor)] — opções adicionais

    Retorna:
        Caminho absoluto do arquivo escolhido, valor de uma opção extra, ou None.
    '''
    mascaras = [mascara] if isinstance(mascara, str) else list(mascara)
    todos = _listar_arquivos(pasta, mascaras)
    arquivos, idx_padrao = _preparar_lista(todos, limite, padrao_recente)

    if not arquivos and not opcoes_extras:
        print(f"\n  {_C.yellow()}⚠️  Nenhum arquivo encontrado em '{pasta}' com padrão '{mascara}'.{_C.reset()}")
        return None

    itens = [(_label_arquivo(f), os.path.abspath(f)) for f in arquivos]
    return _exibir_menu(titulo, itens, idx_padrao=idx_padrao, opcoes_extras=opcoes_extras)


def escolher_pasta(pasta='./', arquivo_obrigatorio='',
                   titulo='Escolha uma pasta',
                   padrao_recente=False, limite=10, opcoes_extras=None):
    '''
    Lista subpastas de `pasta` que contêm `arquivo_obrigatorio` e apresenta menu de seleção.

    Parâmetros:
        pasta: diretório pai onde buscar subpastas
        arquivo_obrigatorio: nome de arquivo que deve existir na subpasta
                             (ou '' para listar todas as subpastas)
        titulo: cabeçalho do menu
        padrao_recente: se True, a subpasta modificada mais recentemente é o padrão
        limite: número máximo de pastas listadas
        opcoes_extras: list[(label, valor)] — opções adicionais

    Retorna:
        Caminho absoluto da pasta escolhida, valor de uma opção extra, ou None.
    '''
    pasta_base = pasta or '.'
    try:
        subpastas = [
            os.path.join(pasta_base, d)
            for d in os.listdir(pasta_base)
            if os.path.isdir(os.path.join(pasta_base, d))
        ]
    except OSError as e:
        print(f"\n  {_C.red()}❌ Erro ao listar '{pasta_base}': {e}{_C.reset()}")
        return None

    if arquivo_obrigatorio:
        subpastas = [d for d in subpastas
                     if os.path.isfile(os.path.join(d, arquivo_obrigatorio))]

    pastas, idx_padrao = _preparar_lista(subpastas, limite, padrao_recente)

    if not pastas and not opcoes_extras:
        aviso = f" contendo '{arquivo_obrigatorio}'" if arquivo_obrigatorio else ""
        print(f"\n  {_C.yellow()}⚠️  Nenhuma subpasta encontrada em '{pasta_base}'{aviso}.{_C.reset()}")
        return None

    def _label_pasta(d):
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(d)).strftime('%Y-%m-%d %H:%M:%S')
        return f'{os.path.basename(d)}/  ({ts})'

    itens = [(_label_pasta(d), os.path.abspath(d)) for d in pastas]
    return _exibir_menu(titulo, itens, idx_padrao=idx_padrao, opcoes_extras=opcoes_extras)

if __name__ == '__main__':
    # Exemplo de uso
    arquivo_escolhido = escolher_arquivo(pasta='./', mascara='*.py', titulo='Selecione um .py para teste', padrao_recente=True, limite=5, opcoes_extras=[('Sair', None)])
    if arquivo_escolhido:
        print(f'Você escolheu: {arquivo_escolhido}')
    else:
        print('Nenhum arquivo escolhido.')