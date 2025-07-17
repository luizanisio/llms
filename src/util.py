# -*- coding: utf-8 -*-

'''
 Autor Luiz Anísio 10/05/2025
 Utilitários simples para facilitar alguns códigos comuns
 '''

import os
import hashlib
from datetime import datetime
import json
import random
import string
from multiprocessing import cpu_count
try:
    import psutil 
    PSUTIL_OK = True
except:
    PSUTIL_OK = False

HASH_BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
class Util():
    @classmethod
    def verifica_versao(cls):
        print(f'Util carregado corretamente em {cls.data_hora_str()}!')
  
    @classmethod
    def hash_file(clss, arquivo, complemento=''):
        # BUF_SIZE is totally arbitrary, change for your app!
        # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python

        md5 = hashlib.md5()
        with open(arquivo, 'rb') as f:
            while True:
                data = f.read(HASH_BUF_SIZE)
                if not data:
                    break
                md5.update(data)

        return f'{md5.hexdigest()}{complemento}'

    @classmethod
    def listar_arquivos(cls, pasta, extensao='txt', inicio=''):
        if not os.path.isdir(pasta):
            msg = f'Não foi encontrada a pasta "{pasta}" para listar os arquivos "{extensao}"'
            raise Exception(msg)
        res = []
        _inicio = str(inicio).lower()
        _extensao = f".{extensao}".lower() if extensao else ''
        for path, dir_list, file_list in os.walk(pasta):
            for file_name in file_list:
                if (not inicio) and file_name.lower().endswith(f"{_extensao}"):
                    res.append(os.path.join(path,file_name))
                elif file_name.lower().endswith(f"{_extensao}") and file_name.lower().startswith(f"{_inicio}"):
                    res.append(os.path.join(path,file_name))
        return res

    @classmethod
    def data_hora_str(cls, datahora = None):
        if not datahora:
           return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
           return datahora.strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def tempo_arquivo(cls, arquivo):
        dt_arquivo = cls.data_arquivo(arquivo)
        if (not dt_arquivo):
            return -1
        return (datetime.now() - dt_arquivo).total_seconds()

    @classmethod
    def data_arquivo_str(cls, arquivo):
        if not os.path.isfile(arquivo):
            return ''
        return Util.data_hora_str( cls.data_arquivo(arquivo) )           

    @classmethod
    def data_arquivo(cls, arquivo):
        if not os.path.isfile(arquivo):
            return None
        return datetime.fromtimestamp(os.path.getmtime(arquivo))

    @classmethod
    def gravar_json(cls, arquivo, dados):
        with open(arquivo, 'w') as f:
            f.write(json.dumps(dados, indent = 2))

    @classmethod
    def ler_json(cls, arquivo, padrao = dict({}) ):
        if os.path.isfile(arquivo):
            with open(arquivo, 'r') as f:
                 dados = f.read()
            if len(dados)>2 and dados.find('{')>=0 and dados[-5:].find('}')>0:
                return json.loads(dados)
        return padrao

    @classmethod
    def gravar_lista_json(cls, arquivo, dados):
        with open(arquivo, 'w') as f:
            for linha in dados:
                f.write(json.dumps(linha) + '\n')

    @classmethod
    def ler_lista_json(cls, arquivo, padrao = []):
        res = padrao
        if os.path.isfile(arquivo):
            res = []
            with open(arquivo, 'r') as f:
                 dados = f.read()
            if len(dados)>2:
               dados = dados.split('\n')
               for linha in dados: 
                   linha = linha.strip()
                   if linha[:1] == '{' and linha[-1:] == '}':
                      res.append(json.loads(linha))
        return res

    # --- NÚMERO DE CPUS -----
    @staticmethod
    def cpus(uma_livre=True):
        num_livres = 1 if uma_livre else 0
        return cpu_count() if cpu_count() < 3 else cpu_count() - num_livres

    @staticmethod
    def get_token():
        tamanho = random.randint(10,15)
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(tamanho))

    @classmethod
    def flatten_listas(cls,lista):
        '''Recebe uma lista com listas dentro e retorna uma lista flat
           - se o valor recebido não for lista, devolve como está
        ''' 
        if not isinstance(lista, list):
            return lista
        # se algum elemento for lista, achata para o primeiro nível
        nova = []
        for item in lista:
            if isinstance(item, list):
                nova.extend(cls.flatten_listas(item))
            else:
                nova.append(item)
        return nova

    @classmethod
    def mensagem_to_json(cls, mensagem:str, padrao = dict({}) ):
        ''' Foco em receber uma mensagem via IA generativa e identificar o json dentro dela 
            Exemplo: dicionario = Util.mensagem_to_json('```json\n{"chave":"valor qualquer", "numero":1}')
        '''
        if isinstance(mensagem, dict):
            return mensagem
        if not isinstance(mensagem, str):
           raise ValueError('mensagem_to_json: parâmetro precisa ser string')
        _mensagem = str(mensagem).strip()
        # limpa resposta ```json ````
        chave_json = mensagem.find('```json\n')
        if chave_json >= 0:
           _mensagem = _mensagem[chave_json+8:]
        else:
           chave_json = mensagem.find('```json')
           _mensagem = _mensagem[chave_json+7:] if chave_json >=0 else _mensagem
            
        chave_ini = _mensagem.find('{')
        chave_fim = _mensagem.rfind('}')
        if len(_mensagem)>2 and chave_ini>=0 and chave_fim>0 and chave_fim > chave_ini:
            _mensagem = _mensagem[chave_ini:chave_fim+1]
            #print(f'MENSAGEM FINAL: {_mensagem}')
            try:
                return json.loads(_mensagem)
            except Exception as e:
                if 'delimiter' not in str(e):
                    raise e
                # corrige aspas internas dentro do json
                return Util.mensagem_to_json(mensagem = Util.escape_json_string_literals(mensagem), padrao = padrao)
                
        return padrao

    @staticmethod
    def hash_string_sha1(texto):
        ''' retorna o sha1 do texto ou json recebido '''
        if isinstance(texto, dict):
           _txt = json.dumps(texto, sort_keys = True).encode("utf-8")
        else:
           _txt = '|'.join(texto) if type(texto) is list else str(texto)
        return hashlib.sha1(_txt.encode('utf-8')).hexdigest()    

    @staticmethod
    def hash_string_md5(texto):
        ''' retorna o md5 do texto ou json recebido '''
        if isinstance(texto, dict):
           _txt = json.dumps(texto, sort_keys = True).encode("utf-8")
        else:
           _txt = '|'.join(texto) if type(texto) is list else str(texto)
        hash_object = hashlib.md5(_txt.encode())
        return str(hash_object.hexdigest())        

    @staticmethod
    def map_thread(func, lista, n_threads=None):
        if (not n_threads) or (n_threads<2):
            n_threads = Util.cpus()
        #print('Iniciando {} threads'.format(n_threads))
        pool = ThreadPool(n_threads)
        pool.map(func, lista)
        pool.close()
        pool.join()

    @staticmethod
    def map_thread_transform(func, lista, n_threads=None, msg_progresso=None):
        ''' recebe uma lista e transforma os dados da própria lista usando a função
            a função vai receber o valor de cada item e o retorno dela vai substituir o valor do item na lista
        '''
        prog=[0]
        def _transforma(i):
            if msg_progresso is not None:
                Util.progress_bar(prog[0], len(lista),f'{msg_progresso} {prog[0]+1}/{len(lista)}')
                prog[0] = prog[0] + 1
            lista[i] = func(lista[i])
        Util.map_thread(func=_transforma, lista= range(len(lista)), n_threads=n_threads)
        if msg_progresso is not None:
            Util.progress_bar(1,1) #finaliza a barra

    @staticmethod
    def progress_bar(current_value, total, msg=''):
        increments = 50
        percentual = int((current_value / total) * 100)
        i = int(percentual // (100 / increments))
        text = "\r[{0: <{1}}] {2:.2f}%".format('=' * i, increments, percentual)
        print('{} {}                     '.format(text, msg), end="\n" if percentual == 100 else "")

    @staticmethod
    def mensagem_inline(msg=''):
        i = random.randint(0,1)
        comp = "  . . ." if i==0 else ' . . .'
        comp=comp.ljust(100)
        print(f'\r{msg}{comp}'.ljust(80), end="\n" if not msg else "" )

    @staticmethod
    def pausa(segundos, progresso = True):
        if segundos ==0:
            return
        if segundos<1:
            segundos =1
        for ps in range(0,segundos):
            time.sleep(1)
            Util.progress_bar(ps,segundos+1,f'Pausa por {segundos-ps}s')
        Util.progress_bar(segundos,segundos,f'Pausa finalizada {segundos}s')

    @classmethod
    def dados_python(cls, retornar_dados_do_hardware:bool = True):
        res = {}
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        res['version'] = python_version
        virtual_env = os_environ.get('VIRTUAL_ENV')
        env_name = None
        if virtual_env:
            # Extraíndo o nome do ambiente a partir do caminho completo
            env_name = os_path.basename(virtual_env)
        if not env_name:
            python_executable_path = sys.executable
            possible_env_path = os_path.dirname(os_path.dirname(python_executable_path))
            if os_path.exists(os_path.join(possible_env_path, 'pyvenv.cfg')) or \
            os_path.exists(os_path.join(possible_env_path, 'conda-meta')):
                env_name = os_path.basename(possible_env_path)
        if not env_name:
            env_name = 'nenhum'
        res['env'] = env_name
        if retornar_dados_do_hardware:
            res['hardware'] = cls.dados_hardware()
        res = {'python': res}
        return res

    @classmethod
    def dados_hardware(cls):
        # pip install psutil
        if not PSUTIL_OK:
            raise Exception('dados_maquina: É necessário instalar o pacote psutil: pip install psutil')
        mem = psutil.virtual_memory()
        memoria_disponivel_gb = mem.available / (1024 ** 3)  # Convertendo bytes para GB
        memoria_total_gb = mem.total / (1024 ** 3)
        uso_cpu_percentual = psutil.cpu_percent(interval=1) # pausa 1s para calcular o uso da CPU
        uso_cpu_percentual_pid = psutil.Process(os_getpid()).cpu_percent(interval=1) # pausa 1s para calcular o uso da CPU do processo
        cpus_fisicas = psutil.cpu_count(logical=False)
        cpus_logicas = psutil.cpu_count(logical=True)
        disco = psutil.disk_usage('/')
        disco_uso_percentual = disco.percent
        return {'cpus_fisicas': cpus_fisicas, 
                'cpus_logicas': cpus_logicas,
                'mem_disponivel_gb': round(memoria_disponivel_gb,3),
                'mem_total_gb' : round(memoria_total_gb,3),
                'disco_uso_%' : round(disco_uso_percentual,3),
                'cpu_uso_%': uso_cpu_percentual,
                'cpu_uso_processo_%': uso_cpu_percentual_pid}

    @classmethod
    def escape_json_string_literals(s: str) -> str:
        ''' Percorre a string 's' e escapa apenas as aspas duplas internas
            não-escapadas dentro de literais JSON.
            Exemplo: texto_json = Util.escape_json_string_literals('{"chave":" valor com "aspas" dentro do texto do campo "chave" "}')
        '''
        if not isinstance(s, str):
           raise TypeError(f'escape_json_string_literals espera receber uma string e recebeu {type(s)}')
        out = []
        in_str = False       # estamos dentro de um literal de string?
        prev_esc = False     # o caractere anterior foi '\', sinalizando escape?
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if c == '"' and not prev_esc:
                # se não estiver escapada, pode ser início/fim de literal ou citação interna
                if not in_str:
                    # abre literal
                    in_str = True
                    out.append(c)
                else:
                    # dentro de literal: decide se fecha ou é interna
                    # olha à frente pelo próximo non-space
                    j = i + 1
                    while j < n and s[j].isspace():
                        j += 1
                    next_char = s[j] if j < n else ''
                    # se for caractere de estrutura JSON, fecha literal
                    if next_char in {',', '}', ']', ':'}:
                        in_str = False
                        out.append(c)
                    else:
                        # caso contrário, é aspas interna → escapa
                        out.append('\\"')
                i += 1
            elif c == '\\':
                # qualquer '\' pode iniciar ou terminar um par de escapes
                out.append(c)
                prev_esc = not prev_esc
                i += 1
            else:
                # caractere normal ou escapado (prev_esc já controla)
                out.append(c)
                prev_esc = False
                i += 1
        return ''.join(out) 
