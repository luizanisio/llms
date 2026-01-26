# -*- coding: utf-8 -*-

'''
 Autor Luiz Anísio 10/05/2025
 Utilitários simples para facilitar alguns códigos comuns
 Fonte: https://github.com/luizanisio/llms/tree/main/src
 '''

from glob import glob
import os, sys
import hashlib
from datetime import datetime, timedelta, timezone
import json
import random
import string
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from cryptography.fernet import Fernet
from typing import List, Optional, Union, Tuple, Set
import shutil
import threading
import time
import regex as re

try:
    import psutil 
    PSUTIL_OK = True
except:
    PSUTIL_OK = False
try:
    from dotenv import load_dotenv
    DOTENV_OK = True
except:
    DOTENV_OK = False

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
    @classmethod
    def cpus(cls, uma_livre=True):
        num_livres = 1 if uma_livre else 0
        return cpu_count() if cpu_count() < 3 else cpu_count() - num_livres

    @classmethod
    def get_token(cls):
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
    def mensagem_to_json(cls, mensagem:str, padrao = dict({}), _corrigir_json_ = True ):
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
            
            # Correção de trailing commas (vírgulas antes de } ou ])
            # Exemplo: {"key": "value",} -> {"key": "value"}
            # _mensagem = re.sub(r',(\s*})', r'\1', _mensagem)  # Remove vírgula antes de }
            # _mensagem = re.sub(r',(\s*\])', r'\1', _mensagem)  # Remove vírgula antes de ]
            
            #print(f'MENSAGEM FINAL: {_mensagem}')
            try:
                return json.loads(_mensagem)
            except Exception as e:
                if (not _corrigir_json_) or 'delimiter' not in str(e):
                    raise e
                # corrige aspas internas dentro do json
                return cls.mensagem_to_json(mensagem = cls.escape_json_string_literals(mensagem), 
                                             padrao = padrao, 
                                             _corrigir_json_ = False)
                
        return padrao

    @classmethod
    def hash_string_sha1(cls, texto):
        ''' retorna o sha1 do texto ou json recebido '''
        if isinstance(texto, dict):
           _txt = json.dumps(texto, sort_keys = True).encode("utf-8")
        else:
           _txt = '|'.join(texto) if type(texto) is list else str(texto)
        return hashlib.sha1(_txt.encode('utf-8')).hexdigest()    

    @classmethod
    def hash_string_md5(cls, texto):
        ''' retorna o md5 do texto ou json recebido '''
        if isinstance(texto, dict):
           _txt = json.dumps(texto, sort_keys = True).encode("utf-8")
        else:
           _txt = '|'.join(texto) if type(texto) is list else str(texto)
        hash_object = hashlib.md5(_txt.encode())
        return str(hash_object.hexdigest())        

    @classmethod
    def map_thread(cls, func, lista, n_threads=None):
        if (not n_threads) or (n_threads<2):
            n_threads = cls.cpus()
        #print('Iniciando {} threads'.format(n_threads))
        pool = ThreadPool(n_threads)
        pool.map(func, lista)
        pool.close()
        pool.join()

    @classmethod
    def map_thread_transform(cls, func, lista, n_threads=None, msg_progresso=None):
        ''' recebe uma lista e transforma os dados da própria lista usando a função
            a função vai receber o valor de cada item e o retorno dela vai substituir o valor do item na lista
        '''
        prog=[0]
        def _transforma(i):
            if msg_progresso is not None:
                cls.progress_bar(prog[0], len(lista),f'{msg_progresso} {prog[0]+1}/{len(lista)}')
                prog[0] = prog[0] + 1
            lista[i] = func(lista[i])
        cls.map_thread(func=_transforma, lista= range(len(lista)), n_threads=n_threads)
        if msg_progresso is not None:
            cls.progress_bar(1,1) #finaliza a barra

    @classmethod
    def progress_bar(cls, current_value, total, msg=''):
        increments = 50
        percentual = int((current_value / total) * 100)
        i = int(percentual // (100 / increments))
        text = "\r[{0: <{1}}] {2:.2f}%".format('=' * i, increments, percentual)
        print('{} {}                     '.format(text, msg), end="\n" if percentual == 100 else "")

    @classmethod
    def mensagem_inline(cls, msg=''):
        i = random.randint(0,1)
        comp = "  . . ." if i==0 else ' . . .'
        comp=comp.ljust(100)
        print(f'\r{msg}{comp}'.ljust(80), end="\n" if not msg else "" )

    @classmethod
    def pausa(cls, segundos, progresso = True):
        if segundos ==0:
            return
        if segundos<1:
            segundos =1
        for ps in range(0,segundos):
            time.sleep(1)
            cls.progress_bar(ps,segundos+1,f'Pausa por {segundos-ps}s')
        cls.progress_bar(segundos,segundos,f'Pausa finalizada {segundos}s')

    @classmethod
    def dados_python(cls, retornar_dados_do_hardware:bool = True):
        res = {}
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        res['version'] = python_version
        virtual_env = os.environ.get('VIRTUAL_ENV')
        env_name = None
        if virtual_env:
            # Extraíndo o nome do ambiente a partir do caminho completo
            env_name = os.path.basename(virtual_env)
        if not env_name:
            python_executable_path = sys.executable
            possible_env_path = os.path.dirname(os.path.dirname(python_executable_path))
            if os.path.exists(os.path.join(possible_env_path, 'pyvenv.cfg')) or \
            os.path.exists(os.path.join(possible_env_path, 'conda-meta')):
                env_name = os.path.basename(possible_env_path)
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
        uso_cpu_percentual_pid = psutil.Process(os.getpid()).cpu_percent(interval=1) # pausa 1s para calcular o uso da CPU do processo
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
    def escape_json_string_literals(cls, s: str) -> str:
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



################################
class UtilCriptografia:
    def __init__(self):
        # Tenta obter a chave de criptografia a partir da variável de ambiente
        self.chave = os.getenv('CHAVE_CRIPT')
        if not self.chave:
            # Se a chave não existir na variável de ambiente, gera uma nova chave
            self.chave = self.nova_chave()
            print('Env CHAVE_CRIPT não encontrado, nova chave FERNET gerada:', self.chave)
        else:
            print('CHAVE FERNET CARREGADA _o/')
        # O Fernet espera a chave como bytes, então garantimos esta conversão
        if isinstance(self.chave, str):
            self.chave = self.chave.encode()
        # Inicializa o objeto Fernet com a chave
        self.fernet = Fernet(self.chave)
    
    def criptografar(self, texto: str) -> str:
        """
        Criptografa uma string e retorna o token criptografado como string.
        """
        # Codifica o texto para bytes e aplica a criptografia
        texto_bytes = texto.encode()
        token = self.fernet.encrypt(texto_bytes)
        # Decodifica o token para string, ideal para armazenamento em DataFrame
        return token.decode()
    
    def decriptografar(self, texto: str) -> str:
        """
        Descriptografa uma string criptografada e retorna o texto original.
        """
        # Codifica a string criptografada em bytes
        token_bytes = texto.encode()
        texto_bytes = self.fernet.decrypt(token_bytes)
        # Retorna o texto decodificado para string
        return texto_bytes.decode()
    
    def nova_chave(self) -> str:
        """
        Gera uma nova chave para o Fernet e a retorna como string.
        """
        chave = Fernet.generate_key()  # chave gerada em bytes
        # Retorna a chave decodificada para string
        return chave.decode()

########################################

try:
    from dotenv import load_dotenv
except ImportError:
    # A exceção só será levantada se o método 'carregar' for chamado.
    def load_dotenv(dotenv_path):
        raise ImportError('Erro de import: considere instalar com "pip install python-dotenv"')

class UtilEnv():
    """
    Classe utilitária para carregar e acessar variáveis de ambiente
    de arquivos .env de forma segura e robusta.
    """    
    @classmethod
    def carregar(cls, arquivos = ['.env','env'], pastas=['./','../','./src/','../src/']):
        """
        Carrega variáveis de ambiente de um arquivo .env.
        Procura por arquivos em diferentes pastas e para no primeiro que encontrar.
        Args:
            arquivos (Union[str, List[str]]): Nome ou lista de nomes de arquivos a procurar.
            pastas (List[str]): Lista de pastas onde procurar os arquivos.
        Returns:
            Optional[str]: O caminho do arquivo .env carregado, ou None se nenhum foi encontrado.
        """        
        arquivos = [str(arquivos)] if not isinstance(arquivos, (list, tuple, set)) else arquivos
        pastas = [str(pastas)] if not isinstance(pastas, (list, tuple, set)) else pastas
        for arquivo in arquivos:
            for pasta in pastas:
                arq = os.path.join(pasta, arquivo)
                if os.path.isfile(arq):
                    load_dotenv(arq)
                    return arq
        return None

    @classmethod
    def get_str(cls, chave: str, padrao: Optional[str] = None) -> Optional[str]:
        """
        Obtém uma variável de ambiente como uma string.

        Args:
            chave (str): A chave da variável de ambiente.
            padrao (Optional[str]): O valor padrão a ser retornado se a chave não for encontrada.

        Returns:
            Optional[str]: O valor da variável ou o padrão.
        """
        valor = os.getenv(chave)
        return valor if valor is not None else padrao

    @classmethod
    def get_int(cls, chave: str, padrao: Optional[int] = None) -> Optional[int]:
        """
        Obtém uma variável de ambiente e a converte para inteiro.

        Args:
            chave (str): A chave da variável de ambiente.
            padrao (Optional[int]): O valor padrão a ser retornado se a chave não for encontrada
                                  ou se a conversão falhar.

        Returns:
            Optional[int]: O valor convertido para inteiro ou o padrão.
        """
        valor_str = os.getenv(chave)
        if valor_str is None:
            return padrao
        
        try:
            return int(valor_str)
        except (ValueError, TypeError):
            # Retorna o padrão se a conversão para int falhar (ex: "abc", "1.5")
            # ou se o valor for inesperado (TypeError, embora raro com getenv).
            return padrao

    @classmethod
    def get_float(cls, chave: str, padrao: Optional[float] = None) -> Optional[float]:
        """
        Obtém uma variável de ambiente e a converte para float.
        Trata tanto ponto (.) quanto vírgula (,) como separador decimal.

        Args:
            chave (str): A chave da variável de ambiente.
            padrao (Optional[float]): O valor padrão a ser retornado se a chave não for encontrada
                                     ou se a conversão falhar.

        Returns:
            Optional[float]: O valor convertido para float ou o padrão.
        """
        valor_str = os.getenv(chave)
        if valor_str is None:
            return padrao
        
        try:
            # Substitui vírgula por ponto para garantir a conversão correta
            valor_normalizado = valor_str.replace(',', '.', 1)
            return float(valor_normalizado)
        except (ValueError, TypeError):
            # Retorna o padrão se a conversão para float falhar (ex: "abc")
            return padrao

############################
class UtilZip():
    @classmethod
    def compactar_pasta(cls, caminho_pasta: str, caminho_zip_saida: Optional[str] = None):
        """
        Compacta uma pasta inteira em um arquivo .zip usando shutil.
    
        Args:
            caminho_pasta (str): O caminho para a pasta que será compactada.
                Ex: './saidas'
            caminho_zip_saida (Optional[str]): O nome do arquivo .zip de saída.
                Se não for fornecido, será usado o mesmo nome da pasta de origem.
                Ex: 'meu_arquivo.zip'
        """
        # 1. Verifica se a pasta de origem realmente existe
        if not os.path.isdir(caminho_pasta):
            print(f"Erro: A pasta '{caminho_pasta}' não foi encontrada.")
            return
    
        # 2. Define o nome do arquivo de saída se não for fornecido
        if caminho_zip_saida is None:
            # Pega o nome base da pasta (ex: './saidas' -> 'saidas')
            base_name = os.path.basename(caminho_pasta)
            caminho_zip_saida = base_name
        else:
            # Remove a extensão .zip se o usuário a incluiu, pois shutil.make_archive a adiciona
            if caminho_zip_saida.endswith('.zip'):
                caminho_zip_saida = caminho_zip_saida[:-4]
    
        try:
            print(f"Compactando a pasta '{caminho_pasta}'...")
            # 3. Usa shutil.make_archive para criar o arquivo zip
            # O primeiro argumento é o nome do arquivo de saída (sem extensão)
            # O segundo é o formato ('zip')
            # O terceiro é a pasta raiz que será compactada
            shutil.make_archive(caminho_zip_saida, 'zip', caminho_pasta)
            print(f"Sucesso! Arquivo '{caminho_zip_saida}.zip' criado.")
        
        except Exception as e:
            print(f"Ocorreu um erro ao compactar: {e}")
    
    
    @classmethod
    def descompactar_arquivo(cls, caminho_zip: str, pasta_destino: str):
        """
        Descompacta um arquivo .zip para uma pasta de destino usando shutil.
    
        Args:
            caminho_zip (str): O caminho para o arquivo .zip que será descompactado.
                Ex: 'saidas.zip'
            pasta_destino (str): A pasta onde o conteúdo será extraído.
                A pasta será criada se não existir.
                Ex: './extracao'
        """
        # 1. Verifica se o arquivo zip realmente existe
        if not os.path.isfile(caminho_zip):
            print(f"Erro: O arquivo '{caminho_zip}' não foi encontrado.")
            return
    
        try:
            print(f"Descompactando '{caminho_zip}' para a pasta '{pasta_destino}'...")
            # 2. Usa shutil.unpack_archive para extrair os arquivos
            shutil.unpack_archive(caminho_zip, pasta_destino, 'zip')
            print(f"Sucesso! Arquivos extraídos em '{pasta_destino}'.")
    
        except shutil.ReadError:
            print(f"Erro: O arquivo '{caminho_zip}' parece não ser um arquivo zip válido ou está corrompido.")
        except Exception as e:
            print(f"Ocorreu um erro ao descompactar: {e}")    

class UtilEnv():
    # IGNORAR_PRINT_DEBUG pode ser usado para ignorar o print em determinados contextos
    IGNORAR_PRINT_DEBUG = False 
    LOCK_ARQUIVO_LOG = threading.Lock()
    
    @classmethod
    def debug(cls, chave = 'DEBUG'):
        return cls.env_true(chave)

    @classmethod
    def desativar_print_debug(cls):
        cls.IGNORAR_PRINT_DEBUG = True

    @classmethod
    def ativar_print_debug(cls):
        cls.IGNORAR_PRINT_DEBUG = False

    @classmethod
    def env_true(cls, chave_env = '', default=False):
        ''' retorna True se a chave existir e for 1,S,SIM,TRUE ou T
            retorna o valor default caso a chave não exista
        '''
        if not chave_env: 
            return False
        _env = os.getenv(chave_env,'')
        return _env.upper().strip() in ('1','S','SIM','TRUE','T') 

    @classmethod
    def teste_rapido(cls):
        _res = os.getenv('TESTE_RAPIDO','')
        return _res.upper() in ('1','S','SIM','TRUE') 

    @classmethod
    def print_debug(cls, msg, incluir_hora:bool = True, incluir_pid:bool = False, grupo = '', pausa = 0):
        if cls.IGNORAR_PRINT_DEBUG or (not cls.debug()):
           return 
        msg_final = f'{datetime.now().strftime("%H:%M:%S")} |' if incluir_hora else ''
        if incluir_pid:
            msg_final += f' <<PID#{os.getpid()}>> |'
        if grupo:
           msg_final = f'{grupo} | {msg_final}'
        print(f'{msg_final}>> {msg}', flush=True)
        cls.pausa_debug(pausa)

    @classmethod
    def file_debug(cls, arquivo:str, valor:str|dict, incluir_hora:bool = True, incluir_pid:bool = False, append = False):
        if not cls.debug():
           return 
        pasta, _ = os.path.split(arquivo)
        if not os.path.isdir(pasta):
            os.makedirs(pasta, exist_ok=True)
        msg_final = json.dumps(valor) if isinstance(valor, dict) else str(valor)
        if incluir_hora:
           msg_final = f'{datetime.now().strftime("%H:%M:%S")} | {msg_final}' if incluir_hora else msg_final
        if incluir_pid:
            msg_final = f'<<PID#{os.getpid()}>> | {msg_final}'
        with cls.LOCK_ARQUIVO_LOG:
            if append and os.path.isfile(arquivo):
                with open(arquivo, 'a') as f:
                    f.write(f'\n{msg_final}')
            else:
                with open(arquivo, 'w') as f:
                    f.write(msg_final)

    @classmethod
    def print_log(cls, msg, incluir_hora:bool = True, incluir_pid:bool = False, grupo = '', pausa_debug = 0):
        msg_final = f'{datetime.now().strftime("%H:%M:%S")} |' if incluir_hora else ''
        if incluir_pid:
            msg_final += f' <<PID#{os.getpid()}>> |'
        if grupo:
           msg_final = f'{grupo} | {msg_final}'
        print(f'{msg_final}>> {msg}', flush=True)
        cls.pausa_debug(pausa_debug)

    @ classmethod
    def pausa_debug(cls, segundos):
        if isinstance(segundos, int) and segundos > 0 and cls.debug():
            sleep(segundos)

    @classmethod
    def print_debug_timer(cls, msg):
        if cls.IGNORAR_PRINT_DEBUG or (not cls.debug('DEBUG_TIMER')):
           return 
        msg_final = f'{datetime.now().strftime("%H:%M:%S")} |'
        msg_final = f'TIMER | {msg_final}'
        print(f'{msg_final}>> {msg}', flush=True)

    @classmethod
    def carregar_env(cls, arquivo = '.env', pastas = None):
        assert DOTENV_OK, 'Instale o pacote: pip install python-dotenv'
        arq_env = UtilArquivos.encontrar_arquivo(arquivo=arquivo, pastas=pastas)
        if arq_env:
            print(f'Env encontrado em {arq_env}')
            from dotenv import load_dotenv
            load_dotenv(arq_env, override=True)
            return True
        print(f'Arquivo Env "{arquivo}" não encontrado')
        return False
    
    @classmethod
    def get_int(cls, chave_env, default=0) -> int:
        try:
            valor = os.getenv(chave_env)
            if isinstance(valor,str):
                return int(valor.strip())
            return default
        except:
            return default

    @classmethod
    def get_str(cls, chave_env, default='') -> str:
        try:
            valor = os.getenv(chave_env)
            if isinstance(valor,str):
                return valor
            return default
        except:
            return default            

class UtilArquivos(object):

    @staticmethod
    def tamanho_arquivo(nome_arquivo):
        if os.path.isfile(nome_arquivo):
           return os.path.getsize(nome_arquivo)
        return 0

    @staticmethod
    def carregar_json(arquivo):
        tipos = ['utf8', 'ascii', 'latin1']
        for tp in tipos:
            try:
                with open(arquivo, encoding=tp) as f:
                    return json.load(f)
            except UnicodeError:
                continue
        with open(arquivo, encoding='latin1', errors='ignore') as f:
            return json.load(f)

    @classmethod
    def encontrar_arquivo(cls, arquivo, pastas = None, incluir_subpastas = False):
        ''' pastas = None ele volta até 5 pastas procurando a pasta ou o arquivo informado
        '''
        from os import walk as os_walk
        _pastas = ['../','../../','../../../','../../../../','../../../../../'] if pastas is None else pastas
        _arq = os.path.split(arquivo)[1]
        subpastas = []
        if incluir_subpastas:
            for root, dirs, files in os_walk('./'):
                for sub in dirs:
                    subpastas.append( os.path.join(root,sub) )
        for pasta in set(list(_pastas) + ['./'] + subpastas):
            if os.path.isfile( os.path.join(pasta, _arq) ):
                return os.path.join(pasta, _arq)
        return None

class UtilArquivos(object):

    @staticmethod
    def tamanho_arquivo(nome_arquivo):
        if os.path.isfile(nome_arquivo):
           return os.path.getsize(nome_arquivo)
        return 0

    @staticmethod
    def carregar_json(arquivo):
        tipos = ['utf8', 'ascii', 'latin1']
        for tp in tipos:
            try:
                with open(arquivo, encoding=tp) as f:
                    return json.load(f)
            except UnicodeError:
                continue
        with open(arquivo, encoding='latin1', errors='ignore') as f:
            return json.load(f)

    @classmethod
    def encontrar_arquivo(cls, arquivo, pastas = None, incluir_subpastas = False):
        ''' pastas = None ele volta até 5 pastas procurando a pasta ou o arquivo informado
        '''
        from os import walk as os_walk
        _pastas = ['../','../../','../../../','../../../../','../../../../../'] if pastas is None else pastas
        _arq = os.path.split(arquivo)[1]
        subpastas = []
        if incluir_subpastas:
            for root, dirs, files in os_walk('./'):
                for sub in dirs:
                    subpastas.append( os.path.join(root,sub) )
        for pasta in set(list(_pastas) + ['./'] + subpastas):
            if os.path.isfile( os.path.join(pasta, _arq) ):
                return os.path.join(pasta, _arq)
        return None

    @staticmethod
    def listar_arquivos(pasta='./', nm_reduzido=False, mascara='*.txt'):
        assert os.path.isdir(pasta), f'listar_arquivos: pasta "{pasta}" não existe'
        arqs = [f for f in glob(fr"{pasta}{mascara}")]
        if nm_reduzido:
            arqs = [os.path.split(arq)[1] for arq in arqs]
        return arqs

class Util(object):

    @staticmethod
    def pausa(segundos, progresso = True):
        if segundos ==0:
            return
        if segundos<1:
            segundos =1
        for ps in range(0,segundos):
            time.sleep(1)
            if progresso:
                increments = 50
                percentual = int((ps / segundos) * 100)
                i = int(percentual // (100 / increments))
                text = "\r[{0: <{1}}] {2:.2f}%".format('=' * i, increments, percentual)
                print('{} Pausa por {}s           '.format(text, segundos-ps), end="")
        if progresso:
            print(f'\rPausa finalizada {segundos}s' + ' ' * 50)

class UtilTextos(object):
    
    @classmethod
    def mensagem_to_json(cls, mensagem:str, padrao = dict({}), _corrigir_json_ = True ):
        ''' O objetivo é receber uma resposta de um modelo LLM e identificar o json dentro dela 
            Exemplo: dicionario = UtilTextos.mensagem_to_json('```json\n{"chave":"valor qualquer", "numero":1}\```')
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
            try:
                return json.loads(_mensagem)
            except json.decoder.JSONDecodeError as e:
                if (not _corrigir_json_):
                    print(f'UtilTextos.mensagem_to_json: retornando padrão - erro ao decodificar string json >>> {str(_mensagem)[:50]}...')
                    return padrao
                # corrige aspas internas dentro do json
                return cls.mensagem_to_json(mensagem = cls.__escape_json_string_literals(mensagem), 
                                             padrao = padrao, 
                                             _corrigir_json_ = False)
                
        return padrao
    
    @classmethod
    def __escape_json_string_literals(cls, s: str) -> str:
        ''' Percorre a string 's' e escapa apenas as aspas duplas internas
            não-escapadas dentro de literais JSON.
        '''
        if not isinstance(s, str):
           raise TypeError(f'escape_json_string_literals espera receber uma string e recebeu {type(s)}')
        out = []
        in_str = False
        prev_esc = False
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if c == '"' and not prev_esc:
                if not in_str:
                    in_str = True
                    out.append(c)
                else:
                    j = i + 1
                    while j < n and s[j].isspace():
                        j += 1
                    next_char = s[j] if j < n else ''
                    if next_char in {',', '}', ']', ':'}:
                        in_str = False
                        out.append(c)
                    else:
                        out.append('\\"')
                i += 1
            elif c == '\\':
                out.append(c)
                prev_esc = not prev_esc
                i += 1
            else:
                out.append(c)
                prev_esc = False
                i += 1
        return ''.join(out)

class UtilDataHora():
    FORMATOS = [
            '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',  # Data, hora e segundos
            '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M',  # Data e hora
            '%Y-%m-%d', '%Y/%m/%d',  # Data
            '%Y-%m-%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S.%f'  # Data, hora, segundos e milissegundos
        ]
    @staticmethod
    def validar_data(data_hora_string):
        #print(f'TESTE: {data_hora_string}')
        if (data_hora_string is None) or len(data_hora_string) < 5:
           return False
        for formato in UtilDataHora.FORMATOS:
            try:
                datetime.strptime(data_hora_string, formato)
                return True
            except ValueError:
                continue  # Tenta o próximo formato
        return False
    
    @staticmethod
    def compara_data1_data2(data1 = None, tipo = '=', data2 = None):
        '''  espera datas no formato string yyyy-mm-dd hh:mm:ss ou datetime
             =, ==, >, <, >=, <=, <>, !
        '''
        _dt1 = '' if data1 is None else data1 
        _dt2 = '' if data2 is None else data2 
        _dt1 = _dt1 if type(_dt1) is str else UtilDataHora.data_hora_str(_dt1)
        _dt2 = _dt2 if type(_dt2) is str else UtilDataHora.data_hora_str(_dt2)
        #print(f'{_dt1} {tipo} {_dt2}')
        if tipo in ('=', '=='):
           return bool(_dt1 == _dt2)
        if tipo in ('<>', '!'):
           return bool(_dt1 != _dt2)
        if tipo == '>=':
           return bool(_dt1 >= _dt2)
        if tipo == '>':
           return bool(_dt1 > _dt2)
        if tipo == '<=':
           return bool(_dt1 <= _dt2)
        if tipo == '<':
           return bool(_dt1 < _dt2)
        return False

    @staticmethod
    def data_hora_str(data_hora=None, somar_dias: int = 0):
        if not (data_hora is None or isinstance(data_hora, datetime)):
            raise ValueError(f'UtilDataHora.data_hora_str recebeu um parâmetro com tipo inválido {type(data_hora)}')
        _data_hora = data_hora if data_hora else datetime.now()
        if isinstance(somar_dias,int) and somar_dias != 0:
           _data_hora = _data_hora + timedelta(days = somar_dias)
        return _data_hora.strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def data_str(data_hora=None, somar_dias: int = 0):
        if not (data_hora is None or isinstance(data_hora, datetime)):
            raise ValueError(f'UtilDataHora.data_str recebeu um parâmetro com tipo inválido {type(data_hora)}')
        _data_hora = data_hora if isinstance(data_hora, datetime) else datetime.now()
        if isinstance(somar_dias,int) and somar_dias != 0:
           _data_hora = _data_hora + timedelta(days = somar_dias)
        return _data_hora.strftime('%Y-%m-%d')

    @staticmethod
    def hora_entre(data_hora = None, hora_inicial = '00:00', hora_final = '23:59'):
        _data_hora = data_hora if data_hora else datetime.now()
        # Converter as strings 'hh:mm' de hora_inicial e hora_final para objetos time
        hora_inicial_obj = datetime.strptime(hora_inicial, "%H:%M").time()
        hora_final_obj = datetime.strptime(hora_final, "%H:%M").time()
        # Extrair apenas a hora da data informada
        hora_data = _data_hora.time()
        # Verificar se a hora está dentro do intervalo especificado
        if hora_inicial_obj <= hora_final_obj:
            # Caso comum: quando hora_inicial é menor ou igual a hora_final
            return hora_inicial_obj <= hora_data <= hora_final_obj
        else:
            # Caso que atravessa a meia-noite: quando hora_inicial é maior que hora_final
            return hora_data >= hora_inicial_obj or hora_data <= hora_final_obj

    @staticmethod
    def hora_na_lista(data_hora = None, lista:list = []):
        # recebe uma lista de inteiros que são as horas aceitas [0~23]
        _data_hora = data_hora if data_hora else datetime.now()
        if isinstance(_data_hora, str):
            _data_hora = UtilDataHora.to_datetime(_data_hora)
        # Extrair apenas a hora da data informada
        hora = _data_hora.time().hour
        # Verificar se a hora está dentro do intervalo especificado
        return int(hora) in [_ for _ in lista if isinstance(_, int)]

    @staticmethod
    def dia_da_semana(data_hora = None, sigla = False):
        _data_hora = data_hora if data_hora else datetime.now()
        # Mapeamento dos números para os dias da semana
        dias = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
        siglas = ["SEQ", "TER", "QUA", "QUI", "SEX", "SÁB", "DOM"]
        # Obter o número do dia da semana
        numero_dia = _data_hora.weekday()
        # Retornar o nome do dia da semana correspondente
        if sigla:
           return siglas[numero_dia] 
        return dias[numero_dia]
    
    @staticmethod
    def to_datetime(data_hora_string):
        if (data_hora_string is None) or len(data_hora_string) < 5:
           return None
        for formato in UtilDataHora.FORMATOS:
            try:
                return datetime.strptime(data_hora_string, formato)
            except ValueError:
                continue  # Tenta o próximo formato
        return None

    @staticmethod
    def somar_dias(data, dias:int):
        _data = data
        if isinstance(data, str):
           _data = UtilDataHora.to_datetime(data)
        elif not isinstance(data, datetime):
           raise ValueError(f'UtilDataHora.somar_dias recebeu um parâmetro com tipo inválido {type(data)} - era esperado str ou datetime')
        return _data + timedelta(days = dias)
    
    @staticmethod
    def segundos_to_str(segundos):
        horas = int(segundos // 3600)
        minutos = int((segundos % 3600) // 60)
        segundos = int(segundos % 60)
        return f"{horas:02d}:{minutos:02d}:{segundos:02d}"    
    
    @staticmethod
    def data_hora_arquivo(arquivo, formato_string = False):
        tempo_criacao = os.path.getctime(arquivo)
        data_hora = datetime.fromtimestamp(tempo_criacao, tz=timezone.utc)
        if not formato_string:
            return data_hora
        return data_hora.strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def segundos_arquivo(arquivo, default = 0):
        if not os.path.isfile(arquivo):
            return default
        tempo_criacao = os.path.getctime(arquivo)
        return time.time() - tempo_criacao

    @staticmethod
    def data_extenso(data = None):
        # Caso nenhuma data seja passada, usa a data e hora atuais
        if isinstance(data, str):
            data = UtilDataHora.to_datetime(data)        
        if data is None:
            data = datetime.now()
        # Lista dos meses em português
        meses_pt = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
                    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
        dia_formatado = f"{data.day:02d}"
        mes_extenso = meses_pt[data.month - 1]
        ano = data.year
        # Exemplo: "17 de fevereiro de 2025"
        return f"{dia_formatado} de {mes_extenso} de {ano}"
