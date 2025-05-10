# -*- coding: utf-8 -*-

'''
 Autor Luiz Anísio 20/11/2022
 Utilitários simples para simplificação de alguns códigos comuns
 '''

import os
import hashlib
from datetime import datetime
import json
import random
import string
from multiprocessing import cpu_count
import regex as re

HASH_BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
class Util():
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
