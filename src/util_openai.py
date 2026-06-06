from time import time
try:
    from openai import OpenAI, RateLimitError, APIConnectionError, AuthenticationError
    MSG_OPENAI = ""
except ImportError:
    MSG_OPENAI = "Módulo 'openai' não encontrado. Instale com 'pip install openai'."
try:
    from together import Together
    MSG_TOGETHER = ""
except ImportError:
    MSG_TOGETHER = "Módulo 'together' não encontrado. Instale com 'pip install together'."
try:
    from openrouter import OpenRouter as OpenRouterClient
    MSG_OPENROUTER = ""
except ImportError:
    MSG_OPENROUTER = "Módulo 'openrouter' não encontrado. Instale com 'pip install openrouter'."
import os
import json
import traceback
import requests

from threading import Lock
LOCK_ARQUIVO_BRUTO = Lock()

'''
 Autor Luiz Anísio 17/10/2025
 Fonte: https://github.com/luizanisio/llms/tree/main/src
 Utilitários para acionar apis generativas da openai, openrouter, together.ai, ollama e vllm
 Realiza tratamento de erros comuns e padroniza retorno json ou texto
 Sempre retorna um json com "resposta" ou com "erro" e "tempo" de processamento

* Para usar no colab, execute a linha:
!curl https://raw.githubusercontent.com/luizanisio/llms/refs/heads/main/src/util_openai.py -o ./util_openai.py

Adicione a(s) chave(s) de API no ambiente do colab:
> PESSOAL_OPENAI_API_KEY='sua-chave-aqui'
> PESSOAL_OPENROUTER_API_KEY='sua-chave-aqui'
Teste:
> from util_openai import get_resposta, teste_resposta
> teste_resposta(as_json=True, modelo='or:openai/gpt-oss-20b')
'''

def get_resposta(prompt:str, papel:str='',
                 modelo='gpt-5-nano', think='l',
                 as_json=True,
                 temperature=0.01,
                 max_tokens=None,
                 max_ctx=None,
                 max_retry=5,
                 timeout=300,
                 api_key=None,
                 raw: bool = False,
                 silencioso: bool = False, **kwargs) -> dict:
    ''' Obtém a resposta do modelo de linguagem para o prompt informado.
        Quando as_json for True, tenta converter a resposta em JSON e retorna uma chave "erro" em caso de falha.
        Qualquer implementação de get_resposta deve seguir essa assinatura para ser usada pelos agentes.
    '''
    UtilLog.verificar_aviso()
    
    tempo = time()
    if 'modelo_think' in kwargs and kwargs['modelo_think']:
        think = kwargs['modelo_think'] 

    # Aplicar a separação do modelo e think o quanto antes
    modelo, think = UtilOA.modelo_think(modelo, think)
    original_model = modelo

    max_retry = max(0, min(10, max_retry))
    
    if isinstance(prompt,(tuple, list)) and len(prompt) > 0:
        messages = prompt
    elif isinstance(prompt,str) and prompt.strip(' \n\t'):    
        messages = []
        if papel and isinstance(papel, str) and papel.strip():
            messages.append({'role':'system', 'content': papel.strip()})
        messages.append({'role':'user', 'content': prompt})
    else:
        raise ValueError('Formato do prompt não reconhecido: deve ser messages ou string')

    # Identificar API pelo prefixo
    tipo_api = 'openai'
    nome_modelo = modelo
    
    if modelo.lower().startswith('or:'):
        tipo_api = 'openrouter'
        nome_modelo = modelo[3:]
    elif modelo.lower().startswith('ol:'):
        tipo_api = 'ollama'
        nome_modelo = modelo[3:]
    elif modelo.lower().startswith('vl:'):
        tipo_api = 'vllm'
        nome_modelo = modelo[3:]
    elif modelo.lower().startswith('tg:'):
        tipo_api = 'together'
        nome_modelo = modelo[3:]
    elif modelo.lower().startswith('oa:'):
        tipo_api = 'openaia'
        nome_modelo = modelo[3:]

    if not silencioso: print(f'Chamada: {original_model}:{think} [{tipo_api}] | max retry = {max_retry} | ')

    # Roteamento para as classes utilitárias
    if tipo_api == 'ollama':
        if nome_modelo.lower() == 'models':
            try: return {'resposta': UtilOllama.models(), 'modelo': 'ol:models', 'tempo': round(time() - tempo, 3)}
            except Exception as e: return {'erro': f'Erro: {str(e)}', 'tempo': round(time() - tempo, 3)}
        if nome_modelo.lower() == 'status':
            try: return {'resposta': UtilOllama.status(), 'modelo': 'ol:status', 'tempo': round(time() - tempo, 3)}
            except Exception as e: return {'erro': f'Erro: {str(e)}', 'tempo': round(time() - tempo, 3)}
            
        ollama_kwargs = dict(messages=messages, modelo=nome_modelo, temperature=temperature, max_tokens=max_tokens, as_json=as_json, raw=raw, timeout=timeout, tempo_inicio=tempo)
        if isinstance(max_ctx, int) and max_ctx > 0:
            ollama_kwargs['num_ctx'] = max_ctx
        return UtilOllama.chat_completion_padronizado(**ollama_kwargs)

    if tipo_api == 'openrouter':
        if MSG_OPENROUTER: raise ImportError(MSG_OPENROUTER)
        return UtilOpenRouter.chat_completion_padronizado(messages=messages, modelo=nome_modelo, temperature=temperature, max_tokens=max_tokens, as_json=as_json, raw=raw, timeout=timeout, think=think, api_key=api_key, silencioso=silencioso, max_retry=max_retry, tempo_inicio=tempo)

    if tipo_api == 'vllm':
        if MSG_OPENAI: raise ImportError(MSG_OPENAI)
        if nome_modelo.lower() == 'models':
            try: return {'resposta': UtilVllm.models(), 'modelo': 'vl:models', 'tempo': round(time() - tempo, 3)}
            except Exception as e: return {'erro': f'Erro: {str(e)}', 'tempo': round(time() - tempo, 3)}
        if nome_modelo.lower() == 'status':
            try: return {'resposta': UtilVllm.status(), 'modelo': 'vl:status', 'tempo': round(time() - tempo, 3)}
            except Exception as e: return {'erro': f'Erro: {str(e)}', 'tempo': round(time() - tempo, 3)}
        return UtilVllmServer.chat_completion_padronizado(messages=messages, modelo=nome_modelo, temperature=temperature, max_tokens=max_tokens, as_json=as_json, raw=raw, timeout=timeout, think=think, api_key=api_key, silencioso=silencioso, max_retry=max_retry, tempo_inicio=tempo)

    if tipo_api == 'together':
        if MSG_TOGETHER: raise ImportError(MSG_TOGETHER)
        return UtilTogether.chat_completion_padronizado(messages=messages, modelo=nome_modelo, temperature=temperature, max_tokens=max_tokens, as_json=as_json, raw=raw, timeout=timeout, think=think, api_key=api_key, silencioso=silencioso, max_retry=max_retry, tempo_inicio=tempo)
        
    if tipo_api == 'openaia':
        return UtilOA().prompt(prompt=messages, sg_modelo=nome_modelo, think=think, as_json=as_json, max_tokens=max_tokens)

    # Fallback para OpenAI nativo
    if MSG_OPENAI: raise ImportError(MSG_OPENAI)
    return UtilOpenAI.chat_completion_padronizado(messages=messages, modelo=nome_modelo, temperature=temperature, max_tokens=max_tokens, as_json=as_json, raw=raw, timeout=timeout, think=think, api_key=api_key, silencioso=silencioso, max_retry=max_retry, tempo_inicio=tempo)


class UtilLog:
    AVISO_LOG_FEITO = False

    @staticmethod
    def verificar_aviso():
        LOG_BRUTO = os.getenv('LOG_BRUTO', '').lower()
        if LOG_BRUTO and LOG_BRUTO not in ('false', '0') and not UtilLog.AVISO_LOG_FEITO:
            UtilLog.AVISO_LOG_FEITO = True
            LOG_BRUTO_ARQUIVO = os.getenv('LOG_BRUTO_ARQUIVO', './log_openai_resposta_bruta.txt')
            print(f'💡 ATENÇÃO: LOG_BRUTO ativado!\b - Arquivo: {LOG_BRUTO_ARQUIVO} ')

    @staticmethod
    def log_bruto(tipo_api: str, modelo: str, messages: list, resposta_bruta: dict, usage_data: dict = None, tempo_str: float = None):
        LOG_BRUTO = os.getenv('LOG_BRUTO', '').lower()
        if not LOG_BRUTO or LOG_BRUTO in ('false', '0'):
            return
            
        if LOG_BRUTO not in ('true', '1', 'all') and tipo_api not in LOG_BRUTO:
            return

        LOG_BRUTO_ARQUIVO = os.getenv('LOG_BRUTO_ARQUIVO', './log_openai_resposta_bruta.txt')

        try:
            with LOCK_ARQUIVO_BRUTO:
                with open(LOG_BRUTO_ARQUIVO, 'a', encoding='utf-8') as f:
                    f.write('#'*60 + '\n')
                    f.write(f'Timestamp: {tempo_str or time()}\n')
                    f.write(f'Modelo: {modelo}\n')
                    f.write('='*40 + '\n')
                    f.write('Prompt enviado:\n')
                    f.write(json.dumps(messages, ensure_ascii=False, indent=2))
                    f.write('\n' + ('-'*40) + '\n')
                    f.write('Resposta bruta:\n')
                    f.write(json.dumps(resposta_bruta, ensure_ascii=False, indent=2))
                    if usage_data:
                        f.write('\n' + ('-'*40) + '\n')
                        f.write('Usage:\n')
                        f.write(json.dumps(usage_data, ensure_ascii=False, indent=2))
                    f.write('\n' + ('-'*80) + '\n')
                    f.write('#'*60 + '\n')
        except Exception:
            pass

class UtilResponse:
    @staticmethod
    def montar_resultado(conteudo: str, usage_data: dict, modelo: str, tempo: float, as_json: bool, resposta_original: dict = None) -> dict:
        resultado = {}
        if as_json:
            try:
                if isinstance(conteudo, str):
                    conteudo_json = UtilJson.mensagem_to_json(conteudo, padrao=None)
                    if conteudo_json is None:
                        resultado['resposta'] = conteudo
                        resultado['erro'] = 'Erro ao extrair JSON da resposta (conteudo=None).'
                        if resposta_original:
                            resultado['erro'] += f' (Dict resposta={resposta_original})'
                        resultado['json'] = False
                    else:
                        resultado['resposta'] = conteudo_json
                        resultado['json'] = True
                        # Limpa erro anterior de parser (ex: no servidor OA)
                        if 'erro' in resultado and 'Erro ao extrair JSON' in str(resultado['erro']):
                            del resultado['erro']
                else:
                    resultado['resposta'] = conteudo
                    resultado['json'] = True
            except Exception as e:
                resultado['resposta'] = conteudo
                resultado['json'] = False
                resultado['erro'] = f'Erro local ao extrair JSON da resposta: {str(e)}'
        else:
            resultado['resposta'] = conteudo
            resultado['json'] = False

        # Garante que chaves padrão existam no usage
        if 'finished_reason' not in usage_data:
            usage_data['finished_reason'] = 'unknown'

        resultado['usage'] = usage_data
        resultado['model'] = modelo
        resultado['tempo'] = round(tempo, 3)
        return resultado

class UtilOpenAI:
    @classmethod
    def chat_completion_padronizado(cls, messages: list, modelo: str, temperature: float = 0.01,
                                    max_tokens: int = None, as_json: bool = True, raw: bool = False,
                                    timeout: int = 300, think: str = None, api_key: str = None,
                                    silencioso: bool = False, max_retry: int = 5, tempo_inicio: float = None):
        tempo = tempo_inicio or time()
        api_key = api_key or os.getenv("PESSOAL_OPENAI_API_KEY")
        if not api_key: return {'erro': 'PESSOAL_OPENAI_API_KEY não encontrada no ambiente', 'model': modelo, 'tempo': round(time() - tempo, 3)}

        client_gpt = OpenAI(api_key=api_key, timeout=timeout)
        args = {'messages': messages, 'model': modelo, 'temperature': temperature, 'timeout': timeout}
        if isinstance(max_tokens, int): args['max_tokens'] = max_tokens
        
        if think:
            _reasoning, _verbosity = UtilOpenRouter._mapear_reasoning(think)
            if 'gpt-5' in modelo.lower():
                args = {k: v for k, v in args.items() if k not in {'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty'}}
            args['reasoning_effort'] = _reasoning
            if isinstance(max_tokens, int): args['max_completion_tokens'] = max_tokens
        if as_json: args['response_format'] = {"type": "json_object"}

        try:
            response = client_gpt.chat.completions.create(**args)
        except AuthenticationError as e: return {'erro': f'Erro de autenticação: {str(e)}', 'model': modelo, 'tempo': round(time() - tempo, 3)}
        except (APIConnectionError, RateLimitError) as e:
            if max_retry > 0:
                import time as time_module
                time_module.sleep(2)
                return cls.chat_completion_padronizado(messages=messages, modelo=modelo, temperature=temperature, max_tokens=max_tokens, as_json=as_json, raw=raw, timeout=timeout, think=think, api_key=api_key, silencioso=silencioso, max_retry=max_retry - 1, tempo_inicio=tempo)
            return {'erro': f'Erro OpenAI (Retry Esgotado): {str(e)}', 'model': modelo, 'tempo': round(time() - tempo, 3)}
        except Exception as e:
            return {'erro': f'Erro OpenAI: {type(e).__name__}: {str(e)}', 'model': modelo, 'tempo': round(time() - tempo, 3)}

        res_dict = response.to_dict()
        UtilLog.log_bruto('openai', modelo, messages, res_dict, tempo_str=tempo)
        if raw:
            res_dict['tempo'] = round(time() - tempo, 3)
            return res_dict

        choices = res_dict.get('choices', [])
        if not choices:
            return {'erro': 'Resposta da OpenAI sem choices', 'model': modelo, 'tempo': round(time() - tempo, 3)}

        conteudo = choices[0].get('message', {}).get('content', '')
        usage_data = res_dict.get('usage', {}) or {}
        comp_det = usage_data.get('completion_tokens_details', {}) or {}
        prompt_det = usage_data.get('prompt_tokens_details', {}) or {}
        
        usage = {
            'prompt_tokens': usage_data.get('prompt_tokens', 0),
            'completion_tokens': usage_data.get('completion_tokens', 0),
            'total_tokens': usage_data.get('total_tokens', 0),
            'cached_tokens': prompt_det.get('cached_tokens', 0),
            'reasoning_tokens': comp_det.get('reasoning_tokens', 0),
            'finished_reason': choices[0].get('finish_reason', 'unknown'),
            'temperature': temperature
        }
        return UtilResponse.montar_resultado(conteudo, usage, res_dict.get('model', modelo), time() - tempo, as_json, res_dict)

class UtilTogether:
    @classmethod
    def chat_completion_padronizado(cls, messages: list, modelo: str, temperature: float = 0.01,
                                    max_tokens: int = None, as_json: bool = True, raw: bool = False,
                                    timeout: int = 300, think: str = None, api_key: str = None,
                                    silencioso: bool = False, max_retry: int = 5, tempo_inicio: float = None):
        tempo = tempo_inicio or time()
        api_key = api_key or os.getenv("PESSOAL_TOGETHER_API_KEY")
        if not api_key: return {'erro': 'PESSOAL_TOGETHER_API_KEY não encontrada no ambiente', 'model': modelo, 'tempo': round(time() - tempo, 3)}

        client_gpt = Together(api_key=api_key)
        args = {'messages': messages, 'model': modelo, 'temperature': temperature, 'timeout': timeout}
        if isinstance(max_tokens, int): args['max_tokens'] = max_tokens
        if think:
            _reasoning, _verbosity = UtilOpenRouter._mapear_reasoning(think)
            args['reasoning_effort'] = _reasoning
        if as_json: args['response_format'] = {"type": "json_object"}

        try:
            response = client_gpt.chat.completions.create(**args)
        except Exception as e:
            if max_retry > 0 and ('connection' in str(e).lower() or 'rate' in str(e).lower()):
                import time as time_module
                time_module.sleep(2)
                return cls.chat_completion_padronizado(messages=messages, modelo=modelo, temperature=temperature, max_tokens=max_tokens, as_json=as_json, raw=raw, timeout=timeout, think=think, api_key=api_key, silencioso=silencioso, max_retry=max_retry - 1, tempo_inicio=tempo)
            return {'erro': f'Erro Together: {type(e).__name__}: {str(e)}', 'model': modelo, 'tempo': round(time() - tempo, 3)}

        try:
            res_dict = response.model_dump()
            usage_data = response.usage.model_dump() if response.usage else {}
        except AttributeError:
            res_dict = response.dict()
            usage_data = response.usage.dict() if response.usage else {}

        UtilLog.log_bruto('together', modelo, messages, res_dict, usage_data=usage_data, tempo_str=tempo)
        if raw:
            res_dict['tempo'] = round(time() - tempo, 3)
            return res_dict

        choices = res_dict.get('choices', [])
        if not choices:
            return {'erro': 'Resposta da Together sem choices', 'model': modelo, 'tempo': round(time() - tempo, 3)}

        conteudo = choices[0].get('message', {}).get('content', '')
        usage_data = res_dict.get('usage', {}) or {}
        comp_det = usage_data.get('completion_tokens_details', {}) or {}
        prompt_det = usage_data.get('prompt_tokens_details', {}) or {}
        
        usage = {
            'prompt_tokens': usage_data.get('prompt_tokens', 0),
            'completion_tokens': usage_data.get('completion_tokens', 0),
            'total_tokens': usage_data.get('total_tokens', 0),
            'cached_tokens': prompt_det.get('cached_tokens', 0),
            'reasoning_tokens': comp_det.get('reasoning_tokens', 0),
            'finished_reason': choices[0].get('finish_reason', 'unknown'),
            'temperature': temperature
        }
        return UtilResponse.montar_resultado(conteudo, usage, res_dict.get('model', modelo), time() - tempo, as_json, res_dict)

class UtilVllmServer:
    @classmethod
    def chat_completion_padronizado(cls, messages: list, modelo: str, temperature: float = 0.01,
                                    max_tokens: int = None, as_json: bool = True, raw: bool = False,
                                    timeout: int = 300, think: str = None, api_key: str = None,
                                    silencioso: bool = False, max_retry: int = 5, tempo_inicio: float = None):
        tempo = tempo_inicio or time()
        url = os.getenv("VLLM_URL", "http://localhost:8000/v1")
        api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        
        client_gpt = OpenAI(api_key=api_key, timeout=timeout, base_url=url)
        args = {'messages': messages, 'model': modelo, 'temperature': temperature, 'timeout': timeout}
        if isinstance(max_tokens, int): args['max_tokens'] = max_tokens
        if as_json: args['response_format'] = {"type": "json_object"}

        try:
            response = client_gpt.chat.completions.create(**args)
        except Exception as e:
            if max_retry > 0 and ('connection' in str(e).lower() or 'rate' in str(e).lower()):
                import time as time_module
                time_module.sleep(2)
                return cls.chat_completion_padronizado(messages=messages, modelo=modelo, temperature=temperature, max_tokens=max_tokens, as_json=as_json, raw=raw, timeout=timeout, think=think, api_key=api_key, silencioso=silencioso, max_retry=max_retry - 1, tempo_inicio=tempo)
            return {'erro': f'Erro vLLM: {type(e).__name__}: {str(e)}', 'model': modelo, 'tempo': round(time() - tempo, 3)}

        res_dict = response.to_dict()
        UtilLog.log_bruto('vllm', modelo, messages, res_dict, tempo_str=tempo)
        if raw:
            res_dict['tempo'] = round(time() - tempo, 3)
            return res_dict

        choices = res_dict.get('choices', [])
        if not choices:
            return {'erro': 'Resposta do vLLM sem choices', 'model': modelo, 'tempo': round(time() - tempo, 3)}

        conteudo = choices[0].get('message', {}).get('content', '')
        usage_data = res_dict.get('usage', {}) or {}
        comp_det = usage_data.get('completion_tokens_details', {}) or {}
        prompt_det = usage_data.get('prompt_tokens_details', {}) or {}
        
        usage = {
            'prompt_tokens': usage_data.get('prompt_tokens', 0),
            'completion_tokens': usage_data.get('completion_tokens', 0),
            'total_tokens': usage_data.get('total_tokens', 0),
            'cached_tokens': prompt_det.get('cached_tokens', 0),
            'reasoning_tokens': comp_det.get('reasoning_tokens', 0),
            'finished_reason': choices[0].get('finish_reason', 'unknown'),
            'temperature': temperature
        }
        return UtilResponse.montar_resultado(conteudo, usage, res_dict.get('model', modelo), time() - tempo, as_json, res_dict)

class UtilJson():
    @classmethod
    def mensagem_to_json(cls, mensagem:str, padrao = dict({}), _corrigir_json_ = True):
        ''' O objetivo é receber uma resposta de um modelo LLM e extrair o objeto JSON contido nela.
        
        Args:
            mensagem (str): A resposta em texto do modelo LLM que pode conter um JSON.
            padrao (dict, opcional): Valor padrão retornado caso a extração ou decodificação falhe. Padrão é {}.
            _corrigir_json_ (bool, opcional): Se True, em caso de erro na decodificação, tenta corrigir aspas internas não escapadas e tenta novamente. Padrão é True.
            
        Returns:
            dict/list: O JSON extraído e decodificado da mensagem, ou o valor de `padrao` em caso de falha.
        '''
        if isinstance(mensagem, dict):
            return mensagem
        if not isinstance(mensagem, str):
           raise ValueError('mensagem_to_json: parâmetro precisa ser string')
        _mensagem = str(mensagem).strip()
        # limpa resposta ```json ````
        chave_json = mensagem.rfind('```json\n')
        if chave_json >= 0:
           _mensagem = _mensagem[chave_json+8:]
        elif mensagem.startswith('json\n'):
           _mensagem = _mensagem[4:] # alguns modelos só colocam json\n no início
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
            except json.decoder.JSONDecodeError as e:
                if (not _corrigir_json_):
                    print(f'UtilJson.mensagem_to_json: retornando padrão - erro ao decodificar string json >>> {str(_mensagem)[:50]}...')
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

def teste_resposta(as_json = True, modelo='gpt-5-nano'):
    if as_json:
        prompt = "Explique a teoria da relatividade de forma com três tópicos curtos no formato JSON. {'topico01': ..., }"
    else:
        prompt = "Explique a teoria da relatividade de forma simples com 50 palavras."
    resposta = get_resposta(prompt, papel="Você é um professor de física.", modelo=modelo, think="low", as_json=as_json)
    if as_json:
       print(f'Resposta JSON: {json.dumps(resposta, ensure_ascii=False, indent=2)}')
    else:
       print(resposta)

def teste_api(openrouter=False):
    # Validação prévia
    if openrouter:
      api_key = os.getenv("PESSOAL_OPENROUTER_API_KEY")
      base_url = "https://openrouter.ai/api/v1"
      modelo='openai/gpt-oss-20b'
      if not api_key:
          print('ERRO: PESSOAL_OPENROUTER_API_KEY não encontrada no ambiente')
          return
    else:
      api_key = os.getenv("PESSOAL_OPENAI_API_KEY")
      base_url = None
      modelo = 'gpt-5-nano'
      if not api_key:
          print('ERRO: PESSOAL_OPENAI_API_KEY não encontrada no ambiente')
          return

    print(f'API Key encontrada: {api_key[:10]}...{api_key[-4:]}')

    try:
        client = OpenAI(api_key=api_key, timeout=30, base_url = base_url)
        print('Cliente OpenAI criado com sucesso.')

        completion = client.chat.completions.create(
                    model=modelo,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"}
                    ])
        print('Resposta recebida com sucesso:')
        print(completion.to_dict())
    except AuthenticationError as e:
        print(f'ERRO DE AUTENTICAÇÃO: {str(e)}')
        if openrouter:
            print('Verifique se a API key está correta e ativa em https://openrouter.ai/')
        else:
            print('Verifique se a API key está correta e ativa em https://platform.openai.com/api-keys')
    except APIConnectionError as e:
        print(f'ERRO DE CONEXÃO: {str(e)}')
        print('Possíveis causas:')
        print('  1. Problema de rede/firewall')
        print('  2. Proxy não configurado')
        print('  3. API da OpenAI indisponível')
    except Exception as e:
        print(f'ERRO INESPERADO: {type(e).__name__}: {str(e)}')


class UtilOllama:
    API_URL = 'http://localhost:11434/api'

    @classmethod
    def chat_completion(cls, messages:list, modelo:str, options:dict=None,
                        num_ctx:int=16*1024,
                        format:str=None, timeout:int=300, api_url:str=None):
        ''' Envia uma requisição de chat para a API nativa do Ollama.
            POST /api/chat
            Parâmetros:
                messages: lista de mensagens no formato [{"role": "...", "content": "..."}]
                modelo: nome do modelo (ex: "llama3", "qwen2.5:1.5b")
                options: dict com opções do modelo (ex: {"temperature": 0.7, "num_predict": 1024})
                num_ctx: tamanho da janela de contexto em tokens (padrão: 16384 = 16k, 32768 = 32k)
                format: "json" para forçar resposta em JSON, ou None
                timeout: timeout da requisição em segundos
                api_url: sobrepõe cls.API_URL se informado
            Retorna o dict nativo do Ollama (ver documentação /api/chat).
        '''
        url = f'{api_url or cls.API_URL}/chat'
        # Monta options com num_ctx como base, permitindo override via options
        _options = {'num_ctx': num_ctx}
        if options:
            _options.update(options)
        payload = {
            'model': modelo,
            'messages': messages,
            'stream': False,
            'options': _options
        }
        if format:
            payload['format'] = format
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @classmethod
    def models(cls, api_url:str=None):
        ''' Lista modelos disponíveis localmente.
            GET /api/tags
            Retorna lista de dicts com info dos modelos.
        '''
        url = f'{api_url or cls.API_URL}/tags'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        modelos = data.get('models', [])
        return [{
            'id': m.get('name', m.get('model', '?')),
            'model': m.get('model', ''),
            'size': m.get('size', 0),
            'modified_at': m.get('modified_at', ''),
            'details': m.get('details', {})
        } for m in modelos]

    @classmethod
    def embedding(cls, input, modelo:str, options:dict=None,
                  timeout:int=60, api_url:str=None):
        ''' Gera embeddings usando a API nativa do Ollama.
            POST /api/embed
            Parâmetros:
                input: texto ou lista de textos
                modelo: nome do modelo de embedding (ex: "all-minilm", "nomic-embed-text")
                options: dict com opções do modelo
                timeout: timeout da requisição em segundos
                api_url: sobrepõe cls.API_URL se informado
            Retorna o dict nativo do Ollama com os embeddings.
        '''
        url = f'{api_url or cls.API_URL}/embed'
        payload = {
            'model': modelo,
            'input': input
        }
        if options:
            payload['options'] = options
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @classmethod
    def status(cls, api_url:str=None):
        ''' Verifica se a api está disponível e retorna a versão do ollama,
            com os modelos instalados e os modelos ativos (carregados em memória).
            Resposta: {"api": True/False, "versao": "x.y.z",
                       "modelos": ["modelo1", ...], "ativos": ["modelo1", ...]}
        '''
        base = api_url or cls.API_URL
        resultado = {'api': False, 'versao': '', 'modelos': [], 'ativos': []}
        try:
            # Versão: GET /api/version
            r = requests.get(f'{base}/version', timeout=5)
            r.raise_for_status()
            resultado['versao'] = r.json().get('version', '?')
            resultado['api'] = True
        except Exception:
            return resultado
        try:
            # Modelos instalados: GET /api/tags
            r = requests.get(f'{base}/tags', timeout=5)
            r.raise_for_status()
            resultado['modelos'] = [m.get('name', '?') for m in r.json().get('models', [])]
        except Exception:
            pass
        try:
            # Modelos ativos (em memória): GET /api/ps
            r = requests.get(f'{base}/ps', timeout=5)
            r.raise_for_status()
            resultado['ativos'] = [m.get('name', '?') for m in r.json().get('models', [])]
        except Exception:
            pass
        return resultado

    @classmethod
    def chat_completion_padronizado(cls, messages:list, modelo:str,
                                    temperature:float=0.01, max_tokens:int=None,
                                    num_ctx:int=16*1024,
                                    as_json:bool=True, raw:bool=False,
                                    timeout:int=300, api_url:str=None,
                                    tempo_inicio:float=None):
        ''' Chama o Ollama via API nativa e retorna a resposta no padrão de get_resposta.
            Parâmetros:
                messages: lista de mensagens [{"role": "...", "content": "..."}]
                modelo: nome do modelo (ex: "llama3", "qwen2.5:1.5b")
                temperature: temperatura do modelo
                max_tokens: número máximo de tokens na resposta (num_predict do Ollama)
                num_ctx: tamanho da janela de contexto em tokens (padrão: 49152 = 48k)
                         Deve comportar entrada + saída. Ex: 32k entrada + 16k saída = 48k
                as_json: se True, tenta converter a resposta em JSON
                raw: se True, retorna o dict bruto do Ollama + tempo
                timeout: timeout da requisição em segundos
                api_url: sobrepõe cls.API_URL se informado
                tempo_inicio: timestamp de início (para cálculo de tempo)
            Retorna dict no padrão: {resposta, usage, model, tempo} ou {erro, model, tempo}
        '''
        tempo = tempo_inicio or time()

        # Monta options e format
        options = {'temperature': temperature, 'num_ctx': num_ctx}
        if isinstance(max_tokens, int):
            options['num_predict'] = max_tokens
        fmt = 'json' if as_json else None

        # Chama a API nativa
        try:
            res_ollama = cls.chat_completion(
                messages=messages, modelo=modelo,
                options=options, format=fmt, timeout=timeout, api_url=api_url
            )
        except Exception as e:
            return {'erro': f'Erro Ollama: {type(e).__name__}: {str(e)}',
                    'model': modelo, 'tempo': round(time() - tempo, 3)}

        UtilLog.log_bruto('ollama', modelo, messages, res_ollama, tempo_str=tempo)
        if raw:
            res_ollama['tempo'] = round(time() - tempo, 3)
            return res_ollama

        conteudo = res_ollama.get('message', {}).get('content', '')
        usage_data = {
            'prompt_tokens': res_ollama.get('prompt_eval_count', 0),
            'completion_tokens': res_ollama.get('eval_count', 0),
            'total_tokens': (res_ollama.get('prompt_eval_count', 0) or 0) + (res_ollama.get('eval_count', 0) or 0)
        }
        comp_det = {}
        prompt_det = {}
        choices = [{'finish_reason': res_ollama.get('done_reason', 'stop')}]
        
        usage = {
            'prompt_tokens': usage_data.get('prompt_tokens', 0),
            'completion_tokens': usage_data.get('completion_tokens', 0),
            'total_tokens': usage_data.get('total_tokens', 0),
            'cached_tokens': prompt_det.get('cached_tokens', 0),
            'reasoning_tokens': comp_det.get('reasoning_tokens', 0),
            'finished_reason': choices[0].get('finish_reason', 'unknown'),
            'temperature': temperature
        }
        return UtilResponse.montar_resultado(conteudo, usage, res_ollama.get('model', modelo), time() - tempo, as_json, res_ollama)

class UtilOpenRouter:
    ''' Utilitário para chamadas ao OpenRouter usando o SDK oficial (openrouter).
        Segue o mesmo padrão de UtilOllama, com chat_completion_padronizado retornando
        o dict no formato padrão de get_resposta.
    '''

    @classmethod
    def _carregar_extra(cls) -> dict:
        ''' Carrega e valida OPENROUTER_EXTRA do ambiente.
            Retorna um dict (vazio se não configurado).
            Levanta ValueError se o JSON for inválido.
        '''
        _or_extra_str = os.getenv('OPENROUTER_EXTRA', '').strip()
        if not _or_extra_str:
            return {}
        try:
            extra = json.loads(_or_extra_str)
            if not isinstance(extra, dict):
                raise ValueError(f'OPENROUTER_EXTRA deve ser um objeto JSON (dict), recebeu {type(extra).__name__}')
            return extra
        except json.JSONDecodeError as e:
            raise ValueError(f'OPENROUTER_EXTRA contém JSON inválido: {e}\nConteúdo: {_or_extra_str}')

    @classmethod
    def _mapear_reasoning(cls, think: str) -> tuple:
        ''' Mapeia o parâmetro think para (reasoning_effort, verbosity).
            Retorna (None, None) se think for vazio/None.
        '''
        if not think:
            return None, None

        _reasoning = think.strip()
        _verbosity = 'low'
        if ':' in _reasoning:
            _reasoning, _verbosity = _reasoning.split(':', 1)

        reasoning_map = {
            'high': 'high', 'alto': 'high', 'h': 'high', 'a': 'high', '+': 'high',
            'medium': 'medium', 'm': 'medium', 'médio': 'medium', 'medio': 'medium',
            'minimal': 'minimal', 'mínimo': 'minimal', 'minimo': 'minimal', 'mi': 'minimal', '0': 'minimal', '-': 'minimal', 'x': 'minimal'
        }
        _reasoning = reasoning_map.get(_reasoning.lower(), 'low')
        _verbosity = reasoning_map.get(_verbosity.lower(), 'low')
        return _reasoning, _verbosity

    @classmethod
    def chat_completion_padronizado(cls, messages: list, modelo: str,
                                    temperature: float = 0.01, max_tokens: int = None,
                                    as_json: bool = True, raw: bool = False,
                                    timeout: int = 300, think: str = None,
                                    api_key: str = None,
                                    silencioso: bool = False,
                                    max_retry: int = 5,
                                    tempo_inicio: float = None) -> dict:
        ''' Chama o OpenRouter via SDK oficial e retorna a resposta no padrão de get_resposta.
            Parâmetros:
                messages: lista de mensagens [{"role": "...", "content": "..."}]
                modelo: nome do modelo (ex: "qwen/qwen3-235b-a22b-2507")
                temperature: temperatura do modelo
                max_tokens: número máximo de tokens na resposta
                as_json: se True, tenta converter a resposta em JSON
                raw: se True, retorna o dict bruto da API + tempo
                timeout: timeout da requisição em milissegundos (convertido internamente)
                think: configuração de reasoning (ex: "high:medium")
                api_key: chave de API do OpenRouter
                silencioso: se True, suprime prints de log
                max_retry: tentativas restantes em caso de erro
                tempo_inicio: timestamp de início (para cálculo de tempo)
            Retorna dict no padrão: {resposta, usage, model, tempo} ou {erro, model, tempo}
        '''
        tempo = tempo_inicio or time()

        api_key = api_key or os.getenv("PESSOAL_OPENROUTER_API_KEY")
        if not api_key:
            return {'erro': 'PESSOAL_OPENROUTER_API_KEY não encontrada no ambiente', 'model': modelo, 'tempo': round(time() - tempo, 3)}

        # Carrega configurações extras do ambiente
        openrouter_extra = cls._carregar_extra()

        # Monta argumentos para client.chat.send
        args = {
            'messages': messages,
            'model': modelo,
            'temperature': temperature,
            'timeout_ms': timeout * 1000,
        }

        if isinstance(max_tokens, int):
            args['max_tokens'] = max_tokens

        # Configuração de reasoning
        _reasoning, _verbosity = cls._mapear_reasoning(think)
        if _reasoning:
            args['reasoning'] = {'effort': _reasoning}
            if _verbosity and _verbosity != 'low':
                args['reasoning']['summary'] = _verbosity
            if isinstance(max_tokens, int):
                args['max_completion_tokens'] = max_tokens
                args.pop('max_tokens', None)
            if not silencioso:
                print(f'UtilOpenRouter: reasoning effort={_reasoning}, verbosity={_verbosity}, max_completion_tokens={args.get("max_completion_tokens")}')
        else:
            if not silencioso:
                print(f'UtilOpenRouter: sem reasoning, max_tokens={args.get("max_tokens")}')

        # Formato de resposta JSON
        if as_json:
            args['response_format'] = {"type": "json_object"}

        # Injeta configurações extras (ex: provider com quantizations)
        if openrouter_extra:
            args.update(openrouter_extra)
            if not silencioso:
                print(f'UtilOpenRouter: extras aplicados >> {openrouter_extra}')

        # Chamada ao SDK
        try:
            with OpenRouterClient(api_key=api_key) as client:
                response = client.chat.send(**args)

        except Exception as e:
            erro_str = str(e).lower()
            # Retry em rate limit ou erro de conexão
            if max_retry > 0 and ('rate' in erro_str or 'limit' in erro_str or 'connection' in erro_str or 'timeout' in erro_str):
                if not silencioso:
                    print(f'UtilOpenRouter: erro recuperável ({type(e).__name__}), tentando novamente... (restantes: {max_retry})')
                import time as time_module
                time_module.sleep(2)
                return cls.chat_completion_padronizado(
                    messages=messages, modelo=modelo,
                    temperature=temperature, max_tokens=max_tokens,
                    as_json=as_json, raw=raw, timeout=timeout,
                    think=think, api_key=api_key,
                    silencioso=silencioso,
                    max_retry=max_retry - 1,
                    tempo_inicio=tempo
                )
            print(f'UtilOpenRouter ERRO: {traceback.format_exc()}')
            return {'erro': f'Erro OpenRouter: {type(e).__name__}: {str(e)}',
                    'model': modelo, 'tempo': round(time() - tempo, 3)}

        # Converte resposta Pydantic para dict
        try:
            res_dict = response.model_dump()
        except Exception:
            res_dict = {}

        UtilLog.log_bruto('openrouter', modelo, messages, res_dict, tempo_str=tempo)
        if raw:
            res_dict['tempo'] = round(time() - tempo, 3)
            return res_dict

        choices = res_dict.get('choices', [])
        if not choices:
            return {'erro': 'Resposta do OpenRouter sem choices', 'model': modelo, 'tempo': round(time() - tempo, 3)}
            
        conteudo = choices[0].get('message', {}).get('content', '')
        usage_data = res_dict.get('usage', {}) or {}
        comp_det = usage_data.get('completion_tokens_details', {}) or {}
        prompt_det = usage_data.get('prompt_tokens_details', {}) or {}

        usage = {
            'prompt_tokens': usage_data.get('prompt_tokens', 0),
            'completion_tokens': usage_data.get('completion_tokens', 0),
            'total_tokens': usage_data.get('total_tokens', 0),
            'cached_tokens': prompt_det.get('cached_tokens', 0),
            'reasoning_tokens': comp_det.get('reasoning_tokens', 0),
            'finished_reason': choices[0].get('finish_reason', 'unknown'),
            'temperature': temperature
        }
        return UtilResponse.montar_resultado(conteudo, usage, res_dict.get('model', modelo), time() - tempo, as_json, res_dict)


class UtilOA:
    ''' Utilitário para consultas no serviço OpenAIA (OA)
    '''
    def __init__(self):
        oa_key = os.getenv("OA_KEY", "") # url::usuario::senha
        self.oa_banco = 'oracle'
        self.oa_url, self.oa_usuario, self.oa_senha = f'{oa_key}::-::-::'.split('::')[:3]
        
        oa_controle_str = os.getenv('OA_CONTROLE', '').strip()
        self.oa_controle = {}
        if oa_controle_str:
            try:
                self.oa_controle = json.loads(oa_controle_str)
            except json.JSONDecodeError as e:
                raise ValueError(f'OA_CONTROLE contém JSON inválido: {e}\nConteúdo: {oa_controle_str}')

    @classmethod
    def modelo_think(cls, modelo_think:str, think:str):
        ''' recebe um modelo e um think e retorna o modelo e o think
            se o think fizer parte do nome do modelo com :think, utiliza ele
        '''
        if not modelo_think or ':' not in modelo_think:
            return modelo_think, think

        valid_tokens = {
            'high', 'alto', 'h', 'a', '+',
            'medium', 'm', 'médio', 'medio',
            'minimal', 'mínimo', 'minimo', 'mi', '0', '-', 'x',
            'low', 'baixo', 'l', 'b'
        }
        partes = modelo_think.split(':')
        extracted = []
        # O modelo pode ter no máximo 2 tokens de think no final (ex: m:l)
        while len(partes) > 1 and len(extracted) < 2:
            ultimo = partes[-1].strip().lower()
            if ultimo in valid_tokens:
                extracted.insert(0, partes.pop())
            else:
                break

        if extracted:
            _modelo = ':'.join(partes)
            _think = ':'.join(extracted)
            return _modelo, _think

        return modelo_think, think

    def prompt(self, prompt:str, sg_modelo:str, think:str = 'm:l', as_json:bool = True, max_tokens:int = None):
        ''' Envia um post para a url do serviço OpenAIA e retorna o resultado no padrão de get_resposta '''
        import time
        import requests
        tempo_inicio = time.time()

        sg_modelo, think = self.modelo_think(sg_modelo, think)
        
        payload = {
            "usuario": self.oa_usuario,
            "senha": self.oa_senha,
            "banco": self.oa_banco,
            "sg_modelo": sg_modelo,
            "think": think,
            "resposta_json": as_json,
            "resumido": True
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        if self.oa_controle:
            payload.update(self.oa_controle)

        if isinstance(prompt, (tuple,list)):
            # se o prompt for uma lista, usa o parâmetro messages
            payload['messages'] = prompt
        else:
            # caso contrário, usa o parâmetro prompt
            payload['prompt'] = prompt

        try:
            response = requests.post(self.oa_url, json=payload, timeout=300)
            if not response.ok:
                return {'erro': f'Erro UtilOA HTTP {response.status_code}: {response.text}', 'model': sg_modelo, 'tempo': round(time.time() - tempo_inicio, 3)}
            res = response.json()
            
            conteudo = res.get('resposta', res.get('response', ''))
            usage_data = res.get('usage', {})
            comp_det = usage_data.get('completion_tokens_details', {}) or {}
            prompt_det = usage_data.get('prompt_tokens_details', {}) or {}
            
            finished_reason = usage_data.get('finished_reason', res.get('finish_reason', 'unknown'))
            
            usage = {
                'prompt_tokens': usage_data.get('prompt_tokens', 0),
                'completion_tokens': usage_data.get('completion_tokens', 0),
                'total_tokens': usage_data.get('total_tokens', 0),
                'cached_tokens': prompt_det.get('cached_tokens', 0),
                'reasoning_tokens': comp_det.get('reasoning_tokens', 0),
                'finished_reason': finished_reason,
                'temperature': payload.get('temperature')
            }

            return UtilResponse.montar_resultado(conteudo, usage, res.get('model', sg_modelo), time.time() - tempo_inicio, as_json, res)

        except Exception as e:
            return {'erro': f'Erro UtilOA: {type(e).__name__}: {str(e)}', 'model': sg_modelo, 'tempo': round(time.time() - tempo_inicio, 3)}


class UtilVllm:
    ''' Utilitário para facilitar consultas de status e modelos em servidores vLLM (API OpenAI Compatível)
    '''
    @classmethod
    def get_base_url(cls, api_url:str=None) -> str:
        base = api_url or os.getenv("VLLM_URL", "http://localhost:8000/v1")
        if base.endswith('/chat/completions'):
             base = base.replace('/chat/completions', '')
        if base.endswith('/'):
             base = base[:-1]
        return base

    @classmethod
    def models(cls, api_url:str=None):
        ''' Lista modelos disponíveis no servidor vLLM.
            GET /models
        '''
        base = cls.get_base_url(api_url)
        url = f'{base}/models'
        
        headers = {}
        api_key = os.getenv("VLLM_API_KEY")
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('data', [])

    @classmethod
    def status(cls, api_url:str=None):
        ''' Verifica se a API vLLM está disponível e retorna os modelos carregados.
        '''
        resultado = {'api': False, 'modelos': []}
        try:
            modelos = cls.models(api_url=api_url)
            resultado['modelos'] = [m.get('id', '?') for m in modelos]
            resultado['api'] = True
        except Exception:
            pass
        return resultado

# ---------------------------------------------------------------------------
# Helpers para detecção e validação de modelos de API remota
# ---------------------------------------------------------------------------

# Prefixos reconhecidos por este pacote e suas descrições
_PREFIXOS_API_REMOTA = {
    'or:': 'OpenRouter',
    'tg:': 'Together.ai',
    'vl:': 'vLLM Server',
    'oa:': 'OpenAIA',
}

def eh_modelo_api_remota(caminho: str) -> bool:
    '''Verifica se o caminho do modelo corresponde a uma API remota conhecida.

    Retorna True se o caminho iniciar com um dos prefixos reconhecidos:
    or: (OpenRouter), tg: (Together.ai), vl: (vLLM Server), oa: (OpenAI).

    Args:
        caminho: caminho ou identificador do modelo (ex: "or:qwen/qwen3.5-35b-a3b")

    Returns:
        True se for modelo de API remota, False caso contrário
    '''
    if not isinstance(caminho, str) or not caminho.strip():
        return False
    caminho_lower = caminho.strip().lower()
    return any(caminho_lower.startswith(p) for p in _PREFIXOS_API_REMOTA)


def extrair_nome_modelo_api(caminho: str) -> str:
    '''Extrai o nome do modelo sem o prefixo de API, para uso em logs e exibição.

    Ex: "or:qwen/qwen3.5-35b-a3b" -> "qwen/qwen3.5-35b-a3b"
        "/caminho/local/modelo" -> "/caminho/local/modelo"

    Args:
        caminho: caminho ou identificador do modelo

    Returns:
        Nome do modelo sem o prefixo, ou o caminho original se não for API remota
    '''
    if not isinstance(caminho, str) or not caminho.strip():
        return caminho or ''
    caminho_strip = caminho.strip()
    caminho_lower = caminho_strip.lower()
    for prefixo in _PREFIXOS_API_REMOTA:
        if caminho_lower.startswith(prefixo):
            return caminho_strip[len(prefixo):]
    return caminho_strip


def validar_modelo_api(caminho: str) -> tuple:
    '''Valida se o caminho do modelo é reconhecido como API remota.

    Verifica se o prefixo é conhecido e se o nome do modelo após o prefixo
    não está vazio. Não faz chamada de rede — é apenas validação local.

    Args:
        caminho: caminho ou identificador do modelo (ex: "or:qwen/qwen3.5-35b-a3b")

    Returns:
        Tupla (ok: bool, mensagem: str):
        - (True, "descrição da API") se o modelo é válido
        - (False, "mensagem de erro") se não for reconhecido ou estiver incompleto
    '''
    if not isinstance(caminho, str) or not caminho.strip():
        return False, 'Caminho do modelo está vazio'

    caminho_strip = caminho.strip()
    caminho_lower = caminho_strip.lower()

    for prefixo, descricao in _PREFIXOS_API_REMOTA.items():
        if caminho_lower.startswith(prefixo):
            nome = caminho_strip[len(prefixo):].strip()
            if not nome:
                return False, f'Nome do modelo está vazio após o prefixo "{prefixo}" ({descricao})'
            return True, f'{descricao} ({nome})'

    return False, f'Prefixo não reconhecido em "{caminho}". Prefixos válidos: {", ".join(_PREFIXOS_API_REMOTA.keys())}'

def _teste_modelo_think():
    testes = ['oa:modelo:h:l', 'ol:qwen2.5:7b-instruct:h:l', 'vl:models:', 'tg:modelo',
              'gpt5-megaplus:mi:low', 'gpt5:m:l', 'gpt-oss:20b:m:l', 'mistral-medium:','qwen3-235b:h:l']    
    for teste in testes:
        try:
            modelo, think = UtilOA.modelo_think(teste,'<nenhum>')
            print(f'Modelo / Think: {teste} >> {modelo} | {think}')
        except Exception as e:
            print(f'Erro ao testar modelo {teste}: {str(e)}')

if __name__ == '__main__':
    import sys
    import util  # garante que a pasta src está no sys.path
    from util import UtilEnv
    if not UtilEnv.carregar_env('.env', pastas=['../', './']):
        print('⚠️ Não foi possível encontrar o arquivo .env para carregar as variáveis de ambiente!')
    # Exemplos de uso:
    # teste_resposta(as_json=True, modelo='or:google/gemma-3-27b-it')  # OpenRouter
    # teste_resposta(as_json=True, modelo='ol:llama3')                 # Ollama local
    #teste_resposta(as_json=True, modelo='or:google/gemma-3-27b-it')
    teste_resposta(as_json=True, modelo='or:qwen/qwen3-235b-a22b-2507:m:l')

    #teste_resposta(as_json=True, modelo='tg:google/gemma-3n-E4B-it:h:h')
    #teste_resposta(as_json=True, modelo='gpt-5-nano')
    
    #_teste_modelo_think()
