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
import os
import json
import traceback
import requests

from threading import Lock
LOCK_ARQUIVO_BRUTO = Lock()
LOG_BRUTO = os.getenv('LOG_BRUTO','').lower()

'''
 Autor Luiz Anísio 17/10/2025
 Fonte: https://github.com/luizanisio/llms/tree/main/src
 Utilitários para acionar apis generativas da openai ou openrouter
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
        * api_key: chave de API para OpenAI ou OpenRouter (se None, tenta obter do ambiente)
                   > PESSOAL_OPENROUTER_API_KEY ou PESSOAL_OPENAI_API_KEY
        * modelo iniciado com or: vai usar o openrouter api
        * modelo iniciado com ol: vai usar o ollama local via UtilOllama (http://localhost:11434/api)
        * raw: se True, retorna o dict bruto da API (sem padronização), apenas com 'tempo' adicionado
        * max_ctx: tamanho máximo da janela de contexto (entrada + saída) em tokens
                   - Para Ollama: define o num_ctx (padrão do método: 49152 = 48k)
                   - Para outras APIs: ignorado no momento
        * think: mi/low/medium/high para modelos com reasoning
                 :low/medium/high para definir verbosity
        Ex. think = 'high:medium' significa reasoning high e verbosity medium
        Retorna um dict com a estrutura (raw=False):
        {   'resposta': dict,  # JSON parseado da resposta da LLM (quando as_json=True)
            'usage': {         # Informações de tokens utilizados
                'prompt_tokens': int,
                'completion_tokens': int,
                'total_tokens': int,
                'cached_tokens': int,
                'reasoning_tokens': int,
                'finished_reason': str,
                'temperature': float      # se o modelo suportar
            },
            'erro': str,        # Presente apenas em caso de erro
            'tempo': float,     # Tempo de resposta em segundos
        }
        Quando raw=True, retorna o dict bruto da API + 'tempo'.
    '''
    tempo = time()
    if 'modelo_think' in kwargs and kwargs['modelo_think']:
        think = kwargs['modelo_think'] 

    # Validação da API key
    if modelo.lower().startswith('or:'):
       if MSG_OPENAI:
          raise ImportError(MSG_OPENAI)
       # openrouter.ai  
       tipo_api = 'openrouter'
       modelo = modelo[3:]
       url = "https://openrouter.ai/api/v1"
       api_key = api_key or os.getenv("PESSOAL_OPENROUTER_API_KEY")
       if not api_key:
          return {'erro': 'PESSOAL_OPENROUTER_API_KEY não encontrada no ambiente', 'model': modelo}
       client_gpt = OpenAI(api_key=api_key, timeout=timeout, base_url=url) 
    elif modelo.lower().startswith('ol:'):
       # ollama local via UtilOllama (API nativa)
       tipo_api = 'ollama'
       modelo = modelo[3:]
       # Caso especial: listar modelos disponíveis
       if modelo.lower() == 'models':
           try:
               res_models = UtilOllama.models()
               return {'resposta': res_models, 'modelo': 'ol:models', 'tempo': round(time() - tempo, 3)}
           except Exception as e:
               return {'erro': f'Erro ao listar modelos Ollama: {str(e)}', 'tempo': round(time() - tempo, 3)}
       # Caso especial: status do Ollama
       if modelo.lower() == 'status':
           try:
               res_status = UtilOllama.status()
               return {'resposta': res_status, 'modelo': 'ol:status', 'tempo': round(time() - tempo, 3)}
           except Exception as e:
               return {'erro': f'Erro ao verificar status do Ollama: {str(e)}', 'tempo': round(time() - tempo, 3)}
    elif modelo.lower().startswith('tg:'):
       if MSG_TOGETHER:
          raise ImportError(MSG_TOGETHER)
       # together.ai
       tipo_api = 'together'
       modelo = modelo[3:]

       api_key = api_key or os.getenv("PESSOAL_TOGETHER_API_KEY")
       if not api_key:
          return {'erro': 'PESSOAL_TOGETHER_API_KEY não encontrada no ambiente', 'model': modelo}
       client_gpt = Together(api_key=api_key) 
    else:
       if MSG_OPENAI:
          raise ImportError(MSG_OPENAI) 
       tipo_api = 'openai'
       api_key = api_key or os.getenv("PESSOAL_OPENAI_API_KEY")
       url = None
       if not api_key:
          return {'erro': 'PESSOAL_OPENAI_API_KEY não encontrada no ambiente', 'model': modelo}

       client_gpt = OpenAI(api_key=api_key, timeout=timeout, base_url=url)

    max_retry = max(0, min(10, max_retry))
    if not silencioso: print(f'Chamada: {modelo}:{think} [{tipo_api}] | max retry = {max_retry} | ')
    
    max_tokens = max(0, min(128*1024, max_tokens)) if isinstance(max_tokens, int) else None

    if isinstance(prompt,(tuple, list)) and len(prompt) > 0:
        messages = prompt
    elif isinstance(prompt,str) and prompt.strip(' \n\t'):    
        messages = []
        if papel and isinstance(papel, str) and papel.strip():
            messages.append({'role':'system', 'content': papel.strip()})
        messages.append({'role':'user', 'content': prompt})
    else:
        raise ValueError('Formato do prompt não reconhecido: deve ser messages ou string')

    # ── Ollama: delega para UtilOllama e retorna ──
    if tipo_api == 'ollama':
        if not silencioso: print(f'get_resposta: Ollama local - modelo {modelo}')
        ollama_kwargs = dict(
            messages=messages, modelo=modelo,
            temperature=temperature, max_tokens=max_tokens,
            as_json=as_json, raw=raw, timeout=timeout,
            tempo_inicio=tempo
        )
        if isinstance(max_ctx, int) and max_ctx > 0:
            ollama_kwargs['num_ctx'] = max_ctx
        return UtilOllama.chat_completion_padronizado(**ollama_kwargs)

    # ── Demais APIs (OpenAI, OpenRouter, Together): monta args ──
    parametros = {
        'messages': messages,
        'model': modelo,
        'temperature': temperature,
        'timeout': timeout
    }
    if isinstance(max_tokens,int):
        parametros['max_tokens'] = max_tokens

    # Configuração de reasoning
    args = parametros.copy()
    if think:
        # Parse reasoning e verbosity (formato: reasoning:verbosity ou apenas reasoning)
        _reasoning = think.strip()
        _verbosity = 'low'
        
        if ':' in _reasoning:
            _reasoning, _verbosity = _reasoning.split(':', 1)
        
        # Mapeia reasoning
        _reasoning = _reasoning.lower()
        reasoning_map = {
            'high': 'high', 'alto': 'high', 'h': 'high', 'a': 'high', '+': 'high',
            'medium': 'medium', 'm': 'medium', 'médio': 'medium', 'medio': 'medium',
            'minimal': 'minimal', 'mínimo': 'minimal', 'minimo': 'minimal', 'mi': 'minimal', '0': 'minimal', '-': 'minimal', 'x': 'minimal'
        }
        _reasoning = reasoning_map.get(_reasoning, 'low')
        
        # Mapeia verbosity
        _verbosity = _verbosity.lower()
        _verbosity = reasoning_map.get(_verbosity, 'low')
        
        # Remove parâmetros incompatíveis com reasoning models
        if tipo_api == 'openai' or 'gpt-5' in modelo.lower():
            print(f'\n===============\nModelo reasoning em uso {modelo}, removendo parâmetros inválidos...\n===============\n')
            args = {k: v for k, v in parametros.items() 
                   if k not in {'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty'}}

        args['reasoning_effort'] = _reasoning
        if isinstance(max_tokens, int):
            args['max_completion_tokens'] = max_tokens
        
        if not silencioso: print(f'get_resposta: usando reasoning_effort={_reasoning} / verbosity={_verbosity} e max_completion_tokens={args.get("max_completion_tokens")} para o modelo {modelo}')
    else:
        if not silencioso: print(f'get_resposta: usando modelo sem reasoning e max_tokens={args.get("max_tokens")} para o modelo {modelo}')

    if as_json:
        args['response_format'] = {"type": "json_object"}

    # ajustes por api
    if tipo_api == 'together':
        args.pop('max_completion_tokens',None)
        assert max_tokens is None or 'max_tokens' in args, 'max_tokens é o parâemtro do Together AI'

    try:
        response = client_gpt.chat.completions.create(**args)
        if tipo_api == 'together':
            res_dict = response.dict()
            conteudo = str(response.choices[0].message.content)
            usage_data = response.usage.dict()
        else:
            res_dict = response.to_dict()
            conteudo = res_dict['choices'][0]['message']['content']

        # Estrutura padronizada de retorno
        resultado = {}
        #print(json.dumps(res_dict, ensure_ascii=False, indent=2))

        if tipo_api in LOG_BRUTO:
            with LOCK_ARQUIVO_BRUTO:
                # grava a resposta no arquivo de log bruto
                with open('log_openai_resposta_bruta.txt', 'a', encoding='utf-8') as f:
                    f.write('#'*60 + '\n')
                    f.write(f'Timestamp: {time()}\n')
                    f.write(f'Modelo: {modelo}\n')
                    f.write('='*40 + '\n')
                    # grava o prompt bruto
                    f.write('Prompt enviado:\n')
                    f.write(json.dumps(messages, ensure_ascii=False, indent=2))
                    f.write('\n' + ('-'*40) + '\n')
                    # grava a resposta bruta
                    f.write('Resposta bruta:\n')
                    if tipo_api == 'together':
                        f.write(json.dumps(res_dict, ensure_ascii=False, indent=2))
                        f.write('\n' + ('-'*40) + '\n')
                        f.write('Usage:\n')
                        f.write(json.dumps(usage_data, ensure_ascii=False, indent=2))
                    else:
                        f.write(json.dumps(res_dict, ensure_ascii=False, indent=2))
                    f.write('\n' + ('-'*80) + '\n')
                    f.write('#'*60 + '\n')

        # Se raw=True, retorna o dict bruto da API com tempo
        if raw:
            res_dict['tempo'] = round(time() - tempo, 3)
            return res_dict

        # Extrai o conteúdo da resposta
        if as_json:
            try:
                conteudo_json = UtilJson.mensagem_to_json(conteudo, padrao=None)
                if conteudo_json is None:
                   resultado['resposta'] = conteudo
                   resultado['erro'] = f'Erro ao extrair JSON da resposta (conteudo=None) (Dict resposta={res_dict}).'
                   resultado['json'] = False
                else:  
                   resultado['resposta'] = conteudo_json
                   resultado['json'] = True
            except Exception as e:
                resultado['resposta'] = conteudo
                resultado['json'] = False
                resultado['erro'] = f'Erro ao extrair JSON da resposta: {str(e)}'
        else:
            resultado['resposta'] = conteudo

        # Extrai informações de uso
        usage_data = res_dict.get('usage', {})
        completion_details = usage_data.get('completion_tokens_details', {}) or {}
        prompt_details = usage_data.get('prompt_tokens_details', {}) or {}

        #print(json.dumps(res_dict,ensure_ascii=False,indent=2))
        #print('-'*40)
        resultado['usage'] = {
            'prompt_tokens': usage_data.get('prompt_tokens', 0),
            'completion_tokens': usage_data.get('completion_tokens', 0),
            'total_tokens': usage_data.get('total_tokens', 0),
            'cached_tokens': prompt_details.get('cached_tokens', 0),
            'reasoning_tokens': completion_details.get('reasoning_tokens', 0),
            'finished_reason': res_dict['choices'][0].get('finish_reason', 'unknown')
        }
        if 'temperature' in args:
            resultado['usage']['temperature'] = args['temperature']

        # Adiciona modelo usado
        resultado['model'] = res_dict.get('model', modelo)

        res = resultado

    except AuthenticationError as e:
        print(f'Erro de autenticação: {str(e)}')
        return {'erro': f'Erro de autenticação: {str(e)}. Verifique se a API key está correta e ativa.', 'model': modelo}
    
    except APIConnectionError as e:
        print(f'Erro de conexão: {str(e)}')
        if max_retry <= 0:
            return {'erro': f'Erro de conexão com API após todas as tentativas: {str(e)}', 'model': modelo}
        print(f'Tentando novamente... (tentativas restantes: {max_retry})')
        import time as time_module
        time_module.sleep(2)
        return get_resposta(prompt=prompt, papel=papel, modelo=modelo, think=think, as_json=as_json,
                           temperature=temperature, max_tokens=max_tokens, max_ctx=max_ctx,
                           max_retry=max_retry - 1, timeout=timeout, api_key=api_key, raw=raw)
    
    except RateLimitError as r:
        print(f'Erro de RateLimit: {str(r)}')
        if max_retry <= 0:
            return {'erro': 'rate limit alcançado, sem mais tentativas', 'model': modelo}
        print(f'Tentando novamente... (tentativas restantes: {max_retry})')
        return get_resposta(prompt=prompt, papel=papel, modelo=modelo, think=think, as_json=as_json,
                           temperature=temperature, max_tokens=max_tokens, max_ctx=max_ctx,
                           max_retry=max_retry - 1, timeout=timeout, api_key=api_key, raw=raw, silencioso=silencioso)
    
    except Exception as e:
        print('ERRO inesperado:', traceback.format_exc())
        return {'erro': f'Erro inesperado: {type(e).__name__}: {str(e)}', 'model': modelo}

    tempo = time() - tempo
    res['tempo'] = round(tempo, 3)
    return res

class UtilJson():
    @classmethod
    def mensagem_to_json(cls, mensagem:str, padrao = dict({}), _corrigir_json_ = True):
        ''' O objetivo é receber uma resposta de um modelo LLM e identificar o json dentro dela
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

        # Log bruto
        try:
            with LOCK_ARQUIVO_BRUTO:
                with open('log_openai_resposta_bruta.txt', 'a', encoding='utf-8') as f:
                    f.write('#'*60 + '\n')
                    f.write(f'Timestamp: {time()}\n')
                    f.write(f'Modelo: ol:{modelo}\n')
                    f.write('='*40 + '\n')
                    f.write('Prompt enviado:\n')
                    f.write(json.dumps(messages, ensure_ascii=False, indent=2))
                    f.write('\n' + ('-'*40) + '\n')
                    f.write('Resposta bruta:\n')
                    f.write(json.dumps(res_ollama, ensure_ascii=False, indent=2))
                    f.write('\n' + ('-'*80) + '\n')
                    f.write('#'*60 + '\n')
        except Exception:
            pass

        # Se raw, retorna o dict nativo do Ollama
        if raw:
            res_ollama['tempo'] = round(time() - tempo, 3)
            return res_ollama

        # Padroniza o retorno para o formato de get_resposta
        conteudo = res_ollama.get('message', {}).get('content', '')
        resultado = {}
        if as_json:
            try:
                conteudo_json = UtilJson.mensagem_to_json(conteudo, padrao=None)
                if conteudo_json is None:
                    resultado['resposta'] = conteudo
                    resultado['erro'] = 'Erro ao extrair JSON da resposta Ollama (conteudo=None).'
                    resultado['json'] = False
                else:
                    resultado['resposta'] = conteudo_json
                    resultado['json'] = True
            except Exception as e:
                resultado['resposta'] = conteudo
                resultado['json'] = False
                resultado['erro'] = f'Erro ao extrair JSON da resposta: {str(e)}'
        else:
            resultado['resposta'] = conteudo

        resultado['usage'] = {
            'prompt_tokens': res_ollama.get('prompt_eval_count', 0),
            'completion_tokens': res_ollama.get('eval_count', 0),
            'total_tokens': (res_ollama.get('prompt_eval_count', 0) or 0) + (res_ollama.get('eval_count', 0) or 0),
            'cached_tokens': 0,
            'reasoning_tokens': 0,
            'finished_reason': res_ollama.get('done_reason', 'stop')
        }
        resultado['usage']['temperature'] = temperature
        resultado['model'] = res_ollama.get('model', modelo)
        resultado['tempo'] = round(time() - tempo, 3)
        return resultado


if __name__ == '__main__':
    import sys 
    sys.path.insert(0, os.path.abspath('..'))
    sys.path.extend(['../src','./src'])
    from util import UtilEnv
    UtilEnv.carregar_env('.env', pastas=['../', './'])
    # Exemplos de uso:
    # teste_resposta(as_json=True, modelo='or:google/gemma-3-27b-it')  # OpenRouter
    # teste_resposta(as_json=True, modelo='ol:llama3')                 # Ollama local
    teste_resposta(as_json=True, modelo='or:google/gemma-3-27b-it')
