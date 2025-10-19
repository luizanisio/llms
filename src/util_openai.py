from time import time
try:
    from openai import OpenAI, RateLimitError, APIConnectionError, AuthenticationError
except ImportError:
    raise ImportError("Módulo 'openai' não encontrado. Instale com 'pip install openai'.")
import os
import json
import traceback

'''
 Autor Luiz Anísio 17/10/2025
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
                 max_retry=5,
                 timeout=120,
                 api_key=None,
                 silencioso: bool = False):
    ''' Obtém a resposta do modelo de linguagem para o prompt informado.
        Quando as_json for True, tenta converter a resposta em JSON e retorna uma chave "erro" em caso de falha.
        Qualquer implementação de get_resposta deve seguir essa assinatura para ser usada pelos agentes.
        * api_key: chave de API para OpenAI ou OpenRouter (se None, tenta obter do ambiente)
                   > PESSOAL_OPENROUTER_API_KEY ou PESSOAL_OPENAI_API_KEY
        * modelo iniciado com or: vai usar o openrouter api
        * think: mi/low/medium/high para modelos com reasoning
                 :low/medium/high para definir verbosity
        Ex. think = 'high:medium' significa reasoning high e verbosity medium
        Retorna um dict com a estrutura:
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
    '''
    tempo = time()

    # Validação da API key
    if modelo.lower().startswith('or:'):
       tipo_api = 'openrouter'
       modelo = modelo[3:]
       url = "https://openrouter.ai/api/v1"
       api_key = api_key or os.getenv("PESSOAL_OPENROUTER_API_KEY")
       if not api_key:
          return {'erro': 'PESSOAL_OPENROUTER_API_KEY não encontrada no ambiente', 'model': modelo}
    else:
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

    parametros = {
        'messages': messages,
        'model': modelo,
        'temperature': temperature,
        'timeout': timeout,
        'max_tokens': max_tokens
    }

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

    try:
        response = client_gpt.chat.completions.create(**args)
        res_dict = response.to_dict()

        # Estrutura padronizada de retorno
        resultado = {}

        # Extrai o conteúdo da resposta
        if as_json:
            try:
                conteudo = res_dict['choices'][0]['message']['content']
                conteudo_json = UtilJson.mensagem_to_json(conteudo)
                resultado['resposta'] = conteudo_json
            except Exception as e:
                resultado['erro'] = f'Erro ao extrair JSON da resposta: {str(e)}'
                resultado['resposta'] = None
        else:
            resultado['resposta'] = res_dict['choices'][0]['message']['content']

        # Extrai informações de uso
        usage_data = res_dict.get('usage', {})
        completion_details = usage_data.get('completion_tokens_details', {})
        prompt_details = usage_data.get('prompt_tokens_details', {})

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
                           temperature=temperature, max_tokens=max_tokens,
                           max_retry=max_retry - 1, timeout=timeout, api_key=api_key)
    
    except RateLimitError as r:
        print(f'Erro de RateLimit: {str(r)}')
        if max_retry <= 0:
            return {'erro': 'rate limit alcançado, sem mais tentativas', 'model': modelo}
        print(f'Tentando novamente... (tentativas restantes: {max_retry})')
        return get_resposta(prompt=prompt, papel=papel, modelo=modelo, think=think, as_json=as_json,
                           temperature=temperature, max_tokens=max_tokens,
                           max_retry=max_retry - 1, timeout=timeout, api_key=api_key, silencioso=silencioso)
    
    except Exception as e:
        print('ERRO inesperado:', traceback.format_exc())
        return {'erro': f'Erro inesperado: {type(e).__name__}: {str(e)}', 'model': modelo}

    tempo = time() - tempo
    res['tempo'] = round(tempo, 3)
    return res

class UtilJson():
    @classmethod
    def mensagem_to_json(cls, mensagem:str, padrao = dict({}), _corrigir_json_ = True ):
        ''' O objetivo é receber uma resposta de um modelo LLM e identificar o json dentro dela
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


if __name__ == '__main__':
    from stjiautilbase.stj_utilitarios import UtilEnv
    UtilEnv.carregar_env('.env', pastas=['../', '../../'])
    assert os.getenv("PESSOAL_OPENAI_API_KEY"), "Chave PESSOAL_OPENAI_API_KEY não encontrada no .env"
    teste_api(openrouter=True)
