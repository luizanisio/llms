# -*- coding: utf-8 -*-

"""
Utilitário para obter respostas de LLMs via STJ OpenAI.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Funções auxiliares para comunicação com modelos de linguagem através da
biblioteca stjiautilbase, com suporte a JSON e resumo de respostas.
"""

from stjiautilbase.stj_openaia import STJOpenAIA
from stjiautilbase.stj_utilitarios import UtilTextos

#MODELO_AGENTES = 'GPT5-srv'
#MODELO_AGENTES_THINK = 'low:low'
MAX_TOKENS = 10000

def resume_resposta(resposta: dict) -> dict:
    """
    Resume a resposta do agente em uma estrutura compacta para logs.
    
    Parâmetros:
    -----------
    resposta : dict
        Dicionário de resposta completo retornado pela LLM
    
    Retorna:
    --------
    dict
        Dicionário resumido contendo:
        - 'resposta': Conteúdo da resposta
        - 'usage': Métricas de tokens (se disponível)
        - 'erro': Mensagem de erro (se houver)
    
    Exemplo:
    --------
    >>> resposta_llm = get_resposta(prompt="teste", modelo="gpt-5")
    >>> resumo = resume_resposta(resposta_llm)
    >>> print(resumo['usage']['total_tokens'])
    """
    if 'erro' in resposta:
        return {'erro': resposta['erro'] }
    _resposta = {'resposta': resposta.get('response')}
    if ('usage' in resposta) and isinstance(resposta['usage'], dict):
        usage = resposta['usage']
        _usage = {}
        _usage['prompt_tokens'] = usage.get('prompt_tokens', 0)
        _usage['completion_tokens'] = usage.get('completion_tokens', 0)
        _usage['total_tokens'] = usage.get('total_tokens', 0)
        _usage['cached_tokens'] = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        _usage['reasoning_tokens'] = usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
        _resposta['usage'] = _usage
    return _resposta

def get_resposta(prompt: str, papel = '', modelo: str = None, modelo_think: str = None, as_json = True):
    """
    Obtém a resposta do modelo de linguagem para o prompt informado.
    
    Esta função utiliza a biblioteca stjiautilbase para comunicação com modelos
    de linguagem, oferecendo suporte a retorno em JSON e controle de tokens.
    
    Parâmetros:
    -----------
    prompt : str
        Texto do prompt a ser enviado ao modelo
    papel : str, opcional
        Papel/contexto do assistente (system message)
    modelo : str, obrigatório
        Identificador do modelo a ser utilizado
    modelo_think : str, opcional
        Nível de reasoning para modelos que suportam
    as_json : bool, padrão True
        Se True, tenta converter a resposta em JSON
    
    Retorna:
    --------
    dict
        Estrutura padronizada:
        - 'resposta': dict ou str, conteúdo da resposta
        - 'usage': dict com tokens (prompt, completion, total, cached, reasoning)
        - 'erro': str, presente apenas em caso de erro
    
    Nota:
    -----
    Qualquer implementação de get_resposta deve seguir essa assinatura 
    para ser compatível com o sistema de agentes.
    
    Exemplo:
    --------
    >>> resposta = get_resposta(
    ...     prompt="Explique inteligência artificial",
    ...     papel="Você é um professor",
    ...     modelo="GPT5-srv",
    ...     as_json=True
    ... )
    >>> if 'erro' not in resposta:
    ...     print(resposta['resposta'])
    """
    if not modelo:
       raise ValueError(f"Parâmetro 'modelo' é obrigatório para get_resposta(...)")
    oa = STJOpenAIA(silencioso=True)
    controle_aia = {'~nolog~': True, 'projeto': 'análise summa espelho'}
    resposta = oa.prompt(prompt, papel = papel, sg_modelo=modelo, think=modelo_think, max_tokens=MAX_TOKENS, retorno_resumido=True,
                         prompt_retorna_json=as_json, sem_erro=True, controle_aia=controle_aia)

    resumo = resume_resposta(resposta)
    if not as_json:
        return resumo
    # converte a resposta em dict
    try:
        resposta_dict = UtilTextos.mensagem_to_json(resposta.get('response',''))
        resumo['resposta'] = resposta_dict
    except Exception as e:
        resumo['erro'] = f'Erro ao converter resposta em JSON: {str(e)}'
    return resumo
