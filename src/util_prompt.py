try:
  import unsloth
  from unsloth import FastModel
  from unsloth.chat_templates import get_chat_template
  import torch
  import transformers
except ImportError as e:
  print(f'''\n\nOCORREU UM ERRO DE IMPORT: {e}
===========================================
= DICA DE PREPARAÇÃO DO AMBIENTE NO COLAB =
===========================================

!curl https://raw.githubusercontent.com/luizanisio/llms/refs/heads/main/util/get_git.py -o ./get_git.py
import get_git
get_git.deps() # instala unsloth, Transformers, Rouge e Levenshtein se precisar
  ''')
  
import json
from copy import deepcopy
from time import time
from enum import Enum

class Modelos(Enum):
    MODELO_GEMMA3_1B = "google/gemma-3-1b-it"   # 2Gb 
    MODELO_GEMMA3_4B = 'google/gemma-3-4b-it'   # 9Gb 
    MODELO_GEMMA3_12B = 'google/gemma-3-12b-it' # 25Gb
    MODELO_GEMMA3_27B = 'google/gemma-3-27b-it' # 39Gb
    MODELO_QWEN_7B = "Qwen/Qwen2.5-7B-Instruct-1M" # 14Gb
    MODELO_JUREMA_7B = "Jurema-br/Jurema-7B"       # 14Gb
    MODELO_QWEN_14B = 'Qwen/Qwen2.5-14B-Instruct-1M'   

    @classmethod
    def listar(cls):
        print("Modelos disponíveis:\n")
        for modelo in cls:
            print(f"- {modelo.name}: {modelo.value}")
        

class Prompt:
      def __init__(self, modelo = Modelos.MODELO_GEMMA3_1B, max_seq_length=4096, cache_dir:str = None, token:str = None):
          self.__pg = None
          self.modelo = UtilLMM.atalhos_modelos(modelo)
          if 'gemma' in str(self.modelo).lower():
             print(f'PromptGemma3: carregando modelo {self.modelo} ..')
             self.__pg = PromptGemma3(modelo=self.modelo, 
                                      max_seq_length=max_seq_length,
                                      cache_dir=cache_dir)
          elif 'qwen' in str(self.modelo).lower() or\
               'jurema' in str(self.modelo).lower():
             print(f'PromptQwen: carregando modelo {self.modelo} ..')
             self.__pg = PromptQwen(modelo=self.modelo, 
                                      max_seq_length=max_seq_length,
                                      cache_dir=cache_dir)

      def verifica_modelo(self):
          if self.__pg is None:
             raise ValueError(f'Não foi carregado um modelo válido! [{self.modelo}]')

      def prompt(self, prompt:str, max_new_tokens = 4096, temperatura = 0.3, detalhar = False):
          self.verifica_modelo()
          return self.__pg.prompt(prompt, max_new_tokens, temperatura, detalhar)

      def prompt_to_json(self, prompt:str, max_new_tokens = 4096, temperatura = 0):
          self.verifica_modelo()
          return self.__pg.prompt_to_json(prompt, max_new_tokens, temperatura)

      @classmethod
      def modelos(cls):
          Modelos.listar()
  
      @classmethod
      def verifica_versao(cls):
          print('============================================')
          print('Transformers:',transformers.__version__, unsloth.__version__)  # deve mostrar 4.53.x
          print('Tourch:', torch.__version__)
          print('============================================')

class PromptGemma3:
  START_T = '<start_of_turn>model'
  END_T = '<end_of_turn>'

  def __init__(self, modelo = Modelos.MODELO_GEMMA3_1B, max_seq_length=4096, cache_dir = None, token = None):
      modelo = UtilLMM.atalhos_modelos(modelo)
      # carregando o modelo
      self.modelo = modelo.value if isinstance(modelo, Modelos) else modelo
      self.max_seq_length = max_seq_length
      self.cache_dir = cache_dir
      self.model, self.tokenizer = FastModel.from_pretrained(
          model_name = self.modelo,
          max_seq_length = max_seq_length, # Choose any for long context!
          load_in_4bit = False,  # 4 bit quantization to reduce memory
          load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
          full_finetuning = False, # [NEW!] We have full finetuning now!
          device_map      = "auto",
          cache_dir       = cache_dir, 
          token = token, # use one if using gated models
      )

      self.tokenizer = get_chat_template(
          self.tokenizer,
          chat_template = "gemma-3",
      )

  def prompt(self, prompt:str, max_new_tokens = 4096, temperatura = 0.3, detalhar = False):
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}]}]
        # gemma 1b tem particularidades
        g1b = '-1b-' in self.modelo
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, 
            tokenizer = g1b, return_tensors="pt"
        )
        if g1b: 
           inputs = inputs.to(self.model.device)
           inputs = {'input_ids': inputs}
        else:
           inputs = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)

        _temperatura = temperatura if isinstance(temperatura, float) else 1.0
        with torch.inference_mode():
              out_ids = self.model.generate(
                  **inputs,
                  max_new_tokens = max_new_tokens,
                  temperature = _temperatura
              )
        res = self._limpar_retorno(self.tokenizer.decode(out_ids[0], skip_special_tokens=False))
        if not detalhar:
           return res
        _input_tokens = inputs["input_ids"].size(1)
        _output_tokens = out_ids.size(1) - _input_tokens
        return {'texto': res, 'input_tokens': _input_tokens, 'output_tokens': _output_tokens}

  def _limpar_retorno(self, txt: str) -> str:
      if self.START_T in txt:
          txt = txt.split(self.START_T, 1)[1]
      if self.END_T and self.END_T in txt:
          txt = txt.split(self.END_T, 1)[0]
      return txt.lstrip("\n ").rstrip("\n ")

  def prompt_to_json(self, prompt:str, max_new_tokens = 4096, temperatura = 0):
      ini = time()
      retorno = self.prompt(prompt, max_new_tokens = max_new_tokens, temperatura=temperatura, detalhar = True)
      res = UtilLMM.mensagem_to_json(retorno.pop('texto',None))
      retorno['time'] = time()-ini
      res.update({'usage': retorno})
      return res

  def exemplo(self):
      _prompt_teste = '''Retorne um json válido com a estrutrua {"mensagem": com a mensagem do usuário, "itens": com uma lista de itens quando ele enumerar algo }
                         Mensagem do usuário: Eu preciso comprar abacaxi, pera e 2L de leite.'''
      r = self.prompt_to_json(_prompt_teste)
      print(json.dumps(r, indent=2, ensure_ascii=False))

# QWEN e JUREMA - O Jurema, em 08/2025, exige token do hugginface
class PromptQwen(PromptGemma3):
    """
    Classe para rodar inferência com modelos Qwen-7B-Chat/Instr.
    Mantém a API idêntica à PromptGemma3.
    """
    # tags de separação tipicamente usadas pelo template qwen-chat
    START_U = '<|im_start|>user'      # início da pergunta do usuário
    START_T = '<|im_start|>assistant' # início da resposta do modelo
    END_T   = '<|im_end|>'            # fim opcional (pode não aparecer)

    def __init__(self,
                 modelo: Modelos = Modelos.MODELO_QWEN_7B,
                 max_seq_length: int = 4096,
                 cache_dir: str | None = None,
                 token = None):
        # carrega o modelo base usando a mesma lógica da superclasse
        super().__init__(modelo=modelo,
                         max_seq_length=max_seq_length,
                         cache_dir=cache_dir,
                         token=token)

        # troca o chat-template para o “qwen2” (disponível no unsloth ≥ 0.4.0)
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen2.5",
        )

    # TODO unir na classe herdada
    def prompt(self,
               prompt: str,
               max_new_tokens: int = 4096,
               temperatura: float = 0.3,
               detalhar: bool = False):

        # Fix: The content field should be a string, not a list of dictionaries.
        messages = [{"role": "user", "content": prompt}]

        # aplica o template já no formato de ids
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        _temperatura = temperatura if isinstance(temperatura, float) else 1.0
        with torch.inference_mode():
            out_ids = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                temperature=_temperatura,
            )

        resposta = self._limpar_retorno(
            self.tokenizer.decode(out_ids[0], skip_special_tokens=False)
        )

        if not detalhar:
            return resposta

        n_in  = inputs.size(1)
        n_out = out_ids.size(1) - n_in
        return {"texto": resposta,
                "input_tokens": n_in,
                "output_tokens": n_out}


##########################################
class UtilLMM():
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

    @classmethod
    def atalhos_modelos(cls, modelo):
        atalho_map = {
            '1': Modelos.MODELO_GEMMA3_1B,    '4': Modelos.MODELO_GEMMA3_4B,
            '12': Modelos.MODELO_GEMMA3_12B,  '27': Modelos.MODELO_GEMMA3_27B,
            'j': Modelos.MODELO_JUREMA_7B,    'j7': Modelos.MODELO_JUREMA_7B,
            'jurema': Modelos.MODELO_JUREMA_7B,
            'q': Modelos.MODELO_QWEN_7B,      'q7': Modelos.MODELO_QWEN_7B,
            'q14': Modelos.MODELO_QWEN_14B
        }
        res = atalho_map.get(str(modelo).lower().strip(), modelo)      
        if res:
           return res
        return atalho_map.get(str(modelo).lower().strip(' bB'), modelo)      
