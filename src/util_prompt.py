try:
  import unsloth
  from unsloth import FastModel
  from unsloth.chat_templates import get_chat_template
  import torch
  import transformers
except ImportError as e:
  print(f'''\n\nOCORREU UM ERRO DE IMPORT: {e}
###########################################  
# DICA DE PREPARAÇÃO DO AMBIENTE NO COLAB
# * copie e rode no início do notebook
###########################################  
arq1 = '/content/pip_unsloth_ok.txt'
arq2 = '/content/pip_transformers_ok.txt'
#clear output
from IPython.display import clear_output
import os
if not os.path.isfile(arq1):
    # Passo 1: Instalar o unsloth primeiro
    !pip install unsloth[colab-new]==2025.7.1
    import unsloth
    clear_output()
    print('Unsloth instalado _o/')
    with open(arq1, 'w') as f:
        f.write('Unsloth instalado _o/')
else:
  print('Unsloth já ok _o/')

if not os.path.isfile(arq2):
    # Passo 2: Instalar outras dependências 
    !pip install --upgrade --force-reinstall --no-cache-dir \
         "transformers>=4.53.0,<4.54.0"
    clear_output()
    print('Transformers instalado _o/')
    with open(arq2, 'w') as f:
        f.write('Transformers instalado _o/')
else:
  print('Transformers já ok _o/')
  ''')
  
import json
from copy import deepcopy
from time import time

MODELO_GEMMA3_1B = "google/gemma-3-1b-it"   # 2Gb disco
MODELO_GEMMA3_4B = 'google/gemma-3-4b-it'   # 9Gb disco
MODELO_GEMMA3_12B = 'google/gemma-3-12b-it' # 25Gb disco
MODELO_GEMMA3_27B = 'google/gemma-3-27b-it' # 

class Prompt:
      def __init__(self, modelo = MODELO_GEMMA3_1B, max_seq_length=4096, cache_dir = None):
          self.__pg = None
          self.modelo = UtilLMM.atalhos_modelos(modelo)
          if 'gemma' in str(self.modelo).lower():
             print(f'PromptGemma3: carregando modelo {self.modelo} ..')
             self.__pg = PromptGemma3(modelo=self.modelo, 
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
      def verifica_versao(cls):
          print('============================================')
          print('Transformers:',transformers.__version__, unsloth.__version__)  # deve mostrar 4.53.x
          print('Tourch:', torch.__version__)
          print('============================================')

class PromptGemma3:
  START_T = '<start_of_turn>model'
  END_T = '<end_of_turn>'

  def __init__(self, modelo = MODELO_GEMMA3_1B, max_seq_length=4096, cache_dir = None):
      modelo = UtilLMM.atalhos_modelos(modelo)
      # carregando o modelo
      self.modelo = modelo
      self.max_seq_length = max_seq_length
      self.cache_dir = cache_dir
      self.model, self.tokenizer = FastModel.from_pretrained(
          model_name = modelo,
          max_seq_length = max_seq_length, # Choose any for long context!
          load_in_4bit = False,  # 4 bit quantization to reduce memory
          load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
          full_finetuning = False, # [NEW!] We have full finetuning now!
          device_map      = "auto",
          cache_dir       = cache_dir, 
          # token = "hf_...", # use one if using gated models
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
        res = self.__limpar_retorno(self.tokenizer.decode(out_ids[0], skip_special_tokens=False))
        if not detalhar:
           return res
        _input_tokens = inputs["input_ids"].size(1)
        _output_tokens = out_ids.size(1) - _input_tokens
        return {'texto': res, 'input_tokens': _input_tokens, 'output_tokens': _output_tokens}

  def __limpar_retorno(self, retorno):
        try:
            texto = retorno.split(self.START_T,1)[1]
            texto = texto.split(self.END_T, 1)[0]
        finally:
            return texto.lstrip("\n").rstrip("\n")

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
            '1b': MODELO_GEMMA3_1B, '1': MODELO_GEMMA3_1B,
            '4b': MODELO_GEMMA3_4B, '4': MODELO_GEMMA3_4B,
            '12b': MODELO_GEMMA3_12B, '12': MODELO_GEMMA3_12B,
            '27b': MODELO_GEMMA3_27B, '27': MODELO_GEMMA3_27B,
        }
        return atalho_map.get(str(modelo).lower().strip(), modelo)      
