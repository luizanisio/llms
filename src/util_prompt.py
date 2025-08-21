import torch
import json
from copy import deepcopy
from time import time
from enum import Enum
import platform
import traceback
# vai guardar os pacotes importados dinamicamente
FASTMODEL = None
GETCHATTEMPLATE = None
AUTTOTOKENIZER:any = None
AUTOMODEL:any = None
AUTOMODELG:any = None
GENCONFIG:any = None

class Modelos(Enum):
    MODELO_GEMMA3_1B = "google/gemma-3-1b-it"   # 2Gb 
    MODELO_GEMMA3_4B = 'google/gemma-3-4b-it'   # 9Gb 
    MODELO_GEMMA3_12B = 'google/gemma-3-12b-it' # 25Gb
    MODELO_GEMMA3_27B = 'google/gemma-3-27b-it' # 39Gb
    MODELO_QWEN_7B = "Qwen/Qwen2.5-7B-Instruct-1M" # 14Gb
    MODELO_JUREMA_7B = "Jurema-br/Jurema-7B"       # 14Gb
    MODELO_QWEN_14B = 'Qwen/Qwen2.5-14B-Instruct-1M'   
    MODELO_DEEPSEEK_1_5B = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    MODELO_DEEPSEEK_70B = 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B' # 150Gb
    MODELO_DEEPSEEK_32B = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
    MODELO_DEEPSEEK_14B = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    MODELO_DEEPSEEK_7B = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    MODELO_DEEPSEEK_8B = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    MODELO_OSS_20B = 'openai/gpt-oss-20b'
    MODELO_OSS_120B = 'openai/gpt-oss-120b'
    MODELO_LLAMA_4_SCOUT_16B = 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
    MODELO_LLAMA_3_3_70B = 'meta-llama/Llama-3.3-70B-Instruct'
    MODELO_LLAMA_3_2_1B = 'meta-llama/Llama-3.2-1B-Instruct'
    MODELO_LLAMA_3_2_3B = 'meta-llama/Llama-3.2-3B-Instruct'

    @classmethod
    def listar(cls):
        print("Modelos disponíveis:\n")
        for modelo in cls:
            print(f"- {modelo.name}: {modelo.value}")

class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)    

class Prompt:
    def __init__(self, modelo:str, max_seq_length:int = 4096, cache_dir:str|None = None, usar_unsloth:bool = False):
        # identificando o modelo
        modelo = UtilLLM.atalhos_modelos(modelo)
        self.modelo = modelo.value if isinstance(modelo, Modelos) else modelo
        _unsloth = ' (Unsloth)' if usar_unsloth else ''
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._num_gpus = torch.cuda.device_count()
        _device = f' {self._num_gpus} x {self._device}' if self._device != 'cpu' else 'cpu'
        print(f'Modelo selecionado: {self.modelo}{_unsloth} | device: {_device}')
        self.otimiza_torch()
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir
        self.usar_unsloth = usar_unsloth
        self._model = None
        self._tokenizer = None
        self._configurar_separadores()
        self.carregar_model_tokenizer(modelo=self.modelo,
                                      max_seq_length=max_seq_length,
                                      cache_dir=cache_dir,
                                      usar_unsloth=usar_unsloth)


    @classmethod
    def otimiza_torch(cls, cache_limit = 128, precision = 'high', auto=True):
        print(f'Otimizando torch:')
        # Aumentar cache do dynamo
        current_cache_limit = torch._dynamo.config.cache_size_limit
        if not auto or current_cache_limit < cache_limit:
            print(f'\t- Alterando cache do dynamo de {current_cache_limit} para {cache_limit}.')
            torch._dynamo.config.cache_size_limit = cache_limit
        else:
            print(f'\t- Cache do dynamo já está adequado ({current_cache_limit}).')
        
        # Performance com F32
        current_precision = torch.get_float32_matmul_precision()
        # Não alteramos se a precisão já for 'high' ou 'highest' (que é ainda melhor)
        if not auto or current_precision not in ('high', 'highest'):
            print(f"\t- Alterando precisão matmul de '{current_precision}' para '{precision}'.")
            torch.set_float32_matmul_precision(precision)
        else:
            print(f"\t- Precisão matmul já está otimizada ('{current_precision}').")
        
        torch.set_float32_matmul_precision(precision)

    def carregar_model_tokenizer(self, modelo:str, max_seq_length:int = 4096, cache_dir:str|None = None, usar_unsloth = False):
        global FASTMODEL, GETCHATTEMPLATE, AUTTOTOKENIZER, AUTOMODEL, AUTOMODELG, GENCONFIG
        ini=time()
        try:
            if usar_unsloth:
                if not FASTMODEL:
                    print('Importando unsloth ... ')
                    import unsloth
                    from unsloth import FastModel
                    from unsloth.chat_templates import get_chat_template
                    from transformers import GenerationConfig
                    FASTMODEL = FastModel
                    GETCHATTEMPLATE = get_chat_template
                    GENCONFIG = GenerationConfig
                model, tokenizer = FASTMODEL.from_pretrained(
                    model_name = modelo,
                    max_seq_length = max_seq_length, # Choose any for long context!
                    load_in_4bit = False,  # 4 bit quantization to reduce memory
                    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
                    full_finetuning = False, # [NEW!] We have full finetuning now!
                    device_map      = "auto",
                    cache_dir       = cache_dir, 
            )
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_kwargs = {"torch_dtype": 'auto', "device_map":'auto'}
                #if self._num_gpus > 1:
                #    print("Múltiplas GPUs detectadas. Usando device_map='auto'.")
                #    model_kwargs["device_map"] = "auto"                
                if (not AUTOMODELG) and self._tipo_modelo == 'gemma':
                    print('Importando tranformers para modelos Gemma ... ')
                    from transformers import Gemma3ForCausalLM
                    AUTOMODELG = Gemma3ForCausalLM
                elif (not AUTOMODEL) and self._tipo_modelo != 'gemma':
                    print('Importando tranformers ... ')
                    from transformers import AutoModelForCausalLM
                    AUTOMODEL = AutoModelForCausalLM
                if not AUTTOTOKENIZER:
                   from transformers import AutoTokenizer, GenerationConfig
                   AUTTOTOKENIZER = AutoTokenizer
                   GENCONFIG = GenerationConfig
                    
                tokenizer = AUTTOTOKENIZER.from_pretrained(modelo)
                if self._tipo_modelo == 'gemma':
                    if UtilLLM.gpu_tesla_t4():
                       model_kwargs.pop('torch_dtype')
                    model = AUTOMODELG.from_pretrained(modelo,**model_kwargs)
                else:
                    model = AUTOMODEL.from_pretrained(modelo,**model_kwargs)
                # Se não for multi-GPU, movemos o modelo para o device correto (cuda:0 ou cpu)
                #if self._num_gpus <= 1:
                #    model.to(self._device)
        except Exception as e:
            UtilLLM.controle_erros(e)
        self._model = model
        self._tokenizer = tokenizer
        print(f'Modelo carregado ({next(model.parameters()).device}): {time()-ini:.1f}s')

    def prompt(self, prompt:str, max_new_tokens:int = 4096, temperatura:float = 0.0, detalhar:bool = False, debug = False):
        if self._tipo_modelo == 'gemma':
            content = [{"type": "text", "text": prompt}]
        else:
            content = prompt
        messages = [{"role": "user", "content": content}, ]
        ini = time()
        inputs = self._tokenizer.apply_chat_template(
          messages,
          add_generation_prompt=True,
          tokenize=True,
          return_dict=True,
          return_tensors="pt",
        )
        #if self._num_gpus == 1:
        #   inputs = inputs.to(self._model.device)
        inputs = self._place_inputs(inputs)
        _temperatura = temperatura if isinstance(temperatura, (float, int)) else 0.2
        _temperatura = min(max(_temperatura,0),1)
        #print(f'########### temperatura: {_temperatura}')
        # configuração da predição
        gen_cfg = GENCONFIG.from_model_config(self._model.config)
        gen_cfg.max_new_tokens = 10
        gen_cfg.min_length = 1
        gen_cfg.temperature = _temperatura
        gen_cfg.top_k = 20 if _temperatura > 0.3 else 2
        gen_cfg.do_sample = bool(_temperatura > 0.3)
        # predição
        with torch.inference_mode():
             outputs = self._model.generate(**inputs, 
                                        max_new_tokens=max_new_tokens,
                                        generation_config = gen_cfg)  

        if debug: print(f'Resposta gerada: {time()-ini:.1f}s')
        res = self._tokenizer.decode(outputs[0], skip_special_tokens=False)
        res, think = self._tratar_retorno(res, debug=debug)
        if not detalhar:
           return res
        _input_tokens = inputs["input_ids"].size(1)
        _output_tokens = outputs.size(1) - _input_tokens
        res = {'texto': res, 'input_tokens': _input_tokens, 'output_tokens': _output_tokens, 'time': time()-ini}
        if think:
           res['think'] = think
        return res

    _STRIP = "\n \t`"
    def _tratar_retorno(self, txt: str, debug = False) -> tuple:
        if debug:
           print('-'*30)
           print(f'RETORNO BRUTO: {txt}')
           print('-'*30)

        # Se os separadores não estiverem configurados, retorna o texto limpo.
        if not self._start_asst:
            return txt.strip(self._STRIP), ''

        # Encontra a última ocorrência do início da resposta do assistente.
        pos_inicio_assistente = txt.rfind(self._start_asst)
        if pos_inicio_assistente == -1:
            return txt.strip(), '' # Retorna se não encontrar o marcador
        # 1. marcador de system
        think_sys = ''
        if self._system:
            pos_inicio_sys = txt.find(self._system)
            if pos_inicio_sys != -1:
                pos_fim_sys = txt.find(self._end_system, pos_inicio_sys)
                if pos_fim_sys != -1:
                    # Extrai o conteúdo dentro das tags <...system > e </  >
                    inicio_conteudo_sys = pos_inicio_sys + len(self._system)
                    think_sys = txt[inicio_conteudo_sys:pos_fim_sys].strip(self._STRIP)
                    txt = txt[pos_fim_sys+len(self._end_system):]
        # 2. Marcador think
        think = ''
        pos_inicio_texto_final = 0
        # 3. Procura e extrai a seção <think> se ela existir para este modelo.
        if self._think:
            pos_inicio_think = txt.find(self._think)
            if pos_inicio_think != -1:
                pos_fim_think = txt.find(self._end_think, pos_inicio_think)
                if pos_fim_think != -1:
                    # Extrai o conteúdo dentro das tags <think> e </think>
                    inicio_conteudo_think = pos_inicio_think + len(self._think)
                    think = txt[inicio_conteudo_think:pos_fim_think].strip(self._STRIP)
                    # remove o think do meio
                    txt = txt[:pos_inicio_think] + txt[pos_fim_think+len(self._end_think):]
        # 4. Encontra o marcador de fim de turno (<|im_end|>, <end_of_turn>, etc.).
        if self._start_asst:
           pos_start = txt.find(self._start_asst) 
           if pos_start != -1:
              txt = txt[pos_start + len(self._start_asst):] 
        pos_fim = txt.find(self._end_turn)
        # 5. Extrai a resposta final.
        if pos_fim == -1:
            # Se não encontrar o marcador de fim, pega tudo
            pass
        else:
            # Caso contrário, pega o trecho entre o início e o marcador de fim.
            txt = txt[:pos_fim]
        # 6. Limpa espaços, quebras de linha e outros caracteres indesejados e retorna.
        if think_sys:
           think = f'{think_sys}\n{think}'.strip(self._STRIP)
        return txt.strip(self._STRIP), think

    def _configurar_separadores(self):
        nome_modelo = str(self.modelo).lower()
        self._start_user = '' # início da pergunta do usuário
        self._start_asst = '' # início da resposta do modelo
        self._think      = ''
        self._end_think  = ''
        self._system      = ''
        self._end_system  = ''
        self._tipo_modelo =''
        if 'gemma' in nome_modelo:
            self._tipo_modelo = 'gemma'
            self._start_user = '<start_of_turn>user'   # início da pergunta do usuário
            self._start_asst = '<start_of_turn>model'  # início da resposta do modelo
            self._end_turn   = '<end_of_turn>'          
        elif 'deepseek' in nome_modelo:
            self._tipo_modelo = 'deepseek'
            self._start_user = '<｜User｜>'      # início da pergunta do usuário
            self._start_asst = '<｜Assistant｜>' # início da resposta do modelo
            self._end_turn   = '<｜end▁of▁sentence｜>'          
            self._think      = '<think>'
            self._end_think  = '</think>'
        elif 'qwen' in nome_modelo or 'jurema' in nome_modelo:
            self._tipo_modelo = 'qwen'
            self._start_user = '<|im_start|>user'      # início da pergunta do usuário
            self._start_asst = '<|im_start|>assistant' # início da resposta do modelo
            self._end_turn   = '<|im_end|>'           
        elif 'gpt' in nome_modelo and 'oss' in nome_modelo:
            self._tipo_modelo = 'gptoss'
            self._system = '<|start|>system<|message|>'
            self._end_system = '<|end|>'
            self._think = '<|start|>assistant<|channel|>analysis<|message|>'
            self._end_think = '<|end|>'
            self._start_user = '<|start|>user<|message|>'      # início da pergunta do usuário
            self._start_asst = '<|start|>assistant<|channel|>final<|message|>' # início da resposta do modelo
            self._end_turn   = '<|return|>'       
        else:
            print(f'Separados não configurados para o Modelo: {self.modelo}')

    def prompt_to_json(self, prompt:str, max_new_tokens = 4096, temperatura = 0, debug = False):
        retorno = self.prompt(prompt, 
                              max_new_tokens = max_new_tokens, 
                              temperatura=temperatura, 
                              detalhar = True, 
                              debug = debug)
        # converte o retorno da chave texto em json - se der erro, fica vazio
        texto = retorno.pop('texto',None)
        try:
            res = UtilLLM.mensagem_to_json(texto, padrao = None)
            if res is None:
                raise ValueError('Response não é um json válido!')
        except (json.JSONDecodeError, ValueError) as e:
            res = {'erro': f'JSONDecodeError: {e}', 'response': texto}
        res['usage'] = retorno
        return res

    def exemplo(self):
        _prompt_teste = '''Retorne um json válido com a estrutrua {"mensagem": com a mensagem do usuário, "itens": com uma lista de itens quando ele enumerar algo }
                            Mensagem do usuário: Eu preciso comprar abacaxi, pera e 2L de leite.'''
        print('-'*30)
        print(f'PROMPT:\n{_prompt_teste}')
        print('-'*30)
        r = self.prompt_to_json(_prompt_teste, debug=True)
        print(f'RESPOSTA TRATADA: prompt_to_json(prompt = "...."):')
        print('=-'*15)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        print('=-'*30)

    @classmethod
    def verifica_versao(cls, mostrar_gpus = True):
        UtilLLM.verifica_versao(mostrar_gpus)

    @classmethod
    def listar_modelos(cls):
        Modelos.listar()
        print('\n* Utilize: Prompt.modelos.MODELO....\n** Ou use os atalhos: 1, 2, 12, j7, d1.5 ... em UtilLLM.ATALHOS')

    @classproperty
    def modelos(cls):
        return Modelos     

    def _is_sharded(self):
        dmap = getattr(self._model, "hf_device_map", None)
        return bool(dmap) and len(dmap) > 1

    def _place_inputs(self, inputs):
        #if self._is_sharded():
        #    return inputs#.to("cpu") # deixa o acelerate decidir o device correto
        try:
            target = self._model.device
        except AttributeError:
            target = next(self._model.parameters()).device
        return inputs.to(target)     

##########################################
class UtilLLM():
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
        # limpa resposta ``json\n{ ... }`` ou pequenas introduções e finalizações antes/depois do json
        chave_ini = _mensagem.find('{')
        chave_fim = _mensagem.rfind('}')
        if len(_mensagem)>2 and chave_ini>=0 and chave_fim>=0 and chave_fim > chave_ini:
            _mensagem = _mensagem[chave_ini:chave_fim+1]
            #print(f'MENSAGEM FINAL: {_mensagem}')
            try:
                return json.loads(_mensagem)
            except Exception as e:
                if (not _corrigir_json_) or 'delimiter' not in str(e):
                    raise e
                # corrige aspas internas dentro do json
                return cls.mensagem_to_json(mensagem = cls.escape_json_string_literals(_mensagem), 
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

    ATALHOS = {
      # Gemma 3
        '1': Modelos.MODELO_GEMMA3_1B,        '4': Modelos.MODELO_GEMMA3_4B,
        '12': Modelos.MODELO_GEMMA3_12B,      '27': Modelos.MODELO_GEMMA3_27B,
      # Jurema 7B / Qwen
        'j': Modelos.MODELO_JUREMA_7B,        'j7': Modelos.MODELO_JUREMA_7B,
        'jurema': Modelos.MODELO_JUREMA_7B,
        'q': Modelos.MODELO_QWEN_7B,          'q7': Modelos.MODELO_QWEN_7B,
        'q14': Modelos.MODELO_QWEN_14B,       
      # DeepSeek
        'd1.5': Modelos.MODELO_DEEPSEEK_1_5B, 'd70': Modelos.MODELO_DEEPSEEK_70B,
        'd32': Modelos.MODELO_DEEPSEEK_32B,   'd14': Modelos.MODELO_DEEPSEEK_14B,
        'd8': Modelos.MODELO_DEEPSEEK_8B,     'd7': Modelos.MODELO_DEEPSEEK_7B,
        'd1': Modelos.MODELO_DEEPSEEK_1_5B,
      # gpt-oss
        'o20': Modelos.MODELO_OSS_20B,        'o120': Modelos.MODELO_OSS_120B,
      # Llama 3.2 3.3 4
        'l1': Modelos.MODELO_LLAMA_3_2_1B,    'l3': Modelos.MODELO_LLAMA_3_2_3B,
        'l70': Modelos.MODELO_LLAMA_3_3_70B,
        'l4s': Modelos.MODELO_LLAMA_4_SCOUT_16B,        }
    @classmethod
    def atalhos_modelos(cls, modelo):
        res = cls.ATALHOS.get(str(modelo).lower().strip(), modelo)      
        if res:
           return res
        return cls.ATALHOS.get(str(modelo).lower().strip(' bB'), modelo)  

    @classmethod
    def print_atalhos(cls):
        lista = list(cls.ATALHOS.items())
        lista.sort()
        print('Atalhos disponveis para os modelos:')
        for atalho, modelo in lista:
            print(f' - {atalho}: {modelo}')

    @classmethod
    def verifica_versao(cls, mostrar_gpus = True):
        global FASTMODEL, GETCHATTEMPLATE, AUTTOTOKENIZER, AUTOMODEL
        try:
            import unsloth
            ver_us=f'Unsloth version: {unsloth.__version__}'
        except:
            ver_us='Unsloth version: não disponível'
        try:
            import transformers
            ver_tr=f'Transformers version: {transformers.__version__}'
        except:
            ver_tr='Transformers version: não disponível'
        print('=' * 30)
        print(f'Torch version: {torch.__version__} | dynamo cache size: {torch._dynamo.config.cache_size_limit}')
        if torch._dynamo.config.cache_size_limit < 32:
           print(' - Considere aumentar o dynamo cache com entradas de tamanhos muito diferentes: torch._dynamo.config.cache_size_limit = 128')
        print(ver_tr)
        print(ver_us)
        print('=' * 30)
        print(f"Plataforma: {platform.system()} {platform.release()}")
        # Precisão para Multiplicação de Matrizes (a que você perguntou)
        _ft32 = torch.backends.cuda.matmul.allow_tf32
        print(f"1. Precisão Matmul: '{torch.get_float32_matmul_precision()}', permite TF32?: {_ft32}")
        if not _ft32:
           print(' - Considere ativar: torch.set_float32_matmul_precision("high")')
        # Flag similar para a biblioteca cuDNN (usada em convoluções, etc.)
        print(f"2. Backend cuDNN permite TF32?: {torch.backends.cudnn.allow_tf32}")
        # Tipo de dado padrão para novos tensores
        print(f"3. DType Padrão (torch.get_default_dtype): {torch.get_default_dtype()}")        
        print('=' * 30)
        if mostrar_gpus:
           cls.mostrar_info_gpus_pytorch()
    
    @classmethod
    def gpu_tesla_t4(cls):
        if not torch.cuda.is_available():
           return False
        nomes = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        nomes = ' '.join(nomes)
        return 'tesla t4' in nomes.lower()

    @classmethod
    def mostrar_info_gpus_pytorch(cls):
        """
        Verifica e exibe o número de GPUs disponíveis, seus nomes, 
        memória total e memória livre.
        """
        try:
            if not torch.cuda.is_available():
                print("CUDA não está disponível. O PyTorch não detectou nenhuma GPU compatível.")
                return
    
            num_gpus = torch.cuda.device_count()
            print(f"Número de GPUs disponíveis: {num_gpus}")
            
            print("-" * 65)
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                
                # torch.cuda.mem_get_info(i) retorna (memória livre, memória total) em bytes para a GPU 'i'
                free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(i)
                
                # Convertendo para Gigabytes (GB)
                total_memory_gb = total_memory_bytes / (1024**3)
                free_memory_gb = free_memory_bytes / (1024**3)
                used_memory_gb = total_memory_gb - free_memory_gb
                
                # Montando a string de saída formatada
                print(f"GPU ID: {i} | Modelo: {gpu_name}")
                print(f"  Memória -> Usada: {used_memory_gb:.2f} GB | Livre: {free_memory_gb:.2f} GB | Total: {total_memory_gb:.2f} GB")
                print("-" * 65)
                
        except Exception as e:
            print(f"Ocorreu um erro ao verificar as GPUs: {e}")
         

    @classmethod
    def controle_erros(cls, e: Exception):
        _msg = str(e).lower()
        _msg_full = str(traceback.format_exc()).lower()
        if 'transformers does not recognize this architecture' in _msg_full:
            msg = '\n'+\
                  '==========================================================================================================\n'+\
                  'Provavelmente a versão do transformers instalada não reconhece o modelo atual, utilize: get_git.deps(unsloth=False) !\n'+\
                  '=========================================================================================================='
            raise ImportError(msg)
        if 'gated repo' in _msg:
            msg = '\n'+\
                  '==========================================================================================================\n'+\
                  'É necessário criar uma variável de ambiente HF_TOKEN com o seu token do HuggingFace com esse modelo ativo!\n'+\
                  '=========================================================================================================='
            raise ImportError(msg)
        if isinstance(e, (ImportError,ModuleNotFoundError)):
            comp = 'get_git.deps(unsloth=True) # instala unsloth,'  if 'unsloth' in _msg else 'get_git.deps() # '
            print(f'''\n\nOCORREU UM ERRO DE IMPORT: {e}
# ===========================================
# = DICA DE PREPARAÇÃO DO AMBIENTE NO COLAB =
# ===========================================

# !curl https://raw.githubusercontent.com/luizanisio/llms/refs/heads/main/util/get_git.py -o ./get_git.py
# import get_git
# {comp} Transformers, Rouge, Levenshtein etc, o que for necessário.
#   ''')
            raise ImportError('dependências não resolvidas!')            

        raise e
