from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, List
import json
import re
'''
 Autor Luiz Anísio Julho/2025
 A solicitação do usuário é inserida dentro da tag <TAREFA>
 A base de conhecimento é inserida dentro da tag <BASE_CONHECIMENTO>
 - Cada item da base de conhecimento tem um titulo, o texto, a página e a url da informação para permitir ao usuário validar a informação.

  A iteração de solução da tarefa é feita dentro da tag <SOLUCAO_TAREFA> e vai ocorrer ao longo de algumas interações com a LLM.
  O prompt vai receber as tags <TAREFA>, <BASE_CONHECIMENTO>, <SOLUCAO_TAREFA>
'''

def verifica_versao():
    print(f'UtilAgentes carregado corretamente em {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}!')
  
@dataclass
class Conhecimento():
  ''' descrição do item da base de conhecimento e seus conteúdos'''
  titulo: str
  texto: str
  pagina: int | None = None
  url: str | None = None
  sugerido_llm: bool | None = False
  score: float = 0

@dataclass
class Servico():
  ''' descrição do serviço e parâmetros necessários '''
  nome: str
  objetivo: str
  quando_usar: str
  parametros: str | None # string definindo os parâmetros
  call_servico: Optional[Callable] # retorna list[Conhecimento]

@dataclass
class Tarefa():
  ''' descrição da tarefa e parâmetros necessários '''
  nome: str
  descricao: str
  objetivo: str
  call_llm: Optional[Callable] # retorna json no formato esperado para saída
  servicos_disponiveis: list[Servico] # lista de serviços existentes
  servicos_sugeridos: List[Dict] = field(default_factory=list) # serviços sugeridos pela llm para enriquecer a base de conhecimento
  conhecimento: list[Conhecimento] = field(default_factory=list) # conhecimento adicionado pela chamada dos serviços
  solucao: str = ''
  progresso: int = 0
  max_iter: int = 5
  max_try: int = 3
  concluida: bool = False

  def print(self, detalhar = False):
      print('=========================================')
      print(f'TAREFA: {self.nome}\nDESCRIÇÃO: {self.descricao}\nOBJETIVO: {self.objetivo}')
      print('---------------------------------------')
      if detalhar:
         print('CONHECIMENTO UTILIZADO:')
         for c in self.conhecimento:
             print(f'\t - {c.titulo} (pg: {c.pagina} | sc: {c.score}): {c.texto}')
         print('---------------------------------------')
      if self.concluida and isinstance(self.solucao,str) and self.solucao:
         print(f'SOLUÇÃO COMPLETA: {self.solucao}')
      elif isinstance(self.solucao,str) and self.solucao:
         print(f'SOLUÇÃO INCOMPLETA: {self.solucao}')
      else:
         print(f'TAREFA SEM SOLUÇÃO!')
      print('=========================================')

@dataclass
class Agente():
  ''' Se for um revisor, herda o prompt de REVISOR incluindo as diretrizes.
      Caso contrário, herda o prompt padrão com as diretrizes
  '''
  nome:str
  diretrizes:str
  revisor: bool | None

class ResolverTarefas():
  ''' Recebe uma lista de tarefas e retorna a solução de cada uma delas.
      A saída de uma tarefa é a entrada de outra.
      Parâmetros serão susbtituídos em descrição, observação e objetivo das tarefas e nas diretrizes dos agentes.
      Se não for definido um ou mais agentes, será incluído um padrão e um revisor
      Sempre será executada a busca de serviços se existirem serviços disponíveis
  '''
  def __init__(self, tarefas: list[Tarefa],
               parametros: dict|None = None,
               agentes: list[Agente] = [],
               debug = True):
    self.tarefas = tarefas
    self.parametros = parametros or {}
    self.agentes = agentes or []
    self.__logs = []
    self.debug = debug
    if not any(self.agentes):
       self.__inserir_agentes_basicos()

  def __inserir_agentes_basicos(self):
     self.agentes = [Agente(nome='Padrão', diretrizes='', revisor=False),
                     Agente(nome='Revisor', diretrizes='', revisor=True) ]

  def resolver(self):
      for tarefa in self.tarefas:
          self.__resolver_tarefa(tarefa)

  def __base_conhecimento(self, tarefa: Tarefa):
      retorno = ''
      itens = tarefa.conhecimento or []
      for item in itens:
          if retorno:
             retorno+= '-------------------------------------\n'
          pagina = f'<PAGINA>{item.pagina}</PAGINA>' if item.pagina else ''
          url = f'<URL>{item.url}</URL>' if item.url else ''
          local = f'{pagina} {url}'.strip() if pagina or url else ''
          retorno += f'# Título: {item.titulo} {local}\n{item.texto}\n'
      return retorno or 'Base de conhecimento sem informações disponíveis.'

  def __catalogo_servicos(self, tarefa: Tarefa):
      texto_servicos = ''
      for servico in tarefa.servicos_disponiveis or []:
          texto_servicos += f'## Serviço "{servico.nome}"\n'
          texto_servicos += f'- objetivo: {servico.objetivo}\n'
          texto_servicos += f'- quando_usar: {servico.quando_usar}\n'
          texto_servicos += f'- parametros: {servico.parametros}\n\n'
      if texto_servicos:
          texto_servicos = f'## Serviços disponíveis:\n{texto_servicos}'
      return texto_servicos

  def __substituicoes(self, prompt:str, tarefa:Tarefa):
          # substituição de parâmetros no objetivo e na descrição da tarefa
          tarefa_desc = tarefa.descricao or ''
          tarefa_obj = tarefa.objetivo or ''
          tarefa_sol = tarefa.solucao or ''
          if isinstance(self.parametros, dict) and any(self.parametros):
             for k, v in self.parametros.items():
                tarefa_desc = tarefa_desc.replace(f'{{{k}}}', str(v))
                tarefa_obj = tarefa_obj.replace(f'{{{k}}}', str(v))
                tarefa_sol = tarefa_sol.replace(f'{{{k}}}', str(v))
          prompt = prompt.replace('<<--tarefa-->>', tarefa_desc)
          prompt = prompt.replace('<<--objetivo-->>', tarefa_obj)
          prompt = prompt.replace('<<--base_conhecimento-->>', self.__base_conhecimento(tarefa))
          prompt = prompt.replace('<<--solucao_tarefa-->>', tarefa_sol)
          prompt = prompt.replace('<<--diretrizes-->>', '') # devem ser substituídas nos agentes se existirem
          if '<<--servicos-->>' in prompt:
             prompt = prompt.replace('<<--servicos-->>', self.__catalogo_servicos(tarefa))
          return prompt

  def __acionar_agente(self, tarefa:Tarefa, agente:Agente):
      ''' aciona o agente para gerar a resposta para a tarefa'''
      if tarefa.call_llm is None:
         raise Exception('tarefa.call_llm não foi definido')
         return
      # estilo do agente
      if agente.revisor:
          prompt = AgentesPrompts.PROMPT_REVISOR
          self.__print_debug(f'ACIONANDO AGENTE REVISOR: {agente.nome}')
      else:
          prompt = AgentesPrompts.PROMP_TAREFA
          self.__print_debug(f'ACIONANDO AGENTE DE TAREFA: {agente.nome}')
      # diretrizes complementares
      if agente.diretrizes:
         diretrizes = f'''# Diretrizes complementares:\n{agente.diretrizes}\n\n**Atenção**, as diretrizes básicas e diretrizes complementares precisam ser seguidas com atenção na realização das tarefas. As diretrizes básicas se sobrepõem às diretrizes complementares em caso de haver contradição entre elas.\n'''
         prompt = prompt.replace('<<--diretrizes-->>', diretrizes)
      else:
        prompt = prompt.replace('<<--diretrizes-->>', '')
      # campos da tarefa
      prompt = self.__substituicoes(prompt, tarefa)
      # acionando a llm
      res = tarefa.call_llm(prompt)
      if self.debug:
         self.__print_debug('AVALIANDO RESPOSTA LLM:\n' + json.dumps(res, indent=2, ensure_ascii=False), destaque=True)
      # avaliando o retorno
      if res.get('solucao_tarefa'):
          tarefa.solucao = res['solucao_tarefa']
          if res.get('contribuicao'):
             self.__log('CONTRIBUIÇÃO', f"Agente {agente.nome}: {res['contribuicao']}")
             tarefa.progresso += 1
          if isinstance(res.get('concluida'), bool) and res.get('concluida'):
             tarefa.concluida = True
             if agente.revisor:
                self.__log('TAREFA REVISADA', f"Agente {agente.nome}")
             else:
                self.__log('TAREFA CONCLUÍDA', f"Agente {agente.nome}")
      else:
          self.__log('AGENTE DORMIU NO PONTO', f"Agente {agente.nome} não fez nada")

  def __descobrir_servicos(self, tarefa: Tarefa, forcado = False):
      ''' aciona magic tools caso self.servico seja None'''
      if tarefa.call_llm is None:
         raise Exception('tarefa.call_llm não foi definido')
         return
      # existem serviços disponíveis?
      if (tarefa.servicos_disponiveis is None) or not any(tarefa.servicos_disponiveis):
         tarefa.servicos_sugeridos = []
         self.__log('DESCOBERTA', f'Nenhum serviço para a tarefa {tarefa.nome}')
         return
      # existem serviços disponíveis, descobrir si algum ajuda a concluir a tarefa
      self.__log('DESCOBERTA', f'Descobrindo serviços para a tarefa {tarefa.nome}')
      res = tarefa.call_llm(self.__substituicoes(AgentesPrompts.PROMPT_TOOLS, tarefa))
      tarefa.servicos_sugeridos = res.get('servicos') or []
      qtd = len(tarefa.servicos_sugeridos)
      self.__log('DESCOBERTA CONCLUÍDA', f'Serviços descobertos para a tarefa {tarefa.nome}: {qtd}')
      if qtd > 0:
          for servico in tarefa.servicos_sugeridos:
              self.__log(f'SERVIÇO SUGERIDO', f'{servico["nome"]} - {servico["motivo"]}')

  def __acionar_servicos_sugeridos(self, tarefa: Tarefa):
      ''' aciona os serviços sugeridos pela llm para enriquecer a base de conhecimento'''
      if len(tarefa.servicos_sugeridos) == 0:
         return
      # limpa conhecimentos anteriores sugeridos por llm
      tarefa.conhecimento = [_ for _ in tarefa.conhecimento or [] if not _.sugerido_llm]
      # Para cada serviço sugerido, executa e adiciona o resultado à base de conhecimento
      tarefa.servicos_sugeridos = tarefa.servicos_sugeridos if isinstance(tarefa.servicos_sugeridos,(list, tuple)) else []
      self.__log('ACIONANDO SERVIÇOS', ','.join([_['nome'] for _ in tarefa.servicos_sugeridos]))
      for svc_llm in tarefa.servicos_sugeridos or []:
          nome = svc_llm.get('nome')
          if not nome:
             continue
          pars = svc_llm.get('pars',{})
          try:
              _pars = json.dumps(pars, ensure_ascii=False)
          except Exception as e:
              self.__log('ERRO PARÂMETROS SERVIÇO', f'{nome}: {str(pars)[:30]} >>>> {e}')
              continue
          _pars = pars[:25] + ' .. ' + pars[-25:] if len(pars) > 60 else _pars
          self.__log('PREPARANDO SERVIÇO', f'{nome}: {_pars}')
          # Encontrar objeto Servico correspondente em servicos_disponiveis, si houver
          for svc in (tarefa.servicos_disponiveis or []):
              if svc.nome == nome and hasattr(svc, 'call_servico') and callable(svc.call_servico):
                  self.__log('SERVIÇO ACIONADO', f'{nome}: {_pars}')
                  try:
                      resultado = svc.call_servico(**pars)  # Executa o serviço
                  except Exception as e:
                      self.__log('ERRO ACIONANDO SERVIÇO', f'{nome}: {str(pars)[:30]} >>>> {e}')
                      continue
                  # Adiciona o texto retornado à base de conhecimento
                  for conhecimento in resultado:
                      if isinstance(conhecimento, Conhecimento):
                         self.__log('CONHECIMENTO', f'{conhecimento.titulo} | pg. {conhecimento.pagina} | score {conhecimento.score}')
                         conhecimento.sugerido_llm = True
                         tarefa.conhecimento.append(conhecimento)

  def __limpar_tags(self):
      for agente in self.agentes:
          agente.diretrizes = AgentesPrompts.limpar_tags(agente.diretrizes) if isinstance(agente.diretrizes, str) else ''
      for tarefa in self.tarefas:
          tarefa.descricao = AgentesPrompts.limpar_tags(tarefa.descricao)  if isinstance(tarefa.descricao, str) else ''
          tarefa.objetivo = AgentesPrompts.limpar_tags(tarefa.objetivo)  if isinstance(tarefa.objetivo, str) else ''

  def __resolver_tarefa(self, tarefa: Tarefa):
      ''' prepara e envia o prompt para a llm da tarefa para gerar a resposta
          se for para usar serviços, executa a verificação de serviços necessários
          Verifica se a tarefa foi concluída e repete até max_iter ser alcançado (se necessário).
      '''
      iter = 0
      servicos_acionados = False
      self.__log('RESOLVENDO TAREFA',f'{tarefa.nome} com {len(tarefa.servicos_disponiveis)} serviços disponíveis')
      self.__limpar_tags()
      while iter < tarefa.max_iter and not tarefa.concluida:
        iter += 1
        # aciona a verificação dos serviços
        if (not servicos_acionados) and isinstance(tarefa.servicos_disponiveis, (tuple,list)):
            self.__descobrir_servicos(tarefa)
            self.__acionar_servicos_sugeridos(tarefa)
            servicos_acionados = True
        # aciona os agentes e revisores
        concluida_por_revisor = False
        for agente in self.agentes:
            self.__acionar_agente(tarefa, agente)
            if tarefa.concluida and agente.revisor:
               break
        if tarefa.concluida:
           self.__log('TAREFA CONCLUÍDA', f'{tarefa.nome}')
        else:
           self.__log('TAREFA NÃO CONCLUÍDA', f'{tarefa.nome}')


  def __log(self, tipo:str, texto: str):
      _tipo = f'{tipo.upper()} - ' if tipo else ''
      txt_log = datetime.now().strftime("%H:%M:%S") + '|' + _tipo + ': ' + texto
      self.__logs.append(txt_log)
      self.__print_debug(txt_log)

  def __print_debug(self, texto:str, destaque=False):
      if destaque:
         txt_log = '\033[92m' + texto + '\033[0m'
      else:
         txt_log = texto
      if self.debug:
         print(txt_log)

##################################################
######## Tools básicos para os Agentes
class AgentesToolsBasicos():
      def __init__(self, textos_conhecimento: list[str],
                   min_score = 1) -> None:
          self.__textos_conhecimento = textos_conhecimento
          self.__textos_processados = self.processar_textos(self.__textos_conhecimento)
          self.__min_score = min_score

      def processar_textos(self, textos:list[str]):
          ''' processamento básico para simplificar
          '''
          res = []
          for texto in textos:
              res.append(self.processar_texto(texto))
          return res

      def processar_texto(self, texto):
          return str(texto).lower().strip()

      def adicionar_textos(self, textos):
          if isinstance(textos,str):
             textos = [textos]
          for texto in textos:
             self.__textos_conhecimento.append(texto)
             self.__textos_processados.append(self.processar_texto(texto))

      def encontrar(self, palavra:str, texto:str):
          if palavra.find(' ') < 0:
             return bool(palavra in texto)
          re_busca = palavra.strip().replace(' ','.{0,50}')
          return bool(re.search(re_busca, texto))

      def busca(self, palavras: list[str] = []):
          if isinstance(palavras, str):
             palavras = palavras.lower().replace(',',';').split(';')
             palavras = [self.processar_texto(palavra) for palavra in palavras ]
          if not isinstance(palavras, list):
              palavras = [str(palavras)]
          palavras_pre = [self.processar_texto(palavra) for palavra in palavras ]
          # print('Buscando:', palavras_pre, 'em', self.__textos_processados)
          conhecimentos = []
          dupla_dados = zip(self.__textos_conhecimento, self.__textos_processados)
          for i, (dado, dado_pre) in enumerate(dupla_dados):
              palavras_ok = 0
              for palavra in palavras_pre:
                  if self.encontrar(palavra,dado_pre):
                    palavras_ok +=1
              if palavras_ok < self.__min_score:
                 continue
              titulo = dado.split(':')[0] if ':' in dado else 'Informação'
              conhecimento = Conhecimento(titulo=titulo, 
                                          texto=dado, 
                                          pagina=i+1,
                                          score = palavras_ok/len(palavras_pre))
              conhecimentos.append(conhecimento)
          # ordena
          conhecimentos.sort(key=lambda x:x.score, reverse=True)
          return conhecimentos

      @classmethod
      def datahora(cls, **agrs):
          ''' no momento, qualquer parâmetro é ignorado '''
          data_resposta = datetime.now()
          # Retorna formato: "03 de agosto de 2025"
          res = data_resposta.strftime("%d de %B de %Y")
          return [Conhecimento(titulo= 'Calendário com dia, mês e ano',
                               texto= f'Hoje é o dia {res}')]

      @classmethod
      def get_servicos_basicos(cls, incluir_exemplos = False, textos_conhecimento = []):
          ''' retorna uma lista de serviços básicos incluindo busca textual
              sobre os textos envidados em "conhecimento" ou no formato base de conhecimento
          '''
          servicos = []
          if incluir_exemplos:
            bc = AgentesToolsExemplo(textos_conhecimento=textos_conhecimento)
            objetivo_base = 'localizar definições e formas de usar produtos e serviços'
          else:
            bc = AgentesToolsBasicos(textos_conhecimento=textos_conhecimento)
            objetivo_base = 'localizar definições, dicas, explicações e formas de usar produtos e serviços'

          sc = Servico(nome='Busca',
                      objetivo= objetivo_base,
                      quando_usar='quando é necessário fazer busca textual na base de conhecimento',
                      parametros='{"palavras": [lista de palavras simples ou compostas]}',
                      call_servico=bc.busca)
          servicos.append(sc)
          sc = Servico(nome='DataHora',
                          objetivo= 'obeter o dia, mês e ano atual',
                          quando_usar='quando é necessário saber a data de hoje, o dia de hoje, mês ou ano atual',
                          parametros='{}',
                          call_servico=bc.datahora)
          servicos.append(sc)
          return servicos


class AgentesToolsExemplo(AgentesToolsBasicos):
      ''' exemplo de pedido de teste:
            "Que dia é hoje e o que é Xibunfa e como podemos usar Xibunfa para limpar o chão?
      '''
      def __init__(self, textos_conhecimento: list[str] = []) -> None:
          super().__init__(textos_conhecimento=[])
          dados = ['Definiçaõ de Xabefa: uma comida típida do povo Xisbicuim e pode ser preparada com camarão e frutas frescas',
                   'Definição de Chão: a base onde pisamos, andamos e construímos nossas residências',
                   'Definiçaõ de Xibunfa: é um produto de limpeza a base de Xabefa',
                   'Como usar Xibunfa: pode ser usada misturando 30% de álcool 76% com 30% de Xibunfa e o resto com vinagre de bacuri']
          self.adicionar_textos(dados)

##################################################
######## Prompts para os Agentes

class AgentesPrompts():
      # tags que devem ser removidas dos textos e pedidos para evitar injeção de dados ou conflitos
      TAGS_REMOVER = ['TAREFA','OBJETIVO','BASE_CONHECIMENTO','SERVICOS','SOLUCAO_TAREFA','PAGINA','URL']

      @classmethod
      def limpar_tags(cls, texto):
          for remover in cls.TAGS_REMOVER:
              texto = texto.replace(f'<{remover}>',f' ')
          return texto

      PROMPT_TOOLS = ''' Você é um agente que identifica serviços úteis para enriquecer o contexto para solução de uma tarefa, seu objetivo é selecionar serviços que podem auxiliar uma LLM a gerar uma resposta como solução para a tarefa.
# Entrada:
- texto dentro da tag <TAREFA>: texto contendo a descrição da tarefa.
- texto dentro da tag <OBJETIVO>: descrição do objetivo que a ser atingido com a solução da tarefa.
- texto dentro da tag <SOLUCAO_TAREFA>: situação atual da solução da tarefa e que os serviços selecionados podem ajudar a atingir o objetivo.
- texto dentro da tag <BASE_CONHECIMENTO>: base de conhecimento já disponível para ajudar a atingir o objetivo.
- servicos dentro da tag <SERVICOS>: lista de serviços disponíveis com nome, objetivo, quando_usar e parâmetros.

# Sua tarefa:
1. Analise a tarefa, objetivo e solução atual da tarefa e selecione os serviços relevantes dentre os serviços disponíveis em <SERVICOS>.
2. Para cada serviço escolhido, preencha os parâmetros com os valores consistentes com o que o serviço espera, extraídos da solicitação do usuário.
3. Em motivo, explique brevemente por que cada serviço foi escolhido.

# Revisão:
- Revise cuidadosamente os parâmetros solicitados de cada serviço para nomeá-los corretamente e preencher os tipos de dados corretamente.

# Saída:
Retorne apenas um JSON válido na estrutura abaixo:

{
  "servicos": [
    {
      "nome": "<nome_do_servico>",
      "pars": { /* parâmetros nomeados */ },
      "motivo": "<razão para acionar o serviço>"
    },
    ...
  ]
}

# Exemplo:
{  "servicos": [
    { "nome": "data",
      "pars": {},
      "motivo": "O usuário perguntou que dia é hoje."
    },
    { "nome": "busca_textual",
      "pars": {
        "palavras": ["compras", "internet", "seguras"]
      },
      "motivo": "O usuário quer saber se compras feitas na internet são seguras."
    }
  ]
}

<TAREFA>
<<--tarefa-->>
</TAREFA>

<OBJETIVO>
<<--objetivo-->>
</OBJETIVO>

<BASE_CONHECIMENTO>
<<--base_conhecimento-->>
</BASE_CONHECIMENTO>

<SOLUCAO_TAREFA>
<<--solucao_tarefa-->>
</SOLUCAO_TAREFA>

<SERVICOS>
<<--servicos-->>
</SERVICOS>
'''

      PROMPT_REVISOR = '''
# Seu papel:
Você é um revisor de tarefas experiente. Seu objetivo é verificar se a solução atual descrita em <SOLUCAO_TAREFA> atinge o objetivo descrito em <OBJETIVO> para a tarefa em <TAREFA>. Use apenas as informações fornecidas (incluindo <BASE_CONHECIMENTO>) e **não invente nada** além do que está disponível.

# Diretrizes básicas
Verifique se a solução em <SOLUCAO_TAREFA> satisfaz completamente o objetivo da tarefa.
Se o objetivo foi atingido, retorne True na chave "concluida", indicando que a tarefa está concluída.
Caso contrário, indique False na chave "concluida" e escreva na chave "solucao_tarefa" uma breve descrição do que não pode ser resolvido em relação ao objetivo, de forma clara e objetiva, descrevendo com poucas palavras que tipo de informação precisaria estar disponível para concluir o objetivo.
Não cite as tags, refira-se à tag <BASE_CONHECIMENTO> como Base de Conhecimento;
<<--diretrizes-->>

# Resposta:
Sua resposta precisa ser um JSON válido com a seguinte estrutura:
- chave `"solucao_tarefa"`: texto final revisado da solução da tarefa (pode ser igual ao recebido, se não houver necessidade de alteração).
- chave "contribuicao": escreva uma frase curta e objetiva explicando o que foi feito para progredir na solução da tarefa
- chave `"concluida"`: valor booleano (`True` ou `False`) indicando se a tarefa foi concluída.

# Estrutura json da resposta:
{"solucao_tarefa": string,
 "contribuicao": string,
 "concluida": boolean
}

# Exemplos de respostas para a chave "solucao_tarefa":
  - Encontrei na Base de Conhecimento a seguinte informação: ....
  - Não encontrei informações sobre o assunto ... mas encontrei informações que podem ajudar como ....
  - Não encontrei informações na base de conhecimento para ajudar sobre ... 

<TAREFA>
<<--tarefa-->>
</TAREFA>

<OBJETIVO>
<<--objetivo-->>
</OBJETIVO>

<BASE_CONHECIMENTO>
<<--base_conhecimento-->>
</BASE_CONHECIMENTO>

<SOLUCAO_TAREFA>
<<--solucao_tarefa-->>
</SOLUCAO_TAREFA>
'''

      PROMPT_EXEMPLO ='Que dia é hoje? Busque informações sobre medicina e IA.'

      # prompt para execução da tarefa
      PROMP_TAREFA='''
# Seu papel:
Você é um especialista em solucionar tarefas com o apoio de dados complementares informados por serviços especializados que estão disponíveis na tag <BASE_CONHECIMENTO>.

# Diretrizes básicas:
Você precisa realizar a tarefa que está descrita na tag <TAREFA> com muita precisão para atingir o objetivo descrito na tag <OBJETIVO>, sem inventar nada, sem explicar se não for solicitado explicitamente, apenas usando dados disponíveis na tag <BASE_CONHECIMENTO> que contém informações retornadas pelos serviços especializados acionados anteriormente, na tag <SOLUCAO_TAREFA> que pode conter passos anteriores realizados para alcançar o objetivo da tarefa.
Você também pode usar informações das tags <TAREFA> e <OBJETIVO> para resolver o que foi solicitado em <TAREFA>
Ao usar informações da base de conhecimento, quando disponível junto com o título da informação, indique a página que está dentro da tag <PAGINA> e a url que está dentro da tag <URL> da informação, no formato (Fl. nnn, url: ... ).
Não cite as tags, refira-se à tag <BASE_CONHECIMENTO> como Base de Conhecimento;
<<--diretrizes-->

# Resposta
Sua resposta precisa ser um json válido com a seguinte estrutura:
- chave "solucao_tarefa": solução atualizada para a tarefa com a análise da tarefa e seu objetivo, consultando as informações da base de conhecimento e atualizando o conteúdo da tag <SOLUCAO_TAREFA>
- chave "contribuicao": escreva uma frase curta e objetiva explicando o que foi feito para progredir na solução da tarefa
- chave "concluida": recebe True se a contribuição fez com que o objetivo tenha sido alcançado e False se o objetivo ainda não foi alcançado

# Estrutura json da resposta:
 {"solucao_tarefa": string,
  "contribuicao": string,
  "concluida": boolean
  }

<TAREFA>
<<--tarefa-->>
</TAREFA>

<OBJETIVO>
<<--objetivo-->>
</OBJETIVO>

<BASE_CONHECIMENTO>
<<--base_conhecimento-->>
</BASE_CONHECIMENTO>

<SOLUCAO_TAREFA>
<<--solucao_tarefa-->>
</SOLUCAO_TAREFA>
'''

