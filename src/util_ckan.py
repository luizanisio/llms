# -*- coding: utf-8 -*-
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/agent-orchestration-2026

Descrição:
-----------
Utilitários CKAN para os notebooks do projeto JAMEX.
Centraliza listagem de recursos, download com cache, processamento de ZIPs
e leitura de JSONs de espelhos e íntegras de acórdãos do STJ.

Hierarquia de classes:
-----------------------
    UtilCkanBase     — infraestrutura compartilhada: filtros, cache, download,
                       mapa de íntegras e métodos utilitários.
    UtilCkan         — foco no espelho do acórdão, com possibilidade de trazer
                       a íntegra junto (cruzamento espelho × íntegra).
    UtilCkanIntegra  — foco exclusivo no dataset de íntegras e seus metadados,
                       sem dependência dos espelhos.

Arquitetura de Mapas:
---------------------
Cada classe mantém arquivos de índice (JSON) que mapeiam documentos
publicados no Portal de Dados Abertos do STJ.

A chave composta é:
    id_mapa = {numeroRegistro}.{YYYYMMDD}.{tipoDecisao}

Quando há duplicatas para um mesmo id_mapa, elas são registradas em
self.duplicados (dict) para análise posterior, sem interromper o processamento.

Como usar:
-----------
    from util_ckan import UtilCkan, UtilCkanIntegra

    # ── UtilCkan: espelhos (com opção de trazer íntegras) ──
    ckan = UtilCkan(
        anos   = {'2023', '2024'},
        orgaos = ['T5', 'T6', 'S3'],
    )
    ckan.atualizar_mapas()
    df = ckan.gerar_dataset_espelhos('../data/exemplo.parquet', incluir_integras=True)

    # ── UtilCkanIntegra: apenas íntegras e metadados ──
    integra = UtilCkanIntegra(
        anos = {'2024'},
    )
    integra.atualizar_mapas()
    df = integra.gerar_dataset_integras('../data/integras.parquet')

Parâmetros comuns do construtor (UtilCkanBase):
------------------------------------------------
    anos                      : set[str] | None  — anos de publicação (YYYY). None = todos.
    datas                     : set|list | None   — datas de publicação específicas.
                                Aceita vários formatos: 'YYYYMMDD', 'YYYY-MM-DD', 'DD/MM/YYYY'.
                                None = sem filtro por data específica (usa apenas ``anos`` se informado).
    classes                   : set[str] | None  — siglas de classes processuais. None = todas.
    registros                 : set[str] | None  — filtrar por numeroRegistro específico.
                                podem ser tuplas (registro, data_publicacao) ou (registro, data_publicacao, tipo_decisao)
    documentos                : set[int] | None  — filtrar por seq_documento_acordao específico.
    download_dir              : Path      — pasta raiz para cache (padrão: downloads_stj).
    timeout                   : int       — timeout HTTP em segundos (padrão: 600).
    atualizar_cache_e_mapas   : int|bool|None
                                    — controla quando os caches e mapas são atualizados:
                                      • None / False  → nunca baixa nada novo; usa exclusivamente o cache local.
                                      • True          → sempre baixa/atualiza; retenta recursos com erro.
                                      • int (minutos) → atualiza os mapas se o mais antigo tiver > N min;
                                                        recursos com erro (.info) são retentados individualmente
                                                        se o respectivo .info for mais antigo que N minutos.
                                                        Padrão: 12 * 60 (12 horas).
                                    Estratégia de cache por recurso (zip/json):
                                      • Arquivo existe com tamanho > 0  → usa cache (nunca re-baixa).
                                      • Arquivo ausente + .info presente → erro anterior; retenta conforme param.
                                      • Arquivo ausente + sem .info      → nunca tentado; baixa se permitido.
                                      • Falha no download                → cria/atualiza .info com detalhes.
                                    Independentemente, atualização é forçada se os arquivos de mapa não existirem.

Parâmetros adicionais de UtilCkan:
-----------------------------------
    colunas : list[str] | None — campos do espelho a importar. None = padrão.
    orgaos  : list[str] | None — siglas dos órgãos (ex: ['T5', 'S3']). None = todos.
"""

import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ─── Constantes de datasets ──────────────────────────────────────────────────

DATASETS_ESPELHOS_PADRAO = [
    ('CE', 'espelhos-de-acordaos-corte-especial'),
    ('S1', 'espelhos-de-acordaos-primeira-secao'),
    ('S2', 'espelhos-de-acordaos-segunda-secao'),
    ('S3', 'espelhos-de-acordaos-terceira-secao'),
    ('T1', 'espelhos-de-acordaos-primeira-turma'),
    ('T2', 'espelhos-de-acordaos-segunda-turma'),
    ('T3', 'espelhos-de-acordaos-terceira-turma'),
    ('T4', 'espelhos-de-acordaos-quarta-turma'),
    ('T5', 'espelhos-de-acordaos-quinta-turma'),
    ('T6', 'espelhos-de-acordaos-sexta-turma'),
]

DATASET_TEXTOS_PADRAO = 'integras-de-decisoes-terminativas-e-acordaos-do-diario-da-justica'

COLUNAS_ESPELHO_PADRAO = [
    'id', 'numeroRegistro', 'siglaClasse', 'descricaoClasse',
    'nomeOrgaoJulgador', 'ministroRelator', 'tipoDeDecisao',
    'dataPublicacao', 'dataDecisao',
    'teseJuridica', 'tema', 'referenciasLegislativas',
    'jurisprudenciaCitada', 'notas', 'termosAuxiliares',
    'informacoesComplementares', 'acordaosSimilares',
]

CKAN_BASE_URL = 'https://dadosabertos.web.stj.jus.br'


# ─── Funções utilitárias de datas ─────────────────────────────────────────────

def _extrair_data_pub_espelho(valor: str) -> str:
    """Extrai YYYYMMDD de string tipo 'DJE        DATA:01/06/2023'."""
    if not valor:
        return ''
    m = re.search(r'(\d{2})/(\d{2})/(\d{4})', valor)
    return f'{m.group(3)}{m.group(2)}{m.group(1)}' if m else ''



def _padronizar_data_filtro(valor: str) -> str:
    """Padroniza data contida nos filtros (registros) para formato YYYYMMDD."""
    if not valor: return ''
    v = str(valor).strip()
    
    # Remove parte de hora caso venha de um datetime/timestamp (ex: 'YYYY-MM-DD 00:00:00' ou 'YYYY-MM-DDTHH:MM:SS')
    v = v.split(' ')[0].split('T')[0]
    
    if re.match(r'^\d{8}$', v): return v
    
    # DD/MM/YYYY ou DD-MM-YYYY
    m = re.match(r'^(\d{2})[-/](\d{2})[-/](\d{4})$', v)
    if m: return f"{m.group(3)}{m.group(2)}{m.group(1)}"
    
    # YYYY-MM-DD ou YYYY/MM/DD
    m = re.match(r'^(\d{4})[-/](\d{2})[-/](\d{2})$', v)
    if m: return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    
    d = _extrair_data_pub_integra(v)
    if d: return d
    d = _extrair_data_pub_espelho(v)
    return d or v

def _extrair_data_pub_integra(valor) -> str:
    """Converte dataPublicacao de metadados de íntegra para YYYYMMDD.

    Aceita:
      - epoch-ms (int): 1685588400000 → 20230601
      - ISO string: '2024-02-08' → 20240208
    """
    if not valor:
        return ''
    # Tenta como string ISO (YYYY-MM-DD)
    if isinstance(valor, str):
        m = re.match(r'(\d{4})-(\d{2})-(\d{2})', valor)
        if m:
            return f'{m.group(1)}{m.group(2)}{m.group(3)}'
    # Tenta como epoch-ms
    try:
        ts = int(valor) / 1000
        return datetime.fromtimestamp(ts).strftime('%Y%m%d')
    except (ValueError, TypeError, OSError):
        return ''


def _gerar_id_mapa(num_registro: str, data_pub_yyyymmdd: str, tipo_decisao: str) -> str:
    """Gera a chave composta que liga espelhos a íntegras."""
    return f'{num_registro}.{data_pub_yyyymmdd}.{tipo_decisao.upper().strip()}'


# ══════════════════════════════════════════════════════════════════════════════
# UtilCkanBase — infraestrutura compartilhada
# ══════════════════════════════════════════════════════════════════════════════

class UtilCkanBase:
    """Classe base com infraestrutura compartilhada para acesso ao Portal de
    Dados Abertos do STJ via API CKAN.

    Responsabilidades:
      - Gerenciamento de filtros (anos, classes, registros, documentos).
      - Cache de downloads (com .info para falhas).
      - Mapa de íntegras (metadados + ZIPs).
      - Controle de duplicatas.
      - Métodos utilitários (leitura de JSON, normalização de texto, etc.).

    As subclasses UtilCkan e UtilCkanIntegra adicionam seus mapas específicos
    e métodos de alto nível.
    """

    def __init__(
        self,
        anos:    Optional[set[str]]    = None,
        datas:   Optional[set]         = None,
        classes: Optional[set[str]]    = None,
        registros: Optional[set] = None,
        documentos: Optional[set] = None,
        tipos_decisao: list[str] | set[str] | str | None = None,
        download_dir: Path              = Path('downloads_stj'),
        base_url:     str               = CKAN_BASE_URL,
        timeout:      int               = 600,
        atualizar_cache_e_mapas: bool | int | None = 12 * 60,
    ):
        """Inicializa a infraestrutura comum.

        Args:
            anos: Anos de interesse (ex: {'2023', '2024'}).
            datas: Datas de publicação específicas. Aceita vários formatos:
                'YYYYMMDD', 'YYYY-MM-DD', 'DD/MM/YYYY'. Ex: {'2023-06-01', '15/06/2023'}.
            classes: Classes de processos (ex: {'AI', 'RE'}).
            registros: Números de registro. Aceita strings ou tuplas
                (registro, data) ou (registro, data, tipo).
            documentos: Sequências de documentos (ex: {123456}).
            tipos_decisao: Tipo de decisão processual (ex: 'acordao').
            download_dir: Diretório raiz para cache.
            base_url: URL base do CKAN.
            timeout: Timeout HTTP em segundos.
            atualizar_cache_e_mapas: Controla atualização de cache e mapas.
                - None / False  → nunca baixa nada novo.
                - True          → sempre baixa/atualiza.
                - int (minutos) → atualiza se o mapa mais antigo tiver > N min.
        """
        if anos:
            self.anos = set()
            for a in anos:
                val = str(a).strip()
                if val.endswith('.0'): val = val[:-2]
                if val.isdigit():
                    self.anos.add(int(val))
        else:
            self.anos = None

        self.datas        = {_padronizar_data_filtro(d) for d in datas} if datas else None
        self.classes      = {c.upper() for c in classes} if classes else None
        
        # Filtros de registros aceitam string ou tuplas (reg, data) ou (reg, data, tipo)
        self.registros = set()
        if registros:
            for r in registros:
                if isinstance(r, str):
                    self.registros.add(r.strip())
                elif isinstance(r, (tuple, list)):
                    if len(r) == 2:
                        self.registros.add((str(r[0]).strip(), _padronizar_data_filtro(r[1])))
                    elif len(r) >= 3:
                        self.registros.add((str(r[0]).strip(), _padronizar_data_filtro(r[1]), str(r[2]).upper().strip()))

        self.documentos = {str(d).strip() for d in documentos} if documentos else None

        self.tipos_decisao = None
        if tipos_decisao:
            import unicodedata
            def _norm(s):
                if not s or (isinstance(s, float) and s != s): return ""
                return unicodedata.normalize('NFKD', str(s)).encode('ASCII', 'ignore').decode('utf-8').strip().lower()
            if isinstance(tipos_decisao, str):
                self.tipos_decisao = {_norm(tipos_decisao)}
            else:
                self.tipos_decisao = {_norm(td) for td in tipos_decisao}

        self.download_dir  = Path(download_dir)
        self.metadados_dir = self.download_dir / 'metadados_integras'
        self.integras_dir  = self.download_dir / 'integras'
        self.base_url     = base_url
        self.timeout      = timeout
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.metadados_dir.mkdir(parents=True, exist_ok=True)
        self.integras_dir.mkdir(parents=True, exist_ok=True)

        # ── Mapa de íntegras (compartilhado) ──
        self._caminho_mapa_integras = self.download_dir / 'mapa_integras.json'
        self._mapa_integras: dict[str, dict] = {}   # id_mapa → registro
        self.duplicados: dict[str, list[dict]] = {}  # id_mapa → lista de ocorrências

        # Preserva o valor original para decisões por recurso individual
        self._param_cache = atualizar_cache_e_mapas

    # ══════════════════════════════════════════════════════════════════════════
    # Filtros
    # ══════════════════════════════════════════════════════════════════════════

    def _passou_filtro_registro(self, num_reg: str, data_pub: str, tipo_decisao: str) -> bool:
        """Verifica se o registro satisfaz os filtros de tuplas de registros."""
        if not self.registros:
            return True
        num = num_reg.strip()
        data = data_pub.strip()
        tipo = tipo_decisao.upper().strip()
        return (
            (num in self.registros) or 
            ((num, data) in self.registros) or 
            ((num, data, tipo) in self.registros)
        )

    def _passou_filtro_ano(self, data_pub: str) -> bool:
        """Verifica se a data de publicação está entre os anos filtrados."""
        if not self.anos:
            return True
        return len(data_pub) >= 4 and data_pub[:4].isdigit() and int(data_pub[:4]) in self.anos

    def _passou_filtro_data(self, data_pub: str) -> bool:
        """Verifica se a data de publicação (YYYYMMDD) está entre as datas filtradas."""
        if not self.datas:
            return True
        return data_pub.strip() in self.datas

    def _passou_filtro_classe(self, sigla_classe: str) -> bool:
        """Verifica se a sigla da classe processual está entre as filtradas."""
        if not self.classes:
            return True
        sigla = sigla_classe.upper()
        return any(re.search(rf'\b{re.escape(c)}\b', sigla) for c in self.classes)

    def _passou_filtro_documento(self, seq_doc, id_mapa: str = '') -> bool:
        """Verifica se o seq_documento está entre os filtrados.
        
        Args:
            seq_doc: Sequência do documento a verificar.
            id_mapa: Chave do mapa (usada para buscar seq no mapa de íntegras).
        """
        if not self.documentos:
            return True
        if seq_doc and str(seq_doc).strip() in self.documentos:
            return True
        # Tenta buscar no mapa de íntegras
        if id_mapa:
            integra = self._mapa_integras.get(id_mapa, {})
            seq_int = str(integra.get('seq_documento', '')).strip()
            if seq_int in self.documentos:
                return True
        return False

    def _passou_filtro_tipo_decisao(self, tipo_decisao: str) -> bool:
        """Verifica se o tipo_decisao (normalizado) está entre os filtrados."""
        if not self.tipos_decisao:
            return True
        if not tipo_decisao:
            return False
        import unicodedata
        t = unicodedata.normalize('NFKD', str(tipo_decisao)).encode('ASCII', 'ignore').decode('utf-8').strip().lower()
        return t in self.tipos_decisao

    # ══════════════════════════════════════════════════════════════════════════
    # Duplicatas
    # ══════════════════════════════════════════════════════════════════════════

    def _registrar_duplicado(self, id_mapa: str, registro_novo: dict, registro_existente: dict, origem: str):
        """Registra uma duplicata detectada durante a indexação."""
        entrada = {
            'id_mapa': id_mapa,
            'origem': origem,
            'existente': registro_existente,
            'duplicado': registro_novo,
        }
        self.duplicados.setdefault(id_mapa, []).append(entrada)

    def obter_duplicados(self, filtro=None) -> dict[str, list[dict]]:
        """Retorna dicionário com os id_mapa que possuem duplicatas e suas ocorrências.

        Args:
            filtro: Opcional. Restringe o resultado aos id_mapa presentes no conjunto
                    informado. Aceita:
                      - set / list  de strings  (id_mapa)
                      - pandas DataFrame com coluna 'id_mapa'
                    Se None, retorna todos os duplicados.
        """
        if filtro is None:
            return self.duplicados
        if isinstance(filtro, pd.DataFrame):
            ids = set(filtro['id_mapa'].dropna()) if 'id_mapa' in filtro.columns else set()
        else:
            ids = set(filtro)
        return {k: v for k, v in self.duplicados.items() if k in ids}

    # ══════════════════════════════════════════════════════════════════════════
    # Mapa de íntegras — construção e persistência
    # ══════════════════════════════════════════════════════════════════════════

    def _carregar_mapa_integras(self):
        """Carrega mapa de íntegras do disco, se existir."""
        if self._caminho_mapa_integras.is_file():
            dados = json.loads(self._caminho_mapa_integras.read_text('utf-8'))
            self._mapa_integras = dados.get('mapa', {})
            for dup in dados.get('duplicados', []):
                self.duplicados.setdefault(dup['id_mapa'], []).append(dup)
            print(f'  📋  Mapa íntegras carregado: {len(self._mapa_integras)} registros')

    def _salvar_mapa_integras(self):
        """Persiste mapa de íntegras no disco."""
        dups = [d for lst in self.duplicados.values() for d in lst if d.get('origem') == 'integra']
        payload = {
            'atualizado_em': datetime.now().isoformat(),
            'total': len(self._mapa_integras),
            'duplicados': dups,
            'mapa': self._mapa_integras,
        }
        self._caminho_mapa_integras.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8'
        )

    def _atualizar_mapa_integras(self):
        """Indexa íntegras usando os JSONs de metadados publicados no CKAN.

        Os JSONs de metadados são recursos separados no dataset de íntegras
        (ex: metadados20240110.json). Não é necessário abrir os ZIPs para indexar.
        Cada metadado contém: seqDocumento, numeroRegistro, dataPublicacao (epoch ms),
        tipoDocumento. Com esses campos, geramos o id_mapa e mapeamos o ZIP/TXT.
        """
        from tqdm.auto import tqdm

        # Arquivos de metadados já indexados
        arquivos_indexados = {
            v.get('arquivo_metadados') for v in self._mapa_integras.values()
            if v.get('arquivo_metadados')
        }

        # Lista JSONs de metadados já em cache local
        jsons_locais = sorted(self.metadados_dir.glob('*.json'))
        novos_locais = [j for j in jsons_locais if j.name not in arquivos_indexados]

        # Tenta listar recursos no CKAN para baixar novos metadados
        recursos_meta = self._listar_recursos_metadados()
        if recursos_meta:
            nomes_locais = {j.name for j in jsons_locais}
            para_baixar = [r for r in recursos_meta if r['name'] not in nomes_locais]
            if para_baixar:
                print(f'  🔄  Baixando {len(para_baixar)} JSON(s) de metadados de íntegras...')
                for r in tqdm(para_baixar, desc='Metadados íntegras'):
                    try:
                        self._baixar(r['url'], r['name'], self.metadados_dir, True)
                    except Exception as e:
                        print(f'  ⚠️  Erro ao baixar {r["name"]}: {e}')
                # Reavalia JSONs locais após download
                jsons_locais = sorted(self.metadados_dir.glob('*.json'))
                novos_locais = [j for j in jsons_locais if j.name not in arquivos_indexados]

        if not novos_locais:
            print(f'  📋  Mapa íntegras já atualizado ({len(self._mapa_integras)} registros)')
            return

        # Montamos lookup de ZIPs disponíveis em cache para verificar
        zips_cache = {z.stem: z.name for z in self.integras_dir.glob('*.zip')}

        print(f'  🔄  Indexando {len(novos_locais)} JSON(s) de metadados...')
        inseridos = 0
        for arq_meta in tqdm(novos_locais, desc='Indexando íntegras'):
            try:
                meta = json.loads(arq_meta.read_text('utf-8', errors='replace'))
                if isinstance(meta, dict):
                    meta = [meta]

                # Extrair data base do nome do arquivo para achar o ZIP
                nome_base = arq_meta.stem
                data_zip = re.sub(r'^metadados(?:Publicacao)?', '', nome_base)
                nome_zip = f'{data_zip}.zip' if data_zip else ''

                for item in meta:
                    num_reg  = str(item.get('numeroRegistro') or '').strip()
                    data_pub = _extrair_data_pub_integra(item.get('dataPublicacao'))
                    tipo_doc = str(item.get('tipoDocumento') or '').strip()
                    seq_doc  = item.get('seqDocumento') or item.get('SeqDocumento')
                    ministro = str(item.get('ministro') or item.get('NM_MINISTRO') or '').strip()

                    if not num_reg or not data_pub or not tipo_doc:
                        continue

                    id_mapa = _gerar_id_mapa(num_reg, data_pub, tipo_doc)
                    arquivo_txt = f'{data_pub}/{seq_doc}.txt'

                    registro = {
                        'id_mapa':             id_mapa,
                        'numero_registro':     num_reg,
                        'data_publicacao':     data_pub,
                        'tipo_decisao':        tipo_doc,
                        'seq_documento':       seq_doc,
                        'arquivo_integra':     nome_zip,
                        'arquivo_txt':         arquivo_txt,
                        'arquivo_metadados':   arq_meta.name,
                        'processo':            str(item.get('processo') or '').strip(),
                        'ministro':            ministro,
                    }

                    if id_mapa in self._mapa_integras:
                        self._registrar_duplicado(id_mapa, registro, self._mapa_integras[id_mapa], 'integra')
                    else:
                        self._mapa_integras[id_mapa] = registro
                        inseridos += 1
            except Exception as e:
                print(f'  ⚠️  Erro ao indexar {arq_meta.name}: {e}')

        print(f'  ✅  Mapa íntegras: +{inseridos} novos → {len(self._mapa_integras)} total')
        self._salvar_mapa_integras()

    # ══════════════════════════════════════════════════════════════════════════
    # Listagem e download — CKAN API e cache
    # ══════════════════════════════════════════════════════════════════════════

    def _listar_recursos(self, dataset_id: str) -> list[dict]:
        """Retorna a lista de recursos de um dataset CKAN."""
        url = f'{self.base_url}/api/3/action/package_show'
        r = requests.get(url, params={'id': dataset_id}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()['result'].get('resources', [])

    def _listar_recursos_metadados(self) -> list[dict]:
        """Lista os JSONs de metadados publicados no dataset de íntegras do CKAN."""
        recursos = []
        try:
            for r in self._listar_recursos(DATASET_TEXTOS_PADRAO):
                nome = r.get('name', '')
                fmt  = r.get('format', '').upper()
                if fmt == 'JSON' or nome.endswith('.json'):
                    if not nome.endswith('.json'):
                        nome = nome + '.json'
                    recursos.append({'name': nome, 'url': r['url']})
        except Exception as e:
            print(f'  ⚠️  Erro ao listar metadados de íntegras no CKAN: {e}')
        return recursos

    def listar_recursos_zip(
        self,
        dataset_id: str = DATASET_TEXTOS_PADRAO,
        anos: Optional[set[str]] = None,
    ) -> list[dict]:
        """Lista recursos ZIP de íntegras, filtrados por ano."""
        anos = anos if anos is not None else self.anos
        recursos: list[dict] = []
        try:
            for r in self._listar_recursos(dataset_id):
                nome = r.get('name', '')
                fmt  = r.get('format', '').upper()
                if not (nome.lower().endswith('.zip') or fmt == 'ZIP'):
                    continue
                if anos and not any(nome.startswith(a) for a in anos):
                    continue
                recursos.append({'name': nome, 'url': r['url']})
        except Exception as e:
            print(f'  ⚠️  Erro ao listar ZIPs: {e}')
        return recursos

    def baixar_integras(self):
        """Baixa todos os ZIPs de íntegras necessários (conforme filtros)."""
        from tqdm.auto import tqdm
        recursos = self.listar_recursos_zip()
        print(f'Baixando {len(recursos)} ZIP(s) de íntegras...')
        for r in tqdm(recursos, desc='ZIPs'):
            try:
                self._baixar(r['url'], r['name'], self.integras_dir, self._param_cache)
            except Exception as e:
                print(f'  ⚠️  {r["name"]}: {e}')

    def baixar_zip(self, recurso: dict) -> Path:
        """Baixa um ZIP de íntegras para o cache."""
        return self._baixar(recurso['url'], recurso['name'],
                            self.integras_dir, self._param_cache)

    def _deve_tentar_recurso(self, caminho_info: Path, param) -> bool:
        """Decide se um recurso com falha anterior (arquivo .info) deve ser retentado.

        Regras:
          - None / False        → False (nunca retenta).
          - True                → True  (sempre retenta).
          - int (minutos)       → True se o .info for mais antigo que ``param`` minutos.
        """
        if param is None or param is False:
            return False
        if param is True:
            return True
        try:
            minutos = int(param)
            if minutos <= 0:
                return False
            idade_min = (datetime.now().timestamp() - caminho_info.stat().st_mtime) / 60
            return idade_min >= minutos
        except (TypeError, ValueError, OSError):
            return False

    def _baixar(self, url: str, nome_arquivo: str, pasta: Path, param) -> Path:
        """Baixa o arquivo para ``pasta`` usando cache local com estratégia por recurso.

        Regras de cache:
          1. Arquivo existe com tamanho > 0 → usa o cache (nunca re-baixa).
          2. Arquivo ausente + .info presente → falha anterior;
             retenta somente se ``_deve_tentar_recurso`` autorizar.
          3. Arquivo ausente + sem .info → nunca tentado;
             baixa se ``param`` for truthy (True ou int > 0).
          4. Falha no download → cria/atualiza .info; remove arquivo parcial.
          5. Sucesso → remove .info se existir.
        """
        caminho      = Path(pasta) / nome_arquivo
        caminho_info = caminho.parent / (caminho.name + '.info')

        # 1. Cache válido
        if caminho.is_file() and caminho.stat().st_size > 0:
            print(f'  [cache] {nome_arquivo:<55}', end='\r', flush=True)
            return caminho

        # 2. Falha anterior registrada no .info
        if caminho_info.is_file():
            if not self._deve_tentar_recurso(caminho_info, param):
                try:
                    info = json.loads(caminho_info.read_text('utf-8'))
                    erro_anterior = info.get('erro', '?')
                    tentativa_em  = info.get('tentativa_em', '?')
                except Exception:
                    erro_anterior, tentativa_em = '?', '?'
                raise FileNotFoundError(
                    f'[erro anterior em {tentativa_em}] {nome_arquivo}: {erro_anterior}'
                )
            # Autorizado a retentar — prossegue para o download

        # 3. Nunca tentado: verifica se download é permitido
        elif not param:
            raise FileNotFoundError(
                f'[ignorado] {nome_arquivo} não está em cache e download está desabilitado'
            )

        # 4. Executa o download
        print(f'  [↓]     {nome_arquivo:<55}', end='\r', flush=True)
        try:
            with requests.get(url, stream=True, timeout=self.timeout) as resp:
                resp.raise_for_status()
                with open(caminho, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        except Exception as exc:
            # Registra a falha no .info e limpa arquivo parcial
            if caminho.is_file():
                try:
                    caminho.unlink()
                except OSError:
                    pass
            info_payload = {
                'url':          url,
                'tentativa_em': datetime.now().isoformat(),
                'erro':         str(exc),
            }
            caminho_info.write_text(
                json.dumps(info_payload, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            raise

        # 5. Sucesso — remove .info se existir
        if caminho_info.is_file():
            try:
                caminho_info.unlink()
            except OSError:
                pass
        return caminho

    # ══════════════════════════════════════════════════════════════════════════
    # Extração de íntegras a partir de ZIPs
    # ══════════════════════════════════════════════════════════════════════════

    def _extrair_integras_de_itens(self, itens: list[dict]) -> dict[str, str]:
        """Extrai textos integrais dos ZIPs para os itens informados.

        Args:
            itens: Lista de dicts com 'id_mapa', 'arquivo_integra' e 'arquivo_txt' / 'seq_documento'.

        Returns:
            dict: id_mapa → texto integral.
        """
        from tqdm.auto import tqdm
        integras: dict[str, str] = {}

        # Agrupa por arquivo ZIP para abrir cada ZIP apenas uma vez
        por_zip: dict[str, list[dict]] = {}
        for item in itens:
            arq = item.get('arquivo_integra', '')
            if arq:
                por_zip.setdefault(arq, []).append(item)

        for nome_zip, itens_zip in tqdm(por_zip.items(), desc='Extraindo íntegras'):
            caminho_zip = self.integras_dir / nome_zip
            if not caminho_zip.is_file():
                print(f'  ⚠️  ZIP não encontrado em cache: {nome_zip}')
                continue
            try:
                with zipfile.ZipFile(caminho_zip) as zf:
                    # Monta lookup de TXTs por stem (seq_documento) para busca flexível
                    txt_por_seq: dict[str, str] = {}
                    for entry in zf.namelist():
                        if entry.endswith('.txt'):
                            txt_por_seq[Path(entry).stem] = entry

                    for item in itens_zip:
                        seq = str(item.get('seq_documento', ''))
                        txt_path = item.get('arquivo_txt', '')
                        # Tenta: 1) caminho exato, 2) busca por seq no lookup
                        if txt_path and txt_path in zf.namelist():
                            integras[item['id_mapa']] = self._normalizar_texto(zf.read(txt_path).decode('utf-8', errors='replace'))
                        elif seq in txt_por_seq:
                            integras[item['id_mapa']] = self._normalizar_texto(zf.read(txt_por_seq[seq]).decode('utf-8', errors='replace'))
            except Exception as e:
                print(f'  ⚠️  Erro ao ler {nome_zip}: {e}')

        return integras

    # ══════════════════════════════════════════════════════════════════════════
    # Resolução de atualização (lógica para mapas em geral)
    # ══════════════════════════════════════════════════════════════════════════

    def _resolver_atualizacao_para_mapas(self, parametro, *caminhos_mapa: Path) -> bool:
        """Determina se os mapas e caches devem ser atualizados.

        Regras (aplicadas nesta ordem):
          1. Algum mapa ausente em disco → True (forçado).
          2. parametro is None or False  → False.
          3. parametro is True           → True.
          4. parametro é int (minutos)   → True se o mapa mais antigo tiver mais de
             ``parametro`` minutos; False caso contrário.
        """
        for cam in caminhos_mapa:
            if not cam.is_file():
                return True
        if parametro is None or parametro is False:
            return False
        if parametro is True:
            return True
        try:
            minutos = int(parametro)
            if minutos <= 0:
                return False
            mtime = min(cam.stat().st_mtime for cam in caminhos_mapa)
            idade_min = (datetime.now().timestamp() - mtime) / 60
            return idade_min >= minutos
        except (TypeError, ValueError):
            return False

    # ══════════════════════════════════════════════════════════════════════════
    # Métodos utilitários estáticos
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _ler_json(caminho: Path) -> list[dict]:
        """Lê JSON com tratamento de encoding. Retorna lista de dicts."""
        for enc in ('utf-8', 'latin-1'):
            try:
                with open(caminho, encoding=enc) as f:
                    dados = json.load(f)
                return dados if isinstance(dados, list) else [dados]
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError:
                conteudo = caminho.read_text(encoding='utf-8', errors='replace')
                if 'sem lançamentos' in conteudo.lower():
                    return []
                raise
        return []

    @staticmethod
    def _formatar_valor(v):
        """Normaliza strings estruturadas (campos curtos) e arredonda floats."""
        if isinstance(v, str):
            return v.replace('\n', ' ').replace('\r', ' ').strip()
        if isinstance(v, float):
            return round(v, 3)
        return v

    @staticmethod
    def _normalizar_texto(v) -> str:
        """Normaliza texto longo (ementa, decisão, íntegra): converte \\r em \\n preservando parágrafos."""
        if not v or not isinstance(v, str):
            return v or ''
        # substitui <br> por quebra de linha pois alguns documentos estão com <br> no lugar de \n
        return v.replace('<br>','\n').replace('\r\n', '\n').replace('\r', '\n').strip()

    @staticmethod
    def _imprimir_resumo(df, caminho_saida: Path, col_texto: str = 'integra'):
        """Imprime estatísticas do dataset gerado."""
        sep = '─' * 55
        total = len(df)
        print(sep)
        print(f'  📄  {caminho_saida}')
        print(f'      {caminho_saida.stat().st_size / 1024**2:.2f} MB | {len(df.columns)} colunas | {total} registros')
        print(sep)
        if col_texto in df.columns:
            com = df[col_texto].str.len().gt(0).sum()
            print(f'  Com {col_texto:<12}: {com:>6}  ({com/total*100:.1f}%)')
            print(f'  Sem {col_texto:<12}: {total - com:>6}  ({(total-com)/total*100:.1f}%)')
            print(sep)
        if 'data_publicacao_iso' in df.columns:
            df_tmp = df.copy()
            df_tmp['_ano'] = df['data_publicacao_iso'].str[:4]
            print('  📅  Por ano de publicação:')
            for ano, grp in df_tmp.groupby('_ano'):
                ct = grp[col_texto].str.len().gt(0).sum() if col_texto in grp.columns else 0
                lbl = f' | {col_texto}: {ct}'
                print(f'      {ano} → {len(grp):>5} registros{lbl}')
            print(sep)
        if 'siglaClasse' in df.columns:
            top = df['siglaClasse'].value_counts().head(10)
            print('  ⚖️  Top classes:')
            for cls, cnt in top.items():
                print(f'      {str(cls):<35} {cnt:>5}')
            print(sep)
        print('  ✅  Concluído!')

    @classmethod
    def _obter_trecho_texto(cls, texto: str, inicio=200, fim=100, quebras=None):
        """Retorna o texto inteiro caso seja menor que inicio+fim ou trecho inicial e final com [..] entre eles.

        Args:
            quebras: se diferente de None, substitui \\n pelo conteúdo de quebras.
        """
        if len(texto) <= inicio + fim:
            res = texto
        else:
            res = texto[:inicio] + ' [..] ' + texto[-fim:]
        return res if quebras is None else res.replace('\n', quebras)

    @classmethod
    def exibir_amostra(cls, df, n: int = 2, titulo: str = 'Amostra'):
        """Exibe até n registros do DataFrame de forma legível."""
        if df is None or df.empty:
            print('⚠️  Nenhum dado disponível.')
            return
        print(f'\n{titulo} ({len(df)} registros totais, exibindo {min(n, len(df))}):')        
        print('═' * 65)
        col_integra = 'integra' if 'integra' in df.columns else None
        col_ementa  = 'ementa'  if 'ementa'  in df.columns else None

        def _texto(v) -> str:
            if v is None:
                return ''
            s = str(v).replace('\r\n', '\n').replace('\r', '\n').strip()
            return s if s not in ('', 'None', 'nan') else ''

        tem_ementa   = df[col_ementa].fillna('').astype(str).str.strip().str.len().gt(0)   if col_ementa  else pd.Series(False, index=df.index)
        tem_integra_ = df[col_integra].fillna('').astype(str).str.strip().str.len().gt(0)  if col_integra else pd.Series(False, index=df.index)

        idx_completos = df.index[tem_ementa & tem_integra_]
        idx_parciais  = df.index[(tem_ementa | tem_integra_) & ~(tem_ementa & tem_integra_)]
        idx_vazios    = df.index[~tem_ementa & ~tem_integra_]

        idx_ordenados = idx_completos.tolist() + idx_parciais.tolist() + idx_vazios.tolist()
        amostra = df.loc[idx_ordenados[:n]]
        for i, row in amostra.iterrows():
            excluir = {'integra', 'ementa', 'decisao'}
            dados = [
                f'  {str(c).ljust(28)}: {v}'
                for c, v in row.items()
                if c not in excluir and str(v) not in ('', '[]', 'None', 'nan')
            ]
            dados.sort()
            [print(d) for d in dados]
            if col_ementa:
                txt = _texto(row.get(col_ementa))
                if txt:
                    print(f'  {"EMENTA:":12}: {cls._obter_trecho_texto(texto=txt, quebras=" // ")}')
            if col_integra:
                txt = _texto(row.get(col_integra))
                if txt:
                    print(f'  {"ÍNTEGRA:":12}: {cls._obter_trecho_texto(texto=txt, quebras=" // ")}')
            print('─' * 65)


# ══════════════════════════════════════════════════════════════════════════════
# UtilCkan — foco nos espelhos de acórdão (com opção de trazer íntegras)
# ══════════════════════════════════════════════════════════════════════════════

class UtilCkan(UtilCkanBase):
    """Acesso ao Portal de Dados Abertos do STJ — foco nos espelhos de acórdão.

    Herda de UtilCkanBase a infraestrutura de filtros, cache, download e
    mapa de íntegras. Adiciona:
      - Mapa de espelhos.
      - Cruzamento espelho × íntegra.
      - Geração de datasets combinados.

    Filtros configurados no construtor são aplicados automaticamente
    nos métodos de alto nível.
    """

    def __init__(
        self,
        anos:    Optional[set[str]]    = None,
        datas:   Optional[set]         = None,
        classes: Optional[set[str]]    = None,
        orgaos:  Optional[list[str]]    = None,
        registros: Optional[set] = None,
        documentos: Optional[set] = None,
        colunas: Optional[list[str]] = None,
        tipos_decisao: list[str] | set[str] | str | None = None,
        download_dir: Path              = Path('downloads_stj'),
        base_url:     str               = CKAN_BASE_URL,
        timeout:      int               = 600,
        atualizar_cache_e_mapas: bool | int | None = 12 * 60,
    ):
        """Inicializa o utilitário CKAN para espelhos.

        Args:
            anos: Anos de interesse (ex: {'2023', '2024'}).
            datas: Datas de publicação específicas (ex: {'2023-06-01', '15/06/2023'}).
            classes: Classes de processos (ex: {'AI', 'RE'}).
            orgaos: Siglas dos órgãos (ex: ['T1', 'T2']). None = todos.
            registros: Números de registro (ex: {'123456'}).
            documentos: Sequências de documentos (ex: {123456}).
            colunas: Colunas a serem extraídas dos espelhos. None = padrão.
            download_dir: Diretório raiz para cache.
            base_url: URL base do CKAN.
            timeout: Timeout HTTP em segundos.
            atualizar_cache_e_mapas: Controla atualização de cache e mapas.
        """
        super().__init__(
            anos=anos, datas=datas, classes=classes, registros=registros,
            documentos=documentos, tipos_decisao=tipos_decisao,
            download_dir=download_dir, base_url=base_url, timeout=timeout,
            atualizar_cache_e_mapas=atualizar_cache_e_mapas
        )

        self.colunas = colunas or COLUNAS_ESPELHO_PADRAO
        self.orgaos = self._validar_orgaos(orgaos)

        self.espelhos_dir = self.download_dir / 'espelhos'
        self.espelhos_dir.mkdir(parents=True, exist_ok=True)

        # ── Mapa de espelhos ──
        self._caminho_mapa_espelhos = self.download_dir / 'mapa_espelhos.json'
        self._mapa_espelhos: dict[str, dict] = {}   # id_mapa → registro

        # Carrega mapas existentes do disco
        self._carregar_mapas()

        # Determina a necessidade de atualização com base no parâmetro e na idade dos mapas
        self.atualizar_cache_e_mapas = self._resolver_atualizacao(atualizar_cache_e_mapas)

        if self.atualizar_cache_e_mapas:
           self.baixar_espelhos()
           self.atualizar_mapas()

    def _validar_orgaos(self, orgaos: Optional[list[str]]) -> list[tuple[str, str]]:
        """Valida e retorna os datasets dos órgãos."""
        if orgaos:
            siglas = {s.upper() for s in orgaos}
            validos = [(s, d) for s, d in DATASETS_ESPELHOS_PADRAO if s in siglas]
            nao_encontrados = siglas - {s for s, _ in validos}
            if nao_encontrados:
                print(f'  ⚠️  Órgãos não reconhecidos: {nao_encontrados}')
            return validos
        return DATASETS_ESPELHOS_PADRAO

    def _resolver_atualizacao(self, parametro) -> bool:
        """Determina se os mapas e caches devem ser atualizados (espelhos + íntegras)."""
        return self._resolver_atualizacao_para_mapas(
            parametro,
            self._caminho_mapa_espelhos,
            self._caminho_mapa_integras,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Mapa de espelhos — construção e consulta
    # ══════════════════════════════════════════════════════════════════════════

    def _carregar_mapas(self):
        """Carrega mapas de espelhos e íntegras do disco, se existirem."""
        if self._caminho_mapa_espelhos.is_file():
            dados = json.loads(self._caminho_mapa_espelhos.read_text('utf-8'))
            self._mapa_espelhos = dados.get('mapa', {})
            for dup in dados.get('duplicados', []):
                self.duplicados.setdefault(dup['id_mapa'], []).append(dup)
            print(f'  📋  Mapa espelhos carregado: {len(self._mapa_espelhos)} registros')
        self._carregar_mapa_integras()

    def _salvar_mapa_espelhos(self):
        """Persiste mapa de espelhos no disco."""
        dups = [d for lst in self.duplicados.values() for d in lst if d.get('origem') == 'espelho']
        payload = {
            'atualizado_em': datetime.now().isoformat(),
            'total': len(self._mapa_espelhos),
            'duplicados': dups,
            'mapa': self._mapa_espelhos,
        }
        self._caminho_mapa_espelhos.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8'
        )

    def atualizar_mapas(self, forcar: bool = False):
        """Atualiza os dois mapas (espelhos + íntegras).

        Por padrão, processa apenas arquivos que ainda não foram indexados.
        Se `forcar=True`, reconstrói ambos os mapas do zero.
        """
        if forcar:
            self._mapa_espelhos = {}
            self._mapa_integras = {}
            self.duplicados = {}
        self._atualizar_mapa_espelhos()
        self._atualizar_mapa_integras()
        n_dups = sum(len(v) for v in self.duplicados.values())
        if n_dups:
            print(f'  ⚠️  {n_dups} duplicata(s) detectada(s) em {len(self.duplicados)} id_mapa(s). '
                  f'Use ckan.obter_duplicados() para analisar.')

    def _atualizar_mapa_espelhos(self):
        """Indexa todos os JSONs de espelhos em cache."""
        from tqdm.auto import tqdm

        arquivos_indexados = {
            v.get('arquivo_espelho') for v in self._mapa_espelhos.values()
            if v.get('arquivo_espelho')
        }

        jsons = sorted(self.espelhos_dir.glob('*.json'))
        novos = [j for j in jsons if j.name not in arquivos_indexados]

        if not novos:
            print(f'  📋  Mapa espelhos já atualizado ({len(self._mapa_espelhos)} registros)')
            return

        print(f'  🔄  Indexando {len(novos)} arquivo(s) de espelhos...')
        inseridos = 0
        for arq in tqdm(novos, desc='Indexando espelhos'):
            orgao = arq.stem.split('_')[0] if '_' in arq.stem else ''
            for item in self._ler_json(arq):
                num_reg = str(item.get('numeroRegistro') or '').strip()
                data_pub_raw = str(item.get('dataPublicacao') or '')
                tipo_decisao = str(item.get('tipoDeDecisao') or '').strip()
                data_pub = _extrair_data_pub_espelho(data_pub_raw)

                if not num_reg or not data_pub or not tipo_decisao:
                    continue

                id_mapa = _gerar_id_mapa(num_reg, data_pub, tipo_decisao)
                id_espelho = str(item.get('id') or '').strip()

                registro = {
                    'id_mapa':          id_mapa,
                    'numero_registro':  num_reg,
                    'data_publicacao':  data_pub,
                    'tipo_decisao':     tipo_decisao,
                    'id_espelho':       id_espelho,
                    'orgao':            orgao,
                    'sigla_classe':     str(item.get('siglaClasse') or ''),
                    'arquivo_espelho':  arq.name,
                }

                if id_mapa in self._mapa_espelhos:
                    self._registrar_duplicado(id_mapa, registro, self._mapa_espelhos[id_mapa], 'espelho')
                else:
                    self._mapa_espelhos[id_mapa] = registro
                    inseridos += 1

        print(f'  ✅  Mapa espelhos: +{inseridos} novos → {len(self._mapa_espelhos)} total')
        self._salvar_mapa_espelhos()

    # ══════════════════════════════════════════════════════════════════════════
    # Consultas com base nos mapas (espelho como mapa principal)
    # ══════════════════════════════════════════════════════════════════════════

    def consultar_mapa(self, filtros: Optional[dict] = None) -> list[dict]:
        """Retorna registros do mapa de espelhos que satisfazem os filtros configurados.

        Combina os filtros do construtor (anos, orgaos, classes, registros,
        documentos) com filtros adicionais opcionais passados como dict.
        """
        resultados = []
        orgaos_siglas = {s for s, _ in self.orgaos}
        todas_siglas  = {s for s, _ in DATASETS_ESPELHOS_PADRAO}
        filtrar_orgao = orgaos_siglas != todas_siglas

        for id_mapa, reg in self._mapa_espelhos.items():
            if not self._passou_filtro_ano(reg.get('data_publicacao', '')):
                continue
            if not self._passou_filtro_data(reg.get('data_publicacao', '')):
                continue
            if filtrar_orgao and reg.get('orgao', '') not in orgaos_siglas:
                continue
            if not self._passou_filtro_classe(reg.get('sigla_classe', '')):
                continue
            if not self._passou_filtro_registro(
                str(reg.get('numero_registro', '')),
                str(reg.get('data_publicacao', '')),
                str(reg.get('tipo_decisao', '')),
            ):
                continue
            if not self._passou_filtro_tipo_decisao(reg.get('tipo_decisao', '')):
                continue
            if not self._passou_filtro_documento(
                reg.get('seq_documento', ''), id_mapa,
            ):
                continue
            # Filtros adicionais
            if filtros:
                skip = False
                for k, v in filtros.items():
                    if reg.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            resultados.append(reg)
        return resultados

    def cruzar_espelhos_integras(self) -> list[dict]:
        """Cruza mapa de espelhos com mapa de íntegras pelo id_mapa.

        Retorna lista de dicts com dados combinados (espelho + íntegra),
        já aplicando todos os filtros configurados na classe.
        """
        espelhos_filtrados = self.consultar_mapa()
        resultado = []
        for reg_espelho in espelhos_filtrados:
            id_mapa = reg_espelho['id_mapa']
            reg_integra = self._mapa_integras.get(id_mapa, {})
            combinado = {**reg_espelho}
            if reg_integra:
                combinado['seq_documento']   = reg_integra.get('seq_documento')
                combinado['arquivo_integra'] = reg_integra.get('arquivo_integra', '')
                combinado['arquivo_txt']     = reg_integra.get('arquivo_txt', '')
                combinado['tem_integra']     = True
            else:
                combinado['seq_documento']   = None
                combinado['arquivo_integra'] = ''
                combinado['arquivo_txt']     = ''
                combinado['tem_integra']     = False
            resultado.append(combinado)
        return resultado

    # ══════════════════════════════════════════════════════════════════════════
    # Métodos de alto nível
    # ══════════════════════════════════════════════════════════════════════════

    def obter_integras(self) -> dict[str, str]:
        """Obtém íntegras indexadas por id_mapa, usando o cruzamento dos mapas.

        Retorna dict: id_mapa → texto integral.
        """
        cruzados = self.cruzar_espelhos_integras()
        itens = [c for c in cruzados if c.get('tem_integra')]
        if not itens:
            print('⚠️  Nenhum item com íntegra encontrado nos mapas.')
            return {}

        integras = self._extrair_integras_de_itens(itens)
        print(f'Íntegras extraídas: {len(integras)} / {len(itens)}')
        return integras

    def obter_espelhos(self) -> dict[str, dict]:
        """Obtém dados completos dos espelhos, indexados por id_mapa.

        Aplica os filtros configurados, lê os JSONs necessários e retorna
        dict: id_mapa → {campos do espelho}.
        """
        items_filtrados = self.consultar_mapa()
        if not items_filtrados:
            print('⚠️  Nenhum espelho encontrado com os filtros informados.')
            return {}

        from tqdm.auto import tqdm
        espelhos: dict[str, dict] = {}

        por_arquivo: dict[str, list[dict]] = {}
        for item in items_filtrados:
            arq = item.get('arquivo_espelho', '')
            if arq:
                por_arquivo.setdefault(arq, []).append(item)

        ids_alvo = {item['id_mapa'] for item in items_filtrados}

        for nome_arq, itens_arq in tqdm(por_arquivo.items(), desc='Lendo espelhos'):
            caminho = self.espelhos_dir / nome_arq
            if not caminho.is_file():
                continue

            ids_neste_arquivo = {it['id_mapa'] for it in itens_arq}
            for item_json in self._ler_json(caminho):
                num_reg = str(item_json.get('numeroRegistro') or '').strip()
                data_pub = _extrair_data_pub_espelho(str(item_json.get('dataPublicacao') or ''))
                tipo = str(item_json.get('tipoDeDecisao') or '').strip()
                if not num_reg or not data_pub or not tipo:
                    continue
                id_mapa = _gerar_id_mapa(num_reg, data_pub, tipo)
                if id_mapa not in ids_neste_arquivo:
                    continue

                reg = {c: self._formatar_valor(item_json.get(c)) for c in self.colunas}
                reg['id_mapa'] = id_mapa
                espelhos[id_mapa] = reg

        print(f'Espelhos extraídos: {len(espelhos)} / {len(ids_alvo)}')
        return espelhos

    def gerar_dataset_espelhos(
        self,
        caminho_saida: Optional[str | Path] = None,
        incluir_integras: bool = False,
        incluir_ementas: bool = True,
        incluir_decisoes: bool = True,
    ):
        """Gera um DataFrame com espelhos + opcionalmente íntegras, usando os mapas.

        Aplica automaticamente os filtros configurados no construtor.
        Retorna o DataFrame gerado. Se caminho_saida for fornecido, salva o parquet.
        """
        from tqdm.auto import tqdm

        if caminho_saida:
            caminho_saida = Path(caminho_saida)
            caminho_saida.parent.mkdir(parents=True, exist_ok=True)

        # ── 1. Cruzar mapas e obter espelhos ──────────────────────────────────
        cruzados = self.cruzar_espelhos_integras()
        if not cruzados:
            print('⚠️  Nenhum registro encontrado com os filtros informados.')
            return None

        print(f'Registros cruzados (espelho × íntegra): {len(cruzados)}')

        por_arquivo: dict[str, list[dict]] = {}
        for item in cruzados:
            arq = item.get('arquivo_espelho', '')
            if arq:
                por_arquivo.setdefault(arq, []).append(item)

        registros: list[dict] = []
        ids_alvo = {c['id_mapa'] for c in cruzados}

        for nome_arq, itens in tqdm(por_arquivo.items(), desc='Espelhos'):
            caminho = self.espelhos_dir / nome_arq
            if not caminho.is_file():
                continue
            ids_neste = {it['id_mapa'] for it in itens}
            cruzados_map = {it['id_mapa']: it for it in itens}

            for item_json in self._ler_json(caminho):
                num_reg = str(item_json.get('numeroRegistro') or '').strip()
                data_pub_raw = str(item_json.get('dataPublicacao') or '')
                data_pub = _extrair_data_pub_espelho(data_pub_raw)
                tipo = str(item_json.get('tipoDeDecisao') or '').strip()
                if not num_reg or not data_pub or not tipo:
                    continue
                id_mapa = _gerar_id_mapa(num_reg, data_pub, tipo)
                if id_mapa not in ids_neste:
                    continue

                reg = {c: self._formatar_valor(item_json.get(c)) for c in self.colunas}
                if incluir_ementas:
                    reg['ementa'] = self._normalizar_texto(item_json.get('ementa'))
                if incluir_decisoes:
                    reg['decisao'] = self._normalizar_texto(item_json.get('decisao'))

                cruzado = cruzados_map[id_mapa]
                reg['id_mapa']                = id_mapa
                reg['orgao']                  = cruzado.get('orgao', '')
                reg['data_publicacao_iso']     = data_pub
                reg['seq_documento_acordao']   = cruzado.get('seq_documento')
                reg['tem_integra']            = cruzado.get('tem_integra', False)
                registros.append(reg)

        print(f'\nRegistros extraídos: {len(registros)}')
        if not registros:
            print('⚠️  Nenhum registro extraído.')
            return None

        df = pd.DataFrame(registros)
        df = df.drop_duplicates(subset=['id_mapa']).reset_index(drop=True)

        # ── 2. Íntegras (opcional) ────────────────────────────────────────────
        if incluir_integras:
            integras = self.obter_integras()
            df['integra'] = df['id_mapa'].map(integras).fillna('')

        # ── 3. Salvamento e resumo ────────────────────────────────────────────
        if caminho_saida:
            df.to_parquet(caminho_saida, index=False)
            self._imprimir_resumo(df, caminho_saida, col_texto='integra')
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # Listagem e download — espelhos
    # ══════════════════════════════════════════════════════════════════════════

    def listar_recursos_espelhos(
        self,
        datasets: Optional[list[tuple[str, str]]] = None,
        anos: Optional[set[str]] = None,
    ) -> list[dict]:
        """Lista recursos JSON de espelhos por órgão julgador."""
        ds   = datasets or self.orgaos
        anos = anos if anos is not None else self.anos
        recursos: list[dict] = []
        for orgao, dataset_id in ds:
            try:
                for r in self._listar_recursos(dataset_id):
                    nome = r.get('name', '')
                    fmt  = r.get('format', '').upper()
                    if not (nome.lower().endswith('.json') or fmt == 'JSON'):
                        continue
                    if anos:
                        m = re.match(r'^(\d{4})\d{4}\.json$', nome, re.IGNORECASE)
                        if not m or m.group(1) not in anos:
                            continue
                    recursos.append({
                        'orgao'      : orgao,
                        'name'       : f'{orgao}_{nome}',
                        'url'        : r['url'],
                        'resource_id': r.get('id', ''),
                    })
            except Exception as e:
                print(f'  ⚠️  Erro ao listar espelhos de {orgao}: {e}')
        return recursos

    def baixar_espelhos(self):
        """Baixa todos os espelhos necessários (conforme filtros)."""
        from tqdm.auto import tqdm
        recursos = self.listar_recursos_espelhos()
        print(f'Baixando {len(recursos)} arquivo(s) de espelhos...')
        for r in tqdm(recursos, desc='Espelhos'):
            try:
                self._baixar(r['url'], r['name'], self.espelhos_dir, self._param_cache)
            except Exception as e:
                print(f'  ⚠️  {r["name"]}: {e}')

    def baixar_espelho(self, recurso: dict) -> Path:
        """Baixa um recurso de espelho para o cache."""
        return self._baixar(recurso['url'], recurso['name'],
                            self.espelhos_dir, self._param_cache)

    # ══════════════════════════════════════════════════════════════════════════
    # Métricas (específicas para espelhos)
    # ══════════════════════════════════════════════════════════════════════════

    def exibir_metricas(self, df, caminho_saida=None):
        """Exibe métricas abrangentes do DataFrame resultante da extração.

        Args:
            df: DataFrame retornado por gerar_dataset_espelhos.
            caminho_saida: caminho do parquet salvo (opcional, exibe info de arquivo).
        """
        sep   = '─' * 55
        total = len(df)

        if caminho_saida:
            p = Path(caminho_saida)
            print(sep)
            print('  ✅  ARQUIVO GERADO')
            print(sep)
            print(f'  Arquivo   : {p}')
            if p.exists():
                print(f'  Tamanho   : {p.stat().st_size / 1024**2:.2f} MB')
            print(f'  Colunas   : {len(df.columns)}  →  {df.columns.tolist()}')

        com_integra  = df['integra'].str.len().gt(0).sum() if 'integra' in df.columns else 0
        sem_integra  = total - com_integra
        com_espelho  = df['id_mapa'].notna().sum() if 'id_mapa' in df.columns else total
        sem_espelho  = total - com_espelho
        dups         = self.obter_duplicados(df)
        n_dups_ids   = len(dups)
        n_dups_total = sum(len(v) for v in dups.values())

        print(sep)
        print('  📊  COBERTURA DOS DADOS')
        print(sep)
        print(f'  Total de registros        : {total:>6}')
        print(f'  Com texto da íntegra      : {com_integra:>6}  ({com_integra/total*100:.1f}%)')
        print(f'  Sem texto da íntegra      : {sem_integra:>6}  ({sem_integra/total*100:.1f}%)')
        print(f'  Com dados de espelho      : {com_espelho:>6}  ({com_espelho/total*100:.1f}%)')
        print(f'  Sem dados de espelho      : {sem_espelho:>6}  ({sem_espelho/total*100:.1f}%)')
        if n_dups_ids:
            print(f'  ⚠️  IDs com duplicatas    : {n_dups_ids:>6}  ({n_dups_total} ocorrência(s) — use ckan.obter_duplicados())')

        tamanhos = (
            df.loc[df['integra'].str.len() > 0, 'integra'].str.len()
            if 'integra' in df.columns
            else pd.Series(dtype=int)
        )
        print(sep)
        print('  📏  TAMANHO DOS TEXTOS INTEGRAIS (chars)')
        print(sep)
        if not tamanhos.empty:
            print(f'  Média                     : {tamanhos.mean():>10.0f}')
            print(f'  Mediana                   : {tamanhos.median():>10.0f}')
            print(f'  Mínimo                    : {tamanhos.min():>10.0f}')
            print(f'  Máximo                    : {tamanhos.max():>10.0f}')
            print(f'  Total de caracteres       : {tamanhos.sum():>10,.0f}')
        else:
            print('  (nenhum texto disponível)')

        data_col = next(
            (c for c in ('data_publicacao_iso', 'publicacao', 'ano') if c in df.columns),
            None,
        )
        if data_col:
            print(sep)
            print('  📅  REGISTROS POR ANO DE PUBLICAÇÃO')
            print(sep)
            tmp = df.copy()
            tmp['_ano'] = df[data_col].astype(str).str[:4]
            agr = tmp.groupby('_ano').agg(
                total=('_ano', 'count'),
                com_integra=('integra', lambda x: x.str.len().gt(0).sum())
                if 'integra' in df.columns
                else ('_ano', lambda x: 0),
            ).sort_index()
            for ano, row in agr.iterrows():
                pct = row['com_integra'] / row['total'] * 100 if row['total'] else 0
                print(f'  {ano}  →  {row["total"]:>5} registros | texto: {row["com_integra"]:>5} ({pct:.0f}%)')

        org_col = next(
            (c for c in ('nomeOrgaoJulgador', 'orgao') if c in df.columns),
            None,
        )
        if org_col:
            print(sep)
            print('  ⚖️   REGISTROS POR ÓRGÃO JULGADOR (top 10)')
            print(sep)
            por_orgao = (
                df[org_col]
                .fillna('(não informado)')
                .value_counts()
                .head(10)
            )
            for org, cnt in por_orgao.items():
                print(f'  {str(org):<35} : {cnt:>5}')

        print(sep)
        print('  ✅  Processamento concluído!')
        print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# UtilCkanIntegra — foco exclusivo nas íntegras e metadados
# ══════════════════════════════════════════════════════════════════════════════

class UtilCkanIntegra(UtilCkanBase):
    """Acesso ao Portal de Dados Abertos do STJ — foco exclusivo nas íntegras.

    Herda de UtilCkanBase a infraestrutura de filtros, cache, download e
    mapa de íntegras. Trabalha apenas com o dataset de íntegras e seus
    metadados, sem dependência dos espelhos.

    Como usar:
        integra = UtilCkanIntegra(
            anos = {'2024'},
            registros = {'202302829818'},
        )
        integra.atualizar_mapas()
        df = integra.gerar_dataset_integras('../data/integras.parquet')

        # Ou obter dict de íntegras:
        textos = integra.obter_integras()  # dict: id_mapa → texto
    """

    def __init__(
        self,
        anos:    Optional[set[str]]    = None,
        datas:   Optional[set]         = None,
        classes: Optional[set[str]]    = None,
        registros: Optional[set] = None,
        documentos: Optional[set] = None,
        tipos_decisao: list[str] | set[str] | str | None = None,
        download_dir: Path              = Path('downloads_stj'),
        base_url:     str               = CKAN_BASE_URL,
        timeout:      int               = 600,
        atualizar_cache_e_mapas: bool | int | None = 12 * 60,
    ):
        """Inicializa o utilitário CKAN para íntegras.

        Args:
            anos: Anos de interesse (ex: {'2023', '2024'}).
            datas: Datas de publicação específicas (ex: {'2023-06-01', '15/06/2023'}).
            classes: Classes de processos (ex: {'AI', 'RE'}).
            registros: Números de registro. Aceita strings ou tuplas.
            documentos: Sequências de documentos (ex: {123456}).
            download_dir: Diretório raiz para cache.
            base_url: URL base do CKAN.
            timeout: Timeout HTTP em segundos.
            atualizar_cache_e_mapas: Controla atualização de cache e mapas.
        """
        super().__init__(
            anos=anos, datas=datas, classes=classes, registros=registros,
            documentos=documentos, tipos_decisao=tipos_decisao, download_dir=download_dir,
            base_url=base_url, timeout=timeout,
            atualizar_cache_e_mapas=atualizar_cache_e_mapas,
        )

        # Carrega mapa de íntegras do disco
        self._carregar_mapa_integras()

        # Determina a necessidade de atualização
        self.atualizar_cache_e_mapas = self._resolver_atualizacao(atualizar_cache_e_mapas)

        if self.atualizar_cache_e_mapas:
            self.baixar_integras()
            self.atualizar_mapas()

    def _resolver_atualizacao(self, parametro) -> bool:
        """Determina se o mapa de íntegras deve ser atualizado."""
        return self._resolver_atualizacao_para_mapas(
            parametro,
            self._caminho_mapa_integras,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Mapas
    # ══════════════════════════════════════════════════════════════════════════

    def atualizar_mapas(self, forcar: bool = False):
        """Atualiza o mapa de íntegras.

        Por padrão, processa apenas arquivos que ainda não foram indexados.
        Se `forcar=True`, reconstrói o mapa do zero.
        """
        if forcar:
            self._mapa_integras = {}
            self.duplicados = {}
        self._atualizar_mapa_integras()
        n_dups = sum(len(v) for v in self.duplicados.values())
        if n_dups:
            print(f'  ⚠️  {n_dups} duplicata(s) detectada(s) em {len(self.duplicados)} id_mapa(s). '
                  f'Use integra.obter_duplicados() para analisar.')

    # ══════════════════════════════════════════════════════════════════════════
    # Consultas com base no mapa de íntegras
    # ══════════════════════════════════════════════════════════════════════════

    def consultar_mapa(self, filtros: Optional[dict] = None) -> list[dict]:
        """Retorna registros do mapa de íntegras que satisfazem os filtros configurados.

        Combina os filtros do construtor (anos, classes, registros, documentos)
        com filtros adicionais opcionais passados como dict.
        """
        resultados = []
        for id_mapa, reg in self._mapa_integras.items():
            if not self._passou_filtro_ano(reg.get('data_publicacao', '')):
                continue
            if not self._passou_filtro_data(reg.get('data_publicacao', '')):
                continue
            if not self._passou_filtro_registro(
                str(reg.get('numero_registro', '')),
                str(reg.get('data_publicacao', '')),
                str(reg.get('tipo_decisao', '')),
            ):
                continue
            if not self._passou_filtro_classe(reg.get('processo', '')):
                continue
            if not self._passou_filtro_tipo_decisao(reg.get('tipo_decisao', '')):
                continue
            if not self._passou_filtro_documento(reg.get('seq_documento', '')):
                continue
            # Filtros adicionais
            if filtros:
                skip = False
                for k, v in filtros.items():
                    if reg.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            resultados.append(reg)
        return resultados

    # ══════════════════════════════════════════════════════════════════════════
    # Métodos de alto nível
    # ══════════════════════════════════════════════════════════════════════════

    def obter_integras(self) -> dict[str, str]:
        """Obtém textos integrais indexados por id_mapa.

        Aplica os filtros configurados no construtor e retorna
        dict: id_mapa → texto integral.
        """
        itens = self.consultar_mapa()
        itens_com_zip = [i for i in itens if i.get('arquivo_integra')]
        if not itens_com_zip:
            print('⚠️  Nenhum item com íntegra encontrado no mapa.')
            return {}

        integras = self._extrair_integras_de_itens(itens_com_zip)
        print(f'Íntegras extraídas: {len(integras)} / {len(itens_com_zip)}')
        return integras

    def gerar_dataset_integras(
        self,
        caminho_saida: Optional[str | Path] = None,
        incluir_texto: bool = True,
    ):
        """Gera um DataFrame com metadados de íntegras + opcionalmente textos.

        Aplica automaticamente os filtros configurados no construtor.
        Retorna o DataFrame gerado. Se caminho_saida for fornecido, salva o parquet.

        Args:
            caminho_saida: Caminho para salvar o parquet. None = não salva.
            incluir_texto: Se True, extrai e inclui o texto integral dos ZIPs.
        """
        if caminho_saida:
            caminho_saida = Path(caminho_saida)
            caminho_saida.parent.mkdir(parents=True, exist_ok=True)

        # ── 1. Consultar mapa de íntegras ────────────────────────────────────
        itens = self.consultar_mapa()
        if not itens:
            print('⚠️  Nenhum registro encontrado com os filtros informados.')
            return None

        print(f'Registros no mapa de íntegras: {len(itens)}')

        # ── 2. Montar DataFrame base com metadados ───────────────────────────
        registros: list[dict] = []
        for item in itens:
            reg = {
                'id_mapa':             item.get('id_mapa', ''),
                'numero_registro':     item.get('numero_registro', ''),
                'data_publicacao':     item.get('data_publicacao', ''),
                'data_publicacao_iso': item.get('data_publicacao', ''),
                'tipo_decisao':        item.get('tipo_decisao', ''),
                'seq_documento':       item.get('seq_documento'),
                'processo':            item.get('processo', ''),
                'ministro':            item.get('ministro', ''),
                'arquivo_integra':     item.get('arquivo_integra', ''),
                'arquivo_metadados':   item.get('arquivo_metadados', ''),
            }
            registros.append(reg)

        df = pd.DataFrame(registros)
        df = df.drop_duplicates(subset=['id_mapa']).reset_index(drop=True)
        print(f'Registros únicos: {len(df)}')

        # ── 3. Textos integrais (opcional) ────────────────────────────────────
        if incluir_texto:
            integras = self.obter_integras()
            df['integra'] = df['id_mapa'].map(integras).fillna('')
            com = df['integra'].str.len().gt(0).sum()
            print(f'Com texto da íntegra: {com} / {len(df)}')

        # ── 4. Salvamento e resumo ────────────────────────────────────────────
        if caminho_saida:
            df.to_parquet(caminho_saida, index=False)
            self._imprimir_resumo(df, caminho_saida, col_texto='integra')
        return df

    def exibir_metricas(self, df, caminho_saida=None):
        """Exibe métricas abrangentes do DataFrame de íntegras.

        Args:
            df: DataFrame retornado por gerar_dataset_integras.
            caminho_saida: caminho do parquet salvo (opcional).
        """
        sep   = '─' * 55
        total = len(df)

        if caminho_saida:
            p = Path(caminho_saida)
            print(sep)
            print('  ✅  ARQUIVO GERADO')
            print(sep)
            print(f'  Arquivo   : {p}')
            if p.exists():
                print(f'  Tamanho   : {p.stat().st_size / 1024**2:.2f} MB')
            print(f'  Colunas   : {len(df.columns)}  →  {df.columns.tolist()}')

        com_integra  = df['integra'].str.len().gt(0).sum() if 'integra' in df.columns else 0
        sem_integra  = total - com_integra
        dups         = self.obter_duplicados(df)
        n_dups_ids   = len(dups)
        n_dups_total = sum(len(v) for v in dups.values())

        print(sep)
        print('  📊  COBERTURA DOS DADOS')
        print(sep)
        print(f'  Total de registros        : {total:>6}')
        print(f'  Com texto da íntegra      : {com_integra:>6}  ({com_integra/total*100:.1f}%)')
        print(f'  Sem texto da íntegra      : {sem_integra:>6}  ({sem_integra/total*100:.1f}%)')
        if n_dups_ids:
            print(f'  ⚠️  IDs com duplicatas    : {n_dups_ids:>6}  ({n_dups_total} ocorrência(s) — use integra.obter_duplicados())')

        tamanhos = (
            df.loc[df['integra'].str.len() > 0, 'integra'].str.len()
            if 'integra' in df.columns
            else pd.Series(dtype=int)
        )
        print(sep)
        print('  📏  TAMANHO DOS TEXTOS INTEGRAIS (chars)')
        print(sep)
        if not tamanhos.empty:
            print(f'  Média                     : {tamanhos.mean():>10.0f}')
            print(f'  Mediana                   : {tamanhos.median():>10.0f}')
            print(f'  Mínimo                    : {tamanhos.min():>10.0f}')
            print(f'  Máximo                    : {tamanhos.max():>10.0f}')
            print(f'  Total de caracteres       : {tamanhos.sum():>10,.0f}')
        else:
            print('  (nenhum texto disponível)')

        data_col = next(
            (c for c in ('data_publicacao_iso', 'data_publicacao') if c in df.columns),
            None,
        )
        if data_col:
            print(sep)
            print('  📅  REGISTROS POR ANO')
            print(sep)
            tmp = df.copy()
            tmp['_ano'] = df[data_col].astype(str).str[:4]
            for ano, grp in tmp.groupby('_ano'):
                ct = grp['integra'].str.len().gt(0).sum() if 'integra' in grp.columns else 0
                print(f'  {ano}  →  {len(grp):>5} registros | texto: {ct:>5} ({ct/len(grp)*100:.0f}%)')

        ministro_col = 'ministro' if 'ministro' in df.columns else None
        if ministro_col:
            print(sep)
            print('  ⚖️   REGISTROS POR MINISTRO (top 10)')
            print(sep)
            por_min = (
                df[ministro_col]
                .fillna('(não informado)')
                .replace('', '(não informado)')
                .value_counts()
                .head(10)
            )
            for min_nome, cnt in por_min.items():
                print(f'  {str(min_nome):<35} : {cnt:>5}')

        print(sep)
        print('  ✅  Processamento concluído!')
        print(sep)


##############################################################################
####### CLI E EXTRAÇÃO EM LOTE
##############################################################################

import os
import sys

try:
    import yaml
except ImportError:
    yaml = None

_YAML_EXEMPLO = """\
# ==========================================================================
# Configuração de Extração de Dados do CKAN
# Executar com: python util_ckan.py --config {nome_arquivo}
# ==========================================================================

# --- Configuração Geral ---
# download_dir: (Opcional) Caminho para salvar a cache e mapas. 
# Se não for absoluto, será resolvido a partir da pasta de execução.
# atualizar_cache_e_mapas: (Opcional) Tempo em minutos para retentar ou True para forçar. Padrão: 720 (12h)
config:
  download_dir: "./dados_stj/downloads"
  atualizar_cache_e_mapas: true

# --- Saída ---
# arquivo: Caminho onde será salvo o arquivo parquet consolidado final.
saida:
  arquivo: "./entrada_experimento.parquet"
  
  # Mapeia colunas diretamente geradas pelo processo do CKAN
  # Colunas disponíveis padrão CKAN: 
  #   id_mapa, ministro, orgao_julgador, tipo_decisao, ementa, decisao, integra, 
  #   data_publicacao, data_publicacao_iso, numero_registro, seq_documento, processo
  # Colunas auto-padronizadas pelo script:
  #   dt_publicacao, ano, sg_classe, num_registro, seq_documento_acordao
  # colunas_ckan:
  #   ministro: num_ministro
  #   dt_publicacao: dt_publicacao
  #   sg_classe: sg_classe
  #   integra: integra
  #   num_registro: num_registro
  #   seq_documento_acordao: seq_documento_acordao
  #   ano: ano
  
  # Preserva colunas (faz merge via left-join) advindas do arquivo_origem configurado nas extrações
  # colunas_origem:
  #   sg_ramo_direito: sg_ramo_direito
  #   pasta: pasta

# --- Sequência de Extrações ---
# Lista de extrações a serem executadas e combinadas.
# tipo: "espelhos" (espelho + opcionalmente integra) ou "integras" (só integra).
# incluir_integras, incluir_ementas, incluir_decisoes: apenas para tipo "espelhos".
# incluir_texto: apenas para tipo "integras".
extracoes:
  - tipo: "espelhos"
    incluir_integras: true
    incluir_ementas: true
    filtros:
      anos: ["2023", "2024"]
      orgaos: ["T5"]
      classes: ["HC", "RHC"]
      tipo_decisao: "acordao"
      
  - tipo: "integras"
    incluir_texto: true
    filtros:
      # Puxa dinamicamente as listas baseadas nos únicos de um arquivo
      arquivo_origem: "./dados/pecas_exportadas_textos.parquet"
      coluna_anos: "ano"
      coluna_documentos: "seq_documento_acordao"
    # colunas_fixas:
    #   - fold: 12
    #   - grupo: "meu grupo extra"
    #   - data: "$hoje"
"""

def carregar_config_ckan(yaml_path: str) -> dict:
    if not yaml:
        print("❌ O pacote 'pyyaml' não está instalado. Instale com: pip install pyyaml")
        sys.exit(1)
        
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Arquivo YAML não encontrado: '{yaml_path}'")

    with open(yaml_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    if not isinstance(config, dict):
        raise ValueError(f"YAML deve conter um dicionário, recebido: {type(config)}")

    # --- Config ---
    cfg = config.get("config", {}) or {}
    download_dir = cfg.get("download_dir", "")
    if download_dir:
        # Resolve path em relação à pasta de execução
        download_dir = os.path.abspath(download_dir)
    else:
        # Padrão na pasta de execução atual
        download_dir = os.path.abspath("downloads_stj")
    cfg["download_dir"] = download_dir
    config["config"] = cfg

    # --- Saída ---
    saida = config.get("saida", {}) or {}
    arquivo_saida = saida.get("arquivo", "")
    if not arquivo_saida:
        raise ValueError("saida.arquivo é obrigatório no YAML")
    saida["arquivo"] = os.path.abspath(arquivo_saida)
    config["saida"] = saida

    # --- Extrações ---
    extracoes = config.get("extracoes", [])
    if not extracoes or not isinstance(extracoes, list):
        raise ValueError("A chave 'extracoes' deve ser uma lista e não pode estar vazia.")
    
    return config

def exibir_view_ckan(yaml_path: str):
    try:
        config = carregar_config_ckan(yaml_path)
    except Exception as e:
        print(f"Erro ao carregar configuração: {e}")
        return

    arquivo_saida = config["saida"]["arquivo"]
    print(f"👁️  Visualizando arquivo de saída: {arquivo_saida}\n")
    if not os.path.isfile(arquivo_saida):
        print(f"⚠️  Arquivo '{arquivo_saida}' não existe. Rode a extração sem --view primeiro.")
        return

    try:
        import pandas as pd
        df = pd.read_parquet(arquivo_saida)
    except Exception as e:
        print(f"Erro ao ler '{arquivo_saida}': {e}")
        return

    print(f"📊 Total de registros: {len(df)}")
    print(f"📑 Colunas presentes: {list(df.columns)}")
    print("-" * 55)
    
    # Usa o UtilsCkanBase para exibir a amostra
    UtilCkanBase.exibir_amostra(df, n=3, titulo="Amostra do Arquivo Final")

def _aplicar_tipos_padrao_ckan(df):
    """Garante que colunas típicas do CKAN tenham o tipo correto."""
    if df is None or df.empty:
        return df
        
    import pandas as pd
    df_out = df.copy()
    
    # seq_documento deve ser Int64 (Integer que suporta Nulos no Pandas)
    for col in ['seq_documento', 'seq_documento_acordao']:
        if col in df_out.columns:
            try:
                df_out[col] = pd.to_numeric(df_out[col], errors='raise').astype('Int64')
            except Exception as e:
                raise ValueError(f"Erro ao converter '{col}' para Inteiro. Verifique o arquivo de origem. Detalhe: {e}")
                
    # Demais identificadores devem ser strings puras (evitar floats como '123.0')
    str_cols = ['id_mapa', 'numero_registro', 'processo', 'ministro', 'orgao_julgador', 'tipo_decisao']
    for col in str_cols:
        if col in df_out.columns:
            if pd.api.types.is_float_dtype(df_out[col]) or pd.api.types.is_numeric_dtype(df_out[col]):
                df_out[col] = pd.to_numeric(df_out[col], errors='coerce').astype('Int64').astype(str).replace('<NA>', '')
            else:
                df_out[col] = df_out[col].astype(str).replace('nan', '')
                
    # Datas (sempre strings)
    date_cols = ['data_publicacao', 'data_publicacao_iso', 'dt_publicacao']
    for col in date_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).str.strip().replace('nan', '')
            
    return df_out


def executar_ckan_batch(yaml_path: str):
    config = carregar_config_ckan(yaml_path)
    
    cfg = config["config"]
    saida = config["saida"]
    extracoes = config["extracoes"]
    
    download_dir = cfg["download_dir"]
    atualizar_cache = cfg.get("atualizar_cache_e_mapas", True)
    
    arquivo_saida = saida["arquivo"]
    
    import pandas as pd
    dfs_resultado = []
    referencia_tipos = {}
    
    print(f"🚀 Iniciando extração em lote (CKAN)")
    print(f"📂 Diretório de download: {download_dir}")
    print(f"💾 Arquivo de saída: {arquivo_saida}\n")
    
    for i, extracao in enumerate(extracoes, 1):
        tipo = str(extracao.get("tipo", "")).strip().lower()
        print(f"--- Extração {i}/{len(extracoes)}: Tipo '{tipo}' ---")
        
        filtros_brutos = extracao.get("filtros", {}) or {}
        
        # Filtros dinâmicos via arquivo parquet
        arq_origem = filtros_brutos.get("arquivo_origem", "")
        df_filtro = None
        if arq_origem:
            import os
            arq_origem = os.path.abspath(arq_origem)
            
            if os.path.isfile(arq_origem):
                print(f"   📥 Carregando filtros dinâmicos de: {arq_origem}")
                import pandas as pd
                df_filtro = pd.read_parquet(arq_origem)
                df_filtro = _aplicar_tipos_padrao_ckan(df_filtro)
                
                col_anos = filtros_brutos.get("coluna_anos")
                if col_anos and col_anos in df_filtro.columns:
                    vals = set(df_filtro[col_anos].dropna().astype(str).unique())
                    filtros_brutos["anos"] = filtros_brutos.get("anos", []) + list(vals)
                    print(f"      - {len(vals)} ano(s) recuperado(s)")
                    
                col_docs = filtros_brutos.get("coluna_documentos")
                if col_docs and col_docs in df_filtro.columns:
                    vals = set(df_filtro[col_docs].dropna().astype(str).unique())
                    filtros_brutos["documentos"] = filtros_brutos.get("documentos", []) + list(vals)
                    print(f"      - {len(vals)} seq_documento(s) recuperado(s)")
                    
                col_regs = filtros_brutos.get("coluna_registros")
                if col_regs and col_regs in df_filtro.columns:
                    vals = set(df_filtro[col_regs].dropna().astype(str).unique())
                    filtros_brutos["registros"] = filtros_brutos.get("registros", []) + list(vals)
                    print(f"      - {len(vals)} registro(s) recuperado(s)")
            else:
                print(f"   ⚠️  Arquivo de origem dinâmico não encontrado: {arq_origem}")

        # Converte anos/classes para set de strings
        anos = set(str(a) for a in filtros_brutos.get("anos", [])) if "anos" in filtros_brutos else None
        datas = set(str(d) for d in filtros_brutos.get("datas", [])) if "datas" in filtros_brutos else None
        classes = set(str(c) for c in filtros_brutos.get("classes", [])) if "classes" in filtros_brutos else None
        orgaos = [str(o) for o in filtros_brutos.get("orgaos", [])] if "orgaos" in filtros_brutos else None
        registros = set(str(r) for r in filtros_brutos.get("registros", [])) if "registros" in filtros_brutos else None
        documentos = set(str(d) for d in filtros_brutos.get("documentos", [])) if "documentos" in filtros_brutos else None
        tipo_decisao_filtro = filtros_brutos.get("tipo_decisao")
        
        df_parcial = None
        
        if tipo == "espelhos":
            ckan = UtilCkan(
                anos=anos, datas=datas, classes=classes, orgaos=orgaos,
                registros=registros, documentos=documentos, tipos_decisao=tipo_decisao_filtro,
                download_dir=Path(download_dir),
                atualizar_cache_e_mapas=atualizar_cache
            )
            df_parcial = ckan.gerar_dataset_espelhos(
                incluir_integras=extracao.get("incluir_integras", False),
                incluir_ementas=extracao.get("incluir_ementas", True),
                incluir_decisoes=extracao.get("incluir_decisoes", True)
            )
        elif tipo == "integras":
            integra = UtilCkanIntegra(
                anos=anos, datas=datas, classes=classes,
                registros=registros, documentos=documentos, tipos_decisao=tipo_decisao_filtro,
                download_dir=Path(download_dir),
                atualizar_cache_e_mapas=atualizar_cache
            )
            df_parcial = integra.gerar_dataset_integras(
                incluir_texto=extracao.get("incluir_texto", True)
            )
        else:
            print(f"⚠️  Tipo de extração desconhecido: '{tipo}'. Ignorando.")
            continue
            
        if df_parcial is not None and not df_parcial.empty:
            df_parcial = _aplicar_tipos_padrao_ckan(df_parcial)
            print(f"✔️  Extração {i} resultou em {len(df_parcial)} registros.")
            
            # --- Inserir Colunas Fixas ---
            col_fixas = extracao.get("colunas_fixas", [])
            if isinstance(col_fixas, list):
                for item in col_fixas:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if isinstance(v,str) and v.strip().lower() in ("$hoje", "$data"):
                                from datetime import date
                                v = date.today().strftime('%Y-%m-%d')
                            elif isinstance(v,str) and v.strip().lower() in ("$agora", "$now"):
                                from datetime import datetime
                                v = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            df_parcial[k] = v
                            
            # --- Fazer Merge com Arquivo Origem (se aplicável) ---
            if df_filtro is not None:
                col_docs = filtros_brutos.get("coluna_documentos")
                col_regs = filtros_brutos.get("coluna_registros")
                
                left_on, right_on = None, None
                if col_docs and col_docs in df_filtro.columns and 'seq_documento' in df_parcial.columns:
                    left_on, right_on = 'seq_documento', col_docs
                    # Já tipados corretamente pelo _aplicar_tipos_padrao_ckan
                elif col_regs and col_regs in df_filtro.columns and 'numero_registro' in df_parcial.columns:
                    left_on, right_on = 'numero_registro', col_regs
                    # Já tipados corretamente pelo _aplicar_tipos_padrao_ckan
                
                if left_on and right_on:
                    print(f"   🔗 Fazendo merge com origem (ON ckan.{left_on} = origem.{right_on})")
                    colunas_origem = saida.get("colunas_origem", {})
                    cols_trazer = []
                    if isinstance(colunas_origem, dict) and colunas_origem:
                        cols_trazer = [c for c in colunas_origem.keys() if c in df_filtro.columns]
                    
                    if cols_trazer:
                        cols_merge = list(set(cols_trazer + [right_on]))
                        # Faz left join e remove a coluna de junção extra (right_on) se ela tiver nome diferente para não poluir
                        df_parcial = pd.merge(df_parcial, df_filtro[cols_merge], how='left', left_on=left_on, right_on=right_on)
                        if right_on != left_on and right_on in df_parcial.columns and right_on not in cols_trazer:
                            df_parcial.drop(columns=[right_on], inplace=True)

            # --- Harmonização Dinâmica de Tipos Extras ---
            for col in df_parcial.columns:
                if col in referencia_tipos:
                    # Tenta converter para o tipo que o primeiro grupo definiu
                    try:
                        df_parcial[col] = df_parcial[col].astype(referencia_tipos[col])
                    except Exception as e:
                        print(f"⚠️  Aviso: Não foi possível harmonizar o tipo da coluna '{col}' ({df_parcial[col].dtype} -> {referencia_tipos[col]}).")
                else:
                    # Registra o tipo para os próximos grupos
                    referencia_tipos[col] = df_parcial[col].dtype

            dfs_resultado.append(df_parcial)
        else:
            print(f"⚠️  Extração {i} não retornou nenhum registro.")
            
    print("\n" + "=" * 55)
    if not dfs_resultado:
        print("❌ Nenhuma extração retornou dados. O arquivo parquet não será gerado.")
        return
        
    print(f"🔄 Concatenando {len(dfs_resultado)} dataframes...")
    df_final = pd.concat(dfs_resultado, ignore_index=True)
    
    if 'id_mapa' in df_final.columns:
        antes = len(df_final)
        df_final = df_final.drop_duplicates(subset=['id_mapa']).reset_index(drop=True)
        depois = len(df_final)
        if antes > depois:
            print(f"🧹 Duplicados removidos por id_mapa: {antes} -> {depois} registros únicos.")
            
    # === PADRONIZAÇÃO AUTOMÁTICA CKAN ===
    # Converte numero_registro para string
    if 'numero_registro' in df_final.columns:
        df_final['num_registro'] = df_final['numero_registro']

    # Cria seq_documento_acordao (mantém Int64)
    if 'seq_documento' in df_final.columns:
        df_final['seq_documento_acordao'] = df_final['seq_documento']

    # Extrai sg_classe de processo
    if 'processo' in df_final.columns:
        df_final['sg_classe'] = df_final['processo'].astype(str).str.split(' ').str[0].str.strip()

    # Formata datas (ano e dt_publicacao a partir de data_publicacao)
    if 'data_publicacao' in df_final.columns:
        dp_str = df_final['data_publicacao'].astype(str).str.strip()
        df_final['ano'] = pd.to_numeric(dp_str.str[:4], errors='coerce').astype('Int64')
        
        mask_8 = dp_str.str.len() == 8
        dt_pub = dp_str.copy()
        dt_pub.loc[mask_8] = dp_str.loc[mask_8].str[:4] + '-' + dp_str.loc[mask_8].str[4:6] + '-' + dp_str.loc[mask_8].str[6:8]
        df_final['dt_publicacao'] = dt_pub
    elif 'data_publicacao_iso' in df_final.columns and 'ano' not in df_final.columns:
        # Fallback de segurança se apenas data_publicacao_iso estiver disponível
        df_final['ano'] = df_final['data_publicacao_iso'].astype(str).str[:4]
    # ====================================
    # Aplica o mapeamento de colunas se configurado
    colunas_ckan = saida.get("colunas_ckan", {})
    colunas_origem = saida.get("colunas_origem", {})
    
    if colunas_ckan or colunas_origem:
        renomear_dict = {}
        cols_manter = []
        
        if isinstance(colunas_ckan, dict):
            for orig, dest in colunas_ckan.items():
                if orig in df_final.columns:
                    renomear_dict[orig] = dest
                    cols_manter.append(orig)
                    
        if isinstance(colunas_origem, dict):
            for orig, dest in colunas_origem.items():
                if orig in df_final.columns:
                    renomear_dict[orig] = dest
                    cols_manter.append(orig)
                    
        # Adiciona colunas fixas que foram definidas no YAML para não serem descartadas
        for extracao in extracoes:
            fixas = extracao.get("colunas_fixas", [])
            if isinstance(fixas, list):
                for f in fixas:
                    if isinstance(f, dict):
                        for k in f.keys():
                            if k in df_final.columns and k not in cols_manter:
                                cols_manter.append(k)
                                
        df_final = df_final[list(set(cols_manter))].rename(columns=renomear_dict)
        print(f"🔀 Colunas finais após mapeamento: {list(df_final.columns)}")
    
    # Garante que a pasta de destino exista
    pasta_saida = os.path.dirname(arquivo_saida)
    if pasta_saida:
        os.makedirs(pasta_saida, exist_ok=True)
        
    print(f"💾 Salvando arquivo consolidado: {arquivo_saida}...")
    df_final.to_parquet(arquivo_saida, index=False)
    
    # Exibe resumo
    UtilCkanBase._imprimir_resumo(df_final, Path(arquivo_saida))
    print("✅ Extração em lote concluída com sucesso.")


##############################################################################
####### EXEMPLOS
##############################################################################

class ExemplosCKan():

    @classmethod
    def exemplo1(cls, atualizar_cache_e_mapas):
        print('=== Exemplo 1: construir mapas + dataset Penal (2024) ===\n')

        ckan = UtilCkan(
            anos   = {'2024','2024'},
            orgaos = ['T5', 'T6', 'S3'],
            atualizar_cache_e_mapas = atualizar_cache_e_mapas,
        )

        # Mostra estatísticas dos mapas
        cruzados = ckan.cruzar_espelhos_integras()
        com_integra = sum(1 for c in cruzados if c['tem_integra'])
        print(f'\nCruzamento: {len(cruzados)} espelhos | {com_integra} com íntegra disponível')

        # Gera dataset
        df = ckan.gerar_dataset_espelhos(
            caminho_saida = Path('../data/exemplo.parquet'),
            incluir_integras = False,
            incluir_ementas = True,
            incluir_decisoes = False,
        )

        # Duplicados
        dups = ckan.obter_duplicados()
        if dups:
            print(f'\n⚠️  {len(dups)} id_mapa(s) com duplicatas')
            for id_mapa, ocorrencias in list(dups.items())[:3]:
                print(f'  {id_mapa}: {len(ocorrencias)} duplicata(s)')

        print('#' * 55)
        cls.print_df(df, ckan._mapa_espelhos, ckan._mapa_integras)


    @classmethod
    def exemplo2(cls, num_registro, atualizar_cache_e_mapas):
        print('=== Exemplo 2: construir mapas + dataset com um número de registro específico ===\n')
        num_registro = num_registro or '202201546162'
        ckan2 = UtilCkan(
            registros= {num_registro},
            atualizar_cache_e_mapas = atualizar_cache_e_mapas,
        )

        df = ckan2.gerar_dataset_espelhos(
            caminho_saida  = Path('../data/exemplo_ementas.parquet'),
            incluir_integras = True,
            incluir_ementas = True,
            incluir_decisoes = True,
        )
        cls.print_df(df, ckan2._mapa_espelhos, ckan2._mapa_integras)

    @classmethod
    def exemplo3_integra(cls, num_registro, atualizar_cache_e_mapas):
        """Exemplo usando UtilCkanIntegra: foco exclusivo nas íntegras."""
        print('=== Exemplo 3: UtilCkanIntegra — íntegras sem espelhos ===\n')
        num_registro = num_registro or '202302829818'
        integra = UtilCkanIntegra(
            registros = {num_registro},
            atualizar_cache_e_mapas = atualizar_cache_e_mapas,
        )

        df = integra.gerar_dataset_integras(
            caminho_saida = Path('../data/exemplo_integras.parquet'),
            incluir_texto = True,
        )
        if df is not None and not df.empty:
            print(f'\nDataset gerado: {len(df)} registros')
            print(df.head())
            if 'integra' in df.columns:
                item = df.iloc[0]
                txt = str(item.get('integra', ''))
                print(f'\n>>> ÍNTEGRA: {txt[:200]} [...] {txt[-100:]}')
        else:
            print('Nenhum registro encontrado.')

    @classmethod
    def print_df(cls, df, mapa_espelhos, mapa_integras):
        if df is None:
            print('Nenhum df informado')
            return
        if len(df) == 0:
            print('Nenhum registro encontrado')
            return
        print(df.head())
        item = df.iloc[0]
        print('-' * 55)
        print(f'Exemplo de ementa ({item.get("id_mapa", "")}):')
        print(f'>>> EMENTA: {str(item["ementa"])[:300]} [...]')

        print('-' * 55)
        print(f'Exemplo de decisão ({item.get("id_mapa", "")}):')
        print(f'>>> DECISÃO: {str(item["decisao"])[:300]} [...]')

        print('-' * 55)
        print(f'Exemplo de integra ({item.get("id_mapa", "")}):')
        print(f'>>> ÍNTEGRA: {str(item["integra"])[:200]} [...] {str(item["integra"])[-100:]}')

        outros_dados = {c:v for c,v in item.items() if c not in {'id_mapa', 'ementa', 'decisao', 'integra'}}
        print('-' * 55)
        print(f'Outros dados ({item.get("id_mapa", "")}):')
        print(outros_dados)
        print('-' * 55)
        print(f'Mapa do espelho: {mapa_espelhos[item.get("id_mapa", "")]}')
        print('-' * 55)
        print(f'Mapa da integra: {mapa_integras[item.get("id_mapa", "")]}')
    

if __name__ == '__main__':
    """Exemplos de uso ou execução CLI."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Utilitários CKAN e CLI de extração em lote")
    parser.add_argument("--config", type=str, help="Caminho para o arquivo YAML de configuração")
    parser.add_argument("--view", action="store_true", help="Apenas visualizar o parquet final gerado sem re-extrair")
    args, unknown = parser.parse_known_args()

    if args.config:
        yaml_path = args.config
        if not os.path.isfile(yaml_path):
            print(f"⚠️  Arquivo '{yaml_path}' não encontrado.")
            resp = input("Deseja criar um arquivo de configuração com exemplo? (s/N): ").strip().lower()
            if resp in ("s", "sim", "y", "yes"):
                with open(yaml_path, "w", encoding="utf-8") as f:
                    f.write(_YAML_EXEMPLO.format(nome_arquivo=os.path.basename(yaml_path)))
                print(f"✅ Arquivo de exemplo criado em: '{yaml_path}'")
                print("   Edite-o com suas configurações e rode o comando novamente.")
            sys.exit(0)
            
        if args.view:
            exibir_view_ckan(yaml_path)
        else:
            executar_ckan_batch(yaml_path)
    else:
        print("💡 Dica: Use --config <arquivo.yaml> para rodar extração em lote.\n")
        atualizar_mapas_e_cache = True
        
        # Exemplo 2: construir mapas e gerar dataset com um número de registro específico
        ExemplosCKan.exemplo2(num_registro = '202302829818', atualizar_cache_e_mapas = atualizar_mapas_e_cache)

