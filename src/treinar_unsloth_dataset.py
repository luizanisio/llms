#!/usr/bin/env python3
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Módulo dedicado à manipulação de datasets para treinamento.
Encapsula lógica de carregamento, validação e preparação de dados.
"""

import os
import random
import json
import pandas as pd
from typing import Dict, List, Any, Optional

# Importando constantes e dataclasses necessárias
from treinar_unsloth_util import (
    ConfigCurriculum,
    ConfigMisc
)

class DatasetTreinamento:
    """
    Gerencia o carregamento, validação e preparação de datasets para treinamento.
    Formato único: curriculum (entrada + saída + divisão).
    """
    
    def __init__(self, config_curriculum: ConfigCurriculum,
                 config_misc: Optional[ConfigMisc] = None):
        """
        Inicializa o gerenciador de datasets.
        
        Args:
            config_curriculum: Configuração do curriculum (entrada, saída, divisão, etc.)
            config_misc: Configurações diversas (log_level, criptografia)
        """
        self.curriculum = config_curriculum
        self.misc = config_misc
        
        # Cache de dados carregados
        self._arquivos_pareados: Optional[List[Dict]] = None
        self._dados_divisao: Optional[pd.DataFrame] = None
        self._cache_dataframes: Dict[str, pd.DataFrame] = {}  # cache de dataframes compartilhados

    # ---------------------------------------------------------------------------
    # Métodos para modo "pastas"
    # ---------------------------------------------------------------------------
    
    def listar_arquivos_por_mascara(self, pasta: str, mascara: str) -> Dict[str, str]:
        """
        Lista arquivos em uma pasta que casam com a máscara glob.
        """
        import glob
        
        if not os.path.isdir(pasta):
            raise ValueError(f"Pasta não encontrada: '{pasta}'")
        
        resultado = {}
        pattern = os.path.join(pasta, mascara)
        
        for caminho_completo in glob.glob(pattern):
            if os.path.isfile(caminho_completo):
                nome_arquivo = os.path.basename(caminho_completo)
                id_arquivo = os.path.splitext(nome_arquivo)[0]
                resultado[id_arquivo] = caminho_completo
        
        return resultado
    
    def parear_arquivos(self) -> List[Dict[str, Any]]:
        """
        Parea arquivos/registros de entrada com saída (gold dataset) pelo ID.
        
        Suporta:
        - Saída como pasta de arquivos ou dataframe
        - Entrada como pasta de arquivos ou dataframe
        """
        if self._arquivos_pareados is not None:
            return self._arquivos_pareados
        
        cc = self.curriculum
        
        # --- Obtém IDs da saída (gold) ---
        usa_saida_df = bool(cc.saida.dataframe)
        usa_entrada_df = bool(cc.entrada.dataframe)
        
        if usa_saida_df:
            ids_saida = set(self._carregar_ids_dataframe(
                cc.saida.dataframe, cc.saida.dataframe_id
            ))
        else:
            arquivos_saida = self.listar_arquivos_por_mascara(
                cc.saida.pasta, cc.saida.mascara
            )
            ids_saida = set(arquivos_saida.keys())
        
        # --- Obtém IDs da entrada ---
        if usa_entrada_df:
            ids_entrada = set(self._carregar_ids_dataframe(
                cc.entrada.dataframe, cc.entrada.dataframe_id
            ))
        else:
            arquivos_entrada = self.listar_arquivos_por_mascara(
                cc.entrada.pasta, cc.entrada.mascara
            )
            ids_entrada = set(arquivos_entrada.keys())
        
        # --- Validação de IDs pareados ---
        if cc.validacao.exigir_ids_pareados:
            ids_sem_par = ids_entrada - ids_saida
            if ids_sem_par:
                ids_lista = sorted(ids_sem_par)[:10]
                extra = f" (+{len(ids_sem_par) - 10} outros)" if len(ids_sem_par) > 10 else ""
                raise ValueError(
                    f"{len(ids_sem_par)} ID(s) de entrada sem correspondente no gold dataset (saída): "
                    f"{ids_lista}{extra}. "
                    f"\n⚠️ Use 'exigir_ids_pareados: false' na seção validacao para ignorar essa verificação."
                )
        
        # --- Pareamento ---
        if usa_entrada_df and usa_saida_df:
            # Ambos via dataframe: interseção de IDs
            ids_comum = ids_entrada & ids_saida
            self._arquivos_pareados = sorted([
                {"id": id_arq, "entrada": None, "predicao": None}
                for id_arq in ids_comum
            ], key=lambda x: x["id"])
            print(f"✅ {len(self._arquivos_pareados)} registro(s) pareados (entrada e saída via dataframe)")
        
        elif usa_entrada_df:
            # Entrada via dataframe, saída via pasta
            ids_comum = ids_entrada & ids_saida
            self._arquivos_pareados = sorted([
                {"id": id_arq, "entrada": None, "predicao": arquivos_saida[id_arq]}
                for id_arq in ids_comum
            ], key=lambda x: x["id"])
            print(f"✅ {len(self._arquivos_pareados)} registro(s) pareados (entrada via dataframe)")
        
        elif usa_saida_df:
            # Entrada via pasta, saída via dataframe
            ids_comum = ids_entrada & ids_saida
            self._arquivos_pareados = sorted([
                {"id": id_arq, "entrada": arquivos_entrada[id_arq], "predicao": None}
                for id_arq in ids_comum
            ], key=lambda x: x["id"])
            print(f"✅ {len(self._arquivos_pareados)} registro(s) pareados (saída via dataframe)")
        
        else:
            # Ambos via pasta
            ids_comum = ids_entrada & ids_saida
            ids_so_entrada = ids_entrada - ids_comum
            ids_so_saida = ids_saida - ids_comum
            
            if ids_so_entrada:
                print(f"⚠️  {len(ids_so_entrada)} arquivo(s) de entrada sem par na saída")
            if ids_so_saida:
                print(f"⚠️  {len(ids_so_saida)} arquivo(s) de saída sem par de entrada")
            
            self._arquivos_pareados = sorted([
                {"id": id_arq, "entrada": arquivos_entrada[id_arq], "predicao": arquivos_saida[id_arq]}
                for id_arq in ids_comum
            ], key=lambda x: x["id"])
            print(f"✅ {len(self._arquivos_pareados)} par(es) de arquivos encontrados")
        
        return self._arquivos_pareados
    
    def _carregar_ids_dataframe(self, caminho: str, col_id: str) -> List[str]:
        """Carrega a lista de IDs de um dataframe parquet."""
        df = self._obter_dataframe(caminho)
        if col_id not in df.columns:
            raise ValueError(f"Coluna '{col_id}' não encontrada no dataframe '{caminho}'")
        return [str(v) for v in df[col_id].tolist()]
    
    def _obter_dataframe(self, caminho: str) -> pd.DataFrame:
        """Obtém um dataframe com cache (para compartilhamento entre entrada/saída)."""
        caminho_abs = os.path.abspath(caminho)
        if caminho_abs not in self._cache_dataframes:
            self._cache_dataframes[caminho_abs] = pd.read_parquet(caminho)
        return self._cache_dataframes[caminho_abs]
    
    def _validar_consistencia_divisao(self) -> None:
        """
        Valida a consistência entre o arquivo de divisão e os arquivos pareados.
        
        Verifica:
        1. Todos os IDs do CSV de divisão devem existir nos arquivos pareados
        2. Todos os IDs dos arquivos pareados devem estar no CSV de divisão
        
        Raises:
            ValueError: Se houver inconsistência entre os arquivos
        """
        if self._dados_divisao is None:
            return  # Nada para validar ainda
        
        arquivos_pareados = self.parear_arquivos()
        ids_pareados = set(arq["id"] for arq in arquivos_pareados)
        ids_divisao = set(self._dados_divisao["id_arquivo"].astype(str))
        
        erros = []
        
        # Validação 1: IDs no CSV mas não nos arquivos pareados
        ids_apenas_csv = ids_divisao - ids_pareados
        if ids_apenas_csv:
            ids_lista = sorted(ids_apenas_csv)
            if len(ids_lista) > 10:
                ids_mostra = ids_lista[:10] + [f"... (+{len(ids_lista) - 10} outros)"]
            else:
                ids_mostra = ids_lista
            erros.append(
                f"❌ {len(ids_apenas_csv)} ID(s) no arquivo de divisão não encontrados nos arquivos pareados:\n"
                f"   {ids_mostra}"
            )
        
        # Validação 2: IDs nos arquivos pareados mas não no CSV
        # (Controlado por validacao.exigir_ids_pareados)
        if self.curriculum.validacao.exigir_ids_pareados:
            ids_apenas_pareados = ids_pareados - ids_divisao
            if ids_apenas_pareados:
                ids_lista = sorted(ids_apenas_pareados)
                if len(ids_lista) > 10:
                    ids_mostra = ids_lista[:10] + [f"... (+{len(ids_lista) - 10} outros)"]
                else:
                    ids_mostra = ids_lista
                erros.append(
                    f"❌ {len(ids_apenas_pareados)} arquivo(s) pareado(s) não encontrados no arquivo de divisão:\n"
                    f"   {ids_mostra}"
                )
        else:
             ids_apenas_pareados = ids_pareados - ids_divisao
             if ids_apenas_pareados:
                 print(f"⚠️  Aviso: {len(ids_apenas_pareados)} arquivos pareados ignorados (não estão na divisão).")
        
        if erros:
            msg_erro = (
                f"\n{'=' * 60}\n"
                f"🚨 ERRO DE CONSISTÊNCIA: Arquivo de divisão e arquivos pareados estão inconsistentes!\n"
                f"{'=' * 60}\n\n"
                + "\n\n".join(erros) +
                f"\n\n{'=' * 60}\n"
                f"💡 SOLUÇÃO: Verifique se os arquivos de entrada/saída correspondem ao arquivo de divisão.\n"
                f"   - Arquivo de divisão: {self.curriculum.divisao.arquivo}\n"
                f"   - Total IDs na divisão: {len(ids_divisao)}\n"
                f"   - Total IDs pareados: {len(ids_pareados)}\n"
                f"\n"\
                f"* você pode remover o arquivo de divisão para que ele seja recriado com uma nova divisão\n"\
                f"{'=' * 60}"
            )
            raise ValueError(msg_erro)
    
    def carregar_ou_criar_divisao(self) -> pd.DataFrame:
        """
        Carrega o arquivo de divisão se existir, ou cria um novo.
        """
        if self._dados_divisao is not None:
            return self._dados_divisao
        
        arquivo_divisao = self.curriculum.divisao.arquivo
        
        # Tenta carregar arquivo existente
        if arquivo_divisao and os.path.isfile(arquivo_divisao):
            print(f"📂 Carregando divisão de: {arquivo_divisao}")
            df = pd.read_csv(arquivo_divisao, sep=None, engine='python')
            
            # Limpa o BOM UTF-8 (\ufeff) e espaços das colunas caso o arquivo seja salvo no formato corrompido ou BOM pelo Excel/Win
            df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
            
            # Migração automática de nomes de colunas antigos
            if "id_arquivo" not in df.columns and "id" in df.columns:
                print("🔄 Migrando coluna 'id' → 'id_arquivo'...")
                df.rename(columns={"id": "id_arquivo"}, inplace=True)
            if "alvo" not in df.columns:
                for col_antiga in ("divisão", "divisao", "divisões", "divisoes", "grupo"):
                    if col_antiga in df.columns:
                        print(f"🔄 Migrando coluna '{col_antiga}' → 'alvo'...")
                        df.rename(columns={col_antiga: "alvo"}, inplace=True)
                        break
            if "id_arquivo" not in df.columns or "alvo" not in df.columns:
                raise ValueError(f"Arquivo de divisão deve ter colunas 'id_arquivo' e 'alvo', nenhuma opção de migração encontrada. Colunas atuais: {df.columns.tolist()}")
            
            self._dados_divisao = df
            
            # Migração automática: avaliacao -> validacao
            mask_antigo = self._dados_divisao["alvo"].isin(["avaliacao", "avaliação", "eval"])
            if mask_antigo.any():
                print(f"🔄 Migrando labels antigos ('avaliacao') para 'validacao'...")
                self._dados_divisao.loc[mask_antigo, "alvo"] = "validacao"
                # Salva atualização
                self._dados_divisao.to_csv(arquivo_divisao, index=False)
                print(f"💾 Arquivo de divisão atualizado: {arquivo_divisao}")
            
            # Em curriculum multi-etapa, a divisão é um subconjunto do dataset global
            # Não validar consistência nesse caso
            self._validar_consistencia_divisao()
            
            # Calcula proporções efetivas
            contagem = self._dados_divisao["alvo"].value_counts(normalize=True).to_dict()
            total_reais = [
                contagem.get('treino', 0.0),
                contagem.get('validacao', 0.0),
                contagem.get('teste', 0.0)
            ]
            self.curriculum.divisao.proporcao_reais = total_reais
            
            return self._dados_divisao
        
        # Cria nova divisão
        print(f"🔀 Criando nova divisão de dados...")
        arquivos_pareados = self.parear_arquivos()
        ids = [arq["id"] for arq in arquivos_pareados]
        
        random.seed(self.curriculum.divisao.seed)
        ids_embaralhados = ids.copy()
        random.shuffle(ids_embaralhados)
        
        n = len(ids_embaralhados)
        prop = self.curriculum.divisao.proporcao
        corte_treino = int(n * prop[0])
        corte_validacao = int(n * (prop[0] + prop[1]))
        
        dados_divisao = []
        for i, id_arq in enumerate(ids_embaralhados):
            if i < corte_treino:
                # Treino: Usado para o aprendizado dos pesos do modelo (backpropagation)
                alvo = "treino"
            elif i < corte_validacao:
                # Validação: Usado durante o treinamento para monitorar métricas e early stopping
                alvo = "validacao"
            else:
                # Teste: Usado APÓS o treinamento (reservado) para avaliação final imparcial
                alvo = "teste"
            dados_divisao.append({"id_arquivo": id_arq, "alvo": alvo})
        
        self._dados_divisao = pd.DataFrame(dados_divisao)
        
        if arquivo_divisao:
            os.makedirs(os.path.dirname(arquivo_divisao) or ".", exist_ok=True)
            self._dados_divisao.to_csv(arquivo_divisao, index=False)
            print(f"💾 Divisão salva em: {arquivo_divisao}")
        
        contagem_abs = self._dados_divisao["alvo"].value_counts()
        contagem_rel = self._dados_divisao["alvo"].value_counts(normalize=True).to_dict()
        
        print(f"   Treino: {contagem_abs.get('treino', 0)} ({contagem_rel.get('treino', 0):.2%}) | Validação: {contagem_abs.get('validacao', 0)} ({contagem_rel.get('validacao', 0):.2%}) | Teste: {contagem_abs.get('teste', 0)} ({contagem_rel.get('teste', 0):.2%})")
        
        # Salva proporções reais no config
        total_reais = [
                contagem_rel.get('treino', 0.0),
                contagem_rel.get('validacao', 0.0),
                contagem_rel.get('teste', 0.0)
            ]
        self.curriculum.divisao.proporcao_reais = total_reais
        
        return self._dados_divisao
    
    def _carregar_conteudo_arquivo(self, caminho: str) -> str:
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(caminho, "r", encoding="latin-1") as f:
                return f.read()

    def _carregar_dataframe_entrada(self) -> Dict[str, str]:
        """Carrega textos de entrada de um dataframe parquet, com decriptografia opcional."""
        entrada = self.curriculum.entrada
        if not entrada.dataframe:
            return {}
        
        df = self._obter_dataframe(entrada.dataframe)
        
        if entrada.dataframe_col not in df.columns:
            raise ValueError(f"Coluna '{entrada.dataframe_col}' não encontrada no dataframe de entrada")
        if entrada.dataframe_id not in df.columns:
            raise ValueError(f"Coluna '{entrada.dataframe_id}' não encontrada no dataframe de entrada")
        
        # Inicializa criptografia se o campo de texto está criptografado
        cripto = self._obter_cripto() if entrada.texto_criptografado else None
        
        mapa_textos = {}
        for _, row in df.iterrows():
            id_registro = str(row[entrada.dataframe_id])
            texto = str(row[entrada.dataframe_col])
            
            if cripto:
                try:
                    texto = cripto.decriptografar(texto)
                except Exception as e:
                    print(f"⚠️  Erro ao decriptografar entrada {id_registro}: {e}")
                    continue
            
            mapa_textos[id_registro] = texto
        
        print(f"📂 Carregados {len(mapa_textos)} textos de entrada do dataframe")
        return mapa_textos

    def _carregar_dataframe_saida(self) -> Dict[str, str]:
        """Carrega textos de saída (gold) de um dataframe parquet, com decriptografia opcional."""
        saida = self.curriculum.saida
        if not saida.dataframe:
            return {}
        
        df = self._obter_dataframe(saida.dataframe)
        
        if saida.dataframe_col not in df.columns:
            raise ValueError(f"Coluna '{saida.dataframe_col}' não encontrada no dataframe de saída")
        if saida.dataframe_id not in df.columns:
            raise ValueError(f"Coluna '{saida.dataframe_id}' não encontrada no dataframe de saída")
        
        cripto = self._obter_cripto() if saida.texto_criptografado else None
        
        mapa_textos = {}
        for _, row in df.iterrows():
            id_registro = str(row[saida.dataframe_id])
            texto = str(row[saida.dataframe_col])
            
            if cripto:
                try:
                    texto = cripto.decriptografar(texto)
                except Exception as e:
                    print(f"⚠️  Erro ao decriptografar saída {id_registro}: {e}")
                    continue
            
            mapa_textos[id_registro] = texto
        
        print(f"📂 Carregados {len(mapa_textos)} textos de saída do dataframe")
        return mapa_textos
    
    def _obter_cripto(self):
        """Retorna instância de criptografia usando a chave configurada em misc."""
        nome_var = self.misc.env_chave_criptografia if self.misc else ""
        if not nome_var:
            raise ValueError(
                "❌ Dados criptografados detectados mas 'misc.env_chave_criptografia' não está configurada"
            )
        chave = os.getenv(nome_var)
        if not chave or not chave.strip():
            raise EnvironmentError(
                f"❌ Variável de ambiente '{nome_var}' não está definida ou está vazia. "
                f"Defina a variável: export {nome_var}=\"sua_chave_fernet\""
            )
        try:
            from util import UtilCriptografia
            os.environ['CHAVE_CRIPT'] = chave
            cripto = UtilCriptografia()
            print("CHAVE FERNET CARREGADA _o/")
            return cripto
        except ImportError as e:
            raise ImportError(
                f"❌ Não foi possível importar UtilCriptografia. "
                f"O módulo 'util' é necessário para decriptografar os dados. Erro: {e}"
            )

    def _montar_prompt(self, conteudo_entrada: str) -> str:
        entrada = self.curriculum.entrada
        if not entrada.prompt_template:
            return conteudo_entrada
        
        template = self._carregar_conteudo_arquivo(entrada.prompt_template)
        
        # Decriptografa o template se necessário
        if entrada.prompt_criptografado:
            cripto = self._obter_cripto()
            template = cripto.decriptografar(template)
        
        tag = entrada.tag_texto
        if tag and tag in template:
            return template.replace(tag, conteudo_entrada)
        
        return template + "\n\n" + conteudo_entrada

    def carregar_mensagens_de_pastas(self, alvo: str = "treino") -> List[Dict[str, Any]]:
        """Carrega mensagens formatadas para treinamento a partir das configurações do curriculum."""
        cc = self.curriculum
        divisao = self.carregar_ou_criar_divisao()
        
        # Filtra IDs pelo alvo
        ids_alvo = set(divisao[divisao["alvo"] == alvo]["id_arquivo"])
        
        arquivos_pareados = self.parear_arquivos()
        arquivos_filtrados = [p for p in arquivos_pareados if p["id"] in ids_alvo]
        
        # Carrega dados de entrada/saída via dataframe se configurado
        mapa_textos_entrada = {}
        if cc.entrada.dataframe:
            mapa_textos_entrada = self._carregar_dataframe_entrada()
        
        mapa_textos_saida = {}
        if cc.saida.dataframe:
            mapa_textos_saida = self._carregar_dataframe_saida()
            
        mensagens = []
        erros = []
        
        from util import UtilTextos as Util
        
        for par in arquivos_filtrados:
            try:
                id_arq = par["id"]
                
                # --- Texto de entrada ---
                if cc.entrada.dataframe:
                    texto_entrada = mapa_textos_entrada.get(id_arq, "")
                    if not texto_entrada:
                        print(f"⚠️  Texto de entrada não encontrado no dataframe para ID: {id_arq}")
                        continue
                else:
                    caminho_entrada = par["entrada"]
                    texto_entrada = self._carregar_conteudo_arquivo(caminho_entrada)
                
                texto_entrada = self._montar_prompt(texto_entrada)
                
                # --- Texto de saída (gold) ---
                if cc.saida.dataframe:
                    conteudo_saida = mapa_textos_saida.get(id_arq, "")
                    if not conteudo_saida:
                        print(f"⚠️  Texto de saída não encontrado no dataframe para ID: {id_arq}")
                        continue
                else:
                    caminho_saida = par["predicao"]
                    conteudo_saida = self._carregar_conteudo_arquivo(caminho_saida)
                
                if cc.validacao.exigir_json_valido:
                     if conteudo_saida.strip().startswith("Error:"):
                         raise ValueError(f"Arquivo de saída contém erro: {conteudo_saida[:100]}")
                     
                     try:
                         json_obj = Util.mensagem_to_json(conteudo_saida)
                         conteudo_saida = json.dumps(json_obj, ensure_ascii=False)
                     except Exception as e:
                         raise ValueError(f"JSON inválido em {id_arq}: {e}")
                
                msg = {
                    "id": id_arq,
                    "messages": [
                        {"role": "user", "content": texto_entrada},
                        {"role": "assistant", "content": conteudo_saida}
                    ]
                }
                mensagens.append(msg)
                
            except Exception as e:
                erros.append((par["id"], str(e)))
                print(f"❌ Erro ao processar {par['id']}: {e}")
        
        if erros:
            print(f"⚠️  {len(erros)} erro(s) ao processar arquivos")
            
        print(f"✅ {len(mensagens)} mensagem(ns) carregada(s)" + (f" para '{alvo}'" if alvo else ""))
        return mensagens

    def mostrar_exemplo(self, titulo: str, msgs_lista: List[Dict]):
        """
        Mostra exemplos formatados de mensagens (primeiro e último).
        """
        if not msgs_lista:
            return
        print(f"\n  👀 {titulo} (primeiro e último):")
        for idx in [0, -1]: # Primeiro e último
            if abs(idx) >= len(msgs_lista) and idx == -1: continue 
            
            msg = msgs_lista[idx]
            print(f"    --- Índice {idx} (ID: {msg.get('id', 'N/A')}) ---")
            
            if 'messages' in msg:
                # User content
                user_msg = next((m for m in msg['messages'] if m['role'] == 'user'), None)
                if user_msg:
                    content = str(user_msg.get('content', ''))
                    inicio = content[:400].replace('\n', ' ')
                    fim = content[-150:].replace('\n', ' ') if len(content) > 400 else ""
                    print(f"    👤 User: {inicio} [...] {fim}")
                
                # Assistant content (resposta)
                assist_msg = next((m for m in msg['messages'] if m['role'] == 'assistant'), None)
                if assist_msg:
                    content = str(assist_msg.get('content', ''))
                    inicio = content[:400].replace('\n', ' ')
                    fim = content[-150:].replace('\n', ' ') if len(content) > 400 else ""
                    print(f"    🤖 Assistant: {inicio} [...] {fim}")
    
    def get_stats(self, dataset: Any, nome: str, max_seq_length: int) -> None:
        """
        Exibe estatísticas de tokens do dataset (originalmente _print_dataset_stats).
        Aceita objeto Dataset do huggingface/datasets.
        """
        if dataset is None or len(dataset) == 0:
            print(f"📊 {nome}: vazio")
            return
        
        try:
            lengths = [len(r.get('input_ids', [])) for r in dataset]
        except:
             print(f"📊 {nome}: {len(dataset)} registros (não tokenizado ou erro ao ler input_ids)")
             return

        if not lengths or max(lengths) == 0:
            print(f"📊 {nome}: {len(dataset)} registros (não tokenizado)")
            return
        
        print(f"📊 {nome}:")
        print(f"   Registros: {len(dataset)}")
        print(f"   Tokens: min={min(lengths)}, max={max(lengths)}, média={sum(lengths)/len(lengths):.0f}")
        
        excedem = sum(1 for l in lengths if l > max_seq_length)
        if excedem > 0:
            print(f"   ⚠️  {excedem} registros excedem max_seq_length={max_seq_length}")
