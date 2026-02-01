

class UtilTikToken:

    def __init__(self):
        self.prepara_tiktoken_cache_files()
        
    def prepara_tiktoken_cache_files(self):
        """
        Copia todos os arquivos de encoding da pasta de cache com o nome esperado pelo cache.
        """
        tiktoken_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiktoken_cache")
        if not os.path.isdir(tiktoken_cache_path):
            tiktoken_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "tiktoken_cache")
        if not os.path.isdir(tiktoken_cache_path):
            print("⚠️ Pasta de cache 'tiktoken_cache' não encontrada na pasta do aplicativo.")
        os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_path
        # URLs oficiais para os encodings suportados
        encodings = {
            "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
            "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
            "p50k_base": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
            "r50k_base": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
        }

        # Lista os arquivos de encodings encontrados na pasta local
        arquivos_na_pasta = os.listdir(tiktoken_cache_path)

        for encoding_name, blob_url in encodings.items():
            # Nome esperado com SHA1 do blob URL
            cache_key = hashlib.sha1(blob_url.encode()).hexdigest()
            caminho_cache = os.path.join(tiktoken_cache_path, cache_key)
            arquivo_original = f"{encoding_name}.tiktoken"
            caminho_original = os.path.join(tiktoken_cache_path, arquivo_original)

            if arquivo_original in arquivos_na_pasta:
                if not os.path.isfile(caminho_cache):
                    shutil.copy(caminho_original, caminho_cache)
                    print(f"✅ Arquivo {caminho_original} copiado para {caminho_cache}.")
                else:
                    print(f"🤖 Arquivo cache {caminho_cache} já existe.")
            else:
                print(f"⚠️ Arquivo {arquivo_original} não encontrado na pasta cache.")

        print("🚀 Todos os arquivos de cache foram processados.")
            
    def TikToken(self, dados, engine):
        # Escolhendo o modelo certo para embeddings
        nome_do_modelo = engine["nome_implantacao"]
        nome_encode = tiktoken.model.encoding_name_for_model(nome_do_modelo)
        nome_base = os.path.splitext(nome_encode)[0]
        UtilEnv.print_debug(f'Buscando base_model para o modelo {nome_do_modelo} = {nome_base}')
        #encoder = tiktoken.encoding_for_model(modelo_embedding)
        #tokens = encoder.encode(dados['texto'])
        
        base_model = tiktoken.get_encoding(nome_base)
        tokens = base_model.encode(dados['texto'])
        return {'qtd_tokens': len(tokens), 'qtd_tokens_unicos': len(set(tokens))}