
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

"""

import os
import shutil
import hashlib
from typing import Dict, Any, Optional, Union

# Tiktoken (OpenAI) - opcional
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Transformers (Hugging Face) - opcional
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Modelos conhecidos que NÃO são da OpenAI (precisam de tokenizador nativo)
# Mapeamento: prefixo do nome do modelo -> modelo HF para carregar tokenizador
MODELOS_NAO_OPENAI = {
    "qwen": None,        # Usa o próprio nome do modelo
    "deepseek": None,    # Usa o próprio nome do modelo
    "gemma": None,       # Usa o próprio nome do modelo  
    "llama": None,       # Usa o próprio nome do modelo
    "mistral": None,     # Usa o próprio nome do modelo
    "phi": None,         # Usa o próprio nome do modelo
}


class UtilTikToken:
    """
    Classe utilitária para contagem de tokens.
    
    Suporta:
    - Modelos OpenAI (via tiktoken): gpt-4, gpt-3.5-turbo, text-embedding-ada-002, etc.
    - Modelos não-OpenAI (via transformers): Qwen, DeepSeek, Gemma, Llama, Mistral, Phi, etc.
    
    Mantém cache de tokenizadores para otimizar múltiplas chamadas.
    """

    def __init__(self, verbose: bool = False):
        """
        Inicializa o utilitário.
        
        Args:
            verbose: Se True, imprime mensagens de debug sobre qual tokenizador está sendo usado.
        """
        self._tiktoken_cache: Dict[str, Any] = {}
        self._hf_tokenizer_cache: Dict[str, Any] = {}
        self._verbose = verbose
        
        if TIKTOKEN_AVAILABLE:
            self._prepara_tiktoken_cache_files()
        
    def _prepara_tiktoken_cache_files(self):
        """
        Copia arquivos de encoding do tiktoken para o cache local.
        Permite uso offline se os arquivos já estiverem baixados.
        """
        tiktoken_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiktoken_cache")
        
        if not os.path.isdir(tiktoken_cache_path):
            alternative_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "tiktoken_cache")
            if os.path.isdir(alternative_path):
                tiktoken_cache_path = alternative_path
            else:
                return

        os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_path
        
        encodings = {
            "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
            "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
            "p50k_base": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
            "r50k_base": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
        }

        try:
            arquivos_na_pasta = os.listdir(tiktoken_cache_path)
        except FileNotFoundError:
            return

        for encoding_name, blob_url in encodings.items():
            cache_key = hashlib.sha1(blob_url.encode()).hexdigest()
            caminho_cache = os.path.join(tiktoken_cache_path, cache_key)
            arquivo_original = f"{encoding_name}.tiktoken"
            caminho_original = os.path.join(tiktoken_cache_path, arquivo_original)

            if arquivo_original in arquivos_na_pasta:
                if not os.path.isfile(caminho_cache):
                    try:
                        shutil.copy(caminho_original, caminho_cache)
                    except OSError:
                        pass
    
    def _is_modelo_nao_openai(self, modelo: str) -> bool:
        """Verifica se o modelo não é da OpenAI e precisa de tokenizador HF."""
        modelo_lower = modelo.lower()
        for prefixo in MODELOS_NAO_OPENAI.keys():
            if prefixo in modelo_lower:
                return True
        return False
    
    def _get_hf_tokenizer(self, modelo: str) -> Optional[Any]:
        """
        Obtém o tokenizador do Hugging Face para um modelo.
        Retorna None se não conseguir carregar.
        """
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        if modelo in self._hf_tokenizer_cache:
            return self._hf_tokenizer_cache[modelo]
        
        try:
            # Tenta carregar o tokenizador do modelo
            tokenizer = AutoTokenizer.from_pretrained(modelo, trust_remote_code=True)
            self._hf_tokenizer_cache[modelo] = tokenizer
            if self._verbose:
                print(f"🔧 Tokenizador HF carregado: {modelo}")
            return tokenizer
        except Exception as e:
            if self._verbose:
                print(f"⚠️ Não foi possível carregar tokenizador HF para '{modelo}': {e}")
            return None
    
    def _get_tiktoken_encoder(self, modelo: str) -> Optional[Any]:
        """Obtém encoder tiktoken para modelos OpenAI."""
        if not TIKTOKEN_AVAILABLE:
            return None
            
        try:
            encoding_name = tiktoken.encoding_name_for_model(modelo)
        except KeyError:
            # Modelo não reconhecido pelo tiktoken, usa fallback
            encoding_name = "cl100k_base"
            
        if encoding_name in self._tiktoken_cache:
            return self._tiktoken_cache[encoding_name]
            
        try:
            encoder = tiktoken.get_encoding(encoding_name)
            self._tiktoken_cache[encoding_name] = encoder
            return encoder
        except Exception:
            return None

    def contar_tokens(self, texto: str, modelo: str = "gpt-4") -> Dict[str, int]:
        """
        Conta tokens de um texto para um determinado modelo.
        
        Para modelos não-OpenAI (Qwen, DeepSeek, Gemma, Llama, etc.), tenta usar
        o tokenizador nativo via Hugging Face. Se não conseguir, faz fallback
        para tiktoken cl100k_base (estimativa aproximada).
        
        Args:
            texto: Texto para contar tokens
            modelo: Nome do modelo (ex: 'gpt-4', 'Qwen/Qwen2.5-1.5B-Instruct', etc.)
            
        Returns:
            Dict com 'qtd_tokens', 'qtd_tokens_unicos' e 'tokenizer_type'
        """
        if not texto:
            return {'qtd_tokens': 0, 'qtd_tokens_unicos': 0, 'tokenizer_type': 'none'}
        
        tokens = []
        tokenizer_type = "unknown"
        
        # 1. Verifica se é modelo não-OpenAI
        if self._is_modelo_nao_openai(modelo):
            # Tenta usar tokenizador HF nativo
            hf_tokenizer = self._get_hf_tokenizer(modelo)
            if hf_tokenizer:
                try:
                    tokens = hf_tokenizer.encode(texto, add_special_tokens=False)
                    tokenizer_type = "huggingface"
                except Exception as e:
                    if self._verbose:
                        print(f"⚠️ Erro ao tokenizar com HF: {e}")
        
        # 2. Se não conseguiu com HF, tenta tiktoken
        if not tokens:
            encoder = self._get_tiktoken_encoder(modelo)
            if encoder:
                try:
                    tokens = encoder.encode(texto, disallowed_special=())
                    tokenizer_type = "tiktoken" if not self._is_modelo_nao_openai(modelo) else "tiktoken_estimativa"
                except Exception:
                    pass
        
        # 3. Fallback final: contagem por caracteres/palavras (muito impreciso)
        if not tokens:
            # Estimativa grosseira: ~4 chars por token (média para inglês)
            tokens = list(range(len(texto) // 4))
            tokenizer_type = "caracteres_estimativa"
        
        return {
            'qtd_tokens': len(tokens),
            'qtd_tokens_unicos': len(set(tokens)),
            'tokenizer_type': tokenizer_type
        }
    
    def get_encoder(self, model_name: str = "gpt-4") -> Any:
        """
        Retorna o encoder/tokenizador para o modelo especificado.
        
        Para modelos não-OpenAI, retorna o tokenizador HF se disponível.
        Caso contrário, retorna encoder tiktoken.
        """
        if self._is_modelo_nao_openai(model_name):
            hf_tok = self._get_hf_tokenizer(model_name)
            if hf_tok:
                return hf_tok
        
        return self._get_tiktoken_encoder(model_name)

    def TikToken(self, dados: Dict[str, Any], engine: Dict[str, Any]) -> Dict[str, int]:
        """
        Método wrapper para compatibilidade com chamadas legadas.
        
        Args:
            dados: Dict contendo a chave 'texto'
            engine: Dict contendo a chave 'nome_implantacao' (nome do modelo)
        """
        texto = dados.get('texto', '')
        modelo = engine.get('nome_implantacao', 'gpt-4')
        result = self.contar_tokens(texto, modelo)
        # Remove campo extra para manter compatibilidade
        return {
            'qtd_tokens': result['qtd_tokens'],
            'qtd_tokens_unicos': result['qtd_tokens_unicos']
        }


if __name__ == "__main__":
    from time import time
    print("=" * 60)
    print("🧪 TESTE DE CONTAGEM DE TOKENS")
    print("=" * 60)
    
    ut = UtilTikToken(verbose=True)
    texto_teste = "Olá, mundo! Este é um teste de contagem de tokens para diferentes modelos de linguagem."
    
    modelos_teste = [
        "gpt-4",
        "gpt-3.5-turbo",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "google/gemma-3-4b-it",
        "meta-llama/Llama-3.2-1B-Instruct",
    ]
    
    print(f"\nTexto: {texto_teste[:50]}...\n")
    
    for modelo in modelos_teste:
        ini = time()
        result = ut.contar_tokens(texto_teste, modelo)
        tempo = time() - ini
        print('. ' * 30)
        print(f"📊 {modelo}"    )
        print(f"   Tokens: {result['qtd_tokens']} | Únicos: {result['qtd_tokens_unicos']} | Tipo: {result['tokenizer_type']}")
        print(f"   Tempo: {tempo:.2f} segundos")
        print()            
