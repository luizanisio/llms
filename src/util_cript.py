import os
from cryptography.fernet import Fernet

class UtilCriptografia:
    def __init__(self):
        # Tenta obter a chave de criptografia a partir da variável de ambiente
        self.chave = os.getenv('CHAVE_CRIPT')
        if not self.chave:
            # Se a chave não existir na variável de ambiente, gera uma nova chave
            self.chave = self.nova_chave()
            print('NOVA CHAVE FERNET GERADA:', self.chave)
        else:
            print('CHAVE FERNET CARREGADA _o/')
        # O Fernet espera a chave como bytes, então garantimos esta conversão
        if isinstance(self.chave, str):
            self.chave = self.chave.encode()
        # Inicializa o objeto Fernet com a chave
        self.fernet = Fernet(self.chave)
    
    def criptografar(self, texto: str) -> str:
        """
        Criptografa uma string e retorna o token criptografado como string.
        """
        # Codifica o texto para bytes e aplica a criptografia
        texto_bytes = texto.encode()
        token = self.fernet.encrypt(texto_bytes)
        # Decodifica o token para string, ideal para armazenamento em DataFrame
        return token.decode()
    
    def decriptografar(self, texto: str) -> str:
        """
        Descriptografa uma string criptografada e retorna o texto original.
        """
        # Codifica a string criptografada em bytes
        token_bytes = texto.encode()
        texto_bytes = self.fernet.decrypt(token_bytes)
        # Retorna o texto decodificado para string
        return texto_bytes.decode()
    
    def nova_chave(self) -> str:
        """
        Gera uma nova chave para o Fernet e a retorna como string.
        """
        chave = Fernet.generate_key()  # chave gerada em bytes
        # Retorna a chave decodificada para string
        return chave.decode()

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv('.env', override=True)    
    
    cr = UtilCriptografia()
    texto = 'esse é um texto para teste de criptografia de dados 123 de oliveira 4'
    texto_c = cr.criptografar(texto)
    texto_d = cr.decriptografar(texto_c)
    print('ORIGINAL:', texto)
    print('CRIPTOGRAFADO:', texto_c)
    print('DECRIPTOGRAFADO:', texto_d)