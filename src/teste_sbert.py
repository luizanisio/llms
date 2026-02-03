
import unittest
import numpy as np
import sys
from unittest.mock import MagicMock, patch
from util_sbert import BERTScoreLike

# Detecta flag "mock" nos argumentos

MODELO = "pequeno"
MOCK_TEST = False
_args_to_remove = []
for _arg in sys.argv:
    if _arg.lower() == "mock":
        MOCK_TEST = True
        _args_to_remove.append(_arg)
    if _arg.lower() == "pequeno":
        MODELO = "pequeno"
        _args_to_remove.append(_arg)
    if _arg.lower() == "medio":
        MODELO = "medio"
        _args_to_remove.append(_arg)
    if _arg.lower() == "grande":
        MODELO = "grande"
        _args_to_remove.append(_arg)
if MOCK_TEST:
    MODELO = 'pequeno'
# Remove o argumento para não confundir o unittest
for _arg in _args_to_remove:
    sys.argv.remove(_arg)

class TestBERTScoreLike(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Inicializa o modelo apenas uma vez para toda a classe de teste.
        Se MOCK_TEST=True, aplica o patch antes de inicializar.
        """
        if MOCK_TEST:
            # Patch na classe SentenceTransformer dentro do módulo util_sbert
            cls.patcher = patch('util_sbert.SentenceTransformer')
            cls.MockSentenceTransformer = cls.patcher.start()
            
            # Configura o comportamento do mock
            cls.mock_model_instance = MagicMock()
            cls.MockSentenceTransformer.return_value = cls.mock_model_instance
            cls.mock_model_instance.encode.side_effect = cls._mock_encode
        
        print("Inicializando modelo SBERT para testes (uma única vez)...")
        # Instancia a classe alvo com o modelo "pequeno"
        cls.bs = BERTScoreLike(modelo=MODELO)

    @classmethod
    def tearDownClass(cls):
        if MOCK_TEST:
            cls.patcher.stop()

    @staticmethod
    def _mock_encode(sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        """
        Gera embeddings controlados.
        """
        vecs = []
        for text in sentences:
            if not text:
                vecs.append([0.0, 0.0, 0.0])
                continue
            
            t = text.strip()
            
            # Lógica de match por tokens explícitos
            is_a = "TOKEN_A" in t or "O céu é azul" in t
            is_b = "TOKEN_B" in t or "A grama é verde" in t 
            is_c = "TOKEN_C" in t
            
            v = [0.0, 0.0, 0.0]
            if is_a: v[0] = 1.0
            if is_b: v[1] = 1.0
            if is_c: v[2] = 1.0
            
            # Caso misto
            if is_a and is_b: 
                v[0] = 1.0
                v[1] = 1.0
            
            if sum(v) == 0:
                v = [0.0001, 0.0001, 0.0001]

            vecs.append(np.array(v, dtype=np.float32))
            
        arr = np.array(vecs)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            
        return arr

    def test_textos_identicos(self):
        """Testa match perfeito (P=1, R=1, F1=1)."""
        if MOCK_TEST:
            texto = "TOKEN_A. TOKEN_A."
        else:
            texto = "O processo foi deferido. O processo foi deferido."

        res = self.bs.comparar_textos(texto, texto, unitizador="sentencas")
        
        # Para textos idênticos, o modelo real também deve dar 1.0 (pois cos(x,x)=1)
        self.assertAlmostEqual(res['P'], 1.0, places=4)
        self.assertAlmostEqual(res['R'], 1.0, places=4)
        self.assertAlmostEqual(res['F1'], 1.0, places=4)

    def test_texto_semelhante_subset(self):
        """
        Candidato: "A"
        Referência: "A. B"
        """
        if MOCK_TEST:
            cand = "TOKEN_A"
            ref = "TOKEN_A. TOKEN_B" 
        else:
            # Usando palavras-chave para minimizar similaridade sintática
            cand = "LEI"
            ref = "LEI. BOLO"

        res = self.bs.comparar_textos(cand, ref, unitizador="sentencas")
        
        if MOCK_TEST:
            self.assertAlmostEqual(res['P'], 1.0, places=4, msg="MOCK Prec 1.0")
            self.assertAlmostEqual(res['R'], 0.5, places=4, msg="MOCK Rec 0.5")
        else:
            # Modelo Real:
            # Precision: "LEI" bate com "LEI" -> 1.0
            # Recall: "LEI" (1.0) e "BOLO" vs "LEI" (baixo).
            # Se o modelo grande for muito "generoso", a similaridade entre LEI e BOLO pode ser ~0.7-0.8.
            # R = (1.0 + 0.8)/2 = 0.9.
            # Vamos aceitar até 0.92, garantindo que não é match perfeito.
            self.assertGreater(res['P'], 0.90, msg="Real P deve ser muito alto pois cand está contido")
            self.assertLess(res['R'], 0.93, msg="Real R deve ser menor que 0.93 pois falta parte do texto")
            self.assertGreater(res['R'], 0.40, msg="Real R não deve ser zero")

    def test_texto_semelhante_superset(self):
        if MOCK_TEST:
            cand = "TOKEN_A. TOKEN_B"
            ref = "TOKEN_A"
        else:
            cand = "LEI. BOLO"
            ref = "LEI"

        res = self.bs.comparar_textos(cand, ref, unitizador="sentencas")

        if MOCK_TEST:
            self.assertAlmostEqual(res['P'], 0.5, places=4)
            self.assertAlmostEqual(res['R'], 1.0, places=4)
        else:
            self.assertGreater(res['R'], 0.90, msg="Real R deve ser alto pois cobre tudo")
            self.assertLess(res['P'], 0.93, msg="Real P deve ser baixo pois tem texto extra irrelevante")

    def test_json_planificado(self):
        """
        Testa a comparação de JSONs.
        """
        if MOCK_TEST:
            j1 = {"chave1": "TOKEN_A"}
            j2 = {"chave1": "TOKEN_A", "chave2": "TOKEN_B"}
        else:
            # Usar valores reais muito distintos
            j1 = {"area": "juridica"}
            j2 = {"area": "juridica", "sobremesa": "pudim"}
        
        res = self.bs.comparar_json(j1, j2, include_key_ctx=True)
        
        if MOCK_TEST:
            self.assertAlmostEqual(res['P'], 1.0, places=4)
            self.assertAlmostEqual(res['R'], 0.5, places=4)
        else:
            # Precision deve ser muito alta (subset quase perfeito)
            self.assertGreater(res['P'], 0.85)
            # Recall deve cair bem
            self.assertLess(res['R'], 0.95)
        
        detalhes = res['detalhes']
        self.assertEqual(detalhes['n_campos_cand'], 1)
        self.assertEqual(detalhes['n_campos_ref'], 2)

    def test_threshold(self):
        """
        Testa o corte por threshold.
        """
        if MOCK_TEST:
            cand = "TOKEN_A TOKEN_B" # vetor misto [1,1,0] -> sim 0.707 com TOKEN_A [1,0,0]
            ref = "TOKEN_A"
            thr_val = 0.8
        else:
            # Frases com similaridade média/baixa
            cand = "Eu gosto de maçã e banana."
            ref = "O trânsito está engarrafado."
            thr_val = 0.8
        
        res_no_thr = self.bs.comparar_textos(cand, ref, threshold=None)
        p_score = res_no_thr['P']
        
        # Define um threshold acima do score obtido para garantir que zera
        # Se for muito baixo, aumentamos
        usar_thr = max(p_score + 0.1, 0.5)
        if usar_thr > 0.95: usar_thr = 0.99
        
        res_thr = self.bs.comparar_textos(cand, ref, threshold=usar_thr)
        
        if MOCK_TEST:
            self.assertEqual(res_thr['P'], 0.0)
        else:
            # Só podemos garantir zerar se o threshold for factível (< 1.0)
            if usar_thr < 1.0:
                self.assertEqual(res_thr['P'], 0.0, f"Deveria zerar com thr {usar_thr} (score original foi {p_score})")

    def test_input_vazio(self):
        res = self.bs.comparar_textos("", "A")
        self.assertEqual(res['P'], 0.0)
        self.assertEqual(res['R'], 0.0)
        
        res_json = self.bs.comparar_json({}, {"a": "A"})
        self.assertEqual(res_json['P'], 0.0)
        self.assertEqual(res_json['R'], 0.0)


if __name__ == '__main__':
    unittest.main()
