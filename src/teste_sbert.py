# Autor: Luiz Anísio
# Fonte: https://github.com/luizanisio/llms/tree/main/src


import unittest
import numpy as np
import sys
import os
import json
import tempfile
import shutil
import threading
import concurrent.futures
from unittest.mock import MagicMock, patch
from util_sbert import BERTScoreLike, SBERTCache, sbert_score

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


class TestBERTScoreLikeGetInstance(unittest.TestCase):
    """
    Testes para o método get_instance (singleton thread-safe).
    Verifica que a mesma instância é retornada e que funciona em ambiente multi-threaded.
    """
    
    @classmethod
    def setUpClass(cls):
        """Limpa o cache de instâncias antes dos testes."""
        BERTScoreLike.clear_instances()
        
        if MOCK_TEST:
            cls.patcher = patch('util_sbert.SentenceTransformer')
            cls.MockSentenceTransformer = cls.patcher.start()
            cls.mock_model_instance = MagicMock()
            cls.MockSentenceTransformer.return_value = cls.mock_model_instance
            cls.mock_model_instance.encode.side_effect = TestBERTScoreLike._mock_encode
    
    @classmethod
    def tearDownClass(cls):
        """Limpa o cache após os testes."""
        BERTScoreLike.clear_instances()
        if MOCK_TEST:
            cls.patcher.stop()
    
    def setUp(self):
        """Limpa o cache antes de cada teste."""
        BERTScoreLike.clear_instances()
    
    def test_get_instance_singleton(self):
        """Verifica que get_instance retorna a mesma instância para o mesmo modelo."""
        print("\n[Singleton] Testando se get_instance retorna mesma instância...")
        
        inst1 = BERTScoreLike.get_instance(MODELO)
        inst2 = BERTScoreLike.get_instance(MODELO)
        
        self.assertIs(inst1, inst2, "get_instance deve retornar a mesma instância")
        print(f"[Singleton] OK - Mesma instância: {id(inst1)} == {id(inst2)}")
    
    def test_get_instance_different_models(self):
        """Verifica que modelos diferentes têm instâncias diferentes."""
        print("\n[Modelos Diferentes] Testando instâncias separadas por modelo...")
        
        inst_pequeno = BERTScoreLike.get_instance("pequeno")
        inst_medio = BERTScoreLike.get_instance("medio")
        
        self.assertIsNot(inst_pequeno, inst_medio, 
                         "Modelos diferentes devem ter instâncias diferentes")
        print(f"[Modelos Diferentes] OK - pequeno: {id(inst_pequeno)}, medio: {id(inst_medio)}")
    
    def test_get_instance_thread_safety(self):
        """
        Verifica thread-safety: múltiplas threads chamando get_instance
        simultaneamente devem obter a mesma instância.
        """
        print("\n[Thread-Safety] Testando acesso concorrente ao get_instance...")
        
        instancias = []
        erros = []
        num_threads = 10
        barrier = threading.Barrier(num_threads)
        
        def worker():
            try:
                barrier.wait()  # Sincroniza todas as threads para começar juntas
                inst = BERTScoreLike.get_instance(MODELO)
                instancias.append(inst)
            except Exception as e:
                erros.append(str(e))
        
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(erros), 0, f"Não deve haver erros: {erros}")
        self.assertEqual(len(instancias), num_threads)
        
        # Todas as instâncias devem ser a mesma
        primeira = instancias[0]
        for i, inst in enumerate(instancias[1:], start=2):
            self.assertIs(inst, primeira, 
                         f"Thread {i} obteve instância diferente!")
        
        print(f"[Thread-Safety] OK - {num_threads} threads obtiveram a mesma instância: {id(primeira)}")
    
    def test_comparar_textos_com_threads(self):
        """
        Executa comparações de texto em paralelo usando get_instance.
        Verifica que os resultados são consistentes.
        """
        print("\n[Threads - Textos] Testando comparar_textos em paralelo...")
        
        if MOCK_TEST:
            textos = [
                ("TOKEN_A. TOKEN_A.", "TOKEN_A. TOKEN_A."),
                ("TOKEN_A", "TOKEN_A. TOKEN_B"),
                ("TOKEN_A. TOKEN_B", "TOKEN_A"),
            ]
        else:
            textos = [
                ("O processo foi deferido.", "O processo foi deferido."),
                ("LEI", "LEI. BOLO"),
                ("LEI. BOLO", "LEI"),
            ]
        
        def comparar_worker(cand, ref):
            bs = BERTScoreLike.get_instance(MODELO)
            return bs.comparar_textos(cand, ref, unitizador="sentencas")
        
        # Executa em paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            # Submete cada par várias vezes
            for _ in range(3):
                for cand, ref in textos:
                    futures.append(executor.submit(comparar_worker, cand, ref))
            
            resultados = [f.result() for f in futures]
        
        # Verifica que todos têm as chaves esperadas
        for res in resultados:
            self.assertIn('P', res)
            self.assertIn('R', res)
            self.assertIn('F1', res)
        
        # Verifica consistência: mesmos inputs devem dar mesmos outputs
        resultados_por_par = {}
        idx = 0
        for _ in range(3):
            for cand, ref in textos:
                chave = (cand, ref)
                if chave not in resultados_por_par:
                    resultados_por_par[chave] = []
                resultados_por_par[chave].append(resultados[idx])
                idx += 1
        
        for chave, lista_res in resultados_por_par.items():
            primeiro_f1 = lista_res[0]['F1']
            for res in lista_res[1:]:
                self.assertAlmostEqual(res['F1'], primeiro_f1, places=6,
                    msg=f"Resultados inconsistentes para {chave[:30]}...")
        
        print(f"[Threads - Textos] OK - {len(resultados)} comparações em paralelo com resultados consistentes")
    
    def test_comparar_json_com_threads(self):
        """
        Executa comparações de JSON em paralelo usando get_instance.
        """
        print("\n[Threads - JSON] Testando comparar_json em paralelo...")
        
        if MOCK_TEST:
            jsons = [
                ({"chave1": "TOKEN_A"}, {"chave1": "TOKEN_A", "chave2": "TOKEN_B"}),
                ({"a": "TOKEN_A", "b": "TOKEN_B"}, {"a": "TOKEN_A", "b": "TOKEN_B"}),
                ({"x": "TOKEN_C"}, {"y": "TOKEN_A"}),
            ]
        else:
            jsons = [
                ({"area": "juridica"}, {"area": "juridica", "sobremesa": "pudim"}),
                ({"nome": "teste", "valor": "abc"}, {"nome": "teste", "valor": "abc"}),
                ({"campo1": "valor diferente"}, {"campo2": "outro valor"}),
            ]
        
        def comparar_json_worker(j1, j2):
            bs = BERTScoreLike.get_instance(MODELO)
            return bs.comparar_json(j1, j2, include_key_ctx=True)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for _ in range(3):
                for j1, j2 in jsons:
                    futures.append(executor.submit(comparar_json_worker, j1, j2))
            
            resultados = [f.result() for f in futures]
        
        for res in resultados:
            self.assertIn('P', res)
            self.assertIn('R', res)
            self.assertIn('F1', res)
            self.assertIn('detalhes', res)
        
        print(f"[Threads - JSON] OK - {len(resultados)} comparações JSON em paralelo")
    
    def test_stress_threads(self):
        """
        Teste de stress: muitas threads fazendo muitas comparações.
        """
        print("\n[Stress] Testando carga alta com múltiplas threads...")
        
        num_threads = 20
        comparacoes_por_thread = 5
        
        if MOCK_TEST:
            texto_base = "TOKEN_A. TOKEN_B."
        else:
            texto_base = "O recurso foi provido parcialmente. A decisão foi reformada."
        
        resultados = []
        erros = []
        
        def stress_worker(thread_id):
            try:
                bs = BERTScoreLike.get_instance(MODELO)
                for i in range(comparacoes_por_thread):
                    res = bs.comparar_textos(texto_base, texto_base)
                    resultados.append((thread_id, i, res['F1']))
            except Exception as e:
                erros.append((thread_id, str(e)))
        
        threads = [threading.Thread(target=stress_worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(erros), 0, f"Erros no teste de stress: {erros}")
        
        total_esperado = num_threads * comparacoes_por_thread
        self.assertEqual(len(resultados), total_esperado,
            f"Esperado {total_esperado} resultados, obteve {len(resultados)}")
        
        # Todos os F1 devem ser iguais (texto idêntico -> 1.0)
        f1_values = [r[2] for r in resultados]
        for f1 in f1_values:
            self.assertAlmostEqual(f1, 1.0, places=4,
                msg="Texto idêntico deve ter F1=1.0")
        
        print(f"[Stress] OK - {total_esperado} comparações em {num_threads} threads sem erros")


class TestSBERTCache(unittest.TestCase):
    """
    Testes para SBERTCache e sbert_score com cache em disco.
    Usa diretório temporário para não afetar caches reais.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix='sbert_cache_test_')
        if MOCK_TEST:
            cls.patcher = patch('util_sbert.SentenceTransformer')
            cls.MockST = cls.patcher.start()
            cls.mock_inst = MagicMock()
            cls.MockST.return_value = cls.mock_inst
            cls.mock_inst.encode.side_effect = TestBERTScoreLike._mock_encode
        # Garante instância carregada
        BERTScoreLike.clear_instances()
        cls.bs = BERTScoreLike.get_instance(MODELO)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)
        BERTScoreLike.clear_instances()
        if MOCK_TEST:
            cls.patcher.stop()

    def _make_cache(self, **kwargs):
        return SBERTCache(modelo=MODELO, cache_dir=self._tmpdir, **kwargs)

    # ── Testes unitários do SBERTCache ────────────────────────────────────

    def test_cache_miss_e_save(self):
        """get_batch retorna miss, save_batch grava arquivo, segundo get_batch acerta."""
        cache = self._make_cache()
        preds = ['Texto A']
        trues = ['Texto B']

        P, R, F1, mi, mp, mt, mm = cache.get_batch(preds, trues)
        self.assertEqual(len(mi), 1, "Primeiro acesso deve ser miss")
        self.assertIsNone(P[0])

        # Simula resultado e salva
        cache.save_batch(mm, [0.8], [0.7], [0.75])

        # Segundo acesso deve acertar
        P2, R2, F12, mi2, _, _, _ = cache.get_batch(preds, trues)
        self.assertEqual(len(mi2), 0, "Segundo acesso deve ser hit")
        self.assertAlmostEqual(P2[0], 0.8)
        self.assertAlmostEqual(R2[0], 0.7)
        self.assertAlmostEqual(F12[0], 0.75)

    def test_cache_ordem_invertida_troca_PR(self):
        """(A,B) e (B,A) usam o mesmo arquivo; P e R são trocados."""
        cache = self._make_cache()

        cache_ab = self._make_cache()
        _, _, _, _, _, _, mm = cache_ab.get_batch(['A'], ['B'])
        cache_ab.save_batch(mm, [0.9], [0.6], [0.72])

        # Busca com ordem invertida
        P, R, F1, mi, _, _, _ = cache.get_batch(['B'], ['A'])
        self.assertEqual(len(mi), 0, "Deve achar cache da ordem inversa")
        # P e R devem estar trocados
        self.assertAlmostEqual(P[0], 0.6, msg="P(B,A) == R(A,B)")
        self.assertAlmostEqual(R[0], 0.9, msg="R(B,A) == P(A,B)")
        self.assertAlmostEqual(F1[0], 0.72)

    def test_cache_validacao_bytes(self):
        """Arquivo com bytes errados é rejeitado (simula colisão de hash)."""
        cache = self._make_cache()
        preds = ['Colisao X']
        trues = ['Colisao Y']

        _, _, _, _, _, _, mm = cache.get_batch(preds, trues)
        cache.save_batch(mm, [0.5], [0.5], [0.5])

        # Corrompe bytes1 no arquivo
        fp = mm[0]['filepath']
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['bytes1'] = 99999
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        P, R, F1, mi, _, _, _ = cache.get_batch(preds, trues)
        self.assertEqual(len(mi), 1, "Bytes corrompidos devem causar miss")
        self.assertIsNone(P[0])

    def test_cache_usar_cache_false(self):
        """Com usar_cache=False tudo é miss mesmo com arquivo presente."""
        cache_w = self._make_cache()
        preds = ['UC False']
        trues = ['Teste']

        _, _, _, _, _, _, mm = cache_w.get_batch(preds, trues)
        cache_w.save_batch(mm, [0.1], [0.2], [0.15])

        cache_no = self._make_cache(usar_cache=False)
        P, R, F1, mi, _, _, _ = cache_no.get_batch(preds, trues)
        self.assertEqual(len(mi), 1)

    def test_cache_atualizar_cache_false(self):
        """Com atualizar_cache=False, save_batch não grava nada."""
        cache = self._make_cache(atualizar_cache=False)
        preds = ['No Save']
        trues = ['Test']

        _, _, _, _, _, _, mm = cache.get_batch(preds, trues)
        cache.save_batch(mm, [0.3], [0.4], [0.35])

        # Arquivo não deve existir
        self.assertFalse(os.path.exists(mm[0]['filepath']))

    def test_cache_batch_misto(self):
        """Batch com um par cacheado e outro não."""
        cache = self._make_cache()

        # Grava primeiro par
        _, _, _, _, _, _, mm = cache.get_batch(['Hit'], ['Par'])
        cache.save_batch(mm, [0.9], [0.8], [0.85])

        # Batch com par existente + par novo
        P, R, F1, mi, mp, mt, mm2 = cache.get_batch(['Hit', 'Miss'], ['Par', 'Novo'])
        self.assertEqual(len(mi), 1, "Apenas o segundo par deve ser miss")
        self.assertEqual(mi[0], 1)
        self.assertAlmostEqual(P[0], 0.9)
        self.assertIsNone(P[1])

    def test_limpar_cache(self):
        """limpar_cache remove todos os arquivos."""
        cache = self._make_cache()
        _, _, _, _, _, _, mm = cache.get_batch(['Limpar'], ['Teste'])
        cache.save_batch(mm, [0.5], [0.5], [0.5])

        n = cache.limpar_cache(verbose=False)
        self.assertGreaterEqual(n, 1)

        # Confirma que o diretório está vazio
        jsons = [f for f in os.listdir(cache.cache_dir) if f.endswith('.json')]
        self.assertEqual(len(jsons), 0)

    # ── Testes de integração com sbert_score ──────────────────────────────

    def test_sbert_score_popula_cache(self):
        """sbert_score grava no cache; segunda chamada retorna do cache sem recalcular."""
        # Aponta env para pasta temporária
        old_env = os.environ.get('SBERT_CACHE_PATH')
        os.environ['SBERT_CACHE_PATH'] = self._tmpdir
        try:
            preds = ['Frase de teste alpha.']
            trues = ['Frase de referência alpha.']

            P1, R1, F11 = sbert_score(preds, trues, modelo=MODELO, decimais=3, verbose=False)
            self.assertEqual(len(P1), 1)
            self.assertGreater(F11[0], 0.0)

            # Segunda chamada — deve vir do cache e retornar os mesmos valores
            P2, R2, F12 = sbert_score(preds, trues, modelo=MODELO, decimais=3, verbose=False)
            self.assertAlmostEqual(P1[0], P2[0], places=5)
            self.assertAlmostEqual(R1[0], R2[0], places=5)
            self.assertAlmostEqual(F11[0], F12[0], places=5)

            # Verifica que arquivo existe no disco
            cache = SBERTCache(modelo=MODELO)
            info = cache._get_key_info(preds[0], trues[0])
            self.assertTrue(os.path.exists(info['filepath']),
                            "Arquivo de cache deve existir após sbert_score")
        finally:
            if old_env is None:
                os.environ.pop('SBERT_CACHE_PATH', None)
            else:
                os.environ['SBERT_CACHE_PATH'] = old_env

    def test_sbert_score_sem_cache_recalcula(self):
        """usar_cache=False força recálculo mas ainda salva se atualizar_cache=True."""
        old_env = os.environ.get('SBERT_CACHE_PATH')
        os.environ['SBERT_CACHE_PATH'] = self._tmpdir
        try:
            preds = ['Recalculo forçado.']
            trues = ['Referência recalculo.']

            P1, R1, F11 = sbert_score(preds, trues, modelo=MODELO, decimais=3,
                                       usar_cache=False, atualizar_cache=True)
            self.assertGreater(F11[0], 0.0)

            # Agora com cache habilitado deve retornar o mesmo
            P2, R2, F12 = sbert_score(preds, trues, modelo=MODELO, decimais=3,
                                       usar_cache=True)
            self.assertAlmostEqual(F11[0], F12[0], places=5)
        finally:
            if old_env is None:
                os.environ.pop('SBERT_CACHE_PATH', None)
            else:
                os.environ['SBERT_CACHE_PATH'] = old_env

    def test_sbert_score_validacoes(self):
        """Testa erros de validação de entrada."""
        with self.assertRaises(TypeError):
            sbert_score("não lista", ["b"])
        with self.assertRaises(ValueError):
            sbert_score(["a"], ["b", "c"])


if __name__ == '__main__':
    unittest.main()
