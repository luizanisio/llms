''' Teste unitário das classes UtilCkan e UtilCkanIntegra.
    Contém diversos cenários de filtros no construtor e suas respectivas
    expectativas de saída após o cruzamento de dados de espelhos e íntegras.
'''
import os
import gc
import unittest
from util_ckan import UtilCkan, UtilCkanIntegra
from threading import Lock
DOWNLOAD_DIR = None
lock_download_dir = Lock()

def get_download_dir():
    import os
    global DOWNLOAD_DIR
    if DOWNLOAD_DIR: return DOWNLOAD_DIR
    with lock_download_dir:
        if DOWNLOAD_DIR: return DOWNLOAD_DIR
        if DOWNLOAD_DIR is None:
            busca = ['../experimentos/summa-experimento/downloads_stj',
                    '../downloads_stj', '../experimentos/downloads_stj',
                    './downloads_stj']
        for b in busca:
            if os.path.exists(b):
                DOWNLOAD_DIR = b
                print(f'DOWNLOAD_DIR: {DOWNLOAD_DIR}')
                break
        if DOWNLOAD_DIR is None:
            DOWNLOAD_DIR = input('Digite o caminho para o diretório de downloads: ')
            if not DOWNLOAD_DIR:
               print('Nenhum caminho fornecido para DOWNLOAD_DIR. Saindo...')
               exit(1)
            print(f'DOWNLOAD_DIR: {DOWNLOAD_DIR}')
    return DOWNLOAD_DIR

class TestUtilCkanFiltros(unittest.TestCase):
    """Testes baseados em dicionários de entrada/saída para facilitar a manutenção."""
    
    @classmethod
    def setUpClass(cls):
        print("Atualizando cache e mapas para os anos dos testes...")
        ckan = UtilCkan(anos={'2022', '2023', '2024', '2025'}, download_dir=get_download_dir(), atualizar_cache_e_mapas=True)
        ckan.atualizar_mapas()
        print("Cache atualizado com sucesso.")

    def test_filtros_diversos(self):
        # Lista de dicionários descrevendo os cenários de teste.
        # parâmetros: argumentos passados para inicializar UtilCkan.
        # esperados : chaves/valores esperados no cruzamento ou no mapa de íntegras correspondente.
        cenarios = [
            {
                'nome': 'Filtro completo por tupla de 3 (registro, data, tipo) e seq_documento',
                'parametros': {
                    'orgaos': ['T2'],
                    'processos': [('202302829818', '20240822', 'ACÓRDÃO')],
                    'documentos': [266239985],
                },
                'esperados': {
                    'id_mapa': '202302829818.20240822.ACÓRDÃO',
                    'orgao': 'T2',
                    'ministro': 'HERMAN BENJAMIN',
                    'tem_integra': True
                }
            },
            {
                'nome': 'Filtro por tupla de 2 (registro, data) - formato com hífen',
                'parametros': {
                    'processos': [('202302829818', '2024-08-22')],
                },
                'esperados': {
                    'id_mapa': '202302829818.20240822.ACÓRDÃO',
                    'orgao': 'T2',
                    'tem_integra': True
                }
            },
            {
                'nome': 'Filtro simplificado apenas pela string do número do registro',
                'parametros': {
                    'processos': ['202302829818'],
                },
                'esperados': {
                    'id_mapa': '202302829818.20240822.ACÓRDÃO',
                    'numero_registro': '202302829818',
                    'tem_integra': True
                }
            },
            {
                'nome': 'Filtro simplificado exclusivo por seq_documento',
                'parametros': {
                    'documentos': [266239985],
                },
                'esperados': {
                    'id_mapa': '202302829818.20240822.ACÓRDÃO',
                    'tem_integra': True
                }
            },
            {
                'nome': 'Filtro por múltiplos processos de datasets distintos com formatos de data variados',
                'parametros': {
                    'processos': [
                        ('202201876389', 1656644400000), 
                        ('202400109342', '2025-11-12')
                    ],
                },
                'esperados': {
                    'id_mapa_multiplos': [
                        '202201876389.20220701.ACÓRDÃO',
                        '202400109342.20251112.ACÓRDÃO'
                    ]
                }
            },
            {
                'nome': 'Filtro por data de publicação (formato ISO)',
                'parametros': {
                    'datas': {'2024-08-22'},
                },
                'esperados': {
                    'data_publicacao': '20240822',
                }
            },
            {
                'nome': 'Filtro por data de publicação (formato DD/MM/YYYY)',
                'parametros': {
                    'datas': {'22/08/2024'},
                },
                'esperados': {
                    'data_publicacao': '20240822',
                }
            },
            {
                'nome': 'Filtro por data de publicação (formato YYYYMMDD)',
                'parametros': {
                    'datas': {'20240822'},
                },
                'esperados': {
                    'data_publicacao': '20240822',
                }
            },
            {
                'nome': 'Filtro combinado: datas + órgão',
                'parametros': {
                    'datas': {'2024-08-22'},
                    'orgaos': ['T2'],
                },
                'esperados': {
                    'data_publicacao': '20240822',
                    'orgao': 'T2',
                }
            },
            {
                'nome': 'Lógica OR (OU) entre processos e documentos',
                'parametros': {
                    'processos': ['202302829818'],
                    'documentos': [146044158],
                },
                'esperados': {
                    'id_mapa_multiplos': [
                        '202302829818.20240822.ACÓRDÃO',
                        '201703248142.20220224.ACÓRDÃO'
                    ]
                }
            },
            {
                'nome': 'Filtro de tipo_decisao restritivo deve barrar processos que não dão match',
                'parametros': {
                    'processos': ['REsp 2045705'],
                    'tipos_decisao': 'acordao'
                },
                'esperados': {
                    'vazio': True
                }
            },
        ]
        
        for cenario in cenarios:
            ckan = None
            resultado = None
            try:
                with self.subTest(cenario=cenario['nome']):
                    # A classe utilizará a restrição imediata no mapeamento via json em disco.
                    ckan = UtilCkan(**cenario['parametros'], download_dir=get_download_dir(), atualizar_cache_e_mapas=False)
                    
                    # Executa o cruzamento limpo
                    resultado = ckan.cruzar_espelhos_integras()
                    
                    esperado = cenario['esperados']
                    
                    if esperado.get('vazio'):
                        self.assertEqual(len(resultado), 0, f"Falhou no {cenario['nome']}: esperava vazio, mas retornou {len(resultado)}.")
                        continue
                    else:
                        # Validamos que pelo menos 1 registro corresponde ao filtro
                        self.assertGreater(len(resultado), 0, f"Falhou no {cenario['nome']}: não retornou resultados.")
    
                    # Tratamento para validação de múltiplos id_mapa simultâneos
                    if 'id_mapa_multiplos' in esperado:
                        ids_retornados = [r.get('id_mapa') for r in resultado]
                        for id_esperado in esperado['id_mapa_multiplos']:
                            self.assertIn(id_esperado, ids_retornados, f"ID {id_esperado} não retornado na busca de múltiplos itens.")
                    else:    
                        # Pega a primeira ocorrência do cruzamento para testes de item único
                        res_dict = resultado[0]
                        
                        # Checagens comuns resultantes
                        if 'id_mapa' in esperado:
                            self.assertEqual(res_dict.get('id_mapa'), esperado['id_mapa'])
                        if 'orgao' in esperado:
                            self.assertEqual(res_dict.get('orgao'), esperado['orgao'])
                        if 'numero_registro' in esperado:
                            self.assertEqual(res_dict.get('numero_registro'), esperado['numero_registro'])
                        if 'data_publicacao' in esperado:
                            self.assertEqual(res_dict.get('data_publicacao'), esperado['data_publicacao'])
                        if 'tem_integra' in esperado:
                            self.assertEqual(res_dict.get('tem_integra'), esperado['tem_integra'])
                            
                        # Alguns dados da íntegra em específico estão no mapa secundário
                        if 'ministro' in esperado:
                            id_mapa = esperado['id_mapa']
                            integra_dict = ckan._mapa_integras.get(id_mapa, {})
                            self.assertEqual(integra_dict.get('ministro'), esperado['ministro'])
            finally:
                del ckan
                del resultado
                gc.collect()


class TestUtilCkanIntegraFiltros(unittest.TestCase):
    """Testes para UtilCkanIntegra — foco exclusivo no dataset de íntegras."""

    @classmethod
    def setUpClass(cls):
        print("Atualizando cache e mapas para os anos dos testes (UtilCkanIntegra)...")
        integra = UtilCkanIntegra(anos={'2022', '2023', '2024', '2025'}, download_dir=get_download_dir(), atualizar_cache_e_mapas=True)
        integra.atualizar_mapas()
        print("Cache atualizado com sucesso.")

    def test_filtros_diversos(self):
        """Testes parametrizados para UtilCkanIntegra com cenários variados."""
        cenarios = [
            {
                'nome': 'Cenário igual à configuração yaml (6 registros)',
                'parametros': {
                    'processos': ["REsp 2046214", "AREsp 2831077", ["202403674719", "2025-01-03"], ["REsp 2045705", "20230320", "DECISÃO"]],
                    'documentos': [289154352, 146044158],
                },
                'esperados': {
                    'qtd_esperada': 6,
                    'multiplos': ['202300025898', '202204045424', '201703248142', '202404413996', '202403674719', '202500067396']
                }
            },
            {
                'nome': 'Filtro por número de registro (string simples)',
                'parametros': {
                    'processos': {'202302829818'},
                },
                'esperados': {
                    'numero_registro': '202302829818',
                    'campos_presentes': ['seq_documento', 'ministro'],
                }
            },
            {
                'nome': 'Filtro por seq_documento',
                'parametros': {
                    'documentos': [266239985],
                },
                'esperados': {
                    'seq_documento': '266239985',
                }
            },
            {
                'nome': 'Filtro por tupla completa (registro, data, tipo)',
                'parametros': {
                    'processos': {('202302829818', '20240822', 'ACÓRDÃO')},
                },
                'esperados': {
                    'id_mapa': '202302829818.20240822.ACÓRDÃO',
                    'numero_registro': '202302829818',
                    'ministro': 'HERMAN BENJAMIN',
                }
            },
            {
                'nome': 'Filtro por tupla de 2 (registro, data) com hífen',
                'parametros': {
                    'processos': {('202302829818', '2024-08-22')},
                },
                'esperados': {
                    'id_mapa': '202302829818.20240822.ACÓRDÃO',
                }
            },
            {
                'nome': 'Filtro por data de publicação (UtilCkanIntegra)',
                'parametros': {
                    'datas': {'2024-08-22'},
                },
                'esperados': {
                    'data_publicacao': '20240822',
                }
            },
            {
                'nome': 'Filtro por data DD/MM/YYYY (UtilCkanIntegra)',
                'parametros': {
                    'datas': {'22/08/2024'},
                },
                'esperados': {
                    'data_publicacao': '20240822',
                }
            },
            {
                'nome': 'Filtro combinado OR (processos + documentos) deve retornar ambos',
                'parametros': {
                    'processos': ['202302829818'],
                    'documentos': [146044158],
                },
                'esperados': {
                    'multiplos': ['202302829818', '201703248142'] # num_registros expected
                }
            },
            {
                'nome': 'Filtro sem atualizar mapas/cache (testa dedução desabilitada)',
                'parametros': {
                    'processos': ['202302829818'],
                },
                'esperados': {
                    'numero_registro': '202302829818',
                },
                'atualizar_cache': False
            },
        ]

        for cenario in cenarios:
            integra = None
            itens = None
            try:
                with self.subTest(cenario=cenario['nome']):
                    # Test_resp showed that atualizar_cache=False works if maps exist
                    atualizar = cenario.get('atualizar_cache', False)
                    integra = UtilCkanIntegra(**cenario['parametros'], download_dir=get_download_dir(), atualizar_cache_e_mapas=atualizar)
                    itens = integra.consultar_mapa()
    
                    esperado = cenario['esperados']
                    
                    if esperado.get('vazio'):
                        self.assertEqual(len(itens), 0, f"Falhou no cenário '{cenario['nome']}': esperava vazio.")
                        continue
                    
                    self.assertGreater(len(itens), 0,
                        f"Falhou no cenário '{cenario['nome']}': não retornou resultados.")
    
                    esperado = cenario['esperados']
                    if 'multiplos' in esperado:
                        regs_retornados = [i.get('numero_registro') for i in itens]
                        if 'qtd_esperada' in esperado:
                            self.assertEqual(len(itens), esperado['qtd_esperada'], f"Falhou no cenário '{cenario['nome']}': esperava {esperado['qtd_esperada']} resultados, obteve {len(itens)}.")
                        for r_esperado in esperado['multiplos']:
                            self.assertIn(r_esperado, regs_retornados)
                        continue
    
                    item = itens[0]
    
                    if 'id_mapa' in esperado:
                        self.assertEqual(item.get('id_mapa'), esperado['id_mapa'])
                    if 'numero_registro' in esperado:
                        self.assertEqual(item.get('numero_registro'), esperado['numero_registro'])
                    if 'seq_documento' in esperado:
                        self.assertEqual(str(item.get('seq_documento')), esperado['seq_documento'])
                    if 'data_publicacao' in esperado:
                        self.assertEqual(item.get('data_publicacao'), esperado['data_publicacao'])
                    if 'ministro' in esperado:
                        self.assertEqual(item.get('ministro'), esperado['ministro'])
                    # Verifica presença de campos obrigatórios
                    for campo in esperado.get('campos_presentes', []):
                        self.assertIn(campo, item,
                            f"Campo '{campo}' ausente no resultado do cenário '{cenario['nome']}'")
            finally:
                del integra
                del itens
                gc.collect()

    def test_obter_integras(self):
        """UtilCkanIntegra.obter_integras() deve retornar textos não vazios."""
        integra = UtilCkanIntegra(
            processos={('202302829818', '20240822', 'ACÓRDÃO')},
            download_dir=get_download_dir(), atualizar_cache_e_mapas=False,
        )
        textos = integra.obter_integras()
        self.assertGreater(len(textos), 0, 'obter_integras() não retornou textos')
        for id_mapa, txt in textos.items():
            self.assertIsInstance(txt, str)
            self.assertGreater(len(txt), 0, f'Texto vazio para id_mapa={id_mapa}')

    def test_multiplos_processos(self):
        """UtilCkanIntegra deve retornar resultados para múltiplos processos."""
        integra = UtilCkanIntegra(
            processos={
                ('202201876389', 1656644400000),
                ('202302829818', '2024-08-22'),
            },
            download_dir=get_download_dir(), atualizar_cache_e_mapas=False,
        )
        itens = integra.consultar_mapa()
        self.assertGreater(len(itens), 1, 'Esperava mais de 1 resultado para múltiplos processos')

        processos_retornados = {i.get('numero_registro') for i in itens}
        self.assertIn('202201876389', processos_retornados)
        self.assertIn('202302829818', processos_retornados)


if __name__ == '__main__':
    unittest.main()

    # testes específicos:
    # python util_ckan_teste.py TestUtilCkanFiltros.test_filtros_diversos TestUtilCkanIntegraFiltros.test_filtros_diversos