from util_sbert import BERTScoreLike
from util_print import print_cores
import time

def main():
    modelos_nomes = ['pequeno', 'medio', 'grande', 'stjiris']
    modelos = {}
    
    print_cores("\n<azul>=== CARREGANDO MODELOS ===</azul>")
    for nome in modelos_nomes:
        alias = 'stjiris/bert-large-portuguese-cased-legal-mlm-mkd-nli-sts-v1' if nome == 'stjiris' else nome
        t0 = time.time()
        print_cores(f"<cinza>Carregando modelo SBERT [{nome}]...</cinza>")
        modelos[nome] = BERTScoreLike(modelo=alias)
        print_cores(f"<verde>Modelo [{nome}] carregado em {time.time()-t0:.1f}s.</verde>")

    textos = [
        ("A decisão foi reformada.", "O acórdão foi modificado."), 
        ("O recurso especial não foi conhecido no Tribunal por ausência de prequestionamento.", 
         "O STJ não conheceu do recurso especial em razão da falta de prequestionamento."), 
        ("O recurso especial não foi conhecido no Tribunal por ausência de prequestionamento.", 
         "O recurso especial foi conhecido e provido, afastando a alegada ausência de prequestionamento."),
        # Novos exemplos demonstrando o domínio jurídico:
        ("Agravo regimental não provido.", 
         "Agravo interno desprovido."), # Sinônimos jurídicos fortes no STJ
        ("Habeas corpus concedido de ofício para anular a condenação.", 
         "Ordem concedida ex officio determinando a nulidade do decreto condenatório."), # Jargão em latim e termos técnicos equivalentes
        ("Ação julgada procedente.", 
         "Ação julgada improcedente.") # Mudança de apenas um prefixo que inverte o resultado jurídico
    ]
    
    print_cores("\n<azul>=== TESTE DE COMPARAÇÃO DE TEXTOS ===</azul>")
    for i, texto in enumerate(textos, 1):
        p1, p2 = texto
        print_cores(f"\n<amarelo>Par {i}:</amarelo>")
        print_cores(f"  <cinza>Texto 1:</cinza> {p1}")
        print_cores(f"  <cinza>Texto 2:</cinza> {p2}")
        print_cores("  " + "-" * 50)
        print_cores(f"  {'Modelo':<15} | {'P':<6} | {'R':<6} | {'F1':<6}")
        print_cores("  " + "-" * 50)
        
        for nome in modelos_nomes:
            m = modelos[nome]
            resultado = m.comparar_textos(p1, p2, metodo="bertscore_like", threshold=0.50)
            f1, p, r = resultado.get('F1', 0), resultado.get('P', 0), resultado.get('R', 0)
            cor = "verde" if f1 >= 0.7 else "vermelho"
            print_cores(f"  {nome:<15} | <{cor}>{p:.3f}</{cor}> | <{cor}>{r:.3f}</{cor}> | <{cor}>{f1:.3f}</{cor}>")

    print_cores("\n<azul>=== TESTE DE COMPARAÇÃO DE JSON ===</azul>")
    testes_json = [
        (
            "Robusto a troca de chave:",
            {"decisao": "negou provimento", "fundamento": "ausência de prova"},
            {"fundamento": "não havia prova suficiente", "decisao": "provimento negado"}
        ),
        (
            "Exemplo com Precision alto e Recall baixo:",
            {"decisao": "negou provimento", "fundamento": "ausência de prova"},
            {"decisao": "negou provimento"}
        ),
        (
            "Exemplo com Precision baixo e Recall alto:",
            {"decisao": "negou provimento", "fundamento": "ausência de prova"},
            {"decisao": "negou provimento", "fundamento": "ausência de prova", "outro": "outro"}
        )
    ]

    for desc, gold, pred in testes_json:
        print_cores(f"\n<amarelo>{desc}</amarelo>")
        print_cores(f"  <cinza>Gold:</cinza> {gold}")
        print_cores(f"  <cinza>Pred:</cinza> {pred}")
        print_cores("  " + "-" * 50)
        print_cores(f"  {'Modelo':<15} | {'P':<6} | {'R':<6} | {'F1':<6}")
        print_cores("  " + "-" * 50)
        
        for nome in modelos_nomes:
            m = modelos[nome]
            resultado = m.comparar_json(pred, gold, include_key_ctx=True, threshold=0.65)
            f1, p, r = resultado.get('F1', 0), resultado.get('P', 0), resultado.get('R', 0)
            cor = "verde" if f1 >= 0.7 else "vermelho"
            print_cores(f"  {nome:<15} | <{cor}>{p:.3f}</{cor}> | <{cor}>{r:.3f}</{cor}> | <{cor}>{f1:.3f}</{cor}>")

if __name__ == '__main__':
    main()
