
'''
 Autor Luiz Anísio 10/11/2025
 Fonte: https://github.com/luizanisio/llms/tree/main/src
 Descrição: Script para calcular BERTScore e ROUGE para pares de frases.
 
'''

try:
    from bert_score import score
    from rouge_score import rouge_scorer
except ImportError:
    print("Instale as bibliotecas necessárias: bert_score e rouge_score")
    raise ImportError("Bibliotecas não instaladas! \n💡Execute: \n   pip install bert_score rouge-score")
pares = [
    ("O gato está no telhado", "O felino está em cima da casa"),
    ("Hoje está ensolarado", "O tempo está bom"),
    ("Ele comprou um carro novo", "Ele adquiriu um veículo recente"),
    ("Vamos almoçar fora?", "Você quer comer em um restaurante?"),
    ("O avião decolou", "O pássaro voou"),
    ("Python é uma linguagem de programação.","Meu hobby favorito é pedalar aos finais de semana"),
    ('Não há o que fazer.\nO que precisas ser feito, feito será!','Não tem nada a ser feito.\nO que deve ser feito, será feito!'),
    ('A inteligência artificial está transformando o mundo.','A IA está mudando o mundo'),
    ('A vida é bela e cheia de surpresas.','A vida bela é e cheia de surpresas.'),
    ('A ordem das coisas pode mudar.\nE mudando, tudo se transforma.','E mudando, tudo se transforma.\nA ordem das coisas pode mudar.'),
]

# Separar frases de referência e de hipótese
hipoteses = [par[0] for par in pares]
referencias = [par[1] for par in pares]

# testa cuda disponível e compatível
try:
    score(['a','a'], ['a','a'], lang="pt", verbose=True, device='cuda')
    device = 'cuda'
    msg_cuda = "🚀 CUDA disponível e compatível!"
except Exception as e:
    device = 'cpu'
    msg_cuda = "🚩CUDA não disponível ou não compatível!"

print('=-'*20)
print(msg_cuda)
print('=-'*20)

# Calcular o BERTScore
P, R, F1 = score(hipoteses, referencias, lang="pt", verbose=True, device=device)

Pr, Rr, Fr = [], [], []
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Exibir os resultados
for i, (h, r) in enumerate(pares):
    print(f"\nPar {i+1}:")
    print(f"\t Hipótese:   {h}")
    print(f"\t Referência: {r}")
    print(f"\t - BERTScore F1: {F1[i].item():.4f}")
    # rouge
    scores = scorer.score(r, h)
    print(f"\t - ROUGE-1 F1: {scores['rouge1'].fmeasure:.4f}")
    print(f"\t - ROUGE-2 F1: {scores['rouge2'].fmeasure:.4f}")
    print(f"\t - ROUGE-L F1: {scores['rougeL'].fmeasure:.4f}")

print('=-'*20)
print(msg_cuda)
print('=-'*20)
