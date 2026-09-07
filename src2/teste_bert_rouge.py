
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

from util_bertscore import bscore

MODELO_OVERRIDE = 'stjiris/bert-large-portuguese-cased-legal-mlm-mkd-nli-sts-v1'

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
    ('Texto muito longo '*2000, 'Texto curto para referenciar'),
    ('Texto falando de abóbora '*2000, 'Texto falando de abóboras '*2000)
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

# Calcular o BERTScore padrão (lang="pt") via bscore (com suporte a janela deslizante)
P, R, F1 = bscore(hipoteses, referencias, lang="pt", verbose=True, device=device, usar_cache=False)

# Calcular o BERTScore Jurídico (model_type=MODELO_OVERRIDE) via bscore
P_leg, R_leg, F1_leg = None, None, None
if MODELO_OVERRIDE:
    print(f"\nCalculando BERTScore com modelo personalizado: {MODELO_OVERRIDE}")
    P_leg, R_leg, F1_leg = bscore(hipoteses, referencias, model_type=MODELO_OVERRIDE, verbose=True, device=device, usar_cache=False)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Exibir os resultados
for i, (h, r) in enumerate(pares):
    h_display = h[:80] + ('...' if len(h) > 80 else '')
    r_display = r[:80] + ('...' if len(r) > 80 else '')
    print(f"\nPar {i+1}:")
    print(f"\t Hipótese:   {h_display}")
    print(f"\t Referência: {r_display}")
    print(f"\t - BERTScore (Padrão) F1:   {F1[i]:.4f}")
    if F1_leg is not None:
        print(f"\t - BERTScore (Jurídico) F1: {F1_leg[i]:.4f}")
    # rouge
    scores = scorer.score(r, h)
    print(f"\t - ROUGE-1 F1: {scores['rouge1'].fmeasure:.4f}")
    print(f"\t - ROUGE-2 F1: {scores['rouge2'].fmeasure:.4f}")
    print(f"\t - ROUGE-L F1: {scores['rougeL'].fmeasure:.4f}")

print('=-'*20)
print(msg_cuda)
print('=-'*20)

print('\n' + '=-'*20)
print("📦 Teste de Mini-Batches (BERTScore)")
print('=-'*20)
# Cria uma grande quantidade de pequenos textos para testar o processamento em mini-batches
pares_batch = [("O gato bebe leite", "O felino ingere laticínio")] * 550
hip_batch = [p[0] for p in pares_batch]
ref_batch = [p[1] for p in pares_batch]

print(f"Calculando {len(pares_batch)} pares (pequenos) para testar divisão em mini-batches...")
P_b, R_b, F1_b = bscore(hip_batch, ref_batch, lang="pt", verbose=True, device=device, usar_cache=False, mini_batch_size=100)
print(f"Sucesso! Processou {len(F1_b)} pares. F1 do primeiro: {F1_b[0]:.4f}")
print('=-'*20)
