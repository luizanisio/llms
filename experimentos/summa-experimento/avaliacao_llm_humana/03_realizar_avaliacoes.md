# Simula GPT5 como Humano
../../../src/realizar_avaliacoes.py --grupos saida_nemo:llm saida_gpth:humano saida_gpt:llm saida_glm:llm saida_mmm3:llm saida_sabia:llm --bayes

# Simula Sabiá como Humano
../../../src/realizar_avaliacoes.py --grupos saida_nemo:llm saida_gpt:llm saida_glm:llm saida_mmm3:llm saida_sabia:humano --bayes

# Avaliação Humana
../../../src/realizar_avaliacoes.py --grupos saida_nemo:llm saida_gpt:llm saida_glm:llm saida_mmm3:llm  saida_sabia:llm saida_humana:humano --bayes

## Teste com um avaliador (simulado)
../../../src/realizar_avaliacoes.py \
  --grupos saida_nemo:llm saida_gpt:llm saida_glm:llm saida_mmm3:llm saida_sabia5:llm saida_humana:humano \
  --bayes --bayes-rope 0.1579 --bayes-rope-decisao 0.0832 \
  --saida analise_parcial_1humano