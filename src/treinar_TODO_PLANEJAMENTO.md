# Guia de Contribuição e Arquitetura do Pipeline

## 🎯 Objetivo Deste Documento
Este guia detalha o funcionamento interno dos scripts contidos dentro do escopo `treinar_*` no núcleo raiz da iniciativa de LLMs. Ele centraliza as decisões e design de código passados, fornece uma referência rápida aos programadores sobre as responsabilidades de negócio modulares que sustentam o pipeline de inferência/treino e aponta as limitações conhecidas atreladas aos backends externos do ecossistema HuggingFace.

## 🛠️ Visão da Arquitetura Interna Padrão

O backend é estritamente Orientado a Objetos mesclando uma série de Helpers e Composers baseados em CLI para garantir modularização limpa das responsabilidades:

1. **Ações Independentes (`treinar_unsloth_actions.py`)**: Desacopla a camada CLI crua dos motores reais de fine-tuning para focar *apenas Treinar*.
2. **Avaliação Fina e Estatísticas (`treinar_unsloth_avaliar.py`)**: Roda isolado garantindo que não há vazamento do cache rico de grafos e Autogradients pesados do Unsloth que derrubam o VRAM no torch.
3. **Motores de Predição Abstratos (`treinar_realizar_predicoes.py`)**: Hierarquia `UtilPredicao` (Base ABC) com implementações estendidas concretas: `UtilPredicaoHF` e robustamente em massa com `UtilPredicaoVLLM`.
4. **Gerenciadores Automáticos (`treinar_unsloth_dataset.py`, `pipeline.py`)**: Roteadores e carregadores da malha de Curriculum. Operam criptos (Fernet), fail-fast checks por YAMLs corrompidos. `CurriculumTracker` rastreia conclusão imutável por sessões.

## 🚧 Limitações Técnicas Operacionais Conhecidas

Ao estender o uso dos pacotes e engines implementados neste escopo, preste atenção aos "Bugs Nativos de Motor" de arquiteturas acopladas de código aberto.
Eles foram atenuados e mitigados com hacks visuais no projeto atual:

### 1. Instruções "Assistente Fake" e Template vLLM Immediate EOS
Modelos com sufixo `Instruct` (e.g. Qwen2.5-7B-Instruct) se alimentados via endpoint de vLLM puros sem formatação perdem o compasso gerando EOS vazio na predição de Lotes (`output_tokens=1`). Por conta disso, todos os motores vLLM injetam a sobreposição forçada das tags (`<|im_start|>user...<|im_end|>\n<|im_start|>assistant`) antes da predição com a função `apply_chat_template`. Use a mesma manobra via SDK de inferências locais avulsas.

### 2. O Teto do QWEN em `max_position_embeddings` e Modelos em Disco (vLLM)
O vLLM barra compilações de contextos maiores que a chave declarada de treinamento original `max_position_embeddings` do `config.json` dos modelos (por exemplo, 32.768 tokens para todos QWEN-Instruct, que apesar de receber context length altos usando rope context, quebravam em start de pipeline VLLM puro pois não identificam isso via HF). A variável Global Root implementada e mitigatória em nosso pipeline é `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. Se o seu modelo em desenvolvimento base não for local (apontando de um disco via Config na raiz), a mitigação com fallback interno falhará causando crash, exigindo referências locais.

## 🚀 Próximas Funcionalidades / Roadmap (Como Extender)

As próximas features ou evoluções estruturais para novos Pesquisadores que entrarem no Hub de desenvolvimento se guiam pelos seguintes North-Stars:

- **Evolução FSDP e Motores Multi-GPU via Data Parallel Simples**
  O HuggingFace Trainer implementa nativamente DDP. No estado atual, mantivemos e forçamos o paralelismo de `device_map="sequential"` local unindo ao hack explícito `UNSLOTH_COMPILE_DISABLE=1` impedindo do Dynamo Torch causar erro global estourando exception nas Fused Cross Entropy Losses que o unsloth aplica de forma chunkada nativamente incompatível na GPU cruzada. Em próximos episódios, deve valer a pena investir em rodar e expor YAML flags para custom DeepSpeed (ZeRO) acoplada numa call `__main__` compatível para paralelismo.
- **Implementar suporte Universal Multimodal (QVL-Visuals ou LLama 3.2 Vision)**
  A malha Dataset e `Curriculum.Entrada` foram pensados ao redor de colunas String (Input/Output). O roadmap almeja que `entrada.formato_imagem` abra uma dimensão onde a API envia tensores de imagem e os motores abstratos de `treinar_realizar_predicoes.py` manipulem embeddings visuais mantendo as mesmas métricas quantitativas limpas da malha preexistente.