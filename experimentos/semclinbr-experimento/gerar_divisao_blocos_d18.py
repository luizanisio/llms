#!/usr/bin/env python3
"""
Gera o CSV de divisão com blocos aleatórios usado pelo protocolo D18.

D18 é o *controle* do D17: mantém exatamente a mesma topologia de etapas
(3 recortes + conjunto completo, 2 épocas cada), trocando apenas o critério
de recorte — dificuldade (D17) por blocos aleatórios (D18). O contraste
D18 vs D17 separa o efeito da *ordenação por dificuldade* do simples efeito
de treinar em blocos com recência.

Por isso os blocos são terços aleatórios: mesma granulometria dos recortes
por dificuldade, igualando o número de instâncias entre os dois protocolos
(2N sobre os recortes + 2N sobre o completo = 4N em ambos).

O CSV de origem NÃO é modificado — a saída é um arquivo novo com a coluna
'bloco' acrescentada. Seed fixa (3407, a mesma dos YAMLs de treinamento)
para que a atribuição seja reprodutível.

Uso:
    python gerar_divisao_blocos_d18.py
"""

import os
import numpy as np
import pandas as pd

SEED = 3407  # mesma seed dos YAMLs de treinamento
BLOCOS = ["b1", "b2", "b3"]

_BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_BASE, "dados", "divisao_Gold_Qwen7B.csv")
DST = os.path.join(_BASE, "dados", "divisao_Gold_Qwen7B_blocos.csv")


def main() -> None:
    df = pd.read_csv(SRC)

    if "bloco" in df.columns:
        raise SystemExit(f"❌ A coluna 'bloco' já existe em {SRC}")

    # np.resize cicla b1,b2,b3,... até len(df), garantindo terços de tamanho
    # máximo equilibrado (diferença de no máximo 1 linha entre blocos);
    # a permutação embaralha a atribuição mantendo esses tamanhos exatos.
    rng = np.random.default_rng(SEED)
    df["bloco"] = rng.permutation(np.resize(np.array(BLOCOS), len(df)))

    df.to_csv(DST, index=False)

    print(f"✅ {len(df):,} linhas → {DST}")
    print("\nTamanho dos blocos:")
    print(df["bloco"].value_counts().sort_index().to_string())
    # Os blocos recortam também o conjunto de validação da etapa (comportamento
    # idêntico ao filtro por dificuldade do D17); a conferência abaixo mostra
    # que a proporção treino/validação/teste se mantém em cada bloco.
    print("\nDistribuição por alvo:")
    print(pd.crosstab(df["bloco"], df["alvo"]).to_string())
    print("\nDistribuição por dificuldade (deve ser ~uniforme: blocos são aleatórios):")
    print(pd.crosstab(df["bloco"], df["dificuldade"]).to_string())


if __name__ == "__main__":
    main()
