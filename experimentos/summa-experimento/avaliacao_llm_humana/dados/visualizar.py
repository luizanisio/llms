import pandas as pd
arquivo = 'entrada_juiz_llm.parquet'
df = pd.read_parquet(arquivo)
print(df)