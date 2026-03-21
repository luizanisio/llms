## Dados dos modelos

warmup_steps: 5 (para fold11)


### Divisão fold 11
- máximo de tokens (qwen 3B): 
  - fácil: 28338 --> 28*1024 + 128 = 28784
  - médio: 29285 --> 28*1024 + 512 = 29184
  - difícil: 32953 --> 35*1024  = 35840
  
RTX3060 
- max_seq_length = 3072
- batch_size = 1

H100 x 2
- max_seq_length = 35840
- batch_size = 8

  
  