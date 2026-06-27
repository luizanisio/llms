# Dicas para rodar os experimentos

## Uso do serviço tmux (similar ao screen)
O tmux é mais leve.

> tmux new -s treino	    #	nova sessão
> tmux attach -t treino     #	reconectar após queda
> tmux ls	                #	listar sessões
> Sair sem	matar: Ctrl+B	depois	D (detach).

No terminal do jupyter notebook ocorrem erros e o script setup_tmux.sh encapsula a solução criando um atalho "tm" para substituir "tmux", funcionando de forma similar. tm ls, tm attach -t treino, tm new -s treino

Opção 2: Pode-se usar o nohup também que é mais simples, mas possui menos funcionalidades. 
> nohup python util_vllm_batch.py --config 05_extracao_b_teste.yaml &
Visualizar a saída:
> tail -n 50 nohup.out
Visualizar a saída em tempo real:
> tail -f nohup.out

# Dicas de ambiente
- preparar o ambiente seguindo algum dos roteiros em requirements.txt ou requirements_sem_versao.txt
