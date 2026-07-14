#!/bin/bash

# verifica a existência de arquivo zip para cada treino e se não existir, compacta a pasta de treino com o mesmo nome da pasta
# Exemplo zip -r "Qwen2.5-1.5B-Instruct(d8).zip" "Qwen2.5-1.5B-Instruct(d8)"
# cria uma lista e varre a lista verificando a falta do arquivo zip para poder compactar a pasta

# Rodar com screen
# Colinha screen
# - ctrl + A + D (detached
# - screen -r     (reattach)
# - screen -D -r nnnn (força reconectar)
# - screen -x nnnn (conecta sem derrubar)
# - screen -list  (lista as sessões)
# - screen -S <nome>  (nomeia a sessão atual)
# - echo $STY    (nome da sessão atual)

echo "Iniciando verificação no diretório: $(pwd)"

# Cria uma lista de pastas no diretório atual
pastas=(*/)

# Verifica se encontrou alguma pasta
if [ "${#pastas[@]}" -eq 0 ] || [ "${pastas[0]}" = "*/" ]; then
    echo "Nenhuma pasta encontrada no diretório atual."
    exit 0
fi

# Varre a lista verificando a falta do arquivo zip
for pasta in "${pastas[@]}"; do
    # Remove a barra no final
    nome="${pasta%/}"
    
    # Pula se não for um diretório válido
    [ ! -d "$nome" ] && continue
    
    arquivo_zip="${nome}.zip"
    
    # Verifica a existência do arquivo zip
    if [ ! -f "$arquivo_zip" ]; then
        echo "Arquivo $arquivo_zip não existe. Compactando..."
        zip -r "$arquivo_zip" "$nome"
    else
        echo "Arquivo $arquivo_zip já existe. Pulando."
    fi
done