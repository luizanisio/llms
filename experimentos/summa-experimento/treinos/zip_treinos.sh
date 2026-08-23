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

# Lista de termos para ignorar (as pastas que contenham algum destes itens em seus nomes não serão processadas)
IGNORAR=("(b)" "(c)" "(d1)" "(d2)" "(d3)" "(d4)" "(d5)" "(d6)" "(d7)" "(d8)")

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
    
    # Verifica se a pasta contém algum termo da lista IGNORAR
    ignorar_pasta=0
    for termo in "${IGNORAR[@]}"; do
        if [[ "$nome" == *"$termo"* ]]; then
            ignorar_pasta=1
            break
        fi
    done
    
    if [ "$ignorar_pasta" -eq 1 ]; then
        echo "Pasta '$nome' contém termo ignorado. Pulando."
        continue
    fi
    
    arquivo_zip="${nome}.zip"
    
    # Verifica a existência do arquivo zip
    if [ ! -f "$arquivo_zip" ]; then
        echo "Arquivo $arquivo_zip não existe. Compactando..."
        zip -r "$arquivo_zip" "$nome"
    else
        echo "Arquivo $arquivo_zip já existe. Pulando."
    fi
done