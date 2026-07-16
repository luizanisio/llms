#!/bin/bash

# Compacta todas as subpastas da pasta atual em um único arquivo zip,
# ignorando qualquer pasta chamada 'chkpt' e seu conteúdo.

NOME_ZIP="treinos_lite.zip"

echo "Iniciando compactação no diretório: $(pwd)"

# Remove o arquivo zip se ele já existir para não adicionar aos arquivos antigos
if [ -f "$NOME_ZIP" ]; then
    echo "Removendo arquivo $NOME_ZIP antigo..."
    rm "$NOME_ZIP"
fi

# Lista de pastas no diretório atual
pastas=(*/)

# Verifica se encontrou alguma pasta
if [ "${#pastas[@]}" -eq 0 ] || [ "${pastas[0]}" = "*/" ]; then
    echo "Nenhuma pasta encontrada no diretório atual."
    exit 0
fi

echo "Compactando pastas em $NOME_ZIP (ignorando pastas chkpt)..."

# Usa o zip com o parâmetro -r (recursivo) e -x para excluir pastas chkpt
# A expressão */chkpt/* exclui o conteúdo da pasta, e */chkpt exclui a própria pasta
zip -r "$NOME_ZIP" "${pastas[@]}" -x "*/chkpt/*" -x "*/chkpt"

echo "Compactação concluída!"
 