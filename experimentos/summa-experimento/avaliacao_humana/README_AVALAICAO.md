# Distribuição de documentos para avaliação humana

---

## 1. Divisão do Gold Set

Revisão da qualificação para incluir estratificação por **grupo de dados** e **faixa de dificuldade**.

### 1.1 Distribuição: 70 documentos

| Grupo       | Fácil | Médio | Difícil | Subtotal |
|-------------|:-----:|:-----:|:-------:|:--------:|
| Treino      |   6   |   6   |    6    |  **18**  |
| Validação   |   5   |   5   |    5    |  **15**  |
| Teste       |   5   |   5   |    5    |  **15**  |
| Inéditos    |   7   |   7   |    8    |  **22**  |
| **Total**   |  23   |  23   |   24    |  **70**  |

- **Total de julgamentos**: 70 × 3 fontes × 3 avaliadores = **630**.
- Classificação de dificuldade pelo escore composto S_i (P30/P70), calculado **após curadoria**.
- Para Inéditos, S_i calculado sobre extrações do modelo base nesses docs novos.

## Configurando o Label Studio

1. Preparação da VPS em Railway

> motivo da escolha: preço e disponibilidade de deploy fácil do Label Studio


```bash
# No computador local (Windows/ WSL):
# Acessar o endereço: http://localhost:8080 (ou o endereço do Docker no host)
#
# 1. Entrar em Settings > Account Settings > API Keys
# 2. Clicar em "Generate key" e copiar o token gerado.
#
# 3. No Docker container (Linux), criar um arquivo .env
cp label-studio/README.md label-studio/.env.prod
# 4. Editar o .env.prod e adicionar:
#    LABEL_STUDIO_API_KEY=token-copiado-do-site
#    LABEL_STUDIO_BASE_URL=http://localhost:8081 (ou o IP público da VPS)
#    LABEL_STUDIO_PROJECT_ID=1 (ou o ID do projeto criado)
```
