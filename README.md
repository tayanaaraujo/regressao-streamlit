📘 Descrição Explicada do Código

Este projeto implementa uma aplicação Streamlit para treinar e testar um modelo de Regressão Linear aplicado a séries temporais, incluindo criação de lags, validação cruzada temporal, compactação com Huffman, e geração de gráficos.

Abaixo está a explicação de cada parte do código.
🔧 1. Configurações Iniciais

Importa as bibliotecas necessárias:

streamlit para interface web.

pandas, numpy para manipulação de dados.

sklearn para modelagem (linear regression, métricas, escalonamento, time series split).

matplotlib para gráficos.

pickle, zipfile, io para manipulação de arquivos.

heapq, Counter para a implementação da codificação de Huffman.

Configura layout da página via st.set_page_config.

Define variáveis globais:

TARGET_COLUMN = "time"

Número de defasagens N_LAG = 5

Número de splits da validação cruzada temporal N_SPLITS = 5

Inicializa variáveis no session_state (modelo, scaler e dados).

🧩 2. Função de Criação de Lags

A função create_lag_features recebe um DataFrame com a coluna time e cria:

t-5, t-4, t-3, t-2, t-1, time


Ou seja, para prever o valor atual, utiliza os 5 valores anteriores.
Ela também remove linhas com NaN decorrentes do deslocamento.

🗜️ 3. Implementação do Algoritmo de Huffman

O código implementa compactação e descompactação usando:

Classe HuffmanNode

Construção da árvore de Huffman conforme frequência dos bytes

Geração dos códigos binários de cada byte

Compactação de qualquer arquivo em bytes

Descompactação reversa

Essa compactação é usada:

Para armazenar o upload do CSV com redução de tamanho

Para baixar as previsões de forma compactada

É um recurso avançado que torna o projeto diferenciado.

📦 4. Funções de I/O

load_data()
Lê CSVs diretamente ou dentro de ZIP.

download_huffman()
Gera um botão para baixar um DataFrame em formato compactado via Huffman.

🧰 5. Barra Lateral

Na sidebar existe:

Botão para resetar toda a sessão, apagando modelo, scaler e dados.

🏋️‍♂️ 6. Seção 1 — Treinamento do Modelo

📌 Upload obrigatório de arquivo .csv ou .zip.

Passos ao fazer upload:

O arquivo é compactado com Huffman e salvo.

Em seguida, é descompactado para leitura:

Mantém coerência ponta-a-ponta da compactação.

O dataframe é processado para criar os lags.

O usuário visualiza parte da amostra.

▶️ Quando o usuário clica em Treinar Modelo:

Separam-se X e y.

Aplica-se normalização (MinMaxScaler).

Executa-se TimeSeriesSplit com 5 folds.

Em cada fold:

Treina um modelo em X_train

Prediz em X_test

Armazena previsões na ordem temporal correta

Calcula as métricas de validação cruzada:

MSE

MAE

R²

Treina-se o modelo final com todos os dados.

Salva:

modelo treinado

scaler

dados de treino

Gera gráfico:

Linha real completa

Previsões separadas por cada fold, com cores diferentes

🔍 7. Seção 2 — Teste / Aplicação

Disponível somente após o modelo ser treinado.

O usuário pode enviar outro arquivo, contendo ou não valores reais.

Processo:

O arquivo é compactado com Huffman (padrão do projeto).

É descompactado e lido como CSV.

São criados os lags.

O usuário visualiza a amostra.

Ao clicar em Executar Previsão:

Aplica o scaler do treino.

Usa o modelo salvo para prever.

Cria um DataFrame com os resultados.

Se existirem valores reais:

Calcula MSE, MAE, R²

Gera:

Gráfico Real vs Predito

Gráfico dos erros absolutos por ponto

Permite baixar o arquivo compactado com Huffman.

📥 8. Download do Treino Compactado

Na barra lateral, o usuário pode baixar o CSV de treino compactado via Huffman, exatamente como foi armazenado.


🎯 Resumo do Fluxo Completo

Upload → Compacta com Huffman → Descompacta → Lê → Cria lags

Treina modelo com validação temporal

Exibe métricas e gráficos por fold

Salva modelo e scaler

Teste com ou sem valores reais

Exibe métricas de teste + gráficos

Permite baixar resultados compactados

Permite baixar dataset de treino compactado
