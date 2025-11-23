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
