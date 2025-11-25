# Regressão Linear em Série Temporal – Treino e Teste Remoto com Streamlit

Este projeto implementa um sistema completo para **treinamento, avaliação e execução remota** de um modelo de **Regressão Linear** aplicado a uma **série temporal**.

A aplicação foi desenvolvida em **Python + Streamlit** e permite o envio de arquivos, tratamento automático dos dados, treinamento, teste e download das previsões.

---

## Funcionalidades Principais

### Upload de arquivo `.csv` para **treino do modelo**
- Aceita arquivos contendo as variáveis `t-1` a `t-5` e o rótulo `time`.
- O modelo é treinado automaticamente após o upload.

---

### Treinamento remoto com Regressão Linear
Após o envio do arquivo de treino, o sistema realiza:

- Normalização **Min–Max**
- Criação de lags da série temporal (5 passos anteriores)
- Separação entre features e rótulos
- Treinamento do modelo `LinearRegression`
- Exibição da **expectativa de desempenho** com:

  - MSE  
  - MAE  
  - R²  

---

### Upload de arquivo `.csv` para **teste do modelo**

O sistema lida com dois cenários:

#### **A) Teste COM rótulo (`time`)**
- Gera previsões
- Avalia o desempenho real
- Exibe métricas (MSE, MAE, R²)
- Mostra gráficos:
  - Real vs Predito
  - Erro Absoluto
- Disponibiliza o arquivo de previsões compactado para download (`.bin`)

#### **B) Teste SEM rótulo**
- Apenas gera as previsões
- Não calcula métricas
- Disponibiliza o arquivo compactado para download

---

### Tratamento de Dados Ponta a Ponta
O sistema executa automaticamente:

- Normalização Min–Max  
- Geração de lags  
- Padronização dos dados  
- Compressão das previsões via **Huffman coding**  
- Reconstrução da base de teste caso faltem features  

---

### Reset do modelo
Um botão “**Resetar Modelo**” permite:

- Limpar os dados da sessão
- Treinar o modelo novamente com outra base

---

## Sobre a Base de Dados

A base representa uma **série temporal** onde o valor atual depende dos **cinco valores anteriores**.  
O objetivo é prever a coluna:

- `time`

As features utilizadas são:
t-1, t-2, t-3, t-4, t-5

## Fluxo Completo da Aplicação

1. Upload do arquivo de treino  
2. Tratamento da base  
3. Treinamento do modelo  
4. Exibição da expectativa de desempenho  
5. Upload do arquivo de teste  
6. Previsão  
7. Avaliação (se houver rótulo)  
8. Download do arquivo compactado  
9. Reset do modelo (opcional)  

---

## Download das Previsões

Após o teste, o sistema disponibiliza:
previsoes_huffman.bin
