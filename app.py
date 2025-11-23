import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import zipfile
import pickle
import heapq
from collections import Counter

# ------------------------------------------------------------------
# Configurações Gerais
# ------------------------------------------------------------------
st.set_page_config(page_title="Projeto Regressão Linear na Nuvem", layout="wide")
st.title("🎓 Treinamento de Modelo em Nuvem (Azure) — Regressão Linear")

TARGET_COLUMN = 'time'
N_LAG = 5

# ------------------------------------------------------------------
# Controle do estado do app
# ------------------------------------------------------------------
if 'modelo' not in st.session_state:
    st.session_state['modelo'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'train_df' not in st.session_state:
    st.session_state['train_df'] = None
if 'compressed_train' not in st.session_state:
    st.session_state['compressed_train'] = None


# ------------------------------------------------------------------
# Criação de Lags (para séries temporais)
# ------------------------------------------------------------------
def create_lag_features(df: pd.DataFrame, n_lag: int):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COLUMN}' não encontrada.")

    df_l = df.copy()
    for i in range(1, n_lag + 1):
        df_l[f't-{i}'] = df_l[TARGET_COLUMN].shift(i)

    df_l.dropna(inplace=True)

    cols = [f't-{i}' for i in range(n_lag, 0, -1)] + [TARGET_COLUMN]
    return df_l[cols]


# ------------------------------------------------------------------
# Compactação Huffman
# ------------------------------------------------------------------
class HuffmanNode:
    def __init__(self, freq, byte=None, left=None, right=None):
        self.freq = freq
        self.byte = byte
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data_bytes: bytes):
    freq = Counter(data_bytes)
    heap = [HuffmanNode(f, b) for b, f in freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        node = heapq.heappop(heap)
        root = HuffmanNode(node.freq, None, node, None)
        heapq.heappush(heap, root)

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = HuffmanNode(n1.freq + n2.freq, None, n1, n2)
        heapq.heappush(heap, merged)

    return heap[0]


def generate_huffman_codes(root):
    codes = {}
    def _gen(node, prefix):
        if node.byte is not None:
            codes[node.byte] = prefix or '0'
            return
        _gen(node.left, prefix + '0')
        _gen(node.right, prefix + '1')
    _gen(root, '')
    return codes


def huffman_compress(data_bytes: bytes):
    root = build_huffman_tree(data_bytes)
    codes = generate_huffman_codes(root)
    bitstr = ''.join(codes[b] for b in data_bytes)

    pad_len = (8 - len(bitstr) % 8) % 8
    if pad_len:
        bitstr += '0' * pad_len

    b_arr = bytearray()
    for i in range(0, len(bitstr), 8):
        b_arr.append(int(bitstr[i:i+8], 2))

    payload = bytes(b_arr)
    return pickle.dumps({'codes': codes, 'pad_len': pad_len, 'payload': payload})


def huffman_decompress(pickled_bytes: bytes):
    store = pickle.loads(pickled_bytes)
    codes = store['codes']
    pad_len = store['pad_len']
    payload = store['payload']

    inv = {v: k for k, v in codes.items()}
    bitstr = ''.join(f'{b:08b}' for b in payload)

    if pad_len:
        bitstr = bitstr[:-pad_len]

    out = bytearray()
    cur = ''
    for bit in bitstr:
        cur += bit
        if cur in inv:
            out.append(inv[cur])
            cur = ''
    return bytes(out)


# ------------------------------------------------------------------
# Função de leitura CSV/ZIP
# ------------------------------------------------------------------
def load_data(uploaded):
    if uploaded.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded, 'r') as zf:
            csvs = [f for f in zf.namelist() if f.endswith('.csv')]
            if len(csvs) != 1:
                st.error("O ZIP deve conter exatamente 1 arquivo CSV.")
                return None
            with zf.open(csvs[0]) as f:
                return pd.read_csv(f)
    else:
        return pd.read_csv(uploaded)


def download_huffman(df, filename):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    compressed = huffman_compress(buf.getvalue().encode('utf-8'))
    st.download_button("📥 Baixar Arquivo Compactado (Huffman)", compressed, filename)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controles")
    if st.button("🔄 Resetar Modelo"):
        st.session_state.clear()
        st.rerun()


# ------------------------------------------------------------------
# 1. Treinamento
# ------------------------------------------------------------------
st.header("1️⃣ Treinamento do Modelo")

uploaded_train = st.file_uploader("Upload do arquivo de treino (.csv ou .zip)", type=['csv', 'zip'])

if uploaded_train:
    raw = uploaded_train.getvalue()
    compressed = huffman_compress(raw)
    st.session_state['compressed_train'] = compressed

    df_raw = pd.read_csv(io.BytesIO(huffman_decompress(compressed)))

    df_train = create_lag_features(df_raw, N_LAG)
    st.subheader("Amostra dos Dados de Treino")
    st.dataframe(df_train.head())

    if st.button("🚀 Treinar Modelo"):
        X = df_train.drop(columns=[TARGET_COLUMN])
        y = df_train[TARGET_COLUMN]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=5)

        model = LinearRegression()
        model.fit(X_scaled, y)

        # validação cruzada real para séries
        preds_cv = cross_val_predict(model, X_scaled, y, cv=tscv)

        mse_cv = mean_squared_error(y, preds_cv)
        mae_cv = mean_absolute_error(y, preds_cv)
        r2_cv = r2_score(y, preds_cv)

        st.session_state['modelo'] = model
        st.session_state['scaler'] = scaler
        st.session_state['train_df'] = df_train

        st.success("Modelo treinado com sucesso!")

        # métricas
        st.subheader("📊 Desempenho Esperado (Validação Cruzada)")
        st.write(f"**MSE:** {mse_cv:.4f}")
        st.write(f"**MAE:** {mae_cv:.4f}")
        st.write(f"**R²:** {r2_cv:.4f}")

        # gráfico treino CV
        st.subheader("📈 Real x Predito (Validação Cruzada)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y.values, label='Real')
        ax.plot(preds_cv, label='Predito (CV)')
        ax.legend()
        ax.set_title("Validação Cruzada — Real x Predito")
        st.pyplot(fig)


# ------------------------------------------------------------------
# 2. Teste / Aplicação
# ------------------------------------------------------------------
st.header("2️⃣ Teste / Aplicação")

if st.session_state['modelo']:
    uploaded_test = st.file_uploader("Upload do teste (.csv ou .zip)", type=['csv', 'zip'])

    if uploaded_test:
        raw_test = uploaded_test.getvalue()
        compressed_test = huffman_compress(raw_test)

        df_test = pd.read_csv(io.BytesIO(huffman_decompress(compressed_test)))
        df_test_lag = create_lag_features(df_test, N_LAG)

        st.subheader("Amostra dos Dados de Teste")
        st.dataframe(df_test_lag.head())

        if st.button("🔍 Executar Previsão"):
            tem_rotulo = TARGET_COLUMN in df_test_lag.columns

            X_test = df_test_lag.drop(columns=[TARGET_COLUMN], errors='ignore')
            y_real = df_test_lag[TARGET_COLUMN] if tem_rotulo else None

            X_test_scaled = st.session_state['scaler'].transform(X_test)
            preds = st.session_state['modelo'].predict(X_test_scaled)

            df_result = df_test_lag.copy()
            df_result["Previsao"] = preds

            st.subheader("📈 Resultados")

            if tem_rotulo:
                mse = mean_squared_error(y_real, preds)
                mae = mean_absolute_error(y_real, preds)
                r2 = r2_score(y_real, preds)

                st.write(f"**MSE (real):** {mse:.4f}")
                st.write(f"**MAE (real):** {mae:.4f}")
                st.write(f"**R² (real):** {r2:.4f}")

                # gráfico: real vs previsto
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(y_real.values, label='Real')
                ax2.plot(preds, label='Predito')
                ax2.legend()
                ax2.set_title("Teste — Real x Predito")
                st.pyplot(fig2)

                # erro absoluto
                abs_error = abs(y_real - preds)
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.bar(range(len(abs_error)), abs_error)
                ax3.set_title("Erro Absoluto por Ponto")
                st.pyplot(fig3)

            else:
                st.info("Arquivo sem rótulos — apenas previsões geradas.")

            download_huffman(df_result, "previsoes_huffman.bin")

else:
    st.warning("⚠️ Treine o modelo antes de fazer o teste.")
