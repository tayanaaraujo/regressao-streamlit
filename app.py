import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
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
st.title("Treinamento de Modelo em Nuvem (Azure): Regressão Linear")

TARGET_COLUMN = 'time'
N_LAG = 5
N_SPLITS = 5  # número de folds para TimeSeriesSplit

# ------------------------------------------------------------------
# Estado do app
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
# Helpers: criação de lags
# ------------------------------------------------------------------
def create_lag_features(df: pd.DataFrame, n_lag: int):
    """
    Espera um DataFrame contendo a coluna 'time' (série). Retorna um DF com colunas:
    time-5, time-4, ..., time-1, time
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COLUMN}' não encontrada.")

    df_l = df.copy()
    for i in range(1, n_lag + 1):
        df_l[f't-{i}'] = df_l[TARGET_COLUMN].shift(i)
    df_l.dropna(inplace=True)
    cols = [f't-{i}' for i in range(n_lag, 0, -1)] + [TARGET_COLUMN]
    return df_l[cols]

# ------------------------------------------------------------------
# Huffman (compactação ponta-a-ponta) - implementação simples
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
# I/O helpers (CSV/ZIP)
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
# Sidebar controles
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Controles")
    if st.button("Resetar Modelo"):
        st.session_state.clear()
        st.rerun()

# ------------------------------------------------------------------
# 1) TREINAMENTO
# ------------------------------------------------------------------
st.header("Treinamento do Modelo")

uploaded_train = st.file_uploader("Upload do arquivo de treino (.csv ou .zip)", type=['csv', 'zip'], key='up_train')

if uploaded_train is not None:
    # ponta-a-ponta: compacta o raw upload
    raw_bytes = uploaded_train.getvalue()
    compressed_raw = huffman_compress(raw_bytes)
    st.session_state['compressed_train'] = compressed_raw

    # descompacta para leitura do CSV
    try:
        df_raw = pd.read_csv(io.BytesIO(huffman_decompress(compressed_raw)))
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()

    try:
        df_train = create_lag_features(df_raw.reset_index(drop=True), N_LAG)
    except Exception as e:
        st.error(f"Erro ao criar lags: {e}")
        st.stop()

    st.subheader("Amostra dos Dados de Treino")
    st.dataframe(df_train.head())

    if st.button("🚀 Treinar Modelo (TimeSeriesSplit CV)"):
        X = df_train.drop(columns=[TARGET_COLUMN])
        y = df_train[TARGET_COLUMN].values  # array 1D

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=N_SPLITS)

        # Vamos fazer validação manual fold-a-fold para obter previsões CV corretas
        preds_cv = np.full_like(y, fill_value=np.nan, dtype=float)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_tr, y_tr = X_scaled[train_idx], y[train_idx]
            X_te = X_scaled[test_idx]

            model_fold = LinearRegression()
            model_fold.fit(X_tr, y_tr)
            preds_fold = model_fold.predict(X_te)

            preds_cv[test_idx] = preds_fold

        # métricas da validação cruzada temporal (comparando y com preds_cv)
        mask = ~np.isnan(preds_cv)
        mse_cv = mean_squared_error(y[mask], preds_cv[mask])
        mae_cv = mean_absolute_error(y[mask], preds_cv[mask])
        r2_cv = r2_score(y[mask], preds_cv[mask])

        # treinar modelo final em todos os dados
        final_model = LinearRegression()
        final_model.fit(X_scaled, y)

        # salvar no session state
        st.session_state['modelo'] = final_model
        st.session_state['scaler'] = scaler
        st.session_state['train_df'] = df_train

        st.success("Modelo treinado e salvo na sessão.")

        # Mostrar métricas esperadas (CV)
        st.subheader("📊 Desempenho Esperado (Validação Cruzada — Temporal)")
        st.write(f"**MSE (esperado):** {mse_cv:.4f}")
        st.write(f"**MAE (esperado):** {mae_cv:.4f}")
        st.write(f"**R² (esperado):** {r2_cv:.4f}")

        # -------------------------
        # Gráfico 3 — combinado por folds (Real + Predições por fold)
        # -------------------------
        st.subheader("Real x Predito (Validação Cruzada por folds)")

        fig, ax = plt.subplots(figsize=(12, 4))
        # linha real inteira
        ax.plot(range(len(y)), y, label='Real', color='black', linewidth=1.2)

        # cores por fold
        cmap = plt.get_cmap('tab10')
        for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(X_scaled)):
            # desenhar a predição do fold como uma linha conectada nos índices de teste
            idxs = list(test_idx)
            preds_fold = preds_cv[test_idx]
            ax.plot(idxs, preds_fold, label=f'Pred Fold {fold+1}', color=cmap(fold % 10), linewidth=2)

        ax.set_title("Validação Cruzada (TimeSeriesSplit) — Real vs Predito por Fold")
        ax.set_xlabel("Índice (tempo crescente)")
        ax.set_ylabel(TARGET_COLUMN)
        ax.legend(loc='upper right', fontsize='small', ncol=2)
        st.pyplot(fig)

# ------------------------------------------------------------------
# 2) TESTE / APLICAÇÃO
# ------------------------------------------------------------------
st.header("Teste do modelo")

if st.session_state.get('modelo') is None:
    st.warning("Treine o modelo antes de usar a etapa de teste.")
else:
    uploaded_test = st.file_uploader("Upload do arquivo de teste (.csv ou .zip) — pode conter ou não rótulos", type=['csv', 'zip'], key='up_test')

    if uploaded_test is not None:
        raw_test = uploaded_test.getvalue()
        compressed_test = huffman_compress(raw_test)  # guardar compressão ponta-a-ponta
        # ler
        try:
            df_test_raw = pd.read_csv(io.BytesIO(huffman_decompress(compressed_test)))
        except Exception as e:
            st.error(f"Erro ao ler CSV de teste: {e}")
            st.stop()

        try:
            df_test = create_lag_features(df_test_raw.reset_index(drop=True), N_LAG)
        except Exception as e:
            st.error(f"Erro ao criar lags no teste: {e}")
            st.stop()

        st.subheader("Amostra dos Dados de Teste Processados")
        st.dataframe(df_test.head())

        if st.button("Executar Previsão (Teste)"):
            X_test = df_test.drop(columns=[TARGET_COLUMN], errors='ignore')
            y_test = df_test[TARGET_COLUMN] if TARGET_COLUMN in df_test.columns else None

            scaler = st.session_state['scaler']
            model = st.session_state['modelo']

            X_test_scaled = scaler.transform(X_test)
            preds_test = model.predict(X_test_scaled)

            df_result = df_test.copy()
            df_result['Previsao'] = preds_test

            st.subheader("Resultados (Teste)")
            if y_test is not None:
                mse_test = mean_squared_error(y_test, preds_test)
                mae_test = mean_absolute_error(y_test, preds_test)
                r2_test = r2_score(y_test, preds_test)
                st.write(f"**MSE (real):** {mse_test:.4f}")
                st.write(f"**MAE (real):** {mae_test:.4f}")
                st.write(f"**R² (real):** {r2_test:.4f}")

                # Gráfico Teste: Real vs Predito
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                ax2.plot(range(len(y_test)), y_test.values, label='Real', color='black', linewidth=1.2)
                ax2.plot(range(len(preds_test)), preds_test, label='Predito', color='tab:orange', linewidth=1.5)
                ax2.set_title("Teste — Real vs Predito")
                ax2.set_xlabel("Índice (tempo crescente)")
                ax2.set_ylabel(TARGET_COLUMN)
                ax2.legend()
                st.pyplot(fig2)

                # Erro absoluto por ponto
                abs_err = np.abs(y_test.values - preds_test)
                fig3, ax3 = plt.subplots(figsize=(12, 3))
                ax3.bar(range(len(abs_err)), abs_err, color='tab:red')
                ax3.set_title("Erro Absoluto por Ponto (Teste)")
                st.pyplot(fig3)
            else:
                st.info("Arquivo de teste sem rótulos — apenas previsões geradas.")

            # download compactado Huffman
            #download_huffman(df_result, "previsoes_huffman.bin")

# ------------------------------------------------------------------
# Opção de download do treino/ teste (compactado)
# ------------------------------------------------------------------
if st.session_state.get('compressed_train') is not None:
    st.sidebar.download_button('Baixar treino (Huffman compactado)', data=st.session_state['compressed_train'], file_name='treino_huffman.bin', mime='application/octet-stream')
    st.sidebar.dowload_button('Baixar teste (Huffman compactado)', data=st.session_state['df_result'], file_name='previsoes_huffman.bin', mime='application/octet-stream')
