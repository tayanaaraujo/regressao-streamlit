# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import pickle
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# HUFFMAN PURO  (SEM GZIP)
# ============================================================

from heapq import heappush, heappop
from collections import Counter


class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol  # byte (0–255) ou None
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freqs):
    heap = []
    for byte_val, f in freqs.items():
        heappush(heap, HuffmanNode(f, symbol=byte_val))

    if len(heap) == 1:
        node = heappop(heap)
        return HuffmanNode(node.freq, left=node)

    while len(heap) > 1:
        n1 = heappop(heap)
        n2 = heappop(heap)
        merged = HuffmanNode(n1.freq + n2.freq, left=n1, right=n2)
        heappush(heap, merged)

    return heappop(heap)


def build_code_table(node, prefix="", table=None):
    if table is None:
        table = {}

    if node.symbol is not None:
        table[node.symbol] = prefix
        return table

    build_code_table(node.left, prefix + "0", table)
    build_code_table(node.right, prefix + "1", table)
    return table


def compress_bytes(data_bytes: bytes) -> bytes:
    freqs = Counter(data_bytes)
    root = build_huffman_tree(freqs)
    table = build_code_table(root)

    # Codifica para uma string binária
    bitstring = "".join(table[b] for b in data_bytes)

    # Padding para múltiplo de 8 bits
    extra_bits = 8 - (len(bitstring) % 8)
    if extra_bits != 8:
        bitstring += "0" * extra_bits
    else:
        extra_bits = 0

    # Converte para bytes
    compressed_data = bytearray()
    for i in range(0, len(bitstring), 8):
        byte = int(bitstring[i:i+8], 2)
        compressed_data.append(byte)

    # Criar header:
    # 1 byte = quantidade de padding
    # 256*4 bytes = tabela de frequências (inteiros de 32 bits)
    header = bytes([extra_bits])
    for i in range(256):
        f = freqs.get(i, 0)
        header += f.to_bytes(4, byteorder='big')  # sempre escreve 4 bytes por símbolo

    return header + bytes(compressed_data)


def decompress_bytes(encoded: bytes) -> bytes:
    # 1) Lê padding
    padding = encoded[0]

    # 2) Lê tabela de frequências (256 inteiros de 4 bytes)
    freqs = {}
    pos = 1
    for byte_val in range(256):
        f = int.from_bytes(encoded[pos:pos+4], byteorder='big')
        pos += 4
        if f > 0:
            freqs[byte_val] = f

    # 3) Reconstrói árvore
    root = build_huffman_tree(freqs)

    # 4) Converte bytes compactados para string binária
    bitstring = ""
    for b in encoded[pos:]:
        bitstring += f"{b:08b}"

    if padding > 0:
        bitstring = bitstring[:-padding]

    # 5) Decodificação
    output = bytearray()
    node = root
    for bit in bitstring:
        node = node.left if bit == "0" else node.right
        if node.symbol is not None:
            output.append(node.symbol)
            node = root

    return bytes(output)


# ============================================================
# Persistência do modelo
# ============================================================

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)


def save_model_and_scaler(model, scaler):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)


def load_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None


def reset_persistent_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        os.remove(SCALER_PATH)
    for f in os.listdir(UPLOADS_DIR):
        try:
            os.remove(os.path.join(UPLOADS_DIR, f))
        except:
            pass


# ============================================================
# Treinamento e aplicação
# ============================================================

def train_model_timeseries(df, n_splits=5):
    if "time" not in df.columns:
        raise ValueError("Coluna 'time' não encontrada.")

    y = df["time"].values
    X = df.drop(columns=["time"]).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()

    tscv = TimeSeriesSplit(n_splits=n_splits)

    r2_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')

    rmse_scores = []
    for tr, ts in tscv.split(X_scaled):
        model.fit(X_scaled[tr], y[tr])
        preds = model.predict(X_scaled[ts])
        rmse_scores.append(np.sqrt(mean_squared_error(y[ts], preds)))

    model.fit(X_scaled, y)
    save_model_and_scaler(model, scaler)

    return np.mean(rmse_scores), np.mean(r2_scores), model, scaler


def apply_model(df, model, scaler):
    if model is None or scaler is None:
        raise RuntimeError("Modelo não carregado.")

    if "time" in df.columns:
        y = df["time"].values
        X = df.drop(columns=["time"]).values
        has_label = True
    else:
        X = df.values
        y = None
        has_label = False

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    results = pd.DataFrame({"predicted": preds})

    if has_label:
        return results, np.sqrt(mean_squared_error(y, preds)), r2_score(y, preds)
    return results, None, None


# ============================================================
# Streamlit Interface
# ============================================================

st.set_page_config(page_title="Regressão Linear TS", layout="wide")
st.title("📈 Regressão Linear — Série Temporal — Huffman Puro")

page = st.sidebar.selectbox("Menu", ["Treinar Modelo", "Testar Modelo", "Resetar Modelo", "Status"])
model, scaler = load_model_and_scaler()

def make_download_link_bytes(data_bytes: bytes, filename: str):
    b64 = base64.b64encode(data_bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Baixar {filename}</a>'


# ============================================================
# Páginas
# ============================================================

if page == "Treinar Modelo":
    st.header("📤 Upload do CSV de Treino")
    uploaded_file = st.file_uploader("Envie o CSV", type=["csv"])

    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        compressed = compress_bytes(raw_bytes)

        save_path = os.path.join(UPLOADS_DIR, f"train_{uploaded_file.name}.bin")
        with open(save_path, "wb") as f:
            f.write(compressed)

        st.markdown(make_download_link_bytes(compressed, f"train_{uploaded_file.name}.bin"), unsafe_allow_html=True)

        try:
            csv_bytes = decompress_bytes(compressed)
            df = pd.read_csv(io.BytesIO(csv_bytes), sep=",")

            # correção de BOM / caracteres invisíveis
            df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

            # tentar renomear automaticamente caso a coluna esteja com lixo
            for c in df.columns:
                if c.replace("\ufeff", "").strip().lower() == "time":
                    df = df.rename(columns={c: "time"})
                    break

        except Exception as e:
            st.error(f"Erro ao descompactar/ler CSV: {e}")
            st.stop()

        st.write(df.head())

        if "time" not in df.columns:
            st.error("CSV precisa conter coluna 'time'.")
        elif st.button("Treinar Modelo"):
            rmse, r2, model, scaler = train_model_timeseries(df)
            st.success("Treinado com sucesso!")
            st.metric("RMSE (val)", f"{rmse:.4f}")
            st.metric("R² (val)", f"{r2:.4f}")


elif page == "Testar Modelo":
    st.header("📤 Upload do CSV de Teste")

    uploaded_file = st.file_uploader("Envie o CSV", type=["csv"])
    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        compressed = compress_bytes(raw_bytes)

        save_path = os.path.join(UPLOADS_DIR, f"test_{uploaded_file.name}.bin")
        with open(save_path, "wb") as f:
            f.write(compressed)

        st.markdown(make_download_link_bytes(compressed, f"test_{uploaded_file.name}.bin"), unsafe_allow_html=True)

        try:
            csv_bytes = decompress_bytes(compressed)
            df = pd.read_csv(io.BytesIO(csv_bytes), sep=",")

            # correção de BOM / caracteres invisíveis
            df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

            # tentar renomear automaticamente caso a coluna esteja com lixo
            for c in df.columns:
                if c.replace("\ufeff", "").strip().lower() == "time":
                    df = df.rename(columns={c: "time"})
                    break

        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

        st.write(df.head())

        if st.button("Executar Teste"):
            if model is None:
                st.error("Treine primeiro.")
            else:
                result, rmse, r2 = apply_model(df, model, scaler)
                st.dataframe(result.head())

                out = io.BytesIO()
                result.to_csv(out, index=False)
                compressed_out = compress_bytes(out.getvalue())
                st.markdown(make_download_link_bytes(compressed_out, "predicoes.bin"), unsafe_allow_html=True)

                if rmse is not None:
                    st.metric("RMSE real", f"{rmse:.4f}")
                    st.metric("R² real", f"{r2:.4f}")


elif page == "Resetar Modelo":
    if st.button("Resetar Tudo"):
        reset_persistent_model()
        st.success("Reset concluído.")


elif page == "Status":
    st.write("Modelo:", os.path.exists(MODEL_PATH))
    st.write("Scaler:", os.path.exists(SCALER_PATH))
    st.write("Uploads:", os.listdir(UPLOADS_DIR))
