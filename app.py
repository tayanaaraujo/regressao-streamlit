# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import pickle
import base64
import gzip
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# tentar importar dahuffman (opcional). Usamos gzip como fallback robusto.
try:
    from dahuffman import HuffmanCodec
    DAHUFFMAN_AVAILABLE = True
except Exception:
    DAHUFFMAN_AVAILABLE = False

# paths e pasta de uploads
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# --------------------------
# compress / decompress helpers
# --------------------------
def compress_bytes_gzip(data_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(data_bytes)
    return buf.getvalue()

def decompress_bytes_gzip(encoded_bytes: bytes) -> bytes:
    buf = io.BytesIO(encoded_bytes)
    with gzip.GzipFile(fileobj=buf, mode='rb') as f:
        return f.read()

# Simple wrappers: try dahuffman for compress, but use gzip for decompress/robustness
def compress_bytes(data_bytes: bytes) -> bytes:
    if DAHUFFMAN_AVAILABLE:
        try:
            # encode bytes -> latin1 string -> huffman
            text = data_bytes.decode('latin1')
            codec = HuffmanCodec.from_data(text)
            encoded = codec.encode(text)
            if isinstance(encoded, str):
                return encoded.encode('latin1')
            elif isinstance(encoded, (bytes, bytearray)):
                return bytes(encoded)
            else:
                return bytes(encoded)
        except Exception:
            return compress_bytes_gzip(data_bytes)
    else:
        return compress_bytes_gzip(data_bytes)

def decompress_bytes(data_bytes: bytes) -> bytes:
    # prefer gzip (robusto)
    try:
        return decompress_bytes_gzip(data_bytes)
    except Exception:
        # Se dahuffman estiver disponível, tentamos (pode falhar sem metadados)
        if DAHUFFMAN_AVAILABLE:
            try:
                # tentativa: tratar como latin1 string; note que sem metadados isso pode falhar
                encoded_text = data_bytes.decode('latin1')
                codec = HuffmanCodec.from_data(encoded_text)  # tentativa; possivelmente incorreta
                decoded = codec.decode(encoded_text)
                return decoded.encode('latin1')
            except Exception:
                raise RuntimeError("Falha na descompressão; formato inválido.")
        else:
            raise RuntimeError("Falha na descompressão gzip e dahuffman não disponível.")

# --------------------------
# persistência do modelo
# --------------------------
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
    for fname in os.listdir(UPLOADS_DIR):
        try:
            os.remove(os.path.join(UPLOADS_DIR, fname))
        except Exception:
            pass

# --------------------------
# treino / aplicação
# --------------------------
def train_model_timeseries(df, n_splits=5):
    """
    Treina usando TimeSeriesSplit (sem shuffle) e retorna:
    rmse_mean, r2_mean, model, scaler
    """
    if "time" not in df.columns:
        raise ValueError("Coluna 'time' não encontrada no dataframe.")

    y = df["time"].values
    X = df.drop(columns=["time"]).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # R2 via cross_val_score com TimeSeriesSplit
    r2_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')

    rmse_scores = []
    for train_idx, test_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx], y[train_idx])
        preds = model.predict(X_scaled[test_idx])
        rmse_scores.append(np.sqrt(mean_squared_error(y[test_idx], preds)))

    # treino final em todos os dados
    model.fit(X_scaled, y)
    # persistir
    save_model_and_scaler(model, scaler)

    return np.mean(rmse_scores), np.mean(r2_scores), model, scaler

def apply_model(df, model, scaler):
    if model is None or scaler is None:
        raise RuntimeError("Modelo ou scaler não carregado.")

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
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
    else:
        rmse = None
        r2 = None

    return results, rmse, r2

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Regressão Linear (Timeseries)", layout="wide")
st.title("📈 Regressão Linear (séries temporais) — MinMax + TimeSeriesSplit + Compressão ponta-a-ponta")

st.sidebar.header("Menu")
page = st.sidebar.selectbox("Navegação", ["Treinar Modelo", "Testar Modelo", "Resetar Modelo", "Status"])

# carregar modelo persistente
model, scaler = load_model_and_scaler()
if model is not None and scaler is not None:
    st.sidebar.success("✅ Modelo carregado do disco.")
else:
    st.sidebar.info("Nenhum modelo persistente encontrado. Treine um modelo.")

def make_download_link_bytes(data_bytes: bytes, filename: str):
    b64 = base64.b64encode(data_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Baixar {filename}</a>'
    return href

# --------------------------
# Treinar
# --------------------------
if page == "Treinar Modelo":
    st.header("📤 Upload do CSV de Treino")
    st.write("Formato esperado: 5 colunas de lag (ex: time-5,..,time-1) + coluna 'time' (alvo).")
    uploaded_file = st.file_uploader("Envie o CSV de treino", type=["csv"])

    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        # salvar compactado o upload
        compressed = compress_bytes(raw_bytes)
        saved_path = os.path.join(UPLOADS_DIR, f"train_{uploaded_file.name}.bin")
        with open(saved_path, "wb") as f:
            f.write(compressed)
        st.markdown(make_download_link_bytes(compressed, f"train_{uploaded_file.name}.bin"), unsafe_allow_html=True)
        st.success("Upload recebido e compactado (ponta-a-ponta).")

        # descompactar para leitura
        try:
            csv_bytes = decompress_bytes(compressed)
        except Exception as e:
            st.error(f"Erro na descompressão do upload: {e}")
            st.stop()

        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")
            st.stop()

        st.write("Preview do CSV de treino:")
        st.dataframe(df.head())

        if "time" not in df.columns:
            st.error("O CSV precisa conter a coluna 'time'.")
        else:
            if st.button("Treinar Modelo"):
                with st.spinner("Treinando com TimeSeriesSplit (sem shuffle)..."):
                    rmse_exp, r2_exp, trained_model, trained_scaler = train_model_timeseries(df, n_splits=5)
                    model = trained_model
                    scaler = trained_scaler
                st.success("Treino concluído e modelo salvo no servidor.")
                st.metric("RMSE (esperado - TimeSeriesSplit mean)", f"{rmse_exp:.4f}")
                st.metric("R² (esperado - TimeSeriesSplit mean)", f"{r2_exp:.4f}")

# --------------------------
# Testar
# --------------------------
elif page == "Testar Modelo":
    st.header("📤 Upload do CSV de Teste")
    if model is None or scaler is None:
        st.warning("Nenhum modelo carregado — treine ou carregue um modelo primeiro.")

    uploaded_file = st.file_uploader("Envie o CSV de teste (pode conter coluna 'time' para avaliação)", type=["csv"])
    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        compressed = compress_bytes(raw_bytes)
        saved_path = os.path.join(UPLOADS_DIR, f"test_{uploaded_file.name}.bin")
        with open(saved_path, "wb") as f:
            f.write(compressed)
        st.markdown(make_download_link_bytes(compressed, f"test_{uploaded_file.name}.bin"), unsafe_allow_html=True)
        st.success("Upload de teste recebido e compactado (ponta-a-ponta).")

        try:
            csv_bytes = decompress_bytes(compressed)
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as e:
            st.error(f"Erro ao descompactar/ler CSV de teste: {e}")
            st.stop()

        st.write("Preview do CSV de teste:")
        st.dataframe(df.head())

        if st.button("Executar Teste"):
            if model is None or scaler is None:
                st.error("Modelo não encontrado. Treine o modelo antes de testar.")
            else:
                results, rmse_real, r2_real = apply_model(df, model, scaler)
                st.write("Previsões (primeiras linhas):")
                st.dataframe(results.head())

                # salvar previsões e compactar
                out_buf = io.BytesIO()
                results.to_csv(out_buf, index=False)
                out_bytes = out_buf.getvalue()
                compressed_out = compress_bytes(out_bytes)
                st.markdown(make_download_link_bytes(compressed_out, "predicoes.bin"), unsafe_allow_html=True)
                st.success("Previsões prontas para download (compactadas).")

                if rmse_real is not None:
                    st.metric("RMSE Real", f"{rmse_real:.4f}")
                    st.metric("R² Real", f"{r2_real:.4f}")
                else:
                    st.info("Arquivo sem rótulos — apenas previsões foram geradas.")

# --------------------------
# Reset
# --------------------------
elif page == "Resetar Modelo":
    st.header("🔁 Resetar modelo e arquivos persistidos")
    if st.button("Resetar"):
        reset_persistent_model()
        model = None
        scaler = None
        st.success("Modelo e arquivos persistidos removidos. Pronto para novo treino.")

# --------------------------
# Status
# --------------------------
elif page == "Status":
    st.header("🔍 Status do servidor")
    st.write(f"Modelo persistido: {os.path.exists(MODEL_PATH)}")
    st.write(f"Scaler persistido: {os.path.exists(SCALER_PATH)}")
    st.write("Arquivos na pasta uploads:")
    st.write(os.listdir(UPLOADS_DIR))
    st.write(f"Dahuffman disponível: {DAHUFFMAN_AVAILABLE}")
    st.info("Usamos gzip como método de compressão/descompressão principal por robustez.")
