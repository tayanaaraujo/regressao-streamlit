import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import io
import zipfile # Para a compactação/descompactação (Requisito de Segurança)

# --- Configuração Inicial ---
st.set_page_config(page_title="Projeto Regressão Linear na Nuvem", layout="wide")
st.title("🎓 Treinamento de Modelo na Nuvem - Azure")
TARGET_COLUMN = 'time' # Coluna alvo da série temporal
N_LAG = 5 # Número de observações anteriores (conforme seu enunciado)

# --- Variáveis de Estado (Memória do App) ---
if 'modelo' not in st.session_state:
    st.session_state['modelo'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None

# --- Funções de Tratamento de Dados ---

@st.cache_data
def create_lag_features(df, n_lag):
    """Cria features de lag para dados de série temporal (t-1, t-2, ...)."""
    df_lag = df.copy()
    for i in range(1, n_lag + 1):
        df_lag[f't-{i}'] = df_lag[TARGET_COLUMN].shift(i)
    # Remove as linhas com NaN resultantes do shift (os N_LAG primeiros)
    df_lag.dropna(inplace=True)
    return df_lag

@st.cache_data
def load_data_zip(uploaded_file, file_type='csv'):
    """Descompacta o arquivo ZIP e lê o CSV (Atende ao requisito Ponto-a-ponta)."""
    if uploaded_file.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, 'r') as zf:
            # Assumimos que há um único CSV dentro do ZIP
            csv_filename = [f for f in zf.namelist() if f.endswith('.csv')][0]
            with zf.open(csv_filename) as csv_file:
                return pd.read_csv(csv_file)
    elif uploaded_file.name.endswith('.csv'):
        # Se for CSV direto, apenas lê
        return pd.read_csv(uploaded_file)
    else:
        st.error(f"Formato de arquivo não suportado. Use .csv ou .zip contendo um .csv.")
        return None

def compress_and_download(df, filename="resultados_previsao.zip"):
    """Compacta o CSV de resultados em ZIP para download (Requisito de Segurança)."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zf:
        # Cria um arquivo CSV dentro do ZIP
        zf.writestr('previsao.csv', csv_data)
        
    st.download_button(
        label="📥 Baixar Resultados (ZIP Compactado)",
        data=zip_buffer.getvalue(),
        file_name=filename,
        mime='application/zip',
    )


# --- Barra Lateral para Reset ---
with st.sidebar:
    st.header("Controles")
    if st.button("🔄 Resetar Modelo"):
        st.session_state['modelo'] = None
        st.session_state['scaler'] = None
        st.info("Modelo e Normalizador resetados. Clique em Run again.")
        st.experimental_rerun()
    
    st.info(f"Modelo: Regressão Linear\nLags de Série Temporal: {N_LAG}")

# -------------------------------------------------------------
# BLOCO 1: TREINAMENTO (Com Normalização e Validação Cruzada)
# -------------------------------------------------------------
st.header("1. Treinamento do Modelo")
arquivo_treino = st.file_uploader("Faça upload do arquivo de Treino (.csv ou .zip)", type=['csv', 'zip'])

if arquivo_treino is not None:
    df_raw = load_data_zip(arquivo_treino)
    if df_raw is None:
        st.stop()
        
    df_treino = create_lag_features(df_raw.reset_index(drop=True), N_LAG)
    
    if df_treino is not None:
        st.subheader("Amostra de Dados Processados")
        st.dataframe(df_treino.head())
        
        if st.button("🚀 Treinar Modelo com Validação Cruzada"):
            try:
                # Separação de features e rótulos
                X_train_clean = df_treino.drop(columns=[TARGET_COLUMN])
                y_train = df_treino[TARGET_COLUMN]
                
                # Normalização Min-Max (Requisito do projeto)
                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train_clean)
                
                # Treino do modelo final (fit em todos os dados)
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
                
                # Validação Cruzada (Para obter EXPECTATIVA de desempenho)
                # Usamos o MSE (Mean Squared Error) como métrica
                mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
                scores = cross_val_score(model, X_train_scaled, y_train, 
                                         cv=5, # 5 Folds
                                         scoring=mse_scorer)
                
                # O cross_val_score retorna valores negativos para o MSE, então pegamos a média.
                mean_mse = -scores.mean()
                
                # Salvar na memória (session_state)
                st.session_state['modelo'] = model
                st.session_state['scaler'] = scaler
                
                st.success("Modelo treinado, scaler ajustado e Validação Cruzada concluída!")
                st.subheader("Desempenho Esperado (Validação Cruzada)")
                st.metric(label="Média do Erro Quadrático Médio (MSE)", value=f"{mean_mse:.4f}")
                
            except Exception as e:
                st.error(f"Erro ao treinar: Verifique se a coluna '{TARGET_COLUMN}' existe no seu CSV. Detalhes: {e}")

# -------------------------------------------------------------
# BLOCO 2: TESTE E APLICAÇÃO (Com Verificação de Rótulos)
# -------------------------------------------------------------
st.header("2. Teste / Aplicação do Modelo")

if st.session_state['modelo'] is not None:
    arquivo_teste = st.file_uploader("Faça upload do arquivo de Teste (.csv ou .zip)", key="teste_uploader", type=['csv', 'zip'])
    
    if arquivo_teste is not None:
        df_teste_raw = load_data_zip(arquivo_teste)
        if df_teste_raw is None:
            st.stop()

        if st.button("🔍 Executar Previsão e Avaliação"):
            try:
                # Cria features de lag nos dados de teste
                df_teste_lagged = create_lag_features(df_teste_raw.reset_index(drop=True), N_LAG)
                
                # --- Preparação dos Dados de Teste ---
                
                # Verificação se tem rótulo (para fins de avaliação)
                tem_rotulo = TARGET_COLUMN in df_teste_lagged.columns
                
                if tem_rotulo:
                    X_test = df_teste_lagged.drop(columns=[TARGET_COLUMN])
                    y_real = df_teste_lagged[TARGET_COLUMN]
                else:
                    # Se não tem rótulo, usa as features de lag criadas, menos a coluna time
                    # que terá NaNs e não será usada, pois não existe na base original
                    X_test = df_teste_lagged.drop(columns=[TARGET_COLUMN], errors='ignore')
                
                # Normalizar usando o scaler TREINADO
                X_test_scaled = st.session_state['scaler'].transform(X_test)
                
                # Prever
                previsoes = st.session_state['modelo'].predict(X_test_scaled)
                
                # Criar DataFrame de resultados
                # Adicionamos as colunas de lag para contexto, mas focamos na previsão
                df_resultado = df_teste_lagged.copy()
                df_resultado['Previsao'] = previsoes
                
                # --- AVALIAÇÃO E DOWNLOAD ---
                st.subheader("Resultados da Previsão")

                if tem_rotulo:
                    # Comparação de desempenho real e esperado
                    mse_real = mean_squared_error(y_real, previsoes)
                    st.metric(label="Desempenho Real (MSE)", value=f"{mse_real:.4f}")
                    st.line_chart(df_resultado[[TARGET_COLUMN, 'Previsao']].tail(100))
                    st.success("Previsão e avaliação de desempenho concluídas.")
                else:
                    # Caso de teste sem rótulos (apenas disponibilizar dados)
                    st.info("Arquivo sem rótulos. Apenas previsões geradas.")

                # Download do CSV (compactado em ZIP)
                compress_and_download(df_resultado, filename='resultados_previsao.zip')
            
            except Exception as e:
                st.error(f"Erro ao executar a previsão: Verifique se o arquivo de teste contém as colunas necessárias e se o número de lags ({N_LAG}) não gerou um erro de dimensão. Detalhes: {e}")

else:
    st.warning("Aguardando o upload do arquivo de treino e o treinamento do modelo (Bloco 1).")
