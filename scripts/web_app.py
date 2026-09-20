"""
App web para visualizar las predicciones del CNN + Conformer 
sobre un nuevo registro EEG subido por el usuario.
"""

import tempfile
from pathlib import Path
import sys 

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_DIR / "results" / "checkpoints" / "CNN_Conformer_best.pt"
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import torch
from sklearn.metrics import f1_score

from data.load_data import load_subject
from data.preprocess import preprocess_signal, create_epochs, create_dataset
from models.conformer import CNNConformer

# Parámetros
SEQ_LEN = 15
CLASS_NAMES = {0: "Wake", 1: "Sleep"}
MODEL_KWARGS = dict(n_channels=2, n_classes=2)

# Configuración inicial de la página
st.set_page_config(page_title="EEG Sleep/Wake", layout="wide")
st.title("EEG Sleep/Wake")


@st.cache_resource
def load_model(checkpoint_path):
    """Carga el modelo entrenado"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU si está disponible
    model = CNNConformer(**MODEL_KWARGS) # arquitectura del modelo
    # Carga los pesos del modelo entrenado
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True) 
    model.load_state_dict(state_dict)
    # Modelo en modo evaluación
    model.to(device).eval()
    return model, device


def normalize_data(X):
    """Normaliza cada canal de la señal"""

    mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32)


def build_sequence(X, center_idx, seq_len):
    """Construye la secuencia de epochs alrededor del epoch central"""

    half = seq_len // 2
    sequence = X[center_idx - half: center_idx + half + 1]
    return np.expand_dims(sequence, axis=0).astype(np.float32)


@torch.no_grad()
def predict_epochs(model, device, X, seq_len):
    """Predice el estado de cada epoch central de la señal"""

    half = seq_len // 2
    centers, predictions = [], []
    for center_idx in range(half, len(X) - half):

        # Construye la secuencia centrada en el epoch actual 
        sequence = build_sequence(X, center_idx, seq_len)
        tensor = torch.from_numpy(sequence).to(device)
        # Obtiene la clase con mayor probabilidad
        pred = torch.argmax(model(tensor), dim=1).item()
        centers.append(center_idx)
        predictions.append(pred)

    return np.array(centers), np.array(predictions)


def format_time(seconds):
    """Convierte segundos a formato HH:MM:SS"""

    seconds = int(seconds)
    # Obtiene horas, minutos y segundos
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def create_result_plot(centers, predictions, y_true):
    """Genera el hipnograma real y los errores de clasificación"""

    # etiquetas reales de los epochs analizados
    real_seq = np.array([int(y_true[c]) for c in centers])
    times = centers * 30 # convierte los índices de los epochs a segundos

    # Hipnograma real
    fig = go.Figure()
    x_green = np.append(times, times[-1] + 30)
    y_green = np.append(real_seq, real_seq[-1])
    fig.add_trace(go.Scatter(
        x=x_green, y=y_green, mode="lines", name="Estado real",
        line=dict(color="lightgreen", width=3, shape="hv"),
        customdata=[format_time(t) for t in x_green],
        hovertemplate="Tiempo: %{customdata}<br>Estado real: %{y}<extra></extra>",
    ))

    # Errores del modelo
    x_err, y_err, custom_err = [], [], []
    for i, center in enumerate(centers):
        real, pred = int(real_seq[i]), int(predictions[i])

        # Solo se representan los epochs clasificados incorrectamente
        if real != pred:
            start, end = center * 30, center * 30 + 30
            x_err += [start, end, None]
            y_err += [real, real, None]
            custom_err += [
                (format_time(start), CLASS_NAMES[real], CLASS_NAMES[pred]),
                (format_time(start), CLASS_NAMES[real], CLASS_NAMES[pred]),
                (None, None, None),
            ]

    if x_err:
        fig.add_trace(go.Scatter(
            x=x_err, y=y_err, mode="markers", name="Errores del modelo",
            line=dict(color="red", width=0.1),
            customdata=custom_err,
            hovertemplate=(
                "<b>Error</b><br>Tiempo: %{customdata[0]}"
                "<br>Real: %{customdata[1]}"
                "<br>Predicción: %{customdata[2]}<extra></extra>"
            ),
        ))

    # Eje Y: Wake / Sleep
    fig.update_yaxes(tickmode="array", tickvals=[0, 1], ticktext=["Wake", "Sleep"], range=[-0.2, 1.2], fixedrange=True)

    # Eje X: segundos (mostrados en formato HH:MM:SS)
    tickvals = np.linspace(0, times[-1] + 30, num=8)
    fig.update_xaxes(title="Tiempo", tickvals=tickvals, ticktext=[format_time(t) for t in tickvals])

    fig.update_layout(hovermode="closest", margin=dict(l=80, r=30, t=65, b=50),)

    return fig


# Sidebar: subida de archivos
st.sidebar.header("Configuración")
psg_file = st.sidebar.file_uploader("PSG (.edf)", type=["edf"])
hyp_file = st.sidebar.file_uploader("Hypnogram (.edf)", type=["edf"])

checkpoint_path = CHECKPOINT_PATH

if psg_file and hyp_file and checkpoint_path.exists():
    if st.button("Analizar señal", type="primary"):
        with st.spinner("Procesando señal y generando predicciones..."):

            # Guarda los archivos subidos en una carpeta temporal
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                psg_path = tmpdir / psg_file.name
                hyp_path = tmpdir / hyp_file.name
                psg_path.write_bytes(psg_file.getbuffer())
                hyp_path.write_bytes(hyp_file.getbuffer())

                raw = load_subject(psg_path, hyp_path) # carga el registro EEG
                raw = preprocess_signal(raw) # preprocesa la señal 
                epochs = create_epochs(raw) # divide la señal en epochs 
                X, y = create_dataset(epochs) # obtiene las señales y sus etiquetas

            # Normaliza el dataset, carga el modelo y genera las predicciones
            X = normalize_data(X)
            model, device = load_model(checkpoint_path)
            centers, predictions = predict_epochs(model, device, X, SEQ_LEN)

        st.subheader("Resultados")
        st.plotly_chart(create_result_plot(centers, predictions, y), use_container_width=True)

        # Métricas
        real_for_predictions = np.array([y[c] for c in centers])
        total_epochs = len(predictions)
        errors = int(np.sum(predictions != real_for_predictions))
        f1_macro = f1_score(real_for_predictions, predictions, average="macro", zero_division=0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Epochs analizados", total_epochs)
        col2.metric("Errores", errors)
        col3.metric("F1-macro", f"{f1_macro * 100:.1f}%")

elif not checkpoint_path.exists():
    st.info("No se ha encontrado modelo")
else:
    st.info("Sube un PSG (.edf) y su Hypnogram (.edf)")