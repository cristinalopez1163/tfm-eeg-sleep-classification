"""
Script principal del pipeline de clasificación de señales EEG:  
   1. Crea o cargar el dataset preprocesado 
   2. Ejecuta los expermientos de clasificación 
        - 2.1 Baseline de Machine Learning (Regresión Logística / Random Forest / XGBoost)
        - 2.2 CNN1D
        - 2.3 CNN1D + BiLSTM
        - 2.4 CNN1D + Transformer 
        - 2.5 CNN1D + Conformer
    3. Guarda los resultados
"""

import logging
import warnings
from pathlib import Path

import mne
import numpy as np
import torch

from data.feature_extraction import extract_features
from data.load_data import get_subject_pairs, load_subject
from data.preprocess import create_dataset, create_epochs, preprocess_signal
from data.sequence_dataset import SequenceEEGDataset, build_valid_centers
from data.torch_dataset import SleepEEGDataset, compute_channel_stats, subject_train_val_test_split
from evaluation.cross_validation import run_cross_validation, summarize_results
from evaluation.train_eval_dl import compute_class_weights, run_multi_seed_experiment
from models.cnn1d import CNN1D
from models.cnn_bilstm import CNNBiLSTM
from models.cnn_transformer import CNNTransformer1D
from models.conformer import CNNConformer
from models.ml import get_models
from utils.labels import CLASS_NAMES, N_CLASSES

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# Experimentos a ejecutar 
RUN_ML = True
RUN_CNN1D = True
RUN_CNN_BILSTM = True
RUN_CNN_TRANSFORMER = True
RUN_CNN_CONFORMER = True
SEQUENCE_MODELS = {
    "CNN_BiLSTM": (CNNBiLSTM, RUN_CNN_BILSTM),
    "CNN_Transformer": (CNNTransformer1D, RUN_CNN_TRANSFORMER),
    "CNN_Conformer": (CNNConformer, RUN_CNN_CONFORMER),
}


# Hiperparámetros
SEEDS = (0, 1, 2)
BATCH_SIZE = 32
LR = 3e-4
MAX_EPOCHS = 100
PATIENCE = 20
SEQ_LEN = 15

# Rutas del proyecto 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "resources" / "raw"
PROCESSED_DIR = BASE_DIR / "resources" / "processed"
LOG_DIR = BASE_DIR / "results" / "logs"
TABLES_DIR = BASE_DIR / "results" / "tables"
CHECKPOINTS_DIR = BASE_DIR / "results" / "checkpoints"
for _dir in (PROCESSED_DIR, LOG_DIR, TABLES_DIR, CHECKPOINTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DATASET_PATH = PROCESSED_DIR / "sleep_dataset.npz"
FEATURES_PATH = PROCESSED_DIR / "ml_features.npz"

def setup_logger(name, filename):
    """Crea o carga, si ya existe, un logger para registrar el estado de la ejecución"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(LOG_DIR / filename, mode="w")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)

    return logger

def build_dataset():
    """Recorre los sujetos en crudo, preprocesa, crea epochs y guarda
    el dataset final en resources/processed/sleep_dataset.npz"""
    X_all, y_all = [], []
    subject_ids, recording_ids = [], []

    # Logger de preprocesado 
    logger = setup_logger("preprocess", "preprocessing.log")
    pairs = get_subject_pairs(DATA_PATH)
    logger.info(f"Sujetos encontrados: {len(pairs)}")

    # Bucle que recorre cada fichero 
    for i, (psg_path, hyp_path) in enumerate(pairs):
        # Extracción del identificador de sujeto y de grabación a partir del nombre de fichero 
        subject_id = Path(psg_path).stem[:5]
        recording_id = Path(psg_path).stem[:6]
        logger.info(f"--- Muestra {i + 1}/{len(pairs)} ({subject_id}, grabación {recording_id}) ---")

        # Carga señal EEG + anotaciones de sueño 
        raw = load_subject(psg_path, hyp_path)
        # Filtrado y limpieza de la señal 
        raw = preprocess_signal(raw)
        # Segmentación en epochs 
        epochs = create_epochs(raw)

        X, y = create_dataset(epochs)
        # Distribución de clases del sujeto 
        unique, counts = np.unique(y, return_counts=True)
        for clase, cantidad in zip(unique, counts):
            logger.info(f"{CLASS_NAMES.get(clase, f'Clase {clase}')}: {cantidad}")

        # Concatenación de epochs del fichero  
        X_all.append(X)
        y_all.append(y)
        # A cada epoch se le asigna el id de sujeto y de grabación 
        subject_ids.extend([subject_id] * len(y))
        recording_ids.extend([recording_id] * len(y))

    # Concatenación de los epochs de todos los sujetos 
    X_all = np.concatenate(X_all, axis=0, dtype=np.float32)
    y_all = np.concatenate(y_all, axis=0, dtype=np.float32)
    subject_ids = np.array(subject_ids)
    recording_ids = np.array(recording_ids)

    # Carga en el logger la información del dataset final 
    logger.info("========= DATASET FINAL =========")
    logger.info(f"Shape X: {X_all.shape} | Shape y: {y_all.shape}")
    # Distribuación de clases
    unique, counts = np.unique(y_all, return_counts=True)
    for clase, cantidad in zip(unique, counts):
        logger.info(f"{CLASS_NAMES.get(clase, f'Clase {clase}')}: {cantidad}")

    # Guarda el dataset en la ruta correspondiente
    np.savez(DATASET_PATH, X=X_all, y=y_all, subject_ids=subject_ids, recording_ids=recording_ids)
    logger.info(f"Dataset guardado en {DATASET_PATH}")

    return X_all, y_all, subject_ids, recording_ids


def load_or_build_dataset():
    """Devuelve el dataset ya procesado. Si no existe, lo crea; si existe lo carga"""
    if DATASET_PATH.exists():
        data = np.load(DATASET_PATH, allow_pickle=True)
        print(f"Dataset ya existente encontrado en {DATASET_PATH}, se reutiliza")
        return data["X"], data["y"], data["subject_ids"], data["recording_ids"]

    return build_dataset()

def save_dl_results(name, results_df, summary, conf_matrix):
    """Guarda, con un formato común, los resultados de los experimentos"""
    prefix = name.lower()
    # Resultados individuales por semilla 
    results_df.to_csv(TABLES_DIR / f"{prefix}_raw_results.csv", index=False)
    # Resumen de las semillas (media +- desviación típica)
    summary.to_csv(TABLES_DIR / f"{prefix}_summary.csv")
    # Matriz de confusión en test
    np.savez(TABLES_DIR / f"{prefix}_confusion_matrix.npz", **{name: conf_matrix})


def run_ml(X, y, subject_ids):
    """Extrae las características manuales, entrena los 
    modelos de Machine Learning y guarda los resultados"""

    # Configuración del logger 
    logger = setup_logger("ml", "ml.log")
    logger.info("===== BASELINE ML =====")

    # Si las características existen, se cargan; si no, se calculan
    if FEATURES_PATH.exists():
        data = np.load(FEATURES_PATH, allow_pickle=True)
        X_feat, feature_names = data["X_feat"], data["feature_names"].tolist()
    else:
        # Cálculo de características
        X_feat, feature_names = extract_features(X, sfreq=100)
        # Se guardan en memoria 
        np.savez_compressed(FEATURES_PATH, X_feat=X_feat, feature_names=np.array(feature_names))

    logger.info(f"Features extraídas: shape {X_feat.shape} -> {feature_names}")

    # Diccionario de modelos a ejecutar
    models = get_models()
    # Ejecución de los modelos 
    results_df, conf_matrices = run_cross_validation(
        X_feat, y, subject_ids, models, logger, n_splits=5, seeds=SEEDS,
    )

    # Se guardan los resultados de cada modelo en memoria 
    results_df.to_csv(TABLES_DIR / "ml_raw_results.csv", index=False)
    summary = summarize_results(results_df)
    summary.to_csv(TABLES_DIR / "ml_summary.csv")
    np.savez(TABLES_DIR / "ml_confusion_matrices.npz", **conf_matrices)

    logger.info(f"===== Resumen (media +- std) =====\n{summary}")
    print(summary)

    return results_df, summary, conf_matrices


def run_cnn1d(X, y, subject_ids):
    """Entrena el modelo CNN1D sobre la señal en crudo, clasificando 
    cada epoch de forma independiente (sin información temporal)"""

    # Configuración del logger 
    logger = setup_logger("cnn1d", "cnn1d.log")
    logger.info("===== CNN1D =====")

    # GPU si está disponible, si no CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # División en train/test/val manteniendo la separación por sujetos 
    train_idx, val_idx, test_idx = subject_train_val_test_split(
        subject_ids, y, test_size=0.15, val_size=0.15, random_state=42)

    # Pesos de cada clase
    class_weights = compute_class_weights(y[train_idx]) 
    logger.info(f"N train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Normalización de los conjuntos 
    mean, std = compute_channel_stats(X[train_idx]) # calculo de la media y desviación típica del conjunto de train
    train_ds = SleepEEGDataset(X[train_idx], y[train_idx], mean, std) 
    val_ds = SleepEEGDataset(X[val_idx], y[val_idx], mean, std)
    test_ds = SleepEEGDataset(X[test_idx], y[test_idx], mean, std)

    n_channels = X.shape[1]
    # Creación de un modelo por cada semilla 
    model_fn = lambda: CNN1D(n_channels=n_channels, n_classes=N_CLASSES)

    # Entrena y evalúa con los hiperparámetros definidos previamente
    results_df, conf_matrix = run_multi_seed_experiment(
        model_fn, train_ds, val_ds, test_ds, device,
        class_weights, SEEDS, BATCH_SIZE,
        MAX_EPOCHS, PATIENCE, LR,
        logger, "CNN1D", CHECKPOINTS_DIR
    )

    # Resultados 
    summary = summarize_results(results_df) # media +- std de las semillas
    save_dl_results("CNN1D", results_df, summary, conf_matrix)
    logger.info(f"===== Resumen (media +- std) sobre test =====\n{summary}")
    print(summary)

    return results_df, summary, conf_matrix


def run_sequence_experiment(name, model_class, X, y, subject_ids, recording_ids, seq_len=SEQ_LEN):
    """
    Entrena y evalúa los modelos con información temporal
    (CNN + BiLSTM, CNN + Transformer y CNN + Conformer)
    """

    # Configuración del logger 
    logger = setup_logger(name.lower(), f"{name.lower()}.log")
    logger.info(f"===== {name} (contexto de {seq_len} epochs) =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Tamaño de la secuencia a cada lado del centro 
    half_window = seq_len // 2
    # Solo se consideran centros cálidos aquellos que tienen una centana completa a su alrededor 
    valid_centers = build_valid_centers(recording_ids, half_window)
    # Los que están demasiado cerca del principio o del final, se descartan 
    logger.info(
        f"Centros válidos: {len(valid_centers)}/{len(y)} epochs "
        f"({len(y) - len(valid_centers)} descartadas por no tener ventana completa)"
    )

    # División en train/test/val manteniendo la separación por sujetos 
    train_rel, val_rel, test_rel = subject_train_val_test_split(
        subject_ids[valid_centers], y[valid_centers],
        test_size=0.15, val_size=0.15, random_state=42,
    )

    train_centers = valid_centers[train_rel]
    val_centers = valid_centers[val_rel]
    test_centers = valid_centers[test_rel]

    # Pesos de cada clase
    class_weights = compute_class_weights(y[train_centers]) 
    logger.info(f"N train={len(train_centers)} val={len(val_centers)} test={len(test_centers)}")

    # Normalización de los conjuntos según las estadísticas calculadas sobre el de train
    train_subjects = set(subject_ids[train_centers])
    train_mask = np.isin(subject_ids, list(train_subjects))
    mean, std = compute_channel_stats(X[train_mask])
    train_ds = SequenceEEGDataset(X, y, train_centers, seq_len, mean, std)
    val_ds = SequenceEEGDataset(X, y, val_centers, seq_len, mean, std)
    test_ds = SequenceEEGDataset(X, y, test_centers, seq_len, mean, std)

    n_channels = X.shape[1]
    # Creación de un modelo por cada semilla 
    model_fn = lambda: model_class(n_channels, N_CLASSES)

    # Entrena y evalúa con los hiperparámetros definidos previamente
    results_df, conf_matrix = run_multi_seed_experiment(
        model_fn, train_ds, val_ds, test_ds, device,
        class_weights, SEEDS, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR,
        logger, name, CHECKPOINTS_DIR
    )

    # Resultados
    summary = summarize_results(results_df) # media +- std sobre el conjunto de test 
    save_dl_results(name, results_df, summary, conf_matrix)
    logger.info(f"===== Resumen (media +- std) sobre test =====\n{summary}")
    print(summary)

    return results_df, summary, conf_matrix


def main():
    """Ejecución completa: crea/carga el dataset y ejecuta los experimentos indicados"""
    X, y, subject_ids, recording_ids = load_or_build_dataset()

    # Machine Learning
    if RUN_ML:
        run_ml(X, y, subject_ids)

    # CNN1D
    if RUN_CNN1D:
        run_cnn1d(X, y, subject_ids)

    # Experimentos secuenciales
    for name, (model_class, should_run) in SEQUENCE_MODELS.items():
        if should_run:
            run_sequence_experiment(name, model_class, X, y, subject_ids, recording_ids)


if __name__ == "__main__":
    main()

