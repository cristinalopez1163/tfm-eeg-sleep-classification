"""
Ejecuta la validación cruzada de los modelos de Machine Learning, 
evalúa cada clasificador y guarda el mejor modelo con sus resultados. 
"""

import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from utils.labels import CLASSES, LABELS, N_CLASSES


# Carpeta donde se guardan los mejores modelos
CHECKPOINT_DIR = (Path(__file__).resolve().parents[2] / "results" / "checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def get_sample_weights(y):
    """Calcula los pesos de forma inversamente proporcional a la frecuencia de clase"""
    classes, counts = np.unique(y, return_counts=True) # clases presentes y su número de muestras
    freq = counts / counts.sum() # frecuencia de cada clase
    weight_per_class = {c: 1.0 / f for c, f in zip(classes, freq)}  # A menor frecuencia, mayor peso
    return np.array([weight_per_class[label] for label in y])


def save_best_model(model_name, model, seed, fold, metrics, n_train, n_test):
    """Guarda el mejor modelo y la información asociada"""

    name = (str(model_name).strip().replace(" ", "_").replace("/", "_"))
    model_path = CHECKPOINT_DIR / f"{name}_best.pkl"
    info_path = CHECKPOINT_DIR / f"{name}_best_info.json"

    # Guarda el modelo entrenado
    joblib.dump(model, model_path)

    # Información del experimento 
    info = {
        "experiment": model_name,
        "seed": int(seed),
        "fold": int(fold),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "model_path": str(model_path),
    }

    # Datos en formato json 
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False,)

    return model_path, info_path


def run_cross_validation(X_feat, y, groups, models_dict, logger, n_splits=5, seeds=(0, 1, 2)):
    """Validación cruzada agrupada por sujeto, repetido para cada semilla y para cada modelo"""

    start_total = time.time()

    all_rows = []

    # Matrices de confusión por modelo 
    conf_matrices = {name: np.zeros((N_CLASSES, N_CLASSES), dtype=int) for name in models_dict}

    # Guardamos el mejor modelo (según F1-macro) de cada algoritmo
    best_models = {name: {"f1_macro": -1.0, "model": None} for name in models_dict}

    logger.info("===== BASELINE ML =====")
    logger.info(f"Semillas: {seeds} | folds por semilla: {n_splits}")

    for name, pipeline in models_dict.items():
        logger.info(f"===== MODELO: {name} =====")
        for seed in seeds:
            # Validación cruzada estratificada por grupo
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            # Asegura que ningún sujeto aparezca en más de un conjunto 
            for fold, (train_idx, test_idx) in enumerate(cv.split(X_feat, y, groups)):

                X_train = X_feat[train_idx]
                X_test = X_feat[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]

                # Crea una copia independiente para cada fold. Así no se reutilizan los pesos
                model = clone(pipeline)
                start = time.time() # tiempo inicial

                sample_weights = get_sample_weights(y_train) # pesos para compensar el desbalanceo de clases 
                model.fit(X_train, y_train, clf__sample_weight=sample_weights) # entrenamiento del modelo 
                train_time_sec = time.time() - start # duración del entrenamiento
                y_pred = model.predict(X_test) # predicciones sobre el conjunto de test 

                # Métricas del modelo 
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                    "cohen_kappa": cohen_kappa_score(y_test, y_pred),
                    "train_time_sec": train_time_sec,
                }
                f1_per_class = f1_score(y_test, y_pred, average=None, labels=LABELS, zero_division=0) # F1 por clase
                for i, class_name in enumerate(CLASSES):
                    metrics[f"f1_{class_name}"] = f1_per_class[i]
                conf_matrices[name] += confusion_matrix(y_test, y_pred, labels=LABELS) # matriz de confusión 

                # Información del experimento 
                row = {"experiment": name, "seed": seed} # nueva fila de resultados 
                row.update(metrics) 
                all_rows.append(row) # añade los resultados del fold

                # Resumen del fold 
                msg = (
                    f"[{name}_seed{seed}_fold{fold}] "
                    f"acc={metrics['accuracy']:.4f} "
                    f"f1_macro={metrics['f1_macro']:.4f} "
                    f"kappa={metrics['cohen_kappa']:.4f} "
                    f"tiempo={train_time_sec:.1f}s"
                )
                print(msg)
                logger.info(msg)

                # Comprueba si el modelo actual obtiene el mejor F1-macro 
                if (metrics["f1_macro"] > best_models[name]["f1_macro"]):
                    # En ese caso, actualiza el mejor modelo del algoritmo
                    best_models[name] = {
                        "f1_macro": metrics["f1_macro"],
                        "model": model,
                        "seed": seed,
                        "fold": fold,
                        "test_metrics": metrics,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                    }

  
    # Resultados globales
    results_df = pd.DataFrame(all_rows)
    summary = summarize_results(results_df)

    logger.info(f"\n{summary}")
    print(summary)

    # Guarda el mejor modelo de cada algoritmo 
    print("\n===== MEJORES MODELOS =====")
    for name, best in best_models.items():
        if best["model"] is None:
            continue

        model_path, _ = save_best_model(name, best["model"], best["seed"], best["fold"], 
                                        best["test_metrics"], best["n_train"], best["n_test"])
        msg = (
            f"[{name}] Mejor modelo -> "
            f"seed={best['seed']} "
            f"fold={best['fold']} "
            f"f1_macro={best['f1_macro']:.4f} "
            f"guardado en {model_path}"
        )
        print(msg)
        logger.info(msg)

    # Tiempo total de ejecución
    total_elapsed = time.time() - start_total

    msg = (f"\nValidación cruzada ML finalizada " f"en {total_elapsed:.1f}s")
    print(msg)
    logger.info(msg)

    return results_df, conf_matrices


def summarize_results(results_df):
    """Calcula media y desviación estándar de las métricas"""

    metric_cols = [c for c in results_df.columns if c not in ("experiment", "seed")]
    # Agrupa los resultados por modelo y calcula la media y desviación de cada métrica 
    return results_df.groupby("experiment")[metric_cols].agg(["mean", "std"])

