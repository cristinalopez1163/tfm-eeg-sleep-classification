"""Genera la comparación de todos los experimentos a partir de los 
resultados guardados durante las ejecuiones de los modelos. Calcula la 
media y desviación típica de cada métrica por experimento y genera las 
matrices de confusión"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carpetas con los resultados de cada experimento 
TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(FIGURES_DIR, exist_ok=True)

CLASS_NAMES = ["Wake", "Sleep"]

def summary_table():
    """Calcula los resultados finales de cada experimento (media +- 
    desviación típica) y guarda la tabla comparativa"""

    # Concatena todos los resultados en un único DataFrame
    csv_paths = glob.glob(os.path.join(TABLES_DIR, "*_raw_results.csv"))
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    metric_cols = [c for c in df.columns if c not in ("experiment", "seed", "model")] # columnas de las métricas

    # Calcula la media y desviación típica de cada métrica, para cada experimento
    summary = df.groupby("experiment")[metric_cols].agg(["mean", "std"]).round(4)

    out_path = os.path.join(TABLES_DIR, "comparison_summary.csv")
    summary.to_csv(out_path)
    print("\n=== Resumen media +- std ===\n", summary)
    print(summary)
    print(f"Tabla resumen guardada en {TABLES_DIR}")

def plot_confusion_matrices():
    """Genera y guarda, para cada experimento, su matriz de confusión normalizada
    (proporción de aciertos y errores respecto al total de la clase)"""

    # Concatena los resultados de los experimentos de ML y DL 
    ml_path = os.path.join(TABLES_DIR, "ml_confusion_matrices.npz") # ML
    npz_paths = glob.glob(os.path.join(TABLES_DIR, "*_confusion_matrix.npz")) # DL 
    npz_paths.append(ml_path)

    for path in npz_paths:
        data = np.load(path)
        for exp_name in data.files:
            cm = data[exp_name].astype(float) # convierte la matriz a float 
            cm_norm = cm / cm.sum(axis=1, keepdims=True) # normaliza respecto al total de muestras

            # Creación de la figura
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1) # mapa de calor 

            # Etiquetas de los ejes
            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_NAMES)
            ax.set_yticklabels(CLASS_NAMES)
            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")
            
            # Nombre del experimento como título 
            ax.set_title(f"Matriz de confusión - {exp_name}")

            # Añade el valor numérico de cada celda
            for i in range(len(CLASS_NAMES)):
                for j in range(len(CLASS_NAMES)):
                    color = "white" if cm_norm[i, j] > 0.5 else "black"
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color)

            # Barra de color 
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            # Guarda la matriz de confusión
            out_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{exp_name}.png")
            fig.savefig(out_path, dpi=150)

    print(f"Matrices de confusión guardadas en {FIGURES_DIR}")


def main():
    """Construye la tabla comparativa y genera las matrices de confusión"""

    summary_table()
    plot_confusion_matrices()


if __name__ == "__main__":
    main()