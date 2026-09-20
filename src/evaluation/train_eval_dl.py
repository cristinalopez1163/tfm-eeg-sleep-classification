"""
Bucle de entrenamiento de los modelos de Deep Learning, 
evaluación de cada clasificador y guardado del mejor modelo 
junto con sus resultados. 
"""

import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from utils.labels import CLASSES, LABELS, N_CLASSES


def compute_class_weights(y):
    """Calcula los pesos por clase, inversamente proporcional 
    a su frecuencia, para usarlo en la función de pérdida"""

    classes, counts = np.unique(y, return_counts=True) # clases presentes y su número de muestras
    freq = counts / counts.sum() # frecuencia de cada clase
    # A menor frecuencia, mayor peso
    weights = 1.0 / freq
    weights = weights / weights.sum() * len(classes) # normalización para que la media de los pesos sea 1

    # Vector de pesos
    weight_tensor = torch.zeros(len(CLASSES), dtype=torch.float32) 
    for c, w in zip(classes, weights):
        weight_tensor[int(c)] = float(w)
    return weight_tensor


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """Evalúa el modelo y devuelve las métricas (accuracy, F1-macro, 
    F1 por clase, índice kappa de cohen y matriz de confusión)"""

    model.eval() # desactiva el modo de entrenamiento 
    all_preds, all_targets = [], [] # predicciones y etiquetas reales
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x) # salida del modelo 
        total_loss += criterion(logits, y).item() * x.size(0) # se acumula la pérdida por batch

        all_preds.append(logits.argmax(dim=1).cpu().numpy()) # clase predicha = logit máximo
        all_targets.append(y.cpu().numpy())

    # Se unen las predicciones de todos los batches 
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Se calculan las métricas
    metrics = {
        "loss": (total_loss / len(loader.dataset)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=LABELS) # F1 por clase 
    for idx, name in enumerate(CLASSES):
        metrics[f"f1_{name}"] = f1_per_class[idx]
    cm = confusion_matrix(y_true, y_pred, labels=LABELS) # matriz de confusión

    return metrics, cm


def train_model(model, train_dataset, val_dataset, device, class_weights, 
                batch_size, max_epochs, patience, lr, logger, run_name):
    """Entrena el modelo aplicando los hiperparámetros correspondientes y 
    devuelve el modelo con los pesos del mejor epoch"""

    # Muestras de train y de validación (el orden varía en cada epoch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)

    # Pesos por clase
    weight_tensor = class_weights.to(device) 
    # Función de pérdida ponderada para descompensar el desbalanceo
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Optimizador y método de reducción adaptativo del learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)

    # Inicializamos las variables 
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    # Bucle de entrenamiento 
    for epoch in range(max_epochs):

        model.train() # activa modo entrenamiento 
        start = time.time()
        running_loss = 0.0

        # Bucle de entrenamiento por batches 
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad() # resetea los gradientes del batch anterior
            loss = criterion(model(x), y) # calcula las predicciones y la funcion de pérdida
            loss.backward() # retropropagación del gradiente 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # limita la magnitud de los gradientes 
            optimizer.step() # actualiza loss pesos del modelo 

            # Se calcula la pérdida ponderada por el tamaño de batch 
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_dataset) # pérdida media del entrenamiento en esa época 
        val_metrics, _ = evaluate(model, val_loader, device, criterion) # evaluación del modelo sobre el conjunto de validación
        scheduler.step(val_metrics["f1_macro"]) # ajusta el lr según el F1-macro de validación

        # Registra la información de la época
        msg = (
            f"[{run_name}] epoch {epoch + 1}/{max_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} tiempo={time.time() - start:.1f}s"
        )
        print(msg)
        logger.info(msg)

        # Si mejora el F1-macro guarda una copia de los pesos 
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1 # cuenta las épocas consecutivas sin mejora 

        # Detiene el entrenamiento si no hay mejora durante varias époxas
        if epochs_no_improve >= patience:
            msg = f"[{run_name}] early stopping en epoch {epoch + 1} (mejor val_f1_macro={best_f1:.4f})"
            print(msg)
            logger.info(msg)
            break

    # Recupera los pesos de la mejor época
    model.load_state_dict(best_state)
    return model


def run_multi_seed_experiment(model_fn, train_dataset, val_dataset, test_dataset, device, class_weights, seeds, 
                              batch_size, max_epochs, patience, lr, logger, experiment_name, checkpoint_dir):
    """Entrena y evalúa el modelo para cada semilla y guarda los mejores resultados"""

    # Inicialización de las variables 
    rows = []
    conf_matrix_total = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    best_f1_macro = -1.0
    best_state_dict = None
    best_info = None

    # Muestras de test 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    weight_tensor = class_weights.to(device) # vector de pesos 
    # Función de pérdida de test
    test_criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Se repite el proceso de entrenamiento y evaluación para cada semilla
    for seed in seeds:

        # Fija la semillas en NumPy, PyTorch y CUDA para garantizar la reproducibilidad
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        run_name = f"{experiment_name}_seed{seed}"
        start = time.time()

        # Se crea un modelo nuevo para cada semilla
        model = model_fn()
        # Entrena el modelo empleando el conjunto de entrenamiento y de validación
        model = train_model(model, train_dataset, val_dataset, device, class_weights, 
                            batch_size, max_epochs, patience, lr, logger, run_name)
        train_time_sec = time.time() - start # duración del entrenamiento

        # Evaluación final sobre el conjunto de test 
        test_metrics, cm = evaluate(model, test_loader, device, test_criterion)
        conf_matrix_total += cm

        # Se añaden las métricas de test a los resultados
        row = {"experiment": experiment_name, "seed": seed, "train_time_sec": train_time_sec}
        row.update(test_metrics)
        rows.append(row)

        msg = f"[{run_name}] TEST -> {test_metrics} (train_time={train_time_sec:.1f}s)" # mensaje de resultados
        print(msg)
        logger.info(msg)

        # Guarda la información del modelo con mejor F1-macro entre semillas
        if test_metrics["f1_macro"] > best_f1_macro:
            best_f1_macro = test_metrics["f1_macro"]
            best_state_dict = copy.deepcopy(model.state_dict())
            best_info = {
                "experiment": experiment_name, "seed": seed,
                "test_metrics": {k: (float(v)) for k, v in test_metrics.items()},
            }

    # Guarda el mejor modelo 
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / f"{experiment_name}_best.pt"
    info_path = checkpoint_dir / f"{experiment_name}_best_info.json"

    torch.save(best_state_dict, model_path) # pesos 
    with open(info_path, "w") as f:
        json.dump(best_info, f, indent=2) # métricas e información asociada 

    msg = (f"[{experiment_name}] Mejor modelo (seed={best_info['seed']}, "
            f"f1_macro={best_f1_macro:.4f}) guardado en {model_path}") # informa de qué modelo se ha guardado
    print(msg)
    logger.info(msg)

    return pd.DataFrame(rows), conf_matrix_total

