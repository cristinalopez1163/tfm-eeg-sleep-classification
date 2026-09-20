"""
Crea el dataset del modelo de Deep Learning sin contexto temporal (CNN1D)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

class SleepEEGDataset(Dataset):
    """Dataset de Pytorch de la señal EEG sin información temporal: 
    cada muestra es un epoch independiente, normalizada por canal"""

    def __init__(self, X, y, mean=None, std=None):
        X = X.astype(np.float32)       

        # Normalización
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        X = (X - mean) / std

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y


def compute_channel_stats(X):
    """Calcula la media y la desviación típica por canal. Estos valores
    se usarán para normalizar el resto de conjuntos"""
    mean = X.mean(axis=(0, 2), keepdims=True, dtype=np.float32)
    std = X.std(axis=(0, 2), keepdims=True, dtype=np.float32)
    std[std == 0] = 1.0 # evita división por 0
    return mean, std


def subject_train_val_test_split(subject_ids, y, test_size=0.15, val_size=0.15, random_state=42):
    """Divide en train/test/val de forma que ningún 
    sujeto aparezca en más de un conjunto a la vez"""

    groups = subject_ids # cada grupo es un sujeto, así sus epochs no se reparten entre conjuntos

    # Primer split: separa test del resto de conjuntos
    split_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(split_test.split(np.zeros(len(y)), y, groups))

    # Segundo split: separa train de val 
    relative_val_size = val_size / (1 - test_size)
    split_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_sub_idx, val_sub_idx = next(
        split_val.split(np.zeros(len(trainval_idx)), y[trainval_idx], groups[trainval_idx]))

    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    # devuelve los índices de los tres conjuntos 
    return train_idx, val_idx, test_idx