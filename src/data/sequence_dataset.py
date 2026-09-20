"""
Crea el dataset de PyTorch utilizado por los modelos secuenciales 
(CNN1D + BiLSTM, CNN1D + Transformer, CNN1D + Conformer)
"""

import numpy as np
import torch
from torch.utils.data import Dataset

def build_valid_centers(recording_ids, half_window):
    """Recorre las grabaciones y selecciona los ínidces que tienen un número suficiente 
    de epochs a cada lado (es decir, tienen suficiente información temporal)"""

    valid_centers = []
    n = len(recording_ids)
    start = 0 # índice de inicio 

    for i in range(1, n + 1):
        if i == n or recording_ids[i] != recording_ids[start]: # final de la grabación o del array
            # Define el bloque de la grabación [start, end)
            end = i  
            block_len = end - start # número de epochs de la grabación

            # Comprueba que existen suficientes epochs a ambos lados para construir la ventana temporal completa
            if block_len >= 2 * half_window + 1:
                # Selección de los índices que pueden actuar como centros evitando los extremos de la grabación 
                block_centers = np.arange(start + half_window, end - half_window)
                valid_centers.append(block_centers)

            # Se pasa al siguiente bloque
            start = i

    return np.concatenate(valid_centers)


class SequenceEEGDataset(Dataset):
    """Dataset de PyTorch que, para cada epoch central válido, devuelve la ventana de 
    seq_len epochs consecutivos (el epoch central y los vecinos a cada lado) junto 
    con la etiqueta del epoch central."""

    def __init__(self, X, y, center_indices, seq_len, mean=None, std=None):
        assert seq_len % 2 == 1, "seq_len debe ser impar (ventana simétrica)"

        self.X = X
        self.y = y
        self.center_indices = np.asarray(center_indices) # índices del centro de la ventana
        self.half_window = seq_len // 2
        self.seq_len = seq_len

        # Media y desviación típica por canal, para normalizar 
        self.mean = mean.astype(np.float32) 
        self.std = std.astype(np.float32) 

    def __len__(self):
        return len(self.center_indices)

    def __getitem__(self, idx):
        center = self.center_indices[idx] # índice del epoch central
        start = center - self.half_window # primer índice de la ventana 
        end = center + self.half_window + 1 # último índice de la ventana

        window = self.X[start:end].astype(np.float32)  # ventana completa

        # Normalización por canal con las estadísticas de train
        window = (window - self.mean) / self.std

        # Ventana y etiqueta asociada
        window = window.astype(np.float32)
        label = np.int64(self.y[center])

        return torch.from_numpy(window), torch.tensor(label)