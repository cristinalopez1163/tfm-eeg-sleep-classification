"""
Define la arquitectura CNN1D + Transformer, del tercer experimento de Deep Learning. 
Reutiliza el bloque CNNFeatureExtractor que actua como módulo 
extractor de características y la secuencia de epochs se modela con un Transformer.
"""

import math
import torch
import torch.nn as nn

from models.cnn1d import CNNFeatureExtractor


class PositionalEncoding(nn.Module):
    """Codificación posicional (porque el mecanismo de self-attention del Transformer no 
    distingue por sí mismo el orden de los epochs en la secuencia)"""

    def __init__(self, d_model, max_len=61):
        super().__init__()

        # Iniciliza la matriz de posiciones 
        pe = torch.zeros(max_len, d_model)
        # Posiciones de cada epoch de la secuencia
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Calcula las frecuencias 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Emplea seno y coseno para codificar las posiciones (funciones suaves que oscilan 
        # entre -1 y 1 para evitar problemas de rango o magnitud)
        pe[:, 0::2] = torch.sin(position * div_term) # funciones seno a las posiciones pares 
        pe[:, 1::2] = torch.cos(position * div_term) # funciones coseno a las posiciones impares
        # Registra la posición
        self.register_buffer("pe", pe.unsqueeze(0))  

    def forward(self, x):
        """Añade la información posicional a las características de la secuencia"""

        # Añade a cada epoch la codificación correspondiente a su posición temporal
        return x + self.pe[:, : x.size(1), :]


class CNNTransformer1D(nn.Module):
    """Modelo CNNTransformer que combina un extractor convolucional de características con 
    un Transformer para incorporar información temopral entre epochs"""

    def __init__(self, n_channels, n_classes):
        super().__init__()

        # Módulo extractor de características que transforma cada epoch en un vector de 512 características
        self.extractor = CNNFeatureExtractor(n_channels)
        # Información temporal de cada epoch 
        self.pos_encoder = PositionalEncoding(512, max_len=15)
        # Capa Transformer que modela las relaciones entre epochs 
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024,
                                                    dropout=0.3, batch_first=True)
        # Construye el Transformer a partir de varias capas encoder 
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Clasificador MLP cargado de transformar la representación temporal en una predicción
        self.classifier = nn.Sequential(

            # Primera capa: de 512 características a 256
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # Segunda capa: de 256 características a 128
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # Tercera capa: de 128 características a 64
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            # Capa de salida: una neurona por clase
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        """Procesa la secuencia de epochs y genera la predicción del epoch central"""

        batch, seq_len, n_channels, n_samples = x.shape # obtiene las dimensiones de la secuencia
        x = x.view(batch * seq_len, n_channels, n_samples) # organiza la entrada para procesar cada epoch con la CNN

        # Extrae las características de cada epoch con la CNN
        feats = self.extractor(x)                 
        feats = feats.view(batch, seq_len, -1)   

        # Añade información sobre la posición temporal
        feats = self.pos_encoder(feats)
        # Procesa la secuencia mediante el Transformer Encoder 
        encoded = self.transformer(feats)        

        # Se toma la representación del epoch central
        center_idx = seq_len // 2

        # Usa esa representación para generar la predicción
        return self.classifier(encoded[:, center_idx, :])