"""
Define la arquitectura CNN1D + BiLSTM, del segundo experimento de Deep Learning. 
Reutiliza el bloque CNNFeatureExtractor que actua como módulo 
extractor de características, pero añade un módulo BiLSTM para procesar la 
ventana de contexto temporal, para que el modelo aproveche la información
de los epochs anteriores y posteriores.  
"""

import torch.nn as nn
from models.cnn1d import CNNFeatureExtractor

class CNNBiLSTM(nn.Module):
    """Modelo con contexto temporal: extractor convolucional CNNFeatureExtractor 
    (aplicado a cada epoch de la ventana)+ BiLSTM (combina la información de la secuencia)
    + clasificador MLP (perceptrón multicapa) sobre el epoch central"""

    def __init__(self, n_channels, n_classes):
        super().__init__()

        # Módulo extractor de características, compartido por todos los epochs de la venta
        self.extractor = CNNFeatureExtractor(n_channels)

        # Dimensión del vector de características
        feat_dim = self.extractor.out_features

        # LSTM bidireccional que procesa la información en ambos sentidos
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=256, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.3)

        # Capas totalmente conectadas encargadas de la clasificación final 
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
        # Vuelve a poner las características en formato secuencial tras haberlas procesado individualmente
        feats = feats.view(batch, seq_len, -1)

        # Procesa la secuencia mediante la BiLSTM
        lstm_out, _ = self.lstm(feats)
        # Se toma la representación del epoch central
        context = lstm_out[:, seq_len // 2, :]

        # Usa esa representación central (con contexto temporal) para generar la predicción
        return self.classifier(context)

