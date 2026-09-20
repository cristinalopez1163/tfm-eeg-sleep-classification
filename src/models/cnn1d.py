"""
Define la arquitectura CNN1D empleada en el primer experimento de Deep Learning. 
Además implementa el bloque CNNFeatureExtractor que actuará como módulo 
extractor de características para todas las arquitecturas de Deep Learning. 
"""

import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    """Extractor convolucional 1D que transforma la señal EEG en un vector de características"""

    def __init__(self, n_channels):
        super().__init__()

        self.features = nn.Sequential(
            # Bloque convolucional 1: extrae características iniciales y reduce la dimensión temporal
            nn.Conv1d(n_channels, 64, kernel_size=50, stride=6, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),

            # Bloque convolucional 2: emplea la representación del bloque previo para combinar los patrones detectados
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Bloque convolucional 3: continúa la extracción de características y reduce de nuevo la dimensión temporal
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.3),

            # Bloque convolucional 4: la red aprende a representar una mayor variedad de patrones de la señal 
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # Bloque convolucional 5: refina las características de alto nivel previas 
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # Bloque convolucional 6: representación final de características antes del average pooling y max pooling
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Average pooling para resumir cada canal mediante el valor medio 
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Max pooling para conservar el valor máximo de cada canal 
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.out_features = 512 # 256 características del average pooling y 256 del max pooling

    def forward(self, x):
        """Define como pasan los datos a través de la red neuronal. 
        Transforma la señal EEG en un vector de 512 características"""

        x = self.features(x) # extrae las características 
        avg = self.avg_pool(x) # resumen global mediante average y max pooling
        mx = self.max_pool(x)
        x = torch.cat([avg, mx], dim=1) # concatena ambas representaciones
        return x.squeeze(-1) # se elimina la última dimensión 


class CNN1D(nn.Module):
    """Modelo completo CNN1D: extractor convolucional de características (CNNFeatureExtractor) + 
    clasificador MLP (perceptrón multicapa), para clasificar cada epoch de forma independiente"""

    def __init__(self, n_channels, n_classes):
        super().__init__()

        # módulo extractor de características que transforma la señal EEG en un vector de 512 características
        self.extractor = CNNFeatureExtractor(n_channels) 

        # Capas totalmente conectadas encargadas de la clasificación
        self.classifier = nn.Sequential(

            # Primera capa: de 512 características a 256
            nn.Linear(self.extractor.out_features, 256),
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
        """Extrae las características de la señal y genera la predicción"""

        feats = self.extractor(x) # extrae la representación de la señal mediante la CNN
        return self.classifier(feats) # utiliza las características extraídas para generar la predicción