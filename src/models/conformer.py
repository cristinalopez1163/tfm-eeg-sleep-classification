"""
Define la arquitectura CNN1D + Conformer, del cuarto experimento de Deep Learning. 
Reutiliza el bloque CNNFeatureExtractor que actua como módulo  extractor de características 
de cada epoch y combina módulos Feed Forward, de autoatención y convolucionales para modelar 
la información temporal de la secuencia (Feed Forward - Atención - Convolución - Feed Forward)
"""

import torch.nn as nn

from models.cnn1d import CNNFeatureExtractor
from models.cnn_transformer import PositionalEncoding


class FeedForwardModule(nn.Module):
    """Bloque feed-forward (capa de propagación hacia adelante) que 
    transforma las características de la secuencia"""

    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(512), # normaliza las características 
            nn.Linear(512, 512 * 4), # amplia la dimensión de las características
            nn.SiLU(), # función de activación
            nn.Dropout(0.3), # regularización
            nn.Linear(512 * 4, 512), # recupera la dimensión original de las características
            nn.Dropout(0.3), # regularización
        )

    def forward(self, x):
        """Transforma las características mediante el bloque feed-forward"""
        return self.net(x)


class MHSAModule(nn.Module):
    """Módulo de autoatención multi-cabeza que modela las relaciones entre epochs"""

    def __init__(self):
        super().__init__()

        self.norm = nn.LayerNorm(512) # normalización
        # Mecanismo de autoatención multi-cabeza (permite relacionar distintas posiciones de la secuencia)
        self.mhsa = nn.MultiheadAttention(512, num_heads=8, dropout=0.3, batch_first=True)
        self.dropout = nn.Dropout(0.3) # regularización

    def forward(self, x):
        """Calcula la representación de la secuencia mediante autoatención"""

        x_norm = self.norm(x) # normaliza
        attn_out, _ = self.mhsa(x_norm, x_norm, x_norm, need_weights=False) # utiliza la propia secuencia 
        return self.dropout(attn_out)


class ConvModule(nn.Module):
    """Módulo convolucional que captura patrones locales en la secuencia. 
    Además de la atención (que ve toda la secuencia), este módulo analiza 
    solo los epochs más cercanos para capturar patrones de corto alcance"""

    def __init__(self):
        super().__init__()

        self.norm = nn.LayerNorm(512) # normalización
        # Primera convolución que amplía la dimensión
        self.pointwise_conv1 = nn.Conv1d(512, 512 * 2, kernel_size=1)
        # Compuerta que decide cuánta información continúa hacia el siguiente bloque 
        self.glu = nn.GLU(dim=1)
        # Segunda convolución que mira solo epochs vecinos 
        self.depthwise_conv = nn.Conv1d(512, 512, kernel_size=7, padding= 7 // 2, groups=512)
        self.batch_norm = nn.BatchNorm1d(512) # normalización
        self.swish = nn.SiLU() # función de activación
        # Tercera convolución que mantiene la dimensión de las características 
        self.pointwise_conv2 = nn.Conv1d(512, 512, kernel_size=1)
        self.dropout = nn.Dropout(0.3) # regularización

    def forward(self, x):
        """Extrae patrones locales de las características de la secuencia"""

        x = self.norm(x) # normaliza antes de aplicar las convoluciones 
        x = x.transpose(1, 2) # adapta el formato           
        x = self.pointwise_conv1(x) # expande los canales  
        x = self.glu(x) # selecciona la información relevante            
        x = self.depthwise_conv(x) # captura patrones locales 
        x = self.batch_norm(x) # normaliza
        x = self.swish(x) # función de activación
        x = self.pointwise_conv2(x) # vuelve a transformar las características a su dimensión original       
        x = self.dropout(x) # regularización

        return x.transpose(1, 2)          

class ConformerBlock(nn.Module):
    """Bloque Conformer completo: Feed Forward - Atención - Convolución - Feed Forward"""
    def __init__(self):
        super().__init__()

        self.ff1 = FeedForwardModule() # Feed Forward
        self.mhsa = MHSAModule() # Autoatención multicabeza (Multihead self-attention)
        self.conv = ConvModule() # Módulo convolucional 
        self.ff2 = FeedForwardModule() # Feed Forward
        self.final_norm = nn.LayerNorm(512)

    def forward(self, x):
        """Procesa la secuencia mediante los bloques conformer"""

        x = x + 0.5 * self.ff1(x) # medio paso feed-forward 
        x = x + self.mhsa(x) # atención (contexto global de la secuencia)
        x = x + self.conv(x) # convolución (contexto local entre epochs vecinos)
        x = x + 0.5 * self.ff2(x) # medio paso feed-forward
        return self.final_norm(x) # normalización final 


class ConformerEncoder(nn.Module):
    """Encoder formado por varios bloques Conformer apilados"""
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([ConformerBlock() for _ in range(2)]) # 2 bloques idénticos

    def forward(self, x):
        """Procesa la secuencia mediante los bloques Conformer"""

        # Aplica cada bloque de forma secuencial 
        for layer in self.layers:
            x = layer(x)
        return x


class CNNConformer(nn.Module):
    """Modelo CNNConformer que combina  un extractor convolucional de características con 
    bloques Conformer para incorporar información temporal entre epochs"""

    def __init__(self, n_channels,n_classes):
        super().__init__()

        # Módulo extractor de características que transforma cada epoch en un vector de 512 características
        self.extractor = CNNFeatureExtractor(n_channels) 
        # Codificación que incorpora la posición temporal de cada epoch
        self.pos_encoder = PositionalEncoding(512, max_len=15)
        # Encoder Conformer 
        self.conformer = ConformerEncoder()
        # Clasificador MLP que genera las predicciones 
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
        # Procesa la secuencia mediante el encoder Conformer 
        encoded = self.conformer(feats)  

        # Se toma la representación del epoch central         
        center_idx = seq_len // 2

        # Usa esa representación para generar la predicción
        return self.classifier(encoded[:, center_idx, :])