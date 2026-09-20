"""
Extracción de características manuales a partir de las señales EEG, utilizadas 
como entrada del baseline de Machine Learning. Por cada epoch y canal se extraen
características en el dominio temporal, frecuencial y parámetros de Hjorth. 
"""

import numpy as np
from scipy import stats
from scipy.signal import welch

# Bandas de frecuencia 
FREQ_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40),}

# Parámetros de hjorth 
def hjorth_params(signal):
    """Calcula movilidad y complejidad de Hjorth"""

    first_deriv = np.diff(signal) # primera derivada de la señal 
    second_deriv = np.diff(first_deriv) # segunda derivada de la señal 

    var_zero = np.var(signal) # varianza de la señal original 
    var_d1 = np.var(first_deriv) # varianza de la primera derivada 
    var_d2 = np.var(second_deriv) # varianza de la segunda derivada

    mobility = np.sqrt(var_d1 / var_zero) # movilidad 
    mobility_d1 = np.sqrt(var_d2 / var_d1) # movilidad calculada sobre la primera derivada 
    complexity = mobility_d1 / mobility # complejidad 

    return {"hjorth_mobility": mobility, "hjorth_complexity": complexity}

def zero_crossing_rate(signal):
    """Calcula cuántas veces la señal cruza el 0"""

    signs = np.sign(signal) # signo de cada muestra (+, - , 0)
    signs[signs == 0] = 1 
    return np.sum(np.diff(signs) != 0) / len(signal) # proporción de cambios de signo respecto a la longitud de la señal 


def time_domain_features(signal):
    """Características del dominio temporal"""

    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "skew": stats.skew(signal),        # asimetría de la distribución
        "kurtosis": stats.kurtosis(signal),  # presencia de picos pronunciados
        "rms": np.sqrt(np.mean(signal ** 2)), # energía de la señal
        "zcr": zero_crossing_rate(signal),
    }


def freq_domain_features(signal, sfreq):
    """Características del dominio de la frecuencia: 
    potencia total y por banda, absoluta y relativa"""

    # Densidad espectral de potencia  (PSD)
    nperseg = min(len(signal), int(2 * sfreq)) # longitud de cada segmento
    # Método de Welch 
    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg) 

    # Potencia total 
    total_power = np.trapezoid(psd, freqs)
    total_power = total_power if total_power > 0 else 1e-12

    feats = {}
    for band_name, (low, high) in FREQ_BANDS.items():
        mask = (freqs >= low) & (freqs < high) # frecuencias dentro de la banda 
        band_power = np.trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0 # potencia absoluta de la banda 
        feats[f"{band_name}_power"] = band_power # potencia absoluta
        feats[f"{band_name}_relpower"] = band_power / total_power # potencia relativa respecto al total 

    return feats


def extract_epoch_features(epoch, sfreq, channel_names):
    """Extrae las características (temporales, frecuenciales, 
    Hjorth) de cada epoch, canal a canal"""

    features = {}

    for ch_idx, ch_name in enumerate(channel_names):
        signal = epoch[ch_idx]

        # tiempo, frecuencia, hjorth
        td = time_domain_features(signal)
        fd = freq_domain_features(signal, sfreq)
        hd = hjorth_params(signal)

        # se guarda cada característica con su nombre
        for name, value in {**td, **fd, **hd}.items():
            features[f"{ch_name}_{name}"] = value

    return features


def extract_features(X, sfreq=100, channel_names=("Fpz-Cz", "Pz-Oz")):
    """Extrae características de todo el conjunto de epochs"""
    n_epochs = X.shape[0]
    feature_names = None
    rows = []

    for i in range(n_epochs):
        # Extracción de características por cada epoch 
        feats = extract_epoch_features(X[i], sfreq, channel_names)
        # Nombre de las características 
        if feature_names is None:
            feature_names = list(feats.keys())
        # Se guardan los valores 
        rows.append([feats[name] for name in feature_names])

    # Conversión de la lista a un array de Numpy 
    X_feat = np.array(rows, dtype=np.float64) # cada fila es un epoch y cada columna una característica
    return X_feat, feature_names