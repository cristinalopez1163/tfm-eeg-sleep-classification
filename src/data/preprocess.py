"""Preprocesado de la señal EEG y construcción de los epochs y del dataset final"""

import mne

mne.set_log_level("ERROR")

def preprocess_signal(raw):
    """
    Preprocesado de la señal EEG 
    """

    # Canales EEG de interés
    eeg_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    eeg_channels = [ch for ch in eeg_channels]

    # Descarta el resto de canales (EOG, EMG, ...)
    raw.pick_channels(eeg_channels)

    raw.filter(0.5, 40) # filtro paso banda (0.5 - 40 Hz) para eliminar frecuencias muy bajas y altas
    raw.notch_filter(49) # filtro notch a 49 Hz para atenuar la interferencia de la red eléctrica

    return raw


def create_epochs(raw):
    """Segmenta la señal continua en epochs de 30 segundos a partir de las anotaciones 
    del hipnograma y agrupa las fases del sueño en 2 claes (Wake / Sleep)"""

    # Clase 0: Vigilia y Clase 1: Sueño (integra fases REM y NREM)
    sleep_stages = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 1,
        'Sleep stage 3': 1,
        'Sleep stage 4': 1,
        'Sleep stage R': 1
    }

    # Creación de los epochs (duración de 30 segundos)
    events, event_id = mne.events_from_annotations(raw, event_id=sleep_stages, chunk_duration=30.0) # eventos
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=0, tmax=30 - 1/raw.info['sfreq'], 
        baseline=None, preload=True, verbose=False)

    return epochs

def create_dataset(epochs):
    """Crea el dataset final (X, y) a partir de los epochs creados previamente"""

    X = epochs.get_data() # señal: (n_epochs, n_canales, n_muestras)
    y = epochs.events[:, -1] # etiqueta (0 o 1)

    event_id = epochs.event_id

    # Elimina etiquetas desconocidas si existen
    unknown_id = event_id.get('Sleep stage ?', None)
    # Se conservan únicamante los epochs con etiquetas conocidas
    if unknown_id is not None:
        mask = y != unknown_id
        X = X[mask]
        y = y[mask]

    return X, y