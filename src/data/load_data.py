"""
Carga los registros en crudo del dataset, empareja cada fichero PSG con 
su hipnograma y asocia cada época a su fase del sueño correspondiente.
"""

import os
import re
import mne 

def get_subject_pairs(data_path):
    """Recorre el directorio y devuelve la lista de tuplas 
    (ruta_psg, ruta_hipnograma) para cada sujeto"""

    # Todos los ficheros del directorio 
    files = os.listdir(data_path)
    psg_files = [f for f in files if "PSG.edf" in f] # solo ficheros de PSG 
    pairs = [] # lista para almacenar las parejas PSG - hipnograma

    for psg in psg_files: 
        match = re.match(r"(SC\d+)", psg) # extrae el identificador del sujeto del nombre del fichero
        if not match:
            continue # se descartan ficheros que no tienen el patrón esperado 
        subject_id = match.group(1) # guarda el identificador del sujeto 

        # Busca el hipnograma correspondiente del sujeto 
        hyp_candidates = [f for f in files if subject_id in f and "Hypnogram" in f]

        # Guarda las rutas del PSG y su hipnograma
        pairs.append((os.path.join(data_path, psg), os.path.join(data_path, hyp_candidates[0])))

    return pairs


def load_subject(psg_path, hypnogram_path):
    """Carga la señal PSG de un sujeto y le añade las anotaciones del hipnograma. 
    Devuelve un objeto Raw de MNE listo para su prepocesamiento"""

    # Carga la señal EEG 
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    # Anotaciones de las fases del sueño 
    annotations = mne.read_annotations(hypnogram_path)
    # Vincula las anotaciones a la señal 
    raw.set_annotations(annotations)

    return raw