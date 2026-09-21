# Clasificación de fases del sueño (Wake vs. Sleep) a partir de señales EEG

Trabajo de Fin de Máster que aborda la clasificación automática de las fases del sueño a partir de señales de electroencefalograma (EEG) como un problema de clasificación binaria (**Wake** frente a **Sleep**). Se compara un enfoque clásico de Machine Learning, basado en características manuales, con distintas arquitecturas de Deep Learning que aprenden directamente de la señal, con el objetivo de evaluar en qué medida cada enfoque mejora la detección de la vigilia frente al sueño.

## Descripción

El trabajo se ha desarrollado sobre el subconjunto **Sleep Cassette** de la base de datos pública [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) (PhysioNet), empleando los canales EEG **Fpz-Cz** y **Pz-Oz** de 78 sujetos, segmentados en epochs de 30 segundos.

- **Baseline de Machine Learning**: Regresión Logística, Random Forest y XGBoost, entrenados sobre características manuales en los dominios temporal y frecuencial, y parámetros de Hjorth.
- **Deep Learning**:
  - **CNN1D**: clasifica cada epoch de forma independiente, a partir de la señal en crudo.
  - **CNN1D + BiLSTM**, **CNN1D + Transformer** y **CNN1D + Conformer**: incorporan contexto temporal mediante ventanas de epochs consecutivos, combinando la misma CNN1D como extractor de características con un mecanismo de modelado de secuencias distinto en cada caso.

Todos los modelos se entrenan y evalúan bajo un mismo protocolo (misma partición por sujeto, mismas semillas y misma estrategia de validación), lo que permite una comparación equitativa entre enfoques.

### Resultados principales

Las arquitecturas de Deep Learning mejoran sistemáticamente la detección de la vigilia frente al sueño respecto al baseline clásico. El mejor modelo es **CNN + Conformer**, con los siguientes resultados sobre el conjunto de test:

| Métrica        | Valor   |
|----------------|---------|
| Accuracy       | 98.34 % |
| F1-macro       | 98.11 % |
| Kappa de Cohen | 96.21 % |

## Estructura del proyecto

```
.
├── literature/                 # papers de referencia consultados
├── requirements.txt
├── resources/                  
│   ├── raw/                    # ficheros .edf originales de Sleep-EDF (PSG + hipnograma)
│   ├── processed/              # dataset ya preprocesado, se genera automáticamente
│   └── metadata/               # metadatos de los sujetos 
|   
├── results/
│   ├── generate_results.py     # genera la tabla comparativa y las matrices de confusión finales
│   ├── logs/                   # logs de cada ejecución 
│   ├── tables/                 # resultados por experimento y tabla resumen 
│   ├── figures/                # matrices de confusión 
│   └── checkpoints/            # pesos de los mejores modelos
├── scripts/
│   └── web_app.py              # demo interactiva con Streamlit (CNN + Conformer)
└── src/
    ├── main.py                 # pipeline principal: dataset + entrenamiento + evaluación
    ├── data/
    │   ├── load_data.py           # carga de PSG + hipnograma (MNE)
    │   ├── preprocess.py          # filtrado y segmentación en epochs de 30s
    │   ├── feature_extraction.py  # características manuales (tiempo, frecuencia, Hjorth)
    │   ├── torch_dataset.py       # Dataset de PyTorch para CNN1D 
    │   └── sequence_dataset.py    # Dataset de PyTorch para modelos secuenciales (BiLSTM/Transformer/Conformer)
    ├── models/
    │   ├── ml.py                  # Regresión Logística, Random Forest, XGBoost
    │   ├── cnn1d.py               # CNN1D y extractor de características (CNNFeatureExtractor)
    │   ├── cnn_bilstm.py          # CNN1D + BiLSTM
    │   ├── cnn_transformer.py     # CNN1D + Transformer
    │   └── conformer.py           # CNN1D + Conformer
    ├── evaluation/
    │   ├── cross_validation.py   # validación cruzada del baseline de ML
    │   └── train_eval_dl.py      # entrenamiento/evaluación de los modelos de DL
    └── utils/
        └── labels.py              # definición de las clases (Wake / Sleep)
```

## Dataset

Este repositorio **no incluye los datos**. Es necesario descargar el subconjunto *Sleep Cassette* de Sleep-EDF Expanded desde PhysioNet y colocar todos los ficheros `.edf` (PSG e hipnogramas) en `resources/raw/`:


## Instalación

Requiere Python 3.10+.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Si vas a entrenar los modelos de Deep Learning con GPU, instala primero la versión de PyTorch correspondiente a tu CUDA desde la [página oficial](https://pytorch.org/get-started/locally/)

## Uso

### 1. Construir el dataset y entrenar los modelos

`src/main.py` construye (o reutiliza, si ya existe) el dataset preprocesado a partir de `resources/raw/`, y ejecuta los experimentos que estén activados. Qué experimentos ejecutar se controla con las variables al inicio del script:

```python
RUN_ML = False
RUN_CNN1D = False
RUN_CNN_BILSTM = False
RUN_CNN_TRANSFORMER = False
RUN_CNN_CONFORMER = True
```

Actívalos según el experimento que quieras realizar y ejecuta:

```bash
python src/main.py
```

El dataset preprocesado se guarda en `resources/processed/sleep_dataset.npz` (se reutiliza en ejecuciones posteriores) y los resultados de cada experimento se guardan en `results/tables/`, con los logs en `results/logs/` y el mejor checkpoint de cada modelo en `results/checkpoints/`.

### 2. Generar la comparación final

Una vez ejecutados los experimentos que se quieran comparar:

```bash
python results/generate_results.py
```

Genera `results/tables/comparison_summary.csv` (media y desviación típica de cada métrica por experimento) y las matrices de confusión en `results/figures/`.

### 3. Demo interactiva

`scripts/web_app.py` permite subir un registro EEG (PSG + hipnograma en `.edf`) y visualizar las predicciones del modelo CNN + Conformer frente al hipnograma real:

```bash
streamlit run scripts/web_app.py
```

Requiere que exista el checkpoint `results/checkpoints/CNN_Conformer_best.pt` (se genera al ejecutar `main.py` con `RUN_CNN_CONFORMER = True`).