"""
Definición de los modelos de Machine Learning para el baseline.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models():
    """
    Crea un diccionario con los tres algoritmos del baseline. 
    """

    models_dict = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()), # estandarización de las características
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)), # clasificador
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()), # estandarización de las características
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", 
                                           n_jobs=-1, random_state=42)), # clasificador
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()), # estandarización de las características
            ("clf", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, objective="binary:logistic",
                                  eval_metric="logloss", n_jobs=-1, random_state=42)) # clasificador 
        ]),
    }
    return models_dict 


