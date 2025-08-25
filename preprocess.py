# preprocess.py
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Colonnes
NUM_COLS: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CAT_COLS: List[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET: str = "target"  # nom attendu par le reste du projet

def build_preprocess() -> ColumnTransformer:
    """
    Construit un ColumnTransformer avec deux pipelines :
      - numérique   : imputation médiane + standardisation
      - catégoriel  : imputation mode + OneHotEncoder (ignore inconnues)
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Remarque: si scikit-learn < 1.2, remplace 'sparse_output=False' par 'sparse=False'
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUM_COLS),
            ("cat", categorical_pipeline, CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

def get_feature_spaces() -> Tuple[List[str], List[str], str]:
    """
    Source de vérité unique : retourne (NUM_COLS, CAT_COLS, TARGET).
    """
    return NUM_COLS, CAT_COLS, TARGET

__all__ = ["build_preprocess", "get_feature_spaces", "NUM_COLS", "CAT_COLS", "TARGET"]
