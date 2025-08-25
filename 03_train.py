# 03_train.py
from pathlib import Path
from typing import Dict, Tuple
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from preprocess import build_preprocess, get_feature_spaces

# === Chemins ===
DATA_PATH = Path("data/heart_clean.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "heart_pipeline.joblib"
MODEL_PKL_PATH = MODEL_DIR / "heart_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

def log(msg: str) -> None:
    print(f"[TRAIN] {msg}")

# --- Data ---
def load_data(path: Path) -> pd.DataFrame:
    log(f"Chargement des données: {path.resolve()}")
    if not path.exists():
        raise FileNotFoundError(
            f"Data clean introuvable: {path}. Exécute d'abord 01_ingestion.py"
        )
    df = pd.read_csv(path)
    log(f"Shape: {df.shape}, Colonnes: {list(df.columns)}")
    return df

def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    _, _, TARGET = get_feature_spaces()
    X = df.drop(columns=[TARGET])
    #  Binarisation: 0 = pas de maladie, 1 = maladie (1..4)
    y = (df[TARGET].astype(int) > 0).astype(int)
    log(f"Split stratifié sur '{TARGET}' (classes: {sorted(y.unique())})")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Modèle ---
def build_model(model_name: str = "logreg") -> Pipeline:
    preprocessor = build_preprocess()
    if model_name == "logreg":
        model = LogisticRegression(max_iter=200, solver="liblinear")
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        raise ValueError("model_name doit être 'logreg' ou 'rf'")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipe

# --- Entraînement + éval ---
def fit_and_eval(pipe: Pipeline, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    log("Entraînement du pipeline...")
    pipe.fit(X_train, y_train)
    log("Prédiction...")
    y_pred = pipe.predict(X_test)

    # Probabilités pour ROC-AUC
    y_score = pipe.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_score)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc),
    }

    log("Rapport de classification:\n" + classification_report(y_test, y_pred))
    log(f"Métriques: {metrics}")
    return metrics

# --- Sauvegarde ---
def save_artifacts(pipe: Pipeline, metrics: Dict[str, float]) -> None:
    log("Sauvegarde des artefacts...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, MODEL_PATH)
    log(f"Modèle sauvegardé -> {MODEL_PATH}")

    # optionnel: version .pkl
    import pickle
    with open(MODEL_PKL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    log(f"Modèle sauvegardé -> {MODEL_PKL_PATH}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log(f"Métriques sauvegardées -> {METRICS_PATH}")

# --- Main ---
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split(df)
    pipe = build_model("logreg")  # baseline = régression logistique
    metrics = fit_and_eval(pipe, X_train, y_train, X_test, y_test)
    save_artifacts(pipe, metrics)
    log("Terminé ")
