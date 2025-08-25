# 04_api.py

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Chemin du modèle entraîné
MODEL_PATH = Path("models/heart_pipeline.joblib")

# Création de l’application FastAPI (avec titre/version)
app = FastAPI(title="Cardio Risk API", version="1.0")

class PatientFeatures(BaseModel):
    """Schéma d’entrée strict validé par Pydantic.
    Field(..., ge=, le=) ajoute des bornes (greater-equal / less-equal).
    """
    age: float = Field(..., ge=0, le=130)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=0, le=400)
    chol: float = Field(..., ge=0, le=1500)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=0, le=300)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=-5, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: float = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=0, le=3)

def load_pipeline():
    """Charge et renvoie le pipeline entraîné.
    Lève une erreur claire si le fichier est absent.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modèle introuvable. Entraîne d'abord via 03_train.py")
    return joblib.load(MODEL_PATH)

@app.on_event("startup")
def _load_model_on_startup():
    """Hook exécuté au démarrage du serveur.
    On précharge le pipeline dans app.state pour éviter de le recharger à chaque requête.
    """
    app.state.pipeline = load_pipeline()

@app.get("/health")
def health():
    """Endpoint de liveness: renvoie {status: ok} si le modèle est prêt."""
    ok = hasattr(app.state, "pipeline") and app.state.pipeline is not None
    return {"status": "ok" if ok else "model_not_loaded"}

@app.post("/predict")
def predict(features: PatientFeatures):
    """Endpoint de prédiction binaire.
    1) Convertit l’entrée validée en DataFrame 1-ligne
    2) Calcule la probabilité de la classe 1 via predict_proba
    3) Applique un seuil 0.5 pour produire un label 0/1
    """
    df = pd.DataFrame([features.model_dump()]) # transforme l’objet Pydantic en dict, puis en DataFrame
    pipe = app.state.pipeline # récupère le pipeline préchargé
    proba = float(pipe.predict_proba(df)[:, 1][0]) # proba de maladie (classe 1)
    label = int(proba >= 0.5) # seuil 0.5 -> 1 sinon 0
    return {"probability": proba, "prediction": label}


# Lancement local: uvicorn api:app --reload --port 8000