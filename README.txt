Cardio Project – Prédiction du risque cardiaque
===========================================================

Contexte
--------
Projet de Data Engineering / Data Science visant à prédire 
la présence d’une maladie cardiaque à partir du dataset 
UCI Heart Disease. 

- Cible (target) :
  0 = pas de maladie
  1 = maladie présente (fusion des classes 1..4)
- Technologies : Python, pandas, scikit-learn, FastAPI, Uvicorn

-----------------------------------------------------------
Structure du projet
-----------------------------------------------------------
cardio_project/
│
├── data/
│   ├── heart.csv            # Données brutes (UCI)
│   └── heart_clean.csv      # Données nettoyées (via 01_ingestion.py)
│
├── models/
│   ├── heart_pipeline.joblib
│   ├── heart_pipeline.pkl
│   └── metrics.json
│
├── ingestion.py          # Nettoyage et validation des données
├── preprocess.py            # Préprocesseur scikit-learn (imputer, scaler, OHE)
├── train.py              # Entraînement + sauvegarde du pipeline
├── api.py                # API FastAPI pour servir le modèle
└── README.txt               # Documentation du projet

-----------------------------------------------------------
Installation
-----------------------------------------------------------
1. Cloner le projet
   git clone git@github.com:Kuniva-ly/cardio_project.git
   cd cardio_project

2. Créer un environnement virtuel
   python3 -m venv .venv
   source .venv/bin/activate

3. Installer les dépendances
   pip install -r requirements.txt

-----------------------------------------------------------
Pipeline d’exécution
-----------------------------------------------------------

1) Ingestion & nettoyage
   python ingestion.py
   --> génère data/heart_clean.csv

2) Entraînement du modèle
   python train.py
   --> génère les artefacts dans models/ :
       - heart_pipeline.joblib
       - heart_pipeline.pkl
       - metrics.json

3) Lancer l’API FastAPI
   uvicorn api:app --reload --port 8000
   --> documentation interactive : http://127.0.0.1:8000/docs
