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
├── 01_ingestion.py          # Nettoyage et validation des données
├── preprocess.py            # Préprocesseur scikit-learn (imputer, scaler, OHE)
├── 03_train.py              # Entraînement + sauvegarde du pipeline
├── 04_api.py                # API FastAPI pour servir le modèle
└── README.txt               # Documentation du projet

-----------------------------------------------------------
Installation
-----------------------------------------------------------
1. Cloner le projet
   git clone https://github.com/<ton-repo>/cardio_project.git
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
   python 01_ingestion.py
   --> génère data/heart_clean.csv

2) Entraînement du modèle
   python 03_train.py
   --> génère les artefacts dans models/ :
       - heart_pipeline.joblib
       - heart_pipeline.pkl
       - metrics.json

3) Lancer l’API FastAPI
   uvicorn 04_api:app --reload --port 8000
   --> documentation interactive : http://127.0.0.1:8000/docs
