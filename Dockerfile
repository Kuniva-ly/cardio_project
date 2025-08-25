# Dockerfile


# Base Python légère
FROM python:3.11-slim


# Dossier de travail dans le conteneur
WORKDIR /app


# Copier les dépendances puis installer (meilleure mise en cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copier les scripts + dossiers (data/models seront montés en volume en dev)
COPY 01_ingestion.py 02_preprocess.py 03_train.py 04_api.py ./
COPY data ./data
COPY models ./models


# Exposer le port FastAPI
EXPOSE 8000


# Commande de démarrage du serveur
CMD ["uvicorn", "04_api:app", "--host", "0.0.0.0", "--port", "8000"]
