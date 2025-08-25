# Dockerfile


# Base Python légère
FROM python:3.11-slim


# Dossier de travail dans le conteneur
WORKDIR /app


# Copier les dépendances puis installer (meilleure mise en cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copier les scripts + dossiers (data/models seront montés en volume en dev)
COPY ingestion.py preprocess.py train.py api.py ./
COPY data ./data
COPY models ./models


# Exposer le port FastAPI
EXPOSE 8000


# Commande de démarrage du serveur
CMD ["uvicorn", "04_api:app", "--host", "0.0.0.0", "--port", "8000"]
