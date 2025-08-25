import pandas as pd
from pathlib import Path
from typing import Dict, List

EXPECTED_SCHEMA: Dict[str, str] = {
"age": "float64", # âge en années
"sex": "int64", # 0=femme, 1=homme (dans ce dataset UCI)
"cp": "int64", # type de douleur thoracique (0-3)
"trestbps": "float64", # pression artérielle au repos
"chol": "float64", # cholestérol sérique (mg/dl)
"fbs": "int64", # glycémie à jeun > 120 mg/dl (1 vrai / 0 faux)
"restecg": "int64", # résultat électrocardiogramme au repos (0-2)
"thalach": "float64", # fréquence cardiaque maximale atteinte
"exang": "int64", # angine induite par l’exercice (1 oui / 0 non)
"oldpeak": "float64", # dépression ST induite par l’exercice
"slope": "int64", # pente du segment ST (0-2)
"ca": "float64", # nb d’artères majeures colorées (0-3) — parfois NaN
"thal": "int64", # thalassémie (0-3 selon la variante de dataset)
"target": "int64" # étiquette: 1=maladie présente, 0=absente
}

RAW_PATH = Path("data/heart.csv")
CLEAN_PATH = Path("data/heart_clean.csv")

def read_csv(path : Path) -> pd.DataFrame:
	"""Lit un CSV en vérifiant l’existence du fichier.
Pourquoi: centraliser la lecture pour tracer/adapter facilement.
Retour: DataFrame pandas.
"""
	if not path.exists() :
		raise FileNotFoundError(f"Fichier introuvable: {path}")
	df = pd.read_csv(path)
	df = df.rename(columns={
        "thalch": "thalach",  
        "num": "target"       
    })
	return df

def validate_and_cast_schema(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
	"""Valide que toutes les colonnes attendues sont présentes et caste les types.
Pourquoi: garantir un schéma stable pour l’entraînement et l’inférence.
"""
	missing = [c for c in schema.keys() if c not in df.columns]
	if missing:
		raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

	df = df[list(schema.keys())].copy()

	for col, dtype in schema.items():
		try:
			df[col] = df[col].astype(dtype)
		except Exception:
			# Si échec (ex: chaîne de caractères non convertible),
			# on essaye de convertir en numérique en forçant les erreurs à NaN	
			df[col] = pd.to_numeric(df[col], errors='coerce')
			# Ensuite on ajuste selon le type cible
			if dtype.startswith('int'):
				df[col] = df[col].fillna(0).astype(dtype)
			elif dtype.startswith('float'):
				df[col] = df[col].astype('float64')
	return df
def basic_cleaning(df: pd.DataFrame, deduplicate_on: List[str] | None = None) -> pd.DataFrame:
	"""Supprime les doublons pour améliorer la qualité des données.
Pourquoi: éviter que des lignes identiques biaisent l’entraînement/les métriques.
"""

	before = len(df)
	df = df.drop_duplicates(subset=deduplicate_on)
	after = len(df)
	print(f"Duplicates supprimés: {before - after}")
	return df

def save_clean(df: pd.DataFrame, path: Path) -> None:
	"""Écrit le DataFrame nettoyé en CSV sans index.
	Pourquoi: figer une version propre de référence pour l’entraînement.
	"""
	path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(path, index=False)
	print(f"Données propres sauvegardées -> {path}")

if __name__ == "__main__":
	df = read_csv(RAW_PATH)
	df = validate_and_cast_schema(df, EXPECTED_SCHEMA)
	df = basic_cleaning(df, deduplicate_on=list(EXPECTED_SCHEMA.keys()))
	save_clean(df, CLEAN_PATH)
