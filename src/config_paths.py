# Gestion centralisée des chemins du projet

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

NETTOYAGE_SCRIPT = BASE_DIR / "src" / "nettoyage_data.py"
TRAIN_SCRIPT = BASE_DIR / "src" / "train_models.py"

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

CLEANED_DATA_PATH = DATA_DIR / "survey_clean.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_random_forest.pkl"

LOGO_PATH = BASE_DIR / "src" / "logo_paris8_couleurs-officielles.png"
