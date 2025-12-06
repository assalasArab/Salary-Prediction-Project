# Fonctions utilitaires pour le modèle et les données

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from config_paths import CLEANED_DATA_PATH, BEST_MODEL_PATH, MODELS_DIR

FEATURES = [
    "Country",
    "FormalEducation",
    "UndergradMajor",
    "YearsCoding",
    "YearsCodingProf",
    "Employment",
    "CompanySize",
    "DevType",
    "LanguageWorkedWith",
]

TARGET = "ConvertedSalary"

NUMERIC_FEATURES = ["YearsCoding", "YearsCodingProf", "CompanySize"]

CATEGORICAL_FEATURES = [
    "Country",
    "FormalEducation",
    "UndergradMajor",
    "Employment",
    "DevType",
    "LanguageWorkedWith",
]


def load_clean_dataset():
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {CLEANED_DATA_PATH}")
    df = pd.read_csv(CLEANED_DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Colonne cible manquante : {TARGET}")
    df = df[(df[TARGET] >= 1000) & (df[TARGET] <= 300000)]
    return df


def build_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_train_test(test_size=0.2, random_state=42):
    df = load_clean_dataset()
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {BEST_MODEL_PATH}")
    return joblib.load(BEST_MODEL_PATH)


def save_model(pipeline):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, BEST_MODEL_PATH)
    print(f"Modèle sauvegardé dans : {BEST_MODEL_PATH}")
