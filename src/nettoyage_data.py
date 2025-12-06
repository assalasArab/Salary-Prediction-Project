# Script de nettoyage du dataset brut vers survey_clean.csv

import re
import numpy as np
import pandas as pd

from config_paths import DATA_DIR, CLEANED_DATA_PATH


def convert_years(x):
    """Convertit les chaînes d'années d'expérience en valeurs numériques."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip()

    if "Less than" in x:
        return 0.5
    if "More than" in x:
        return 30.0

    nums = re.findall(r"\d+", x)
    if not nums:
        return np.nan

    nums = [int(n) for n in nums]
    return sum(nums) / len(nums)


def convert_company_size(x):
    """Convertit la taille d’entreprise en nombre moyen d’employés."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()

    if "to" in s:
        nums = re.findall(r"\d+", s)
        if len(nums) >= 2:
            a, b = int(nums[0]), int(nums[1])
            return (a + b) / 2

    if "more" in s or "or more" in s:
        nums = re.findall(r"\d+", s)
        if nums:
            return float(nums[0])

    if "fewer than" in s:
        nums = re.findall(r"\d+", s)
        if nums:
            return float(nums[0]) / 2

    try:
        return float(s)
    except ValueError:
        return np.nan


def main():
    raw_path = DATA_DIR / "survey_results_public.csv"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Fichier brut introuvable : {raw_path}"
        )

    df = pd.read_csv(
        raw_path,
        na_values=["NA", "NaN", ""],
        low_memory=False,
    )

    cols = [
        "Country",
        "FormalEducation",
        "UndergradMajor",
        "YearsCoding",
        "YearsCodingProf",
        "Employment",
        "CompanySize",
        "DevType",
        "LanguageWorkedWith",
        "ConvertedSalary",
    ]

    df = df[cols].copy()
    df = df.dropna(subset=["ConvertedSalary"])

    df["YearsCoding"] = df["YearsCoding"].apply(convert_years)
    df["YearsCodingProf"] = df["YearsCodingProf"].apply(convert_years)
    df["CompanySize"] = df["CompanySize"].apply(convert_company_size)

    print("Aperçu après nettoyage :")
    print(df.head())
    print()
    print(df[["YearsCoding", "YearsCodingProf", "CompanySize"]].describe())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"\nFichier nettoyé sauvegardé dans : {CLEANED_DATA_PATH}")


if __name__ == "__main__":
    main()
