# Entraînement des modèles et choix du meilleur

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from model_utils import (
    build_preprocessor,
    get_train_test,
    save_model,
)


def main():
    print("Entraînement des modèles")

    X_train, X_test, y_train, y_test = get_train_test()
    preprocessor = build_preprocessor()

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=42,
        ),
    }

    results = []
    best_model_name = None
    best_r2 = -np.inf
    best_pipeline = None

    for name, model in models.items():
        print(f"=== Modèle : {name} ===")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        scoring = {
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
        }

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=scoring,
            n_jobs=-1,
        )

        cv_mae = -cv_results["test_mae"]
        cv_rmse = np.sqrt(-cv_results["test_mse"])
        cv_r2 = cv_results["test_r2"]

        print("Validation croisée (5-fold) :")
        print(f"  MAE moyen  : {cv_mae.mean():8.2f} (+/- {cv_mae.std():.2f})")
        print(f"  RMSE moyen : {cv_rmse.mean():8.2f} (+/- {cv_rmse.std():.2f})")
        print(f"  R² moyen   : {cv_r2.mean():8.4f} (+/- {cv_r2.std():.4f})")

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\nÉvaluation sur le set de test :")
        print(f"  MAE  = {mae:10.2f}")
        print(f"  RMSE = {rmse:10.2f}")
        print(f"  R²   = {r2:10.4f}\n")

        results.append((name, mae, rmse, r2))

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    print("\nRÉSULTATS COMPARATIFS (Test)")
    for name, mae, rmse, r2 in results:
        print(f"{name:15s} | MAE = {mae:10.2f} | RMSE = {rmse:10.2f} | R² = {r2:6.4f}")

    print(f"\nMeilleur modèle (selon R² sur le test) : {best_model_name}")

    if best_pipeline is not None:
        save_model(best_pipeline)


if __name__ == "__main__":
    main()
