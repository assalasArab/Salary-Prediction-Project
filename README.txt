README – Démarrage du projet

Prérequis

Python 3.10 ou supérieur

pip installé

Le fichier survey_results_public.csv doit être placé dans le dossier data/

Installation des dépendances
Dans un terminal, exécuter :
pip install -r requirements.txt

Étapes pour lancer le projet

Étape 1 : Nettoyage du dataset
python src/nettoyage_data.py
Le fichier nettoyé survey_clean.csv sera généré dans le dossier data/.

Étape 2 : Entraînement du modèle
python src/train_models.py
Le meilleur modèle sera enregistré dans models/best_random_forest.pkl

Étape 3 : Lancement de l’interface graphique
python main_app.py
L’application se lance et permet d’utiliser la prédiction et les graphiques.

Résumé

nettoyer → entraîner → lancer l’interface

tous les fichiers sont dans src/, sauf le main_app.py

le modèle final est automatiquement chargé par l’application