Salary Prediction – End-to-End Machine Learning Pipeline

This repository showcases a full ML workflow aimed at predicting developer salaries using structured survey data.
The project is divided into several modules to ensure clarity, scalability, and reproducibility.

 Project Structure
/src
  ├── nettoyage_data.py      # Data cleaning & preprocessing
  ├── analyse_data.py        # Exploratory visualizations
  ├── entrainement_model.py  # Model training & evaluation
  ├── test_model.py          # User salary prediction test
  └── utils/                 # Helper functions
/data
/results
/models

 Technical Workflow

Data Cleaning

Handling missing entries

Normalizing categories

Removing salary outliers

Encoding categorical variables

Exploratory Data Analysis

Salary distribution

Correlation between skills, experience, and income

Detection of anomalies

Model Training

Tested algorithms: Linear Regression, RandomForest, GradientBoosting

Hyperparameter tuning

Cross-validation

Saving trained models

Evaluation

Metrics: R², RMSE, MAE

Comparison of model performance

Visualization of results

Testing Interface

Script allowing users to input parameters and get a predicted salary

 Objective

Understand salary determinants and build a reliable prediction model using real-world developer data.
