Salary Prediction 

This repository showcases a full ML workflow aimed at predicting developer salaries using structured survey data.
The project is divided into several modules to ensure clarity, scalability, and reproducibility.

## Project Structure
/src
  ├── nettoyage_data.py
  
  ├── analyse_data.py
  
  ├── entrainement_model.py
  
  ├── test_model.py
  
  └── utils/
/data
/results
/models

## Workflow

### Data Cleaning
- Handling missing values
- Normalizing categorical variables
- Removing outliers
- Encoding features

### Exploratory Data Analysis
- Salary distribution
- Correlation analysis
- Detection of anomalies
- Feature impact visualizations

### Model Training
- Linear Regression, RandomForest, GradientBoosting
- Hyperparameter tuning
- Cross-validation
- Model saving

### Evaluation
- R², RMSE, MAE
- Performance comparison
- Result visualizations

### Testing
- User input script for salary prediction

