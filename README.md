# GitHub Stars Prediction Model

A machine learning project for predicting the number of stars a GitHub repository will receive based on various repository characteristics. The model utilizes regression techniques, and includes data preprocessing, model training, and evaluation processes.

## Project Overview

The goal of this project is to predict the number of stars a GitHub repository will receive, based on various characteristics such as the number of forks, issues, commits, and other repository-related features. The model also analyzes these factors to provide insights into what contributes to a repository's popularity.

## Key Features

### Data Preprocessing:
- **Handling Missing Values and Outliers**: Cleaned the data to handle missing values and outliers effectively.
- **Log Transformation**: Applied a logarithmic transformation to both the target variable **'Stars'** and numerical features such as **'Forks'** to normalize their distributions and handle skewness.

### Model Training:
- Trained multiple machine learning models, including **Lasso Regression**, **Random Forest**, and **XGBoost**.
- Evaluated model performance using **RMSE** (Root Mean Squared Error).

### Hyperparameter Tuning:
- Used **GridSearchCV** with **5-fold cross-validation** to optimize the models' hyperparameters and improve performance.

### Error Analysis:
- Analyzed residuals (errors between predicted and actual values) to identify areas of improvement and refine the model.
- Investigated patterns in prediction errors to enhance future iterations.

## Installation

To set up this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/github-stars-prediction.git
    cd github-stars-prediction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

Make sure you have Python 3.x installed along with the following libraries:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

## Usage

To run the model training and evaluation, execute the following:

```bash
python train_model.py
