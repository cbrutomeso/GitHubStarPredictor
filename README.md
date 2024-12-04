# GitHub Stars Prediction Model

A machine learning project for predicting the number of stars a GitHub repository will receive based on various repository characteristics. The model utilizes regression techniques, and includes data preprocessing, model training, and evaluation processes.

## Project Overview

The goal of this project is to predict the number of stars a GitHub repository will receive, based on various characteristics such as the number of forks, issues, commits, and other repository-related features. The model also analyzes these factors to provide insights into what contributes to a repository's popularity.

## Key Features

### Data Preprocessing:
- **Handling Missing Values and Outliers**: Cleaned the data to handle missing values and outliers effectively.
- **Log Transformation**: Applied a logarithmic transformation to both the target variable **'Stars'** and numerical features such as **'Forks'** to normalize their distributions and handle skewness.

### Model Training and Hyperparameter Tuning:
- Trained multiple machine learning models, including **Lasso Regression**, **Random Forest**, and **XGBoost**.
- Used **GridSearchCV** with **5-fold cross-validation** to optimize the models' hyperparameters and improve performance.
- Evaluated model performance using **RMSE** (Root Mean Squared Error) and other relevant metrics.

## Tools Used

- **Python** (Version 3.x): The programming language used to implement the model and preprocess the data.
- **Jupyter Notebook** (Version 6.x): Used for interactive development and experimentation.
- **pandas** (Version 1.x): For data manipulation and analysis.
- **numpy** (Version 1.x): For numerical operations and handling arrays.
- **scikit-learn** (Version 0.x): For building machine learning models, performing cross-validation, and hyperparameter tuning with **GridSearchCV**.
- **xgboost** (Version 1.x): For training the XGBoost model, which is one of the regression techniques used.
- **matplotlib** (Version 3.x) & **seaborn** (Version 0.x): For data visualization, including plotting the distribution of features and evaluating model performance.
- **os** (Version x.x): For interacting with the operating system and handling file paths.
- **datetime** (Version x.x): For handling date and time operations during data preprocessing.

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
