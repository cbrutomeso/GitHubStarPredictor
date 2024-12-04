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

- **Python**: The programming language used to implement the model and preprocess the data.
- **Jupyter Notebook**: Used for interactive development and experimentation.
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations and handling arrays.
- **scikit-learn**: For building machine learning models, performing cross-validation, and hyperparameter tuning with **GridSearchCV**.
- **xgboost**: For training the XGBoost model, which is one of the regression techniques used.
- **matplotlib** & **seaborn**: For data visualization, including plotting the distribution of features and evaluating model performance.
- **os**: For interacting with the operating system and handling file paths.
- **datetime**: For handling date and time operations during data preprocessing.

