# GitHub Stars Prediction Model

A machine learning project for predicting the number of stars a GitHub repository will receive based on various repository characteristics. The model utilizes regression techniques, and includes data preprocessing, model training, and evaluation processes.

## Project Overview

The goal of this project is to predict the number of stars a GitHub repository will receive, based on various characteristics such as the number of forks, issues, commits, and other repository-related features. The model also analyzes these factors to provide insights into what contributes to a repository's popularity.

## Dataset Variables

1. **Name**: The name of the GitHub repository.
2. **Description**: A brief description of the repository's purpose or content.
3. **URL**: The URL link to the repository.
4. **Created At**: The date and time when the repository was created.
5. **Updated At**: The date and time when the repository was last updated.
6. **Homepage**: The homepage or website associated with the repository, if available.
7. **Size**: The size of the repository in kilobytes.
8. **Stars**: The number of stars the repository has received from GitHub users.
9. **Forks**: The number of times the repository has been forked by other users.
10. **Issues**: The number of open issues in the repository.
11. **Language**: The programming language(s) used in the repository.
12. **License**: The type of license associated with the repository, if any.
13. **Topics**: A list of topics or tags associated with the repository for categorization.
14. **Has Issues**: A boolean value indicating whether the repository has issues enabled.
15. **Has Projects**: A boolean value indicating whether the repository has projects enabled.
16. **Has Downloads**: A boolean value indicating whether the repository allows downloads.
17. **Has Wiki**: A boolean value indicating whether the repository has a wiki page.
18. **Has Pages**: A boolean value indicating whether the repository has GitHub Pages (static website) enabled.
19. **Has Discussions**: A boolean value indicating whether the repository has discussions enabled.
20. **Is Fork**: A boolean value indicating whether the repository is a fork of another repository.
21. **Is Archived**: A boolean value indicating whether the repository is archived (no longer maintained).
22. **Is Template**: A boolean value indicating whether the repository is a template repository.
23. **Default Branch**: The name of the default branch in the repository (e.g., "main" or "master").

## Key Features

### Data Preprocessing:
- **Handling Missing Values and Outliers**: Cleaned the data to handle missing values and outliers effectively.
- **Log Transformation**: Applied a logarithmic transformation to both the target variable **'Stars'** and numerical features such as **'Forks'** to normalize their distributions and handle skewness.

### Model Training and Hyperparameter Tuning:
- Trained multiple machine learning models, including **Lasso Regression**, **Random Forest**, and **XGBoost**.
- Used **GridSearchCV** with **5-fold cross-validation** to optimize the models' hyperparameters and improve performance.
- Evaluated model performance using **RMSE** (Root Mean Squared Error) and other relevant metrics.

## Tools Used

- **Python**
- **Jupyter Notebook**
- **pandas**
- **numpy**
- **scikit-learn**
- **xgboost**
- **matplotlib** & **seaborn**
- **os**
- **datetime**

