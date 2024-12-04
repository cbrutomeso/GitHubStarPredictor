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

## Jupyter Notebooks

This project includes two Jupyter Notebooks that provide a detailed walkthrough of the data analysis and machine learning model development process. Each notebook is structured to explain and demonstrate the key steps in the project, from data preprocessing to model training and evaluation.

### Notebooks:

1. **data_processing.ipynb**:
   - filepath: notebooks/data_processing.ipynb
   - This notebook focuses on loading the dataset, handling missing values, and performing exploratory data analysis (EDA).
   - It includes the following key features:
     - **Handling Missing Values and Outliers**: Cleaned the data to handle missing values and outliers effectively.
     - **Log Transformation**: Applied a logarithmic transformation to both the target variable **'Stars'** and numerical features such as **'Forks'** to normalize their distributions and handle skewness.
   - Various visualizations are created to understand the distribution of features and their relationships with the target variable.

2. **model_training.ipynb**:
   - filepath: notebooks/model_training.ipynb
   - This notebook covers the training of machine learning models, including **Lasso Regression**, **Random Forest**, and **XGBoost**.
   - It includes the following key features:
     - **Model Training**: Trained multiple machine learning models.
     - **Hyperparameter Tuning**: Used **GridSearchCV** with **5-fold cross-validation** to optimize the models' hyperparameters and improve performance.
     - **Performance Evaluation**: Evaluated model performance using **RMSE** (Root Mean Squared Error) and other relevant metrics.
   - The results of different models are compared to determine the best-performing approach.

Each notebook is independent, allowing users to follow the step-by-step analysis and adjust or extend the code as needed. To run the notebooks, install the required dependencies and follow the instructions provided within each notebook.

## Summary Results

The best-performing model in this project was the **Random Forest** model. The results obtained on the test set are as follows:

- The **Test RMSE** for **Random Forest** was **0.5544**, slightly better than **XGBoost** at **0.5580**, and significantly better than **Lasso Regression** at **0.6543**, indicating that Random Forest minimized prediction error the most among the three models.
- The **Test R²** value was **0.6846**, which suggests that the model explains approximately 68.46% of the variance in the target variable, **Stars**.
- Regarding feature importance, the results suggest that the **number of forks a repository plays a significant role in determining the number of stars** it will receive, with an importance value of **0.7118**. Other important features include the time since the last update, the number of issues, the repository's age, and whether or not it has discussions enabled.

