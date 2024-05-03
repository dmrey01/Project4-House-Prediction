# --IN PROGRESS-- Project-4: House Price Prediction using various ML techniques

<p align="center">
  <img width="400" height="400" src="https://github.com/AAlbers341/project1_coffee_analysis/assets/137431770/e9f9f422-f019-46ba-9148-bd1f56829aaf">
</p>

## Table of Contents
1. [Tech Stack](#1-tech-stack)
2. [Data Sources](#2-data-sources)
3. [Background](#3-background)
4. [What is El Nino and La Nina?](#4-what-is-el-nino-and-la-nina)
5. [EDA Notebooks](#5-eda-notebooks)
6. [Conclusion](#6-conclusion)

## Project Overview
This project focuses on developing a predictive model for housing prices using various machine learning techniques. **The primary goal is to accurately predict house sales prices based on a comprehensive dataset that includes numerous features related to property characteristics such as area, quality, condition, and location.**

## Results Summary

### Below is the performance table. 

RMSE Calculations and Top Models:
- The Gradient Boosting Regressor clearly stands out with detailed RMSE analysis and adjustments in model complexity, suggesting it as one of the most effective models in the notebook.
- The Lasso Regression model stood out its implementation and detailed analysis.
- The Random Forest model was also highlighted due to its low RMSE value.


## Tech Stack
| Python Library  | Usage |
| ------------- | ------------- |
|![image](https://github.com/AAlbers341/project1_coffee_analysis/assets/137431770/8841ac4f-3cbe-49c0-a2f4-c0acb1100224)  | Data manipulation and analysis  |
| ![image](https://github.com/AAlbers341/project1_coffee_analysis/assets/137431770/4bf0c856-4875-4ff1-8ebd-03c0e66ebd14) | Numerical computations  |
| ![image](https://github.com/AAlbers341/project1_coffee_analysis/assets/137431770/0edbd134-7e50-4c13-9a74-4f6b8350ba57) | 2D plotting  |
| ![image](https://github.com/AAlbers341/project1_coffee_analysis/assets/137431770/76cae272-4bcc-4ab2-a376-880cd1d28190) | High-level plotting API   |
| ![image](https://github.com/AAlbers341/project1_coffee_analysis/assets/137431770/3e7c0369-6e67-4080-81bc-ee7d214f80c0) |  Scientific and technical computing  |

## Dataset
The dataset utilized in this project contains detailed information on residential home sales. It includes a wide range of features, such as:

- **OverallQual**: Overall material and finish quality
- **GrLivArea**: Above grade (ground) living area square feet
- **GarageCars**: Size of garage in car capacity
- **TotalBsmtSF**: Total square feet of basement area
- **FullBath**: Full bathrooms above grade

These and many other features provide a nuanced view of each home's attributes, contributing significantly to the model's predictive capabilities.

## Data Sources
| Data Source  | Usage |
| ------------- | ------------- |
| [Arabica Coffee Dataset - St. Louis FED](https://fred.stlouisfed.org/series/PCOFFOTMUSDM)  | Coffee prices since 1990s to present day for Arabica  |

## Models Used
Several models were explored and evaluated for their effectiveness in predicting house prices:

Lasso Regression: Utilized for its ability to perform feature selection by shrinking the coefficients of less important features to zero.
Gradient Boosting Regressor: Chosen for its ability to handle non-linear data and provide robust predictive power through ensemble learning.
Random Forest Regressor: Selected for its versatility and reliability, providing excellent results particularly in terms of handling a variety of data types and feature importance determination.
Key Results
Model Comparison: The models were evaluated based on the Root Mean Square Error (RMSE), with the Gradient Boosting and Random Forest models showing particularly promising results.
Feature Importance: Analysis revealed that certain features, such as OverallQual and GrLivArea, were highly predictive of house prices, guiding feature selection and engineering efforts.
Optimization: Parameter tuning was performed, particularly for the Random Forest and Gradient Boosting models, to enhance model accuracy and efficiency.

8 Machine Learning Algorithms were ran in this project.

- Supervised
    - Classification:
    - Regression: Linear Regression / Ridge / Lasso / ElasticNet
- Unsupervised
    - Clustering: ESupport Vector Machine
- Ensemble Methods
    - RandomForest / XGBoost / LGBM / CatBoost


## Conclusion
This project successfully demonstrates the application of advanced machine learning techniques to real-world data. The models developed not only provide accurate predictions of house prices but also offer insights into the factors that most significantly affect residential property values.

Instructions for Use
Setup: Clone the repository and install required Python packages.
Data Preparation: Load the data and perform necessary preprocessing steps as outlined in the notebooks.
Model Training and Evaluation: Follow the notebooks to train and evaluate the models. Adjust parameters as needed to optimize performance.
Deployment: Utilize the best-performing model to make predictions on new data.