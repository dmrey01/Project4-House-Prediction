# --IN PROGRESS-- Project-4: House Price Prediction using various ML techniques :house_with_garden:	

<p align="center">
  <img width="500" height="500" src="https://github.com/dmrey01/Project-4/blob/main/4.Other/house_pic.jpg">
</p>

## Table of Contents
1. [Project Overview](#project-overview)
2. [Results Summary](#results-summary)
3. [Dataset](#dataset)
4. [Data Source](#data-source)
5. [Tech Stack](#tech-stack)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Data Visualizations](#data-visualizations)
8. [Machine Learning Modeling](#machine-learning-modeling)
9. [Blending All Models](#blending-all-models)
10. [Evaluation and Submissions](#evaluation-and-submissions)
11. [Conclusion](#Conclusion)

## Project Overview
This project focuses on developing a predictive model for housing prices using various machine learning techniques. The primary goal is to **accurately predict house sales prices based on a comprehensive dataset that includes numerous features related to property characteristics** such as area, quality, condition, and location.

## Results Summary

Several models were explored and evaluated for their effectiveness in predicting house prices. 

**Key Results**
- **Model Comparison:** The models were evaluated based on the **Root Mean Square Error (RMSE)**, with the *Gradient Boosting, Lasso Regression and Random Forest models* showing particularly promising results.
- **Feature Importance:** Analysis revealed that certain features, such as *OverallQual and GrLivArea*, were highly predictive of house prices, guiding feature selection and engineering efforts.
- **Optimization:** **Parameter tuning was performed**, particularly for the *Random Forest and Gradient Boosting models*, to enhance model accuracy and efficiency.

## Performance Table

| **ML Technique**  | **RMSE Error** |
| ------------- | ------------- |
| Lasso Regression Model  | 0.1540  |
| Random Forest Model  | 0.1540  |
| Gradient Boosting Model  | 0.1431  |
| Linear Regression Model  | 0.1897  |
| Decision Tree Regressor Model  | 0.2103  |
| Random Forest Regressor Model  | 0.1546  |
| Support Vector Regression (SVR) Model  | 0.4021  |
| K-Nearest Neighbors Regressor Model  | 0.2006  |
| Simple Average Blend  | 0.1631  |

<p align="left">
  <img width="650" height="500" src="https://github.com/dmrey01/Project-4/blob/main/4.Other/ml_model_by_rmse.jpg">
</p>

### Model Summary with RMSE Error

1. **Gradient Boosting Model (0.1431):**
- **Performance:** Best performing model.
- **Interpretation:** This model has the lowest RMSE, indicating it has the best predictive accuracy among the models listed. Gradient Boosting effectively handles both linear and non-linear relationships by building an ensemble of weak prediction models, typically decision trees.

2. **Lasso Regression Model (0.1540):**
- **Performance:** Joint second best with Random Forest.
- **Interpretation:** Lasso (Least Absolute Shrinkage and Selection Operator) performs both variable selection and regularization to enhance the prediction accuracy. Its performance is strong, especially considering it helps in reducing overfitting in models with many features.

3. **Random Forest Model (0.1540) and Random Forest Regressor Model (0.1546):**
- **Performance:** Very close to Lasso, almost identical RMSE for one variant.
- **Interpretation:** Shows robust performance, slightly varied between implementations. Random Forest is effective due to its method of averaging multiple deep decision trees, trained on different parts of the same training set, which reduces overfitting and variance.

4. **Simple Average Blend (0.1631):**
- **Performance:** Mid-tier performance.
- **Interpretation:** This technique involves averaging the predictions of multiple models. It's a straightforward ensemble method that can yield decent results by blending different model outputs, reducing variance but not always capturing complex patterns effectively.

5. **Linear Regression Model (0.1897):**
- **Performance:** Lower middle performance.
- **Interpretation:** Shows moderate predictive power. Linear regression assumes a linear relationship between features and the target variable, which might not always hold true in complex datasets like house pricing, explaining the higher RMSE.

6. **K-Nearest Neighbours Regressor Model (0.2006):**
- **Performance:** Lower performance.
- **Interpretation:** This model uses feature similarity to predict values of new data points, which indicates it may struggle with high dimensional data or where feature scaling is not adequately managed.

7. **Decision Tree Regressor Model (0.2103):**
- **Performance:** Lower performance among tree-based models.
- **Interpretation:** Prone to overfitting, especially without adequate depth constraints or pruning. Decision Trees can capture complex patterns but often don’t generalize well.

8. **Support Vector Regression (SVR) Model (0.4021):**
- **Performance:** Worst performance.
- **Interpretation:** This model, especially with default settings, may not perform well with large data sets or when features influencing the target are nonlinear. Its high RMSE suggests that it might not have captured the complex relationships in the data effectively.

## Dataset
The datasets utilized in this project contains detailed information on residential home sales reported in **Ames, Iowa**. Datasets are broken up into two parts: **train and test**.

### 1. Train Dataset Analysis

**The train.csv dataset includes data on 1,460 properties, described by 81 features.** Here are some key insights and observations:

**General Structure**
- **Total Entries:** 1,460
- **Total Features:** 81, including a wide range of property characteristics such as zoning, lot size, street type, building type, quality and condition ratings, and many more detailed attributes related to the property’s interior and exterior.

**Notable Features**
- **SalePrice:** The target variable for prediction, with values ranging from **$34,900 to $755,000**, and a mean of approximately **$180,921**.

- **MSSubClass, MSZoning, and Neighborhood:** These features categorize properties by type, zoning, and location within Ames, Iowa.

- **YearBuilt and YearRemodAdd:** Indicating the age of the property and the last remodeling year.
- **OverallQual and OverallCond:** Rating scales from 1 to 10 that assess the overall material and finish quality, and the overall condition of the property.
- **TotalBsmtSF and GrLivArea:** Key area measurements in square feet for basement and above-ground living area.

**Categorical and Numeric Distribution**
- Many features are **categorical** (e.g., Street, LotShape, Utilities), requiring encoding for machine learning applications.
- **Numeric** data spans across discrete and continuous variables, with different scales and distributions, which may need standardization or normalization.

```ruby
train_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>

### 2. Test Dataset Analysis

The test dataset includes data on **1,459 properties with the same 80 features** as in the train dataset (excluding *SalePrice*). Here are some key observations:

**General Structure**
- **Total Entries:** 1,459

- **Total Features:** 80 (one less than the train set due to the absence of the target variable SalePrice).

```ruby
# Display basic info
print(train_df.info())
print("***Train dataset shape is {}***".format(train_df.shape))

print(test_df.info())
print("***Test dataset shape is {}***".format(test_df.shape))
```

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 81 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   object 
 3   LotFrontage    1201 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   object 
 6   Alley          91 non-null     object 
 7   LotShape       1460 non-null   object 
 8   LandContour    1460 non-null   object 
 9   Utilities      1460 non-null   object 
 10  LotConfig      1460 non-null   object 
 11  LandSlope      1460 non-null   object 
 12  Neighborhood   1460 non-null   object 
 13  Condition1     1460 non-null   object 
 14  Condition2     1460 non-null   object 
 15  BldgType       1460 non-null   object 
 16  HouseStyle     1460 non-null   object 
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
 20  YearRemodAdd   1460 non-null   int64  
 21  RoofStyle      1460 non-null   object 
 22  RoofMatl       1460 non-null   object 
 23  Exterior1st    1460 non-null   object 
 24  Exterior2nd    1460 non-null   object 
 25  MasVnrType     588 non-null    object 
 26  MasVnrArea     1452 non-null   float64
 27  ExterQual      1460 non-null   object 
 28  ExterCond      1460 non-null   object 
 29  Foundation     1460 non-null   object 
 30  BsmtQual       1423 non-null   object 
 31  BsmtCond       1423 non-null   object 
 32  BsmtExposure   1422 non-null   object 
 33  BsmtFinType1   1423 non-null   object 
 34  BsmtFinSF1     1460 non-null   int64  
 35  BsmtFinType2   1422 non-null   object 
 36  BsmtFinSF2     1460 non-null   int64  
 37  BsmtUnfSF      1460 non-null   int64  
 38  TotalBsmtSF    1460 non-null   int64  
 39  Heating        1460 non-null   object 
 40  HeatingQC      1460 non-null   object 
 41  CentralAir     1460 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1460 non-null   int64  
 44  2ndFlrSF       1460 non-null   int64  
 45  LowQualFinSF   1460 non-null   int64  
 46  GrLivArea      1460 non-null   int64  
 47  BsmtFullBath   1460 non-null   int64  
 48  BsmtHalfBath   1460 non-null   int64  
 49  FullBath       1460 non-null   int64  
 50  HalfBath       1460 non-null   int64  
 51  BedroomAbvGr   1460 non-null   int64  
 52  KitchenAbvGr   1460 non-null   int64  
 53  KitchenQual    1460 non-null   object 
 54  TotRmsAbvGrd   1460 non-null   int64  
 55  Functional     1460 non-null   object 
 56  Fireplaces     1460 non-null   int64  
 57  FireplaceQu    770 non-null    object 
 58  GarageType     1379 non-null   object 
 59  GarageYrBlt    1379 non-null   float64
 60  GarageFinish   1379 non-null   object 
 61  GarageCars     1460 non-null   int64  
 62  GarageArea     1460 non-null   int64  
 63  GarageQual     1379 non-null   object 
 64  GarageCond     1379 non-null   object 
 65  PavedDrive     1460 non-null   object 
 66  WoodDeckSF     1460 non-null   int64  
 67  OpenPorchSF    1460 non-null   int64  
 68  EnclosedPorch  1460 non-null   int64  
 69  3SsnPorch      1460 non-null   int64  
 70  ScreenPorch    1460 non-null   int64  
 71  PoolArea       1460 non-null   int64  
 72  PoolQC         7 non-null      object 
 73  Fence          281 non-null    object 
 74  MiscFeature    54 non-null     object 
 75  MiscVal        1460 non-null   int64  
 76  MoSold         1460 non-null   int64  
 77  YrSold         1460 non-null   int64  
 78  SaleType       1460 non-null   object 
 79  SaleCondition  1460 non-null   object 
 80  SalePrice      1460 non-null   int64  
dtypes: float64(3), int64(35), object(43)
memory usage: 924.0+ KB
None
***Train dataset shape is (1460, 81)***
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1459 entries, 0 to 1458
Data columns (total 80 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1459 non-null   int64  
 1   MSSubClass     1459 non-null   int64  
 2   MSZoning       1455 non-null   object 
 3   LotFrontage    1232 non-null   float64
 4   LotArea        1459 non-null   int64  
 5   Street         1459 non-null   object 
 6   Alley          107 non-null    object 
 7   LotShape       1459 non-null   object 
 8   LandContour    1459 non-null   object 
 9   Utilities      1457 non-null   object 
 10  LotConfig      1459 non-null   object 
 11  LandSlope      1459 non-null   object 
 12  Neighborhood   1459 non-null   object 
 13  Condition1     1459 non-null   object 
 14  Condition2     1459 non-null   object 
 15  BldgType       1459 non-null   object 
 16  HouseStyle     1459 non-null   object 
 17  OverallQual    1459 non-null   int64  
 18  OverallCond    1459 non-null   int64  
 19  YearBuilt      1459 non-null   int64  
 20  YearRemodAdd   1459 non-null   int64  
 21  RoofStyle      1459 non-null   object 
 22  RoofMatl       1459 non-null   object 
 23  Exterior1st    1458 non-null   object 
 24  Exterior2nd    1458 non-null   object 
 25  MasVnrType     565 non-null    object 
 26  MasVnrArea     1444 non-null   float64
 27  ExterQual      1459 non-null   object 
 28  ExterCond      1459 non-null   object 
 29  Foundation     1459 non-null   object 
 30  BsmtQual       1415 non-null   object 
 31  BsmtCond       1414 non-null   object 
 32  BsmtExposure   1415 non-null   object 
 33  BsmtFinType1   1417 non-null   object 
 34  BsmtFinSF1     1458 non-null   float64
 35  BsmtFinType2   1417 non-null   object 
 36  BsmtFinSF2     1458 non-null   float64
 37  BsmtUnfSF      1458 non-null   float64
 38  TotalBsmtSF    1458 non-null   float64
 39  Heating        1459 non-null   object 
 40  HeatingQC      1459 non-null   object 
 41  CentralAir     1459 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1459 non-null   int64  
 44  2ndFlrSF       1459 non-null   int64  
 45  LowQualFinSF   1459 non-null   int64  
 46  GrLivArea      1459 non-null   int64  
 47  BsmtFullBath   1457 non-null   float64
 48  BsmtHalfBath   1457 non-null   float64
 49  FullBath       1459 non-null   int64  
 50  HalfBath       1459 non-null   int64  
 51  BedroomAbvGr   1459 non-null   int64  
 52  KitchenAbvGr   1459 non-null   int64  
 53  KitchenQual    1458 non-null   object 
 54  TotRmsAbvGrd   1459 non-null   int64  
 55  Functional     1457 non-null   object 
 56  Fireplaces     1459 non-null   int64  
 57  FireplaceQu    729 non-null    object 
 58  GarageType     1383 non-null   object 
 59  GarageYrBlt    1381 non-null   float64
 60  GarageFinish   1381 non-null   object 
 61  GarageCars     1458 non-null   float64
 62  GarageArea     1458 non-null   float64
 63  GarageQual     1381 non-null   object 
 64  GarageCond     1381 non-null   object 
 65  PavedDrive     1459 non-null   object 
 66  WoodDeckSF     1459 non-null   int64  
 67  OpenPorchSF    1459 non-null   int64  
 68  EnclosedPorch  1459 non-null   int64  
 69  3SsnPorch      1459 non-null   int64  
 70  ScreenPorch    1459 non-null   int64  
 71  PoolArea       1459 non-null   int64  
 72  PoolQC         3 non-null      object 
 73  Fence          290 non-null    object 
 74  MiscFeature    51 non-null     object 
 75  MiscVal        1459 non-null   int64  
 76  MoSold         1459 non-null   int64  
 77  YrSold         1459 non-null   int64  
 78  SaleType       1458 non-null   object 
 79  SaleCondition  1459 non-null   object 
dtypes: float64(11), int64(26), object(43)
memory usage: 912.0+ KB
None
***Test dataset shape is (1459, 80)***


### Summary and Insights 

Both the train and test datasets are quite consistent in terms of the types of features and data quality issues (e.g., missing values). **The test set lacks the SalePrice feature** since it's meant for making predictions in a typical modeling task.

**These are the features provide a nuanced view of each home's attributes, contributing significantly to the model's predictive capabilities.**

- **OverallQual**: Overall material and finish quality
- **GrLivArea**: Above grade (ground) living area square feet
- **GarageCars**: Size of garage in car capacity
- **TotalBsmtSF**: Total square feet of basement area
- **FullBath**: Full bathrooms above grade

![alt text](image-3.png)

## Data Source
| Data Source  | Description |
| ------------- | ------------- |
| [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)  | Predict sales prices and practice feature engineering, RFs, and gradient boosting  |

## Tech Stack
| Name  | Usage |
| ------------- | ------------- |
|![image](https://github.com/AAlbers341/project1_coffee_analysis/assets/137431770/8841ac4f-3cbe-49c0-a2f4-c0acb1100224)  | Data manipulation and analysis  |
| ![image](https://github.com/AAlbers341/project1_coffee_analysis/assets/137431770/4bf0c856-4875-4ff1-8ebd-03c0e66ebd14) | Numerical computations  |
| ![image](https://github.com/AAlbers341/project1_coffee_analysis/assets/137431770/0edbd134-7e50-4c13-9a74-4f6b8350ba57) | 2D plotting  |
| ![image](https://github.com/dmrey01/Project-4/blob/main/4.Other/scikit.jpg) | Machine learning library   |
| ![image](https://github.com/dmrey01/Project-4/blob/main/4.Other/seaborn.jpg) | Library for creating statistical graphics and visualizing data  |
| ![image](https://github.com/dmrey01/Project-4/blob/main/4.Other/tableau.jpg) | Visual analytics platform  |
| ![image](https://github.com/dmrey01/Project-4/blob/main/4.Other/google_colab.jpg) | Hosted Jupyter Notebook service  |
| ![image](https://github.com/dmrey01/Project-4/blob/main/4.Other/power_point.jpg) | Presentation program  |
| ![image](https://github.com/dmrey01/Project-4/blob/main/4.Other/github.jpg) | Developer platform  |

## Exploratory Data Analysis (EDA)


```ruby
train_df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8, color='green');
```
![alt text](image-1.png)

```ruby
test_df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
```

![alt text](image-2.png)

## Data Visualizations

## Machine Learning Modeling
:

Lasso Regression: Utilized for its ability to perform feature selection by shrinking the coefficients of less important features to zero.
Gradient Boosting Regressor: Chosen for its ability to handle non-linear data and provide robust predictive power through ensemble learning.
Random Forest Regressor: Selected for its versatility and reliability, providing excellent results particularly in terms of handling a variety of data types and feature importance determination.
Key Results
Model Comparison: 
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

## Blending All Models

## Evaluation and Submissions

## Conclusion
This project successfully demonstrates the application of advanced machine learning techniques to real-world data. The models developed not only provide accurate predictions of house prices but also offer insights into the factors that most significantly affect residential property values.