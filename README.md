# Insurance Claim Prediction Project

## Project ID: PRCP-1010-InsClaimPred

### Overview

This project aims to address the challenge of an imbalanced insurance dataset to accurately predict which customers are likely to file a claim. The dataset contains a large number of features, many of which are anonymized, posing a significant challenge for traditional analysis. The project uses various machine learning techniques to build a robust predictive model and provides strategic recommendations for the insurance marketing team.

## Problem Statement

**The project is divided into two main tasks:**

Predictive Model: Develop a predictive model to identify customers likely to claim insurance.

Marketing Suggestions: Provide data-driven suggestions to the insurance marketing team to improve product adoption among customers.

## Dataset

The dataset used is train.csv, which contains:
595,212 rows and 59 columns.

Target Column: The target column is highly imbalanced:

**Class 0 (No Claim): 96.36%**

**Class 1 (Claim): 3.64%**

## Methodology
The project follows a comprehensive machine learning pipeline:

## Data Preprocessing:

**Irrelevant Column Dropping:** The id column was dropped as it holds no predictive value for the model.

**Feature Scaling:** The data was scaled using StandardScaler to ensure that all features contribute equally to the model training process.

## Addressing Data Imbalance:

Due to the significant imbalance in the target variable, several techniques were explored to prevent the model from becoming biased towards the majority class (no claims).

**RandomOverSampler:** This technique was applied to the training data to increase the number of samples in the minority class (claims), balancing the class distribution.

**SMOTE (Synthetic Minority Over-sampling Technique):** This advanced oversampling method was used to create synthetic data points for the minority class, providing a more robust training set.

## Dimensionality Reduction:

**Principal Component Analysis (PCA):** To handle the high number of features and reduce model complexity, PCA was applied. The analysis showed that just two principal components could explain a significant portion of the data's variance. The data was reduced to these two components for some models.

## Model Training and Evaluation:

#### Multiple classification models were trained and evaluated on the preprocessed data, including:

**Logistic Regression:** A baseline model for binary classification.

**Decision Tree Classifier:** A tree-based model that showed better performance than logistic regression.

**XGBoost Classifier:** A powerful gradient boosting model that yielded the best performance among traditional machine learning models.

**Sequential Model (Neural Network):** To further improve predictions, a Sequential deep learning model was built with multiple dense layers and dropout layers. This model demonstrated strong performance and was selected as the final predictive solution.

**Evaluation Metrics:** The models were evaluated using accuracy_score, confusion_matrix, and classification_report.

## Key Challenges

**Data Imbalance:** The skewed distribution of the target variable was the primary challenge, leading to overfitting and poor performance in initial models like Logistic Regression.

**Anonymized Features:** The lack of descriptive feature names prevented traditional EDA, making it difficult to gain insights into feature importance and relationships from the outset.

**High Dimensionality:** The large number of features added complexity and increased the time required for model training.

## Strategic Suggestions for the Marketing Team

#### Based on the predictive model and data analysis, the following suggestions are provided to the insurance marketing team:

**Educate Customers on Claim Process & Benefits:** Create campaigns to build trust and explain the value of the insurance by simplifying the claim process.

**Target Likely Buyers:** Use the model to identify high-probability customers for targeted promotions and upselling, while using educational content for low-probability customers.

**Personalize Insurance Plans:** Develop customizable plans and offer discounts to low-risk customers to encourage more sign-ups.

**Incentivize First-Time Buyers:** Attract new customers with special offers and "no-claim bonus" programs to reward them for maintaining their insurance.

**Address Psychological Barriers:** Use relatable storytelling in marketing campaigns to show how insurance can help during unexpected events, overcoming the belief that "it won't happen to me."

**Follow Up with Non-Claimers:** Conduct surveys or analysis to understand why a large number of customers do not file claims, helping to improve product and communication strategies.

## Code & Libraries

#### The project was developed using Python and standard data science libraries:

**numpy**

**pandas**

**seaborn**

**matplotlib**

**scikit-learn**

**imblearn**

**xgboost**

**tensorflow**
