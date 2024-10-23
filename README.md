# Customer Churn Analysis & Prediction

### [Click here to use the Churn Prediction App](https://card-churn-prediction-2024.streamlit.app/)

## Overview

Customer churn prediction is essential for businesses to identify which customers are at risk of leaving, allowing them to take proactive measures. This project uses machine learning techniques to build predictive models that classify whether a customer will churn based on various demographic, transactional, and engagement features.

The main objectives of this project are:
- Data exploration and preprocessing to prepare the data for machine learning.
- Building, tuning, and evaluating several machine learning models.
- Selecting the best model for customer churn prediction.
- Deploying the final model as a web application for real-time churn prediction.

## Table of Contents
1. [Project Overview](#overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Libraries Used](#libraries-used)
6. [Data Preprocessing](#data-preprocessing)
7. [Modeling Techniques](#modeling-techniques)
8. [Model Evaluation](#model-evaluation)
9. [Deployment](#deployment)
10. [Results](#results)
11. [Conclusion](#conclusion)

## Dataset

The dataset used in this project, **credit_card_churn.csv**, contains 21 features related to customer demographics, credit card usage, and engagement. The key target variable is the `Attrition_Flag`, which indicates whether a customer has churned or not.

### Key Features:
- **Customer_Age**: The age of the customer.
- **Gender**: The gender of the customer (M or F).
- **Dependent_count**: The number of dependents that a customer has.
- **Education_Level**: The educational qualification of the customer.
- **Marital_Status**: The marital status of the customer (Married, Single, Divorced).
- **Income_Category**: The income category of the customer (e.g., Less than $40K, $40K-$60K, $60K-$80K).
- **Card_Category**: The type of credit card (Blue, Gold, Silver, Platinum).
- **Months_on_book**: The number of months the customer has had their account open.
- **Total_Relationship_Count**: The total number of products held by the customer.
- **Months_Inactive_12_mon**: The number of months the customer has been inactive in the last 12 months.
- **Contacts_Count_12_mon**: The number of contacts with the customer in the last 12 months.
- **Credit_Limit**: The credit limit on the customer’s card.
- **Total_Revolving_Bal**: The total revolving balance on the customer’s credit card.
- **Avg_Open_To_Buy**: The average open-to-buy credit available on the credit card.
- **Total_Amt_Chng_Q4_Q1**: The change in the total transaction amount between Q4 and Q1.
- **Total_Trans_Amt**: The total transaction amount in the last 12 months.
- **Total_Trans_Ct**: The total number of transactions in the last 12 months.
- **Total_Ct_Chng_Q4_Q1**: The change in the total transaction count between Q4 and Q1.
- **Avg_Utilization_Ratio**: The average credit card utilization ratio (percentage of the credit limit used).

  
Target variable:
- **Attrition_Flag**: 0 = Existing customer, 1 = Attrited customer.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MahmoudSaad21/Card-Churn-Prediction.git
   ```

2. Install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. You can run the main Python script `customer_churn_analysis.py` to reproduce the analysis and model-building process.

## Project Structure

```
.
├── customer_churn_analysis.py       # Main Python script for analysis and model building
├── credit_card_churn.csv            # Dataset for customer churn analysis
├── models                           # Folder containing trained models (.pkl files)
├── scaler                           # Folder containing scaler (.pkl files)
├── encoder                          # Folder containing encoders (.pkl files)
├── README.md                        # Documentation for the project
├── requirements.txt                 # List of dependencies
├── app.py                           # Streamlit app for churn prediction
├── sampled_data.csv                 # sampled data for teasting
├── mlruns                           # Folder containing model evaluation results
```

## Libraries Used

This project utilizes several libraries for data manipulation, machine learning, and visualization:

- **Pandas**: Data manipulation and analysis.
- **Numpy**: Numerical computations.
- **Matplotlib & Seaborn**: Data visualization and plotting.
- **Scikit-learn**: Machine learning algorithms and model evaluation tools.
- **Imbalanced-learn**: For handling imbalanced datasets with **SMOTEENN** (a combination of oversampling and undersampling).
- **XGBoost & LightGBM**: Gradient boosting algorithms for classification.
- **TQDM**: Progress bar for tracking lengthy operations.
- **Joblib & Pickle**: Model serialization for saving and loading trained models.
- **Streamlit**: Web framework for deploying the model as an interactive app.

## Data Preprocessing

Preprocessing steps performed before modeling:
1. **Handling Missing Values**: Visualizing missing data using `missingno` and filling missing values accordingly.
2. **Encoding Categorical Variables**: Using **OneHotEncoder** and **LabelEncoder** to convert categorical features like `Gender`, `Education_Level`, `Marital_Status`, etc., into numerical format.
3. **Scaling**: Normalizing numerical features such as `Credit_Limit` and `Avg_Utilization_Ratio` using **StandardScaler** to ensure they are on the same scale.
4. **Handling Imbalance**: Using **SMOTEENN** (Synthetic Minority Oversampling Technique and Edited Nearest Neighbors) to deal with the imbalanced target variable (`Attrition_Flag`), ensuring that the model doesn't favor the majority class.

## Modeling Techniques

We used multiple models to predict customer churn, including:

1. **Logistic Regression**: A baseline linear model.
2. **K-Nearest Neighbors (KNN)**: A non-parametric method that classifies based on proximity.
3. **Decision Tree**: A model that splits data recursively to create decision rules.
4. **Random Forest**: An ensemble of decision trees to improve performance.
5. **AdaBoost**: An adaptive boosting technique for classification.
6. **Gradient Boosting**: Sequentially builds models to minimize errors.
7. **XGBoost**: An efficient and scalable implementation of gradient boosting.
8. **Naive Bayes**: Based on Bayes' theorem for probabilistic classification.
9. **LightGBM**: A gradient boosting framework focusing on large datasets with better performance.

## Model Evaluation

The models were evaluated based on various metrics to determine their performance:
- **Accuracy**: Proportion of correct predictions.
- **Recall**: Ability to detect true positives (churned customers).
- **Confusion Matrix**: Breakdown of predicted vs actual classes.
- **Classification Report**: Provides precision, recall, F1-score, and support for each class.

### Hyperparameter Tuning
- **GridSearchCV** was used for hyperparameter tuning to identify the best parameters for each model.

### Feature Importance
- **XGBoost** provides a feature importance plot, highlighting which features contribute most to predicting customer churn.

## Deployment

### Web Application

The final churn prediction model was deployed as a web application using **Streamlit**. The app allows users to upload new customer data or enter the data manually and predicts whether they are likely to churn. The key features of the deployment include:

1. **User Input**: Users can upload a CSV file containing customer data or manually entering customer data, which is processed by the app.
2. **Model Selection**: Users can select the prediction model (e.g., Random Forest, XGBoost, Logistic Regression) based on their preference.
3. **Prediction Output**: The app provides a classification output indicating whether the customer is likely to churn or remain.

## Results

After evaluating multiple models, **XGBoost** and **Random Forest** demonstrated the highest accuracy and recall. These models are most suitable for predicting customer churn due to their robustness and ability to handle non-linear relationships in the data.

## Conclusion

This project demonstrates how machine learning can be effectively used to predict customer churn. By building and deploying multiple models, we provide an end-to-end solution from data preprocessing to real-time prediction through a web interface. The deployment of the model as a Streamlit app offers a scalable and accessible way for businesses to integrate churn prediction into their operations.

## Team Members

- **Mahmoud Saad Mahmoud**  
  Email: mahmoud.saad.mahmoud.11@gmail.com

- **zeyad ahmed mostafa**  
  Email: ziada00700@gmail.com 

- **Mohamed Badr Mohamed**  
  Email: bdr00637@gmail.com
  
- **Alaa Ghalib Othman**  
  Email: ghalibalaa29@gmail.com 

- **Ahmed Mohamed Tawfik**  
  Email: ahmad970816@gmail.com 

