import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler

# Load the models
models = {
    "Logistic Regression": joblib.load('logistic_model.pkl'),
    "Random Forest": joblib.load('random_forest_model.pkl'),
    "Decision Tree": joblib.load('decision_tree_model.pkl'),
    "Naive Bayes": joblib.load('naive_bayes_model.pkl'),
}

# Function to preprocess data
def preprocess_data(data, drop_features):
    data.drop(columns=drop_features, inplace=True)

    mapping = {'Attrited Customer': 1, 'Existing Customer': 0}
    data['Attrition_Flag'] = data['Attrition_Flag'].replace(mapping)

    X = data

    # Encoding categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scaling features
    columns_to_scale = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                        'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
                        'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                        'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

    scaler = StandardScaler()
    X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])

    return X

# Streamlit app layout
st.title("Customer Attrition Prediction")

st.markdown("""
This application predicts customer attrition using multiple models. 
You can either enter customer data manually or upload a CSV file. 
Choose a model and specify features to drop before making predictions.
""")

# Model selection
model_name = st.selectbox("Select Model", options=list(models.keys()))

# Data entry method
data_input_method = st.radio("Data Input Method", ("Manual Entry", "Upload CSV"))

# Manual data entry
if data_input_method == "Manual Entry":
    data = {}
    # Input fields for all the necessary customer attributes
    data['Customer_Age'] = st.number_input("Customer_Age", min_value=0)
    data['Gender'] = st.selectbox("Gender", ["M", "F"])
    data['Dependent_count'] = st.number_input("Dependent_count", min_value=0)
    data['Education_Level'] = st.selectbox("Education_Level", ["High School", "Graduate", "Uneducated"])
    data['Marital_Status'] = st.selectbox("Marital_Status", ["Married", "Single", "Unknown"])
    data['Income_Category'] = st.selectbox("Income_Category", ["Less than $40K", "$60K - $80K", "$80K - $120K"])
    data['Card_Category'] = st.selectbox("Card_Category", ["Blue", "Gold", "Platinum"])
    data['Months_on_book'] = st.number_input("Months_on_book", min_value=0)
    data['Credit_Limit'] = st.number_input("Credit_Limit", min_value=0)
    data['Total_Revolving_Bal'] = st.number_input("Total_Revolving_Bal", min_value=0)
    data['Avg_Open_To_Buy'] = st.number_input("Avg_Open_To_Buy", min_value=0)
    data['Total_Amt_Chng_Q4_Q1'] = st.number_input("Total_Amt_Chng_Q4_Q1", min_value=0)
    data['Total_Trans_Amt'] = st.number_input("Total_Trans_Amt", min_value=0)
    data['Total_Trans_Ct'] = st.number_input("Total_Trans_Ct", min_value=0)
    data['Total_Ct_Chng_Q4_Q1'] = st.number_input("Total_Ct_Chng_Q4_Q1", min_value=0)
    data['Avg_Utilization_Ratio'] = st.number_input("Avg_Utilization_Ratio", min_value=0.0, max_value=1.0)

    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])

else:  # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

# Features to drop
features_to_drop = st.multiselect("Select features to drop (optional)", 
                                    options=[
                                        'CLIENTNUM',
                                        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                                        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
                                    ])

# Preprocess data and predict
if st.button("Predict"):
    try:
        X = preprocess_data(input_df, features_to_drop)
        
        # Make predictions using the selected model
        selected_model = models[model_name]
        predictions = selected_model.predict(X)

        # Show predictions
        st.write("Predictions (Attrition_Flag):")
        st.write(predictions)
        
    except Exception as e:
        st.error(f"Error in processing: {e}")
