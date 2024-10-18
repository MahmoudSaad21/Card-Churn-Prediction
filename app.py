import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the models
models = {
    "Gradient Boosting": joblib.load('models/Gradient Boosting.pkl'),
    "Random Forest": joblib.load('models/Random Forest.pkl'),
    "XGBoost": joblib.load('models/XGBoost.pkl'),
    "Logistic Regression": joblib.load('models/Logistic Regression.pkl'),
    "Decision Tree": joblib.load('models/Decision Tree Classifier.pkl'),
    "Naive Bayes": joblib.load('models/Naive Bayes.pkl'),
    "LightGBM": joblib.load('models/LightGBM.pkl'),
    "K-Nearest Neighbors": joblib.load('models/K-Nearest Neighbors.pkl'),
    "AdaBoost": joblib.load('models/AdaBoost.pkl'),
}

# List of columns used during model training
expected_columns = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
                    'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book',
                    'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
                    'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
                    'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

# Function to preprocess data
def preprocess_data(data):
    # Specify features to drop
    scaler = joblib.load('scaler/scaler.pkl')
    le_card = joblib.load('encoder/Card_Category_encoder.pkl')
    le_edu = joblib.load('encoder/Education_Level_encoder.pkl')
    le_gender = joblib.load('encoder/Gender_encoder.pkl')
    le_income = joblib.load('encoder/Income_Category_encoder.pkl')
    le_marital = joblib.load('encoder/Marital_Status_encoder.pkl')

    drop_features = ['CLIENTNUM',
                     'Attrition_Flag',
                     'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                     'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
                     ]
    
    # Drop specified features if they exist
    data.drop(columns=[col for col in drop_features if col in data.columns], inplace=True)

    # Encoding categorical variables
    data['Card_Category'] = le_card.transform(data['Card_Category'])
    data['Education_Level'] = le_edu.transform(data['Education_Level'])
    data['Gender'] = le_gender.transform(data['Gender'])
    data['Income_Category'] = le_income.transform(data['Income_Category'])
    data['Marital_Status'] = le_marital.transform(data['Marital_Status'])
    # Scaling features
    columns_to_scale = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                        'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
                        'Total_Revolving_Bal', 'Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
                        'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

    data[columns_to_scale] = scaler.transform(data[columns_to_scale])
    # Ensure the columns are in the expected order
    data = data[expected_columns]
    return data

# Streamlit app layout
st.title("Customer Attrition Prediction")

st.markdown(""" 
This application predicts customer attrition using multiple models.
You can either enter customer data manually or upload a CSV file.
Choose a model before making predictions.
""")

# Model selection
model_name = st.selectbox("Select Model", options=list(models.keys()))

# Data entry method
data_input_method = st.radio("Data Input Method", ("Manual Entry", "Upload CSV"))

# Manual data entry
if data_input_method == "Manual Entry":
    data = {}

    # Input fields for all the necessary customer attributes with descriptions
    data['Customer_Age'] = st.number_input("Customer_Age", min_value=0.0, step=1.0, format="%.4f", 
                                          help="The age of the customer in years.")
    data['Gender'] = st.selectbox("Gender", ["M", "F"], 
                                  help="The gender of the customer. M for male, F for female.")
    data['Dependent_count'] = st.number_input("Dependent_count", min_value=0.0, step=1.0, format="%.4f",
                                              help="The number of dependents the customer has.")
    data['Education_Level'] = st.selectbox("Education_Level", 
                                       ["College", "Doctorate", "Graduate", "High School", "Post-Graduate", "Uneducated", "Unknown"], 
                                       help="The highest level of education attained by the customer.")
    data['Marital_Status'] = st.selectbox("Marital_Status", 
                                      ["Divorced", "Married", "Single", "Unknown"], 
                                      help="The marital status of the customer.")
    data['Income_Category'] = st.selectbox("Income_Category", 
                                       ["$120K +", "$40K - $60K", "$60K - $80K", "$80K - $120K", "Less than $40K", "Unknown"], 
                                       help="The annual income category of the customer.")
    data['Card_Category'] = st.selectbox("Card_Category", 
                                     ["Blue", "Gold", "Platinum", "Silver"], 
                                     help="The type of card the customer holds.")
    data['Months_on_book'] = st.number_input("Months_on_book", min_value=0.0, step=1.0, format="%.4f", 
                                            help="The total number of months the customer has been with the bank.")
    data['Credit_Limit'] = st.number_input("Credit_Limit", min_value=0.0, step=0.0001, format="%.4f", 
                                          help="The credit limit of the customer.")
    data['Total_Relationship_Count'] = st.number_input("Total_Relationship_Count", min_value=0.0, step=1.0, format="%.4f", 
                                                      help="The total number of products the customer has with the bank.")
    data['Months_Inactive_12_mon'] = st.number_input("Months_Inactive_12_mon", min_value=0.0, step=1.0, format="%.4f", 
                                                    help="The number of months the customer was inactive in the last 12 months.")
    data['Contacts_Count_12_mon'] = st.number_input("Contacts_Count_12_mon", min_value=0.0, step=1.0, format="%.4f", 
                                                    help="The number of contacts made by the customer in the last 12 months.")
    data['Total_Revolving_Bal'] = st.number_input("Total_Revolving_Bal", min_value=0.0, step=0.0001, format="%.4f", 
                                                  help="The total revolving balance on the customer's account.")
    data['Avg_Open_To_Buy'] = st.number_input("Avg_Open_To_Buy", min_value=0.0, step=0.0001, format="%.4f", 
                                              help="The average amount the customer is willing to spend on")
    data['Total_Amt_Chng_Q4_Q1'] = st.number_input("Total_Amt_Chng_Q4_Q1", min_value=0.0, step=0.0001, format="%.4f", 
                                                  help="The change in transaction amount from Q4 to Q1.")
    data['Total_Trans_Amt'] = st.number_input("Total_Trans_Amt", min_value=0.0, step=0.0001, format="%.4f", 
                                              help="The total transaction amount.")
    data['Total_Trans_Ct'] = st.number_input("Total_Trans_Ct", min_value=0.0, step=1.0, format="%.4f", 
                                            help="The total number of transactions.")
    data['Total_Ct_Chng_Q4_Q1'] = st.number_input("Total_Ct_Chng_Q4_Q1", min_value=0.0, step=0.0001, format="%.4f", 
                                                  help="The change in transaction count from Q4 to Q1.")
    data['Avg_Utilization_Ratio'] = st.number_input("Avg_Utilization_Ratio", min_value=0.0, max_value=1.0, step=0.0001, format="%.4f", 
                                                    help="The average utilization ratio of the credit line.")
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])

else:  # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

# Preprocess data and predict
if st.button("Predict"):
    try:
        data = preprocess_data(input_df)

        # Make predictions using the selected model
        selected_model = models[model_name]
        predictions = selected_model.predict(data)
        # Show predictions
        st.write("Predictions (Attrition_Flag):")
        st.write(predictions)

    except Exception as e:
        st.error(f"Error in processing: {e}")
