import streamlit as st
import pandas as pd
import pickle

#Loading the Model
with open('financial_inclusion_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define label encoders (these should match what you used during training)
label_encoders = {
    'location_type': {'Rural': 0, 'Urban': 1},
    'cellphone_access': {'No': 0, 'Yes': 1},
    'gender_of_respondent': {'Female': 0, 'Male': 1},
    'relationship_with_head': {
        'Head of Household': 0,
        'Spouse': 1,
        'Child': 2,
        'Parent': 3,
        'Other relative': 4,
        'Other non-relatives': 5
    },
    'marital_status': {
        'Married/Living together': 0,
        'Single/Never Married': 1,
        'Widowed': 2,
        'Divorced/Seperated': 3
    },
    'education_level': {
        'No formal education': 0,
        'Primary education': 1,
        'Secondary education': 2,
        'Vocational/Specialised training': 3,
        'Tertiary education': 4,
        'Other/Dont know/RTA': 5
    },
    'job_type': {
        'Self employed': 0,
        'Government Dependent': 1,
        'Formally employed Private': 2,
        'Informally employed': 3,
        'Formally employed Government': 4,
        'Farming and Fishing': 5,
        'Remittance Dependent': 6,
        'Other Income': 7,
        'Dont Know/Refuse to answer': 8
    }
}

st.title("FINANCIAL INCLUSION MODEL")

#  Creating the Input Form

st.header("User Information")
location_type = st.selectbox("Location Type", ['Rural', 'Urban'])
cellphone_access = st.selectbox("Cellphone Access", ['No', 'Yes'])
gender = st.selectbox("Gender", ['Female', 'Male'])
relationship = st.selectbox("Relationship with Head", [
    'Head of Household', 'Spouse', 'Child', 'Parent', 'Other relative', 'Other non-relatives'
])
marital_status = st.selectbox("Marital Status", [
    'Married/Living together', 'Single/Never Married', 'Widowed', 'Divorced/Seperated'
])
education = st.selectbox("Education Level", [
    'No formal education', 'Primary education', 'Secondary education',
    'Vocational/Specialised training', 'Tertiary education', 'Other/Dont know/RTA'
])
job = st.selectbox("Job Type", [
    'Self employed', 'Government Dependent', 'Formally employed Private',
    'Informally employed', 'Formally employed Government', 'Farming and Fishing',
    'Remittance Dependent', 'Other Income', 'Dont Know/Refuse to answer'
])
household_size = st.number_input("Household Size", min_value=1, max_value=20, value=3)
age = st.number_input("Age", min_value=16, max_value=100, value=30)

if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'location_type': [label_encoders['location_type'][location_type]],
        'cellphone_access': [label_encoders['cellphone_access'][cellphone_access]],
        'household_size': [household_size],
        'age_of_respondent': [age],
        'gender_of_respondent': [label_encoders['gender_of_respondent'][gender]],
        'relationship_with_head': [label_encoders['relationship_with_head'][relationship]],
        'marital_status': [label_encoders['marital_status'][marital_status]],
        'education_level': [label_encoders['education_level'][education]],
        'job_type': [label_encoders['job_type'][job]]
    })

    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Display result
    if prediction[0] == 1:
        st.success(f"THIS PERSON IS LIKELY TO HAVE A BANK ACCOUNT (AND THE CONFIDENCE LEVEL IS: {probability[0][1] * 100:.2f}%)")
    else:
        st.warning(f"THIS PERSON IS LIKELY TO HAVE A BANK ACCOUNT (AND THE CONFIDENCE LEVEL IS: {probability[0][0] * 100:.2f}%)")
