import streamlit as st
import pandas as pd

# Load your data
data = pd.read_csv('unique_patterns.csv')


st.title("Disease Prediction App")

# Get the list of unique symptoms (columns starting from the second column)
unique_symptoms = data.columns[1:]

# Create a multiselect widget to select multiple symptoms
selected_symptoms = st.multiselect("Select Symptoms:", unique_symptoms)

# Create a dictionary to store diseases associated with each symptom
diseases_associated_with_symptoms = {}

# Check if the selected symptoms rule out any diseases and keep track of associated diseases
for symptom in selected_symptoms:
    diseases_associated_with_symptoms[symptom] = set(data[data[symptom] == 1]['Disease'].unique())

if diseases_associated_with_symptoms:

    # Find mutual diseases associated with all selected symptoms
    mutual_diseases = set.intersection(*map(set, diseases_associated_with_symptoms.values()))

    # Display the mutual diseases
    st.write("From the selected symptoms, we think you may have these:")
    for disease in mutual_diseases:
        st.write(disease)
else:
    
    st.write("Please enter in your symptom(s). Pretty Please.")
    
