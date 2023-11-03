import streamlit as st
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load the KNN model
loaded_model = joblib.load('KNNTest.pkl')  # Replace 'KNNTest.pkl' with your KNN model file

# Load your DataFrame
df = pd.read_csv('unique_patterns.csv')  # Replace 'unique_patterns.csv' with the actual filename or path

# Create a Streamlit web app
st.title('KNN Model Deployment')
st.write('Enter the features for prediction:')

# Input fields for features with dropdown boxes
abdominal_pain = st.selectbox('Abdominal Pain', ['Yes', 'No'])
nausea = st.selectbox('Nausea', ['Yes', 'No'])

if st.button('Predict'):
    # Map the user's selected values to numeric values
    abdominal_pain_mapping = {'Yes': 1, 'No': 0}
    nausea_mapping = {'Yes': 1, 'No': 0}

    # Create a feature vector from user input
    user_input = [abdominal_pain_mapping.get(abdominal_pain, 0), nausea_mapping.get(nausea, 0)]

    try:
        # Make predictions using the loaded KNN model
        prediction = loaded_model.predict([user_input])[0]
        st.write(f'Predicted Class: {prediction}')
    except Exception as e:
        st.write(f'Error: {e}')
