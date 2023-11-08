import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the trained classifier from a joblib file
clf = joblib.load('rf_model.sav')

# Load your data (e.g., final_df) here
df = pd.read_csv('final_df.csv')

# Assuming your X-axis features are stored in 'X' and the disease column is 'disease'

# Create a multiselect widget for selecting features
selected_features = st.multiselect('Select Features',df.columns)

if st.button('Predict Probabilities'):
    if not selected_features:
        st.warning('Please select at least one feature.')
    else:
        # Filter the dataset to include only selected features
        X_selected = df[selected_features]

        # Use the trained model to predict probabilities
        disease_probabilities = clf.predict_proba(X_selected)

        # Display the disease probabilities for each row
        st.write("Disease Probabilities:")
        disease_probabilities_df = pd.DataFrame(disease_probabilities, columns=clf.classes_)
        st.dataframe(disease_probabilities_df)

# You can add more Streamlit UI elements for a better user experience
