import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

# Load your dataset
df = pd.read_csv('combined_disease_df_final.csv')

# Encode the target variable 'disease' using LabelEncoder
le = LabelEncoder()
df['disease_code'] = le.fit_transform(df['disease'])

# Separate features and target variable
X = df.drop(columns=['disease', 'count of disease occurrence', 'disease_code'])
y = df['disease_code']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Create a Streamlit app
st.title("Symptom-Based Disease Probability Predictor")

# Create a multiselect dropdown for selecting symptoms
selected_symptoms = st.multiselect("Select Symptoms", X.columns)

# Sort selected symptoms in ascending order
selected_symptoms.sort()

# Create a Predict button
if st.button("Predict"):
    # Create a new instance with selected symptoms
    new_instance = pd.DataFrame(0, index=[0], columns=X.columns)
    new_instance[selected_symptoms] = 1

    # Make predictions using predict_proba for the new instance
    proba = clf.predict_proba(new_instance)

    # Extract the predicted probabilities for each disease
    disease_probabilities = pd.DataFrame({
        'Disease': le.inverse_transform(clf.classes_),
        'Probability': proba[0] * 100  # Convert probabilities to percentages
    })

    # Display the top 5 probabilities in a bar chart
    top5_diseases = disease_probabilities.nlargest(5, 'Probability')

    # Streamlit app
    st.title('Top 10 Disease Probabilities')

    # Create a bar chart
    fig, ax = plt.subplots()
    bars = ax.bar(top5_diseases['Disease'], top5_diseases['Probability'])
    ax.set_xlabel('Predicted Diseases')
    ax.set_ylabel('Probability (%)')

    # Rotate x-axis labels at a 45-degree angle and set font size to 10
    ax.set_xticklabels(top5_diseases['Disease'], rotation=90, fontsize=10)

    # Display the numbers on top of each bar as whole percentages
    for bar in bars:
        yval = bar.get_height()
        percentage_label = f'{int(yval)}%'
        plt.text(bar.get_x() + bar.get_width()/2, yval, percentage_label, ha='center', va='bottom', fontsize=10)

    # Display the chart using Streamlit's st.pyplot
    st.pyplot(fig)
