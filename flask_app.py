from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
clf = joblib.load('random_forest_model.pkl')

# Load the symptom weights from the CSV
df1 = pd.read_csv('Symptom-severity.csv')
weights_dict = df1.set_index('Symptom')['weight'].to_dict()

df = pd.read_csv('dataset.csv')

@app.route('/predict', methods=['POST'])
def predict():
    # Get symptoms from the request
    data = request.json
    symptoms = data['symptoms']
    
    # Debug: Print the received symptoms
    print(f"Received symptoms: {symptoms}")

    # Create a dataframe with the symptoms
    symptoms_df = pd.DataFrame([symptoms], columns=df.columns[1:])
    
    # Debug: Print the symptoms DataFrame
    print(f"Symptoms DataFrame:\n{symptoms_df}")
    
    # One-hot encode the symptoms
    encoded_list = [pd.get_dummies(symptoms_df[col], prefix="", prefix_sep="") for col in symptoms_df.columns]
    encoded_df = pd.concat(encoded_list, axis=1)
    
    # Debug: Print the one-hot encoded DataFrame
    print(f"One-hot encoded DataFrame:\n{encoded_df}")
    
    # Replace 1s in the encoded dataframe with the respective weights
    for symptom, weight in weights_dict.items():
        if symptom in encoded_df.columns:
            encoded_df[symptom] = encoded_df[symptom].replace(1, weight)
    
    # Debug: Print the weighted encoded DataFrame
    print(f"Weighted encoded DataFrame:\n{encoded_df}")
    
    # Make a prediction
    prediction = clf.predict(encoded_df)
    
    # Debug: Print the prediction
    print(f"Prediction: {prediction[0]}")
    
    # Return the prediction
    return jsonify({'disease': prediction[0]})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
