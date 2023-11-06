import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the saved Random Forest classifier
loaded_model = joblib.load('rf_model.joblib')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://png.pngtree.com/background/20210709/original/pngtree-black-gold-border-minimalistic-background-picture-image_952105.jpg");
             background-size: 100% 100%;
             background-attachment: fixed;  /* Ensures the background doesn't scroll */
         }}
         </style>
         """,
         unsafe_allow_html=True
    )

add_bg_from_url()


# Define a function to make probability predictions
def predict_disease_probs(features):
    prediction_probs = loaded_model.predict_proba([features])
    return prediction_probs

# Define the function to encode disease names
def encode_disease(index):
    disease_names = [
        'Migraine', 'Hepatitis C', 'Impetigo', 'Pneumonia', 'Peptic ulcer disease',
        'Drug Reaction', 'Hypothyroidism', 'Gastroenteritis', 'Tuberculosis', 'Psoriasis',
        'Varicose veins', 'Chronic cholestasis', 'Hepatitis B', 'Hepatitis D', 'Typhoid',
        'Hypoglycemia', 'Alcoholic hepatitis', 'GERD', 'Paralysis (brain hemorrhage)', 'Arthritis',
        'Cervical spondylosis', 'Jaundice', 'Urinary tract infection', 'Bronchial Asthma', 'Osteoarthristis',
        'Malaria', 'Heart attack', 'Hepatitis E', 'Common Cold', 'Allergy', 'AIDS', 'Diabetes',
        'Chicken pox', 'Fungal infection', '(Vertigo) Paroxysmal Positional Vertigo', 'Hyperthyroidism',
        'Acne', 'Dengue', 'Hypertension', 'Dimorphic hemorrhoids (piles)', 'Hepatitis A'
    ]

    if 0 <= index < len(disease_names):
        return disease_names[index]
    else:
        return None  # Return None for out-of-range indices

# Streamlit app
st.title('Disease Prediction App')

# Create input fields for feature values
feature_names = [
    "abdominal_pain", "abnormal_menstruation", "acidity", "acute_liver_failure", "altered_sensorium", "anxiety", 
    "back_pain", "belly_pain", "blackheads", "bladder_discomfort", "blister", "blood_in_sputum", "bloody_stool", 
    "blurred_and_distorted_vision", "breathlessness", "brittle_nails", "bruising", "burning_micturition", 
    "chest_pain", "chills", "cold_hands_and_feets", "coma", "congestion", "constipation", "continuous_feel_of_urine", 
    "continuous_sneezing", "cough", "cramps", "dark_urine", "dehydration", "depression", "diarrhoea", 
    "foul_smell_ofurine", "distention_of_abdomen", "dizziness", "drying_and_tingling_lips", "enlarged_thyroid", 
    "excessive_hunger", "extra_marital_contacts", "family_history", "fast_heart_rate", "fatigue", "fluid_overload", 
    "spotting_urination", "headache", "high_fever", "hip_joint_pain", "history_of_alcohol_consumption", 
    "increased_appetite", "indigestion", "inflammatory_nails", "internal_itching", "irregular_sugar_level", 
    "irritability", "irritation_in_anus", "joint_pain", "knee_pain", "lack_of_concentration", "lethargy", 
    "loss_of_appetite", "loss_of_balance", "loss_of_smell", "malaise", "mild_fever", "mood_swings", 
    "movement_stiffness", "mucoid_sputum", "muscle_pain", "muscle_wasting", "muscle_weakness", "nausea", 
    "neck_pain", "nodal_skin_eruptions", "obesity", "pain_behind_the_eyes", "pain_during_bowel_movements", 
    "pain_in_anal_region", "painful_walking", "palpitations", "passage_of_gases", "patches_in_throat", "phlegm", 
    "polyuria", "prominent_veins_on_calf", "puffy_face_and_eyes", "pus_filled_pimples", 
    "receiving_blood_transfusion", "receiving_unsterile_injections", "red_sore_around_nose", 
    "red_spots_over_body", "redness_of_eyes", "restlessness", "runny_nose", "rusty_sputum", "scurring", 
    "shivering", "silver_like_dusting", "sinus_pressure", "skin_peeling", "skin_rash", "slurred_speech", 
    "small_dents_in_nails", "spinning_movements", "dischromic_patches", "stiff_neck", "stomach_bleeding", 
    "stomach_pain", "sunken_eyes", "sweating", "swelled_lymph_nodes", "swelling_joints", 
    "swelling_of_stomach", "swollen_blood_vessels", "swollen_extremeties", "swollen_legs", 
    "throat_irritation", "toxic_look_(typhos)", "ulcers_on_tongue", "unsteadiness", 
    "visual_disturbances", "vomiting", "watering_from_eyes", "weakness_in_limbs", "weakness_of_one_body_side", 
    "weight_gain", "weight_loss", "yellow_crust_ooze", "yellow_urine", "yellowing_of_eyes", "yellowish_skin", 
    "itching"
]

# Create a dropdown multiselect widget for feature values with an empty list as the default
selected_features = st.multiselect(
    'Select Features',
    feature_names,
    default=[]
)

# Create a dictionary to store feature values
feature_values = {feature_name: 1 if feature_name in selected_features else 0 for feature_name in feature_names}

# Make probability predictions
if st.button('Predict'):
    if not selected_features:
        st.warning('Please select at least one feature before predicting.')
    else:
        prediction_probs = predict_disease_probs(list(feature_values.values()))

        # Convert probabilities to percentages
        percentages = [prob * 100 for prob in prediction_probs[0]]

        # Create a DataFrame to store the results
        results_df = pd.DataFrame({
            'Disease': [encode_disease(i) for i in range(len(loaded_model.classes_))],
            'Probability': percentages
        })

        # Filter and display only diseases with probabilities > 0.0
        filtered_results = results_df[results_df['Probability'] > 0.0]

        # Create a red-themed bar chart with transparency and percentages on top of each bar
        fig, ax = plt.subplots()
        bars = ax.bar(filtered_results['Disease'], filtered_results['Probability'], alpha=0.5, color='red')
        ax.set_xticklabels(filtered_results['Disease'], rotation=90, fontsize=10, color='black')
        plt.xlabel('Disease', color='white')
        plt.ylabel('Probability (%)', color='white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), ha='center', va='bottom', fontsize=5, color='white')

        # Set the y-axis to show ticks from 0 to 100 with an increment of 10
        ax.set_yticks(range(0, 101, 10))

        # Set the background color of the plot to black
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        st.pyplot(fig)
