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
             background-image: url("https://t4.ftcdn.net/jpg/02/95/44/01/360_F_295440125_BZ9c8sWssOHp13AbB1rx9vGkQalEy1nz.jpg");
             background-size: 100% 100%;
             background-attachment: fixed;  /* Ensures the background doesn't scroll */
         }}
         </style>
         """,
         unsafe_allow_html=True
    )

add_bg_from_url()






import streamlit as st

# Banner
st.markdown("# Revolutionizing Disease Prediction with Advanced Data Science")
st.markdown("Bridging the gap between data-driven insights and healthcare expertise for better diagnostic support.")

# Introduction
st.markdown("## Introduction")
st.markdown("A quick dive into how we're pushing the frontiers of preliminary diagnostics with a data-informed approach to identifying potential diseases based on symptoms.")

# About
st.markdown("## About")
st.markdown("### Project Overview")
st.markdown("A visionary project aimed to harness symptom data for predictive analysis, aimed at supporting healthcare professionals and patients with early diagnosis.")
st.markdown("Utilization of a rich dataset covering a multitude of diseases and corresponding symptoms, underscoring the diverse and complex nature of disease symptomatology.")

# Data Science Pipeline
st.markdown("## Data Science Pipeline")
st.markdown("1. **Data Acquisition:**")
st.markdown("   - The data encompasses a comprehensive collection of symptoms and their related diseases, forming a robust foundation for the predictive model.")
st.markdown("2. **Data Cleaning:**")
st.markdown("   - Rigorous preprocessing steps were taken to ensure quality and consistency, involving cleaning, encoding, and transforming data for optimal model input.")
st.markdown("3. **Data Exploration:**")
st.markdown("   - Insightful visualizations were crafted, like the frequency distribution of symptoms and disease co-occurrence, highlighting key patterns and relationships.")
st.markdown("4. **Feature Engineering:**")
st.markdown("   - The process included creating new features to enhance the model's ability to learn complex patterns from the data.")
st.markdown("5. **Modeling:**")
st.markdown("   - A meticulous evaluation of various models was performed, with details provided on training and validation accuracies:")
st.markdown("     - Baseline Model for reference.")
st.markdown("     - Random Forest with a test accuracy of 89.58%.")
st.markdown("     - Logistic Regression with notable validation accuracy.")
st.markdown("     - KNN providing a satisfactory baseline but with room for improvement.")
st.markdown("6. **Model Evaluation:**")
st.markdown("   - Metrics such as accuracy, precision, and recall were adopted for a comprehensive performance assessment.")
st.markdown("   - Detailed visualizations from the notebook, such as confusion matrices, will illustrate the models' performance clearly.")

# Results
st.markdown("## Results")
st.markdown("### Model Findings")
st.markdown("Emphasis on the Random Forest model which emerged as the most promising, delivering an impressive test accuracy and demonstrating robust generalization capabilities.")
st.markdown("### Statistical Analysis")
st.markdown("The model's ability to discern distinct patterns in symptom-disease relationships was validated through rigorous statistical analysis.")

# Interactive Prediction
st.markdown("## Interactive Prediction")
st.markdown("### Symptom Selector")
st.markdown("A feature enabling users to select their current symptoms from a comprehensive list and receive an instant predictive analysis.")
st.markdown("### Model Prediction")
st.markdown("A detailed explanation of how the Random Forest model utilizes the input symptoms to predict possible diseases, with a note on the importance of professional medical consultation for accurate diagnosis.")








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

        # Sort the results by probability in descending order
        sorted_results = filtered_results.sort_values(by='Probability', ascending=False)

        # Create a red-themed bar chart with transparency and percentages on top of each bar
        fig, ax = plt.subplots()
        bars = ax.bar(sorted_results['Disease'], sorted_results['Probability'], alpha=0.5, color='red')
        ax.set_xticklabels(sorted_results['Disease'], rotation=90, fontsize=10, color='black')
        plt.xlabel('Disease', color='white')
        plt.ylabel('Probability (%)', color='white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), ha='center', va='bottom', fontsize=5, color='white')



        # Set the background color of the plot to black
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        st.pyplot(fig)
        
        
# Conclusion and Future Work
st.markdown("## Conclusion and Future Work")
st.markdown("### Summary")
st.markdown("A concise overview of the project's success, highlighting the Random Forest model's high accuracy and the potential of machine learning in healthcare.")
st.markdown("### Recommendations")
st.markdown("Advocating for the model's usage in a controlled clinical environment to complement medical expertise, with a call for continued collaboration for refinement and validation.")
st.markdown("### Next Steps")
st.markdown("Planned enhancements such as user interface improvements for the symptom selector and dataset expansion to encompass rarer conditions and ongoing model optimization based on user feedback and medical advancements.")

# Footer
st.markdown("## Footer")
st.markdown("### Contact Information")
st.markdown("Channels for reaching out for potential partnerships, contributions, or inquiries into the model and its applications.")
st.markdown("### Legal Disclaimer")
st.markdown("A statement clarifying that the predictions made by the model are not a substitute for professional medical diagnosis or treatment.")
st.markdown("### Credits")
st.markdown("Acknowledgements to the data scientists, healthcare experts, and all who contributed to the success of this project.")
