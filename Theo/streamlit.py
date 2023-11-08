import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the saved Random Forest classifier
loaded_model = joblib.load('rf_model.sav')

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
st.markdown("To develop a machine learning model that accurately predicts the disease based on the reported symptoms.")

# Introduction
st.markdown("## Background")
st.markdown("Diagnosis of diseases often requires expertise and thorough medical examination. However, for common ailments or as a preliminary diagnostic tool, we aim to utilize symptom data to predict potential diseases. This tool can assist healthcare professionals as a reference, guide patients in understanding their conditions, or even help in telemedicine where immediate physical diagnosis isn't feasible.")

# About
st.markdown("## About")
st.markdown("### Project Overview")
st.markdown("A visionary project aimed to harness symptom data for predictive analysis, aimed at supporting healthcare professionals and patients with early diagnosis.")
st.markdown("Utilization of a rich dataset covering a multitude of diseases and corresponding symptoms, underscoring the diverse and complex nature of disease symptomatology.")

st.markdown("### Expected Outcomes") 
st.markdown("A robust model that can predict the disease based on the given symptoms.")
st.markdown("An understanding of the relationship between symptoms and diseases – which symptoms are strong indicators of specific diseases.")
st.markdown("A baseline comparison to understand the efficacy of our model.")

st.markdown("### Constraints")
st.markdown("The model is not a replacement for professional medical advice. It's a supplementary tool to assist in understanding symptoms.")
st.markdown("The accuracy of the model is contingent on the quality and comprehensiveness of the data.")


# Data Science Pipeline
st.markdown("## Data Science Pipeline")
st.markdown("1. **Data Acquisition:**")
st.markdown("   - The data encompasses a comprehensive collection of symptoms and their related diseases through webscraping from WHO and Columbia University, forming a robust foundation for the predictive model.")
st.markdown("2. **Data Cleaning:**")
st.markdown("   - Rigorous preprocessing steps were taken to ensure quality and consistency, involving cleaning, encoding, and transforming data for optimal model input.")
st.markdown("3. **Data Exploration:**")
st.markdown("   - Insightful visualizations were crafted, like the frequency distribution of symptoms and disease co-occurrence, highlighting key patterns and relationships.")
st.markdown("4. **Feature Engineering:**")
st.markdown("   - The process included creating new features to enhance the model's ability to learn complex patterns from the data.")
st.markdown("5. **Modeling:**")
st.markdown("   - A meticulous evaluation of various models was performed, with details provided on training and validation accuracies:")
st.markdown("     - Baseline Model Accuracy: 0.0208")
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









def encode_disease(index):
    disease_names = ["Alzheimer's disease", 'Pneumocystis carinii pneumonia', 'accident cerebrovascular', 'hiv infections', 'adenocarcinoma', 'adhesion', 'affect labile', 'anemia', 'anxiety state', 'aphasia', 'arthritis', 'asthma', 'bacteremia', 'benign prostatic hypertrophy', 'bipolar disorder', 'bronchitis', 'oral candidiasis', 'carcinoma', 'cardiomyopathy', 'cellulitis', 'cholecystitis', 'biliary calculus', 'chronic alcoholic intoxication', 'chronic kidney failure', 'chronic obstructive airway disease', 'cirrhosis', 'colitis', 'confusion', 'coronary heart disease', 'decubitus ulcer', 'deep vein thrombosis', 'degenerative polyarthritis', 'deglutition disorder', 'dehydration', 'delirium', 'delusion', 'dementia', 'dependence', 'depressive disorder', 'diabetes', 'diverticulitis', 'diverticulosis', 'edema pulmonary', 'pericardial effusion body substance', 'embolism pulmonary', 'emphysema pulmonary', 'encephalopathy', 'endocarditis', 'epilepsy', 'exanthema', 'failure heart', 'failure heart congestive', 'failure kidney', 'fibroid tumor', 'gastritis', 'gastroenteritis', 'gastroesophageal reflux disease', 'glaucoma', 'gout', 'hemiparesis', 'hemorrhoids', 'hepatitis', 'hepatitis B', 'hepatitis C', 'hernia', 'hernia hiatal', 'hyperbilirubinemia', 'hypercholesterolemia', 'hyperglycemia', 'hyperlipidemia', 'hypertension pulmonary', 'hypertensive disease', 'hypoglycemia', 'hypothyroidism', 'ileus', 'incontinence', 'infection', 'infection urinary tract', 'influenza', 'insufficiency renal', 'ischemia', 'ketoacidosis diabetic', 'kidney disease', 'kidney failure acute', 'lymphatic diseases', 'lymphoma', 'carcinoma breast', 'carcinoma of lung', 'carcinoma prostate', 'malignant neoplasms', 'primary malignant neoplasm', 'carcinoma colon', 'manic disorder', 'melanoma', 'migraine disorders', 'mitral valve insufficiency', 'myocardial infarction', 'neoplasm', 'neoplasm metastasis', 'neuropathy', 'neutropenia', 'obesity', 'obesity morbid', 'osteomyelitis', 'osteoporosis', 'overload fluid', 'pancreatitis', 'pancytopenia', 'paranoia', 'parkinson disease', 'paroxysmal dyspnea', 'peripheral vascular disease', 'personality disorder', 'pneumonia', 'pneumonia aspiration', 'pneumothorax', 'primary carcinoma of the liver cells', 'psychotic disorder', 'pyelonephritis', 'respiratory failure', 'schizophrenia', 'sepsis (invertebrate)', 'sickle cell anemia', 'spasm bronchial', 'stenosis aortic valve', 'suicide attempt', 'tachycardia sinus', 'thrombocytopaenia', 'thrombus', 'tonic-clonic seizures', 'transient ischemic attack', 'tricuspid valve insufficiency', 'ulcer peptic', 'upper respiratory infection', 'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']
    if 0 <= index < len(disease_names):
        return disease_names[index]
    else:
        return None  # Return None for out-of-range indices


# Streamlit app
st.title('Disease Prediction App')

# Create input fields for feature values
feature_names = [
    "spotting_urination", "achalasia", "ambidexterity", "ascites", "asthenia", 
    "behavior_showing_increased_motor_activity", "breakthrough_pain", "chills", "clonus", "cough", 
    "cyanosis", "diarrhoea", "drool", "drowsiness", "drowsiness^sleepy", "dysarthria", "breathlessness", 
    "dyspnea_on_exertion", "ecchymosis", "energy_increased", "erythema", "extreme_exhaustion", "fall", 
    "fecaluria", "feeling_suicidal", "foul_smell_ofurine", "high_fever", "flatulence", "bleeding", 
    "hallucinations_auditory", "hematuria", "heme_positive", "hemiplegia", "hot_flush", "hyperkalemia", 
    "hypokinesia", "hypoproteinemia", "inappropriate_affect", "intoxication", "left_atrial_hypertrophy", 
    "lesion", "loose_associations", "mass_in_breast", "mass_of_body_structure", "mediastinal_shift", 
    "mental_status_changes", "moan", "nightmare", "orthopnea", "out_of_breath", "pain", "abdominal_pain", 
    "pain_chest", "pain,__chest", "para_2", "paraparesis", "paresthesia", "pericardial_friction_rub", 
    "pleuritic_pain", "polyuria", "prostatism", "pulsus_paradoxus", "rale", "seizure", "dyspnea", 
    "breathlessness", "sore_to_touch", "speech_slurred", "suicidal", "swelling", "dischromic_patches", 
    "thicken", "transaminitis", "tremor", "uncoordination", "unresponsiveness", "unsteady_gait", 
    "urgency_of_micturition", "nausea_and_vomiting", "wheezing", "anxiety", "yellow_sputum", 
    "abnormal_menstruation", "acidity", "acute_liver_failure", "altered_sensorium", "back_pain", 
    "belly_pain", "blackheads", "bladder_discomfort", "blister", "blood_in_sputum", "bloody_stool", 
    "blurred_and_distorted_vision", "brittle_nails", "bruising", "burning_micturition", "chest_pain", 
    "cold_hands_and_feets", "coma", "congestion", "constipation", "continuous_feel_of_urine", 
    "continuous_sneezing", "cramps", "dark_urine", "dehydration", "depression", "dischromic _patches", 
    "distention_of_abdomen", "dizziness", "drying_and_tingling_lips", "enlarged_thyroid", 
    "excessive_hunger", "extra_marital_contacts", "family_history", "fast_heart_rate", "fatigue", 
    "fluid_overload", "foul_smell_of urine", "headache", "hip_joint_pain", "history_of_alcohol_consumption", 
    "increased_appetite", "indigestion", "inflammatory_nails", "internal_itching", "irregular_sugar_level", 
    "irritability", "irritation_in_anus", "joint_pain", "knee_pain", "lack_of_concentration", "lethargy", 
    "loss_of_appetite", "loss_of_balance", "loss_of_smell", "malaise", "mild_fever", "mood_swings", 
    "movement_stiffness", "mucoid_sputum", "muscle_pain", "muscle_wasting", "muscle_weakness", 
    "nausea", "neck_pain", "nodal_skin_eruptions", "obesity", "pain_behind_the_eyes", 
    "pain_during_bowel_movements", "pain_in_anal_region", "painful_walking", "palpitations", 
    "passage_of_gases", "patches_in_throat", "phlegm", "prominent_veins_on_calf", "puffy_face_and_eyes", 
    "pus_filled_pimples", "receiving_blood_transfusion", "receiving_unsterile_injections", 
    "red_sore_around_nose", "red_spots_over_body", "redness_of_eyes", "restlessness", "runny_nose", 
    "rusty_sputum", "scurring", "shivering", "silver_like_dusting", "sinus_pressure", "skin_peeling", 
    "skin_rash", "slurred_speech", "small_dents_in_nails", "spinning_movements", "spotting_ urination", 
    "stiff_neck", "stomach_bleeding", "stomach_pain", "sunken_eyes", "sweating", "swelled_lymph_nodes", 
    "swelling_joints", "swelling_of_stomach", "swollen_blood_vessels", "swollen_extremeties", "swollen_legs", 
    "throat_irritation", "toxic_look_(typhos)", "ulcers_on_tongue", "unsteadiness", 
    "visual_disturbances", "vomiting", "watering_from_eyes", "weakness_in_limbs", "weakness_of_one_body_side", 
    "weight_gain", "weight_loss", "yellow_crust_ooze", "yellow_urine", "yellowing_of_eyes", "yellowish_skin", 
    "itching", "pneumonoultramicroscopicsilicovolcanoconiosis", "other"
]


# Create a dropdown multiselect widget for feature values with an empty list as the default
selected_features = st.multiselect(
    'Select Features',
    feature_names,
    default=[]
)

# Define a function to make probability predictions
def predict_disease_probs(features):
    prediction_probs = loaded_model.predict_proba([features])
    return prediction_probs


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
        sorted_results = filtered_results.sort_values(by='Probability', ascending=False).head(5)

        # Create a red-themed bar chart with transparency and percentages on top of each bar
        fig, ax = plt.subplots()
        bars = ax.bar(sorted_results['Disease'], sorted_results['Probability'], alpha=0.5, color='red')
        ax.set_xticklabels(sorted_results['Disease'], rotation=45, fontsize=10, color='black')
        plt.xlabel('Disease', color='white')
        plt.ylabel('Probability (%)', color='white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), ha='center', va='bottom', fontsize=15, color='white')



        # Set the background color of the plot to black
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        st.pyplot(fig)
        
        

# Conclusion and Recommendations
conclusion_and_recommendations = """
### Conclusion

The Disease Symptoms Prediction Model project aimed to leverage symptom data to predict potential diseases, assisting healthcare professionals and patients in preliminary diagnostics. This objective was approached through the development of machine learning models, utilizing a dataset encompassing a variety of diseases and their associated symptoms.

**Achievement of Goals:**
- We successfully developed a machine learning model, with the Random Forest classifier emerging as the most accurate, significantly outperforming the baseline model.
- The relationship between symptoms and diseases was elucidated through statistical analysis, confirming that certain symptoms such as abdominal pain and vomiting are strong indicators of specific conditions like Alcoholic Hepatitis and Chronic Cholestasis, respectively.
- A baseline model was established, providing a reference point for evaluating the effectiveness of more sophisticated predictive models.

**Key Findings:**
- Statistical significance was identified between certain symptoms and diseases, validating the model's capability to capture these relationships.
- Symptom frequency analysis and N-gram visualizations provided deeper insights into common and distinctive symptom patterns.
- The Random Forest model, with a test accuracy of 89.58%, was identified as the most promising predictive model in our trials.

**Recommendations:**
- Due to its high validation and test accuracies, the Random Forest model is recommended for initial deployment in a controlled environment to gauge real-world efficacy.
- Collaboration with medical professionals is advised to interpret the model's predictions and to incorporate their feedback for refinement.

**Next Steps:**
- **Integration into User-Friendly Platforms:** The next phase involves creating a user interface for the model, making it accessible to end-users who can report symptoms and receive disease predictions.
- **Dataset Expansion:** To improve the model's comprehensiveness and accuracy, we plan to include a broader range of diseases, especially rare conditions, to enhance predictive capabilities.
- **Continuous Model Improvement:** We aim to continuously refine the model by incorporating medical professional feedback and adjusting it according to the latest medical research and data.

**"If I Had More Time, I Would...":**
- **Explore Advanced Models:** Experiment with deep learning and ensemble methods to potentially uncover complex patterns in symptom-disease relationships that simpler models might miss.
- **Conduct a Thorough Hyperparameter Tuning:** Allocate more time to fine-tune the models, especially KNN, to ensure that we are not overlooking a potentially suitable model due to suboptimal parameters.
- **Implement a Feedback Loop:** Develop a system to collect user and professional feedback on the model’s predictions to facilitate ongoing learning and improvement.
- **Focus on Interpretability:** Devote efforts to make the model's decision process more transparent, aiding healthcare professionals in understanding the rationale behind predictions, which is crucial for medical applications.

By adhering to these next steps and considerations, the project can make substantial progress towards its goal of becoming a reliable, assistive tool in the diagnostic process.
"""

# Display the conclusion and recommendations
st.markdown(conclusion_and_recommendations)


# Footer
st.markdown("## Footer")
st.markdown("### Contact Information")
st.markdown("Channels for reaching out for potential partnerships, contributions, or inquiries into the model and its applications.")
st.markdown("### Legal Disclaimer")
st.markdown("A statement clarifying that the predictions made by the model are not a substitute for professional medical diagnosis or treatment.")
st.markdown("### Credits")
st.markdown("Acknowledgements to the data scientists, healthcare experts, and instructors at CodeUp LLC who contributed to the success of this project.")


# List of LinkedIn profiles
linkedin_profiles = [
    "https://www.linkedin.com/in/thoai-hung-nguyen/",
    "https://www.linkedin.com/in/theodore-quansah/",
    "https://www.linkedin.com/in/nishabista/",
    "https://www.linkedin.com/in/marcelino-salazar/"
]

# Display LinkedIn profiles as clickable links
st.markdown("### LinkedIn Profiles")
for profile in linkedin_profiles:
    st.markdown(f"[{profile}]({profile})")

