import streamlit as st
import pandas as pd
import joblib

# Load your trained K-Nearest Neighbors model
loaded_model = joblib.load('rf_model.sav')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
    )

add_bg_from_url()

# Create a Streamlit web app
st.title("Disease Prediction App")





symptoms = [
    "Abdominal pain",
    "Abnormal menstruation",
    "Abscess (bacterial)",
    "Acidity",
    "Achalasia",
    "Altered sensorium",
    "Anxiety",
    "Back pain",
    "Belly pain",
    "Behavior showing increased motor activity",
    "Blackheads",
    "Bladder discomfort",
    "Blister",
    "Blood in sputum",
    "Bloody stool",
    "Blurred and distorted vision",
    "Brittle nails",
    "Bruising",
    "Burning micturition",
    "Breathlessness",
    "Chest pain",
    "Chills",
    "Clonus",
    "Cold hands and feet",
    "Coma",
    "Congestion",
    "Constipation",
    "Continuous feel of urine",
    "Continuous sneezing",
    "Cough",
    "Cramps",
    "Cyanosis",
    "Dark urine",
    "Dehydration",
    "Depression",
    "Diarrhea",
    "Dischromic patches",
    "Distention of abdomen",
    "Dizziness",
    "Drying and tingling lips",
    "Dysarthria",
    "Dyspnea",
    "Dyspnea on exertion",
    "Ecchymosis",
    "Energy increased",
    "Erythema",
    "Extreme exhaustion",
    "Fecaluria",
    "Feeling, suicidal",
    "Feeling suicidal",
    "Flatulence",
    "Fluid overload",
    "Foul smell of urine",
    "Foul smell of urine",
    "Hallucinations (auditory)",
    "Headache",
    "Hematuria",
    "Heme positive",
    "Hemiplegia",
    "Hot flush",
    "Hyperkalemia",
    "Hypokinesia",
    "Hypoproteinemia",
    "Inappropriate affect",
    "Increased appetite",
    "Indigestion",
    "Inflammatory nails",
    "Internal itching",
    "Irregular sugar level",
    "Irritability",
    "Irritation in anus",
    "Itching",
    "Joint pain",
    "Knee pain",
    "Lack of concentration",
    "Lethargy",
    "Loss of appetite",
    "Loss of balance",
    "Loss of smell",
    "Malaise",
    "Mass in breast",
    "Mass of body structure",
    "Mediastinal shift",
    "Mental status changes",
    "Moan",
    "Nightmare",
    "Orthopnea",
    "Out of breath",
    "Pain",
    "Pain (abdominal)",
    "Pain (chest)",
    "Pain (chest)",
    "Para 2",
    "Spotting urination",
    "Paresthesia",
    "Pericardial friction rub",
    "Pleuritic pain",
    "Polyuria",
    "Prostatism",
    "Pulsus paradoxus",
    "Rale",
    "Seizure",
    "Sore to touch",
    "Speech slurred",
    "Suicidal",
    "Swelling",
    "Systolic murmur",
    "Thicken",
    "Transaminitis",
    "Tremor",
    "Uncoordination",
    "Unresponsiveness",
    "Unsteady gait",
    "Urgency of micturition",
    "Nausea and vomiting",
    "Wheezing",
    "Yellow sputum",
    "Yellow crust ooze",
    "Yellow urine",
    "Yellowing of eyes",
    "Yellowish skin"
]



encoded_inputs = {
    "Abdominal pain": 0,
    "Abnormal menstruation": 1,
    "Abscess (bacterial)": 2,
    "Acidity": 3,
    "Achalasia": 4,
    "Altered sensorium": 5,
    "Anxiety": 6,
    "Back pain": 7,
    "Belly pain": 8,
    "Behavior showing increased motor activity": 9,
    "Blackheads": 10,
    "Bladder discomfort": 11,
    "Blister": 12,
    "Blood in sputum": 13,
    "Bloody stool": 14,
    "Blurred and distorted vision": 15,
    "Brittle nails": 16,
    "Bruising": 17,
    "Burning micturition": 18,
    "Breathlessness": 19,
    "Chest pain": 20,
    "Chills": 21,
    "Clonus": 22,
    "Cold hands and feet": 23,
    "Coma": 24,
    "Congestion": 25,
    "Constipation": 26,
    "Continuous feel of urine": 27,
    "Continuous sneezing": 28,
    "Cough": 29,
    "Cramps": 30,
    "Cyanosis": 31,
    "Dark urine": 32,
    "Dehydration": 33,
    "Depression": 34,
    "Diarrhea": 35,
    "Dischromic patches": 36,
    "Distention of abdomen": 37,
    "Dizziness": 38,
    "Drying and tingling lips": 39,
    "Dysarthria": 40,
    "Dyspnea": 41,
    "Dyspnea on exertion": 42,
    "Ecchymosis": 43,
    "Energy increased": 44,
    "Erythema": 45,
    "Extreme exhaustion": 46,
    "Fecaluria": 47,
    "Feeling, suicidal": 48,
    "Feeling suicidal": 49,
    "Flatulence": 50,
    "Fluid overload": 51,
    "Foul smell of urine": 52,
    "Hallucinations (auditory)": 53,
    "Headache": 54,
    "Hematuria": 55,
    "Heme positive": 56,
    "Hemiplegia": 57,
    "Hot flush": 58,
    "Hyperkalemia": 59,
    "Hypokinesia": 60,
    "Hypoproteinemia": 61,
    "Inappropriate affect": 62,
    "Increased appetite": 63,
    "Indigestion": 64,
    "Inflammatory nails": 65,
    "Internal itching": 66,
    "Irregular sugar level": 67,
    "Irritability": 68,
    "Irritation in anus": 69,
    "Itching": 70,
    "Joint pain": 71,
    "Knee pain": 72,
    "Lack of concentration": 73,
    "Lethargy": 74,
    "Loss of appetite": 75,
    "Loss of balance": 76,
    "Loss of smell": 77,
    "Malaise": 78,
    "Mass in breast": 79,
    "Mass of body structure": 80,
    "Mediastinal shift": 81,
    "Mental status changes": 82,
    "Moan": 83,
    "Nightmare": 84,
    "Orthopnea": 85,
    "Out of breath": 86,
    "Pain": 87,
    "Pain (abdominal)": 88,
    "Pain (chest)": 89,
    "Para 2": 90,
    "Spotting urination": 91,
    "Paresthesia": 92,
    "Pericardial friction rub": 93,
    "Pleuritic pain": 94,
    "Polyuria": 95,
    "Prostatism": 96,
    "Pulsus paradoxus": 97,
    "Rale": 98,
    "Seizure": 99,
    "Sore to touch": 100,
    "Speech slurred": 101,
    "Suicidal": 102,
    "Swelling": 103,
    "Systolic murmur": 104,
    "Thicken": 105,
    "Transaminitis": 106,
    "Tremor": 107,
    "Uncoordination": 108,
    "Unresponsiveness": 109,
    "Unsteady gait": 110,
    "Urgency of micturition": 111,
    "Nausea and vomiting": 112,
    "Wheezing": 113,
    "Yellow sputum": 114,
    "Yellow crust ooze": 115,
    "Yellow urine": 116,
    "Yellowing of eyes": 117,
    "Yellowish skin": 118
}




    



user_inputs = {
    "Abdominal pain: ['Abdominal pain']",
    "Abnormal menstruation: ['Abnormal menstruation']",
    "Abscess (bacterial): ['Abscess (bacterial)']",
    "Acidity: ['Acidity']",
    "Achalasia: ['Achalasia']",
    "Altered sensorium: ['Altered sensorium']",
    "Anxiety: ['Anxiety']",
    "Back pain: ['Back pain']",
    "Belly pain: ['Belly pain']",
    "Behavior showing increased motor activity: ['Behavior showing increased motor activity']",
    "Blackheads: ['Blackheads']",
    "Bladder discomfort: ['Bladder discomfort']",
    "Blister: ['Blister']",
    "Blood in sputum: ['Blood in sputum']",
    "Bloody stool: ['Bloody stool']",
    "Blurred and distorted vision: ['Blurred and distorted vision']",
    "Brittle nails: ['Brittle nails']",
    "Bruising: ['Bruising']",
    "Burning micturition: ['Burning micturition']",
    "Breathlessness: ['Breathlessness']",
    "Chest pain: ['Chest pain']",
    "Chills: ['Chills']",
    "Clonus: ['Clonus']",
    "Cold hands and feet: ['Cold hands and feet']",
    "Coma: ['Coma']",
    "Congestion: ['Congestion']",
    "Constipation: ['Constipation']",
    "Continuous feel of urine: ['Continuous feel of urine']",
    "Continuous sneezing: ['Continuous sneezing']",
    "Cough: ['Cough']",
    "Cramps: ['Cramps']",
    "Cyanosis: ['Cyanosis']",
    "Dark urine: ['Dark urine']",
    "Dehydration: ['Dehydration']",
    "Depression: ['Depression']",
    "Diarrhea: ['Diarrhea']",
    "Dischromic patches: ['Dischromic patches']",
    "Distention of abdomen: ['Distention of abdomen']",
    "Dizziness: ['Dizziness']",
    "Drying and tingling lips: ['Drying and tingling lips']",
    "Dysarthria: ['Dysarthria']",
    "Dyspnea: ['Dyspnea']",
    "Dyspnea on exertion: ['Dyspnea on exertion']",
    "Ecchymosis: ['Ecchymosis']",
    "Energy increased: ['Energy increased']",
    "Erythema: ['Erythema']",
    "Extreme exhaustion: ['Extreme exhaustion']",
    "Fecaluria: ['Fecaluria']",
    "Feeling, suicidal: ['Feeling, suicidal']",
    "Feeling suicidal: ['Feeling suicidal']",
    "Flatulence: ['Flatulence']",
    "Fluid overload: ['Fluid overload']",
    "Foul smell of urine: ['Foul smell of urine']",
    "Foul smell of urine: ['Foul smell of urine']",
    "Hallucinations (auditory): ['Hallucinations (auditory)']",
    "Headache: ['Headache']",
    "Hematuria: ['Hematuria']",
    "Heme positive: ['Heme positive']",
    "Hemiplegia: ['Hemiplegia']",
    "Hot flush: ['Hot flush']",
    "Hyperkalemia: ['Hyperkalemia']",
    "Hypokinesia: ['Hypokinesia']",
    "Hypoproteinemia: ['Hypoproteinemia']",
    "Inappropriate affect: ['Inappropriate affect']",
    "Increased appetite: ['Increased appetite']",
    "Indigestion: ['Indigestion']",
    "Inflammatory nails: ['Inflammatory nails']",
    "Internal itching: ['Internal itching']",
    "Irregular sugar level: ['Irregular sugar level']",
    "Irritability: ['Irritability']",
    "Irritation in anus: ['Irritation in anus']",
    "Itching: ['Itching']",
    "Joint pain: ['Joint pain']",
    "Knee pain: ['Knee pain']",
    "Lack of concentration: ['Lack of concentration']",
    "Lethargy: ['Lethargy']",
    "Loss of appetite: ['Loss of appetite']",
    "Loss of balance: ['Loss of balance']",
    "Loss of smell: ['Loss of smell']",
    "Malaise: ['Malaise']",
    "Mass in breast: ['Mass in breast']",
    "Mass of body structure: ['Mass of body structure']",
    "Mediastinal shift: ['Mediastinal shift']",
    "Mental status changes: ['Mental status changes']",
    "Moan: ['Moan']",
    "Nightmare: ['Nightmare']",
    "Orthopnea: ['Orthopnea']",
    "Out of breath: ['Out of breath']",
    "Pain: ['Pain']",
    "Pain (abdominal): ['Pain (abdominal)']",
    "Pain (chest): ['Pain (chest)']",
    "Pain (chest): ['Pain (chest)']",
    "Para 2: ['Para 2']",
    "Spotting urination: ['Spotting urination']",
    "Paresthesia: ['Paresthesia']",
    "Pericardial friction rub: ['Pericardial friction rub']",
    "Pleuritic pain: ['Pleuritic pain']",
    "Polyuria: ['Polyuria']",
    "Prostatism: ['Prostatism']",
    "Pulsus paradoxus: ['Pulsus paradoxus']",
    "Rale: ['Rale']",
    "Seizure: ['Seizure']",
    "Sore to touch: ['Sore to touch']",
    "Speech slurred: ['Speech slurred']",
    "Suicidal: ['Suicidal']",
    "Swelling: ['Swelling']",
    "Systolic murmur: ['Systolic murmur']",
    "Thicken: ['Thicken']",
    "Transaminitis: ['Transaminitis']",
    "Tremor: ['Tremor']",
    "Uncoordination: ['Uncoordination']",
    "Unresponsiveness: ['Unresponsiveness']",
    "Unsteady gait: ['Unsteady gait']",
    "Urgency of micturition: ['Urgency of micturition']",
    "Nausea and vomiting: ['Nausea and vomiting']",
    "Wheezing: ['Wheezing']",
    "Yellow sputum: ['Yellow sputum']",
    "Yellow crust ooze: ['Yellow crust ooze']",
    "Yellow urine: ['Yellow urine']",
    "Yellowing of eyes: ['Yellowing of eyes']",
    "Yellowish skin: ['Yellowish skin']"
}

    
    
symptoms = list(encoded_inputs.keys())

# Allow the user to select symptoms
selected_options = st.multiselect("Select symptoms:", symptoms)

# Create user inputs based on the selected symptoms
user_inputs = {}
for symptom in selected_options:
    user_inputs[symptom] = [symptom]

# Create an array of zeros with the same number of features as the model
# (assuming the model's features match the symptoms)
input_data = [0] * len(encoded_inputs)

# Update the input_data with 1 at the positions corresponding to the selected symptoms
for symptom in selected_options:
    input_data[encoded_inputs[symptom]] = 1

# Predict the probabilities for different disease classes
prediction = loaded_model.predict_proba([input_data])

# Display the predicted probabilities
st.write("Predicted Probabilities for Different Diseases:")
for disease, probability in zip(loaded_model.classes_, prediction[0]):
    st.write(f"{disease}: {probability:.2f}")