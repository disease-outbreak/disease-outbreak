import streamlit as st
import pandas as pd
import joblib

# Load your trained K-Nearest Neighbors model
loaded_model = joblib.load('KNNTest.sav')

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

# Input features from the user
abdominalpain = st.checkbox("Abdominal Pain")
abnormalmenstruation = st.checkbox("Abnormal Menstruation")
acidity = st.checkbox("Acidity")
acuteliverfailure = st.checkbox("Acute Liver Failure")
alteredsensorium = st.checkbox("Altered Sensorium")
anxiety = st.checkbox("Anxiety")
backpain = st.checkbox("Back Pain")
bellypain = st.checkbox("Belly Pain")
blackheads = st.checkbox("Blackheads")
# Checkbox for 'bladderdiscomfort'
bladderdiscomfort = st.checkbox("Bladder Discomfort")

# Checkbox for 'blister'
blister = st.checkbox("Blister")

# Checkbox for 'bloodinsputum'
bloodinsputum = st.checkbox("Blood in Sputum")

# Checkbox for 'bloodystool'
bloodystool = st.checkbox("Bloody Stool")

# Checkbox for 'blurredanddistortedvision'
blurredanddistortedvision = st.checkbox("Blurred and Distorted Vision")

# Checkbox for 'breathlessness'
breathlessness = st.checkbox("Breathlessness")

# Checkbox for 'brittlenails'
brittlenails = st.checkbox("Brittle Nails")

# Checkbox for 'bruising'
bruising = st.checkbox("Bruising")

# Checkbox for 'burningmicturition'
burningmicturition = st.checkbox("Burning Micturition")

# Checkbox for 'chestpain'
chestpain = st.checkbox("Chest Pain")

# Checkbox for 'chills'
chills = st.checkbox("Chills")

# Checkbox for 'coldhandsandfeets'
coldhandsandfeets = st.checkbox("Cold Hands and Feet")

# Checkbox for 'coma'
coma = st.checkbox("Coma")

# Checkbox for 'congestion'
congestion = st.checkbox("Congestion")

# Checkbox for 'constipation'
constipation = st.checkbox("Constipation")

# Checkbox for 'continuousfeelofurine'
continuousfeelofurine = st.checkbox("Continuous Feel of Urine")

# Checkbox for 'continuoussneezing'
continuoussneezing = st.checkbox("Continuous Sneezing")

# Checkbox for 'cough'
cough = st.checkbox("Cough")

# Checkbox for 'cramps'
cramps = st.checkbox("Cramps")

# Checkbox for 'darkurine'
darkurine = st.checkbox("Dark Urine")

# Checkbox for 'dehydration'
dehydration = st.checkbox("Dehydration")

# Checkbox for 'depression'
depression = st.checkbox("Depression")

# Checkbox for 'diarrhoea'
diarrhoea = st.checkbox("Diarrhoea")

# Checkbox for 'dischromic patches'
dischromic_patches = st.checkbox("Dischromic Patches")

# Checkbox for 'distentionofabdomen'
distentionofabdomen = st.checkbox("Distention of Abdomen")

# Checkbox for 'dizziness'
dizziness = st.checkbox("Dizziness")

# Checkbox for 'dryingandtinglinglips'
dryingandtinglinglips = st.checkbox("Drying and Tingling Lips")

# Checkbox for 'enlargedthyroid'
enlargedthyroid = st.checkbox("Enlarged Thyroid")

# Checkbox for 'excessivehunger'
excessivehunger = st.checkbox("Excessive Hunger")

# Checkbox for 'extramaritalcontacts'
extramaritalcontacts = st.checkbox("Extramarital Contacts")

# Checkbox for 'familyhistory'
familyhistory = st.checkbox("Family History")

# Checkbox for 'fastheartrate'
fastheartrate = st.checkbox("Fast Heart Rate")

# Checkbox for 'fatigue'
fatigue = st.checkbox("Fatigue")

# Checkbox for 'fluidoverload'
fluidoverload = st.checkbox("Fluid Overload")

# Checkbox for 'foulsmellof urine'
foulsmellof_urine = st.checkbox("Foul Smell of Urine")

# Checkbox for 'headache'
headache = st.checkbox("Headache")

# Checkbox for 'highfever'
highfever = st.checkbox("High Fever")

# Checkbox for 'hipjointpain'
hipjointpain = st.checkbox("Hip Joint Pain")

# Checkbox for 'historyofalcoholconsumption'
historyofalcoholconsumption = st.checkbox("History of Alcohol Consumption")

# Checkbox for 'increasedappetite'
increasedappetite = st.checkbox("Increased Appetite")

# Checkbox for 'indigestion'
indigestion = st.checkbox("Indigestion")

# Checkbox for 'inflammatorynails'
inflammatorynails = st.checkbox("Inflammatory Nails")

# Checkbox for 'internalitching'
internalitching = st.checkbox("Internal Itching")

# Checkbox for 'irregularsugarlevel'
irregularsugarlevel = st.checkbox("Irregular Sugar Level")

# Checkbox for 'irritability'
irritability = st.checkbox("Irritability")

# Checkbox for 'irritationinanus'
irritationinanus = st.checkbox("Irritation in Anus")

# Checkbox for 'itching'
itching = st.checkbox("Itching")

# Checkbox for 'jointpain'
jointpain = st.checkbox("Joint Pain")

# Checkbox for 'kneepain'
kneepain = st.checkbox("Knee Pain")

# Checkbox for 'lackofconcentration'
lackofconcentration = st.checkbox("Lack of Concentration")

# Checkbox for 'lethargy'
lethargy = st.checkbox("Lethargy")

# Checkbox for 'lossofappetite'
lossofappetite = st.checkbox("Loss of Appetite")

# Checkbox for 'lossofbalance'
lossofbalance = st.checkbox("Loss of Balance")

# Checkbox for 'lossofsmell'
lossofsmell = st.checkbox("Loss of Smell")

# Checkbox for 'malaise'
malaise = st.checkbox("Malaise")

# Checkbox for 'mildfever'
mildfever = st.checkbox("Mild Fever")

# Checkbox for 'moodswings'
moodswings = st.checkbox("Mood Swings")

# Checkbox for 'movementstiffness'
movementstiffness = st.checkbox("Movement Stiffness")

# Checkbox for 'mucoidsputum'
mucoidsputum = st.checkbox("Mucoid Sputum")

# Checkbox for 'musclepain'
musclepain = st.checkbox("Muscle Pain")

# Checkbox for 'musclewasting'
musclewasting = st.checkbox("Muscle Wasting")

# Checkbox for 'muscleweakness'
muscleweakness = st.checkbox("Muscle Weakness")

# Checkbox for 'nausea'
nausea = st.checkbox("Nausea")

# Checkbox for 'neckpain'
neckpain = st.checkbox("Neck Pain")

# Checkbox for 'nodalskineruptions'
nodalskineruptions = st.checkbox("Nodal Skin Eruptions")

# Checkbox for 'obesity'
obesity = st.checkbox("Obesity")

# Checkbox for 'painbehindtheeyes'
painbehindtheeyes = st.checkbox("Pain Behind the Eyes")

# Checkbox for 'painduringbowelmovements'
painduringbowelmovements = st.checkbox("Pain During Bowel Movements")

# Checkbox for 'painfulwalking'
painfulwalking = st.checkbox("Painful Walking")

# Checkbox for 'paininanalregion'
paininanalregion = st.checkbox("Pain in Anal Region")

# Checkbox for 'palpitations'
palpitations = st.checkbox("Palpitations")

# Checkbox for 'passageofgases'
passageofgases = st.checkbox("Passage of Gases")

# Checkbox for 'patchesinthroat'
patchesinthroat = st.checkbox("Patches in Throat")

# Checkbox for 'phlegm'
phlegm = st.checkbox("Phlegm")

# Checkbox for 'polyuria'
polyuria = st.checkbox("Polyuria")

# Checkbox for 'prominentveinsoncalf'
prominentveinsoncalf = st.checkbox("Prominent Veins on Calf")

# Checkbox for 'puffyfaceandeyes'
puffyfaceandeyes = st.checkbox("Puffy Face and Eyes")

# Checkbox for 'pusfilledpimples'
pusfilledpimples = st.checkbox("Pus-Filled Pimples")

# Checkbox for 'receivingbloodtransfusion'
receivingbloodtransfusion = st.checkbox("Receiving Blood Transfusion")

# Checkbox for 'receivingunsterileinjections'
receivingunsterileinjections = st.checkbox("Receiving Unsterile Injections")

# Checkbox for 'rednessofeyes'
rednessofeyes = st.checkbox("Redness of Eyes")

# Checkbox for 'redsorearoundnose'
redsorearoundnose = st.checkbox("Red Sore Around Nose")

# Checkbox for 'redspotsoverbody'
redspotsoverbody = st.checkbox("Red Spots Over Body")

# Checkbox for 'restlessness'
restlessness = st.checkbox("Restlessness")

# Checkbox for 'runnynose'
runnynose = st.checkbox("Runny Nose")

# Checkbox for 'rustysputum'
rustysputum = st.checkbox("Rusty Sputum")

# Checkbox for 'scurring'
scurring = st.checkbox("Scurring")

# Checkbox for 'shivering'
shivering = st.checkbox("Shivering")

# Checkbox for 'silverlikedusting'
silverlikedusting = st.checkbox("Silver-Like Dusting")

# Checkbox for 'sinuspressure'
sinuspressure = st.checkbox("Sinus Pressure")

# Checkbox for 'skinpeeling'
skinpeeling = st.checkbox("Skin Peeling")

# Checkbox for 'skinrash'
skinrash = st.checkbox("Skin Rash")

# Checkbox for 'slurredspeech'
slurredspeech = st.checkbox("Slurred Speech")

# Checkbox for 'smalldentsinnails'
smalldentsinnails = st.checkbox("Small Dents in Nails")

# Checkbox for 'spinningmovements'
spinningmovements = st.checkbox("Spinning Movements")

# Checkbox for 'spotting urination'
spotting_urination = st.checkbox("Spotting Urination")

# Checkbox for 'stiffneck'
stiffneck = st.checkbox("Stiff Neck")

# Checkbox for 'stomachbleeding'
stomachbleeding = st.checkbox("Stomach Bleeding")

# Checkbox for 'stomachpain'
stomachpain = st.checkbox("Stomach Pain")

# Checkbox for 'sunkeneyes'
sunkeneyes = st.checkbox("Sunken Eyes")

# Checkbox for 'sweating'
sweating = st.checkbox("Sweating")

# Checkbox for 'swelledlymphnodes'
swelledlymphnodes = st.checkbox("Swelled Lymph Nodes")

# Checkbox for 'swellingjoints'
swellingjoints = st.checkbox("Swelling Joints")

# Checkbox for 'swellingofstomach'
swellingofstomach = st.checkbox("Swelling of Stomach")

# Checkbox for 'swollenbloodvessels'
swollenbloodvessels = st.checkbox("Swollen Blood Vessels")

# Checkbox for 'swollenextremeties'
swollenextremeties = st.checkbox("Swollen Extremities")

# Checkbox for 'swollenlegs'
swollenlegs = st.checkbox("Swollen Legs")

# Checkbox for 'throatirritation'
throatirritation = st.checkbox("Throat Irritation")

# Checkbox for 'toxiclook(typhos)'
toxiclook_typhos = st.checkbox("Toxic Look (Typhos)")

# Checkbox for 'ulcersontongue'
ulcersontongue = st.checkbox("Ulcers on Tongue")

# Checkbox for 'unsteadiness'
unsteadiness = st.checkbox("Unsteadiness")

# Checkbox for 'visualdisturbances'
visualdisturbances = st.checkbox("Visual Disturbances")

# Checkbox for 'vomiting'
vomiting = st.checkbox("Vomiting")

# Checkbox for 'wateringfromeyes'
wateringfromeyes = st.checkbox("Watering from Eyes")

# Checkbox for 'weaknessinlimbs'
weaknessinlimbs = st.checkbox("Weakness in Limbs")

# Checkbox for 'weaknessofonebodyside'
weaknessofonebodyside = st.checkbox("Weakness of One Body Side")

# Checkbox for 'weightgain'
weightgain = st.checkbox("Weight Gain")

# Checkbox for 'weightloss'
weightloss = st.checkbox("Weight Loss")

# Checkbox for 'yellowcrustooze'
yellowcrustooze = st.checkbox("Yellow Crust Ooze")

# Checkbox for 'yellowingofeyes'
yellowingofeyes = st.checkbox("Yellowing of Eyes")

# Checkbox for 'yellowishskin'
yellowishskin = st.checkbox("Yellowish Skin")

# Checkbox for 'yellowurine'
yellowurine = st.checkbox("Yellow Urine")


# Create a DataFrame from the user inputs
user_inputs = {
    "abdominalpain": [abdominalpain],
    "abnormalmenstruation": [abnormalmenstruation],
    "acidity": [acidity],
    "acuteliverfailure": [acuteliverfailure],
    "alteredsensorium": [alteredsensorium],
    "anxiety": [anxiety],
    "backpain": [backpain],
    "bellypain": [bellypain],
    "blackheads": [blackheads],
    "bladderdiscomfort": [bladderdiscomfort],
    "blister": [blister],
    "bloodinsputum": [bloodinsputum],
    "bloodystool": [bloodystool],
    "blurredanddistortedvision": [blurredanddistortedvision],
    "breathlessness": [breathlessness],
    "brittlenails": [brittlenails],
    "bruising": [bruising],
    "burningmicturition": [burningmicturition],
    "chestpain": [chestpain],
    "chills": [chills],
    "coldhandsandfeets": [coldhandsandfeets],
    "coma": [coma],
    "congestion": [congestion],
    "constipation": [constipation],
    "continuousfeelofurine": [continuousfeelofurine],
    "continuoussneezing": [continuoussneezing],
    "cough": [cough],
    "cramps": [cramps],
    "darkurine": [darkurine],
    "dehydration": [dehydration],
    "depression": [depression],
    "diarrhoea": [diarrhoea],
    "dischromic patches": [dischromic_patches],
    "distentionofabdomen": [distentionofabdomen],
    "dizziness": [dizziness],
    "dryingandtinglinglips": [dryingandtinglinglips],
    "enlargedthyroid": [enlargedthyroid],
    "excessivehunger": [excessivehunger],
    "extramaritalcontacts": [extramaritalcontacts],
    "familyhistory": [familyhistory],
    "fastheartrate": [fastheartrate],
    "fatigue": [fatigue],
    "fluidoverload": [fluidoverload],
    "foulsmellof urine": [foulsmellof_urine],
    "headache": [headache],
    "highfever": [highfever],
    "hipjointpain": [hipjointpain],
    "historyofalcoholconsumption": [historyofalcoholconsumption],
    "increasedappetite": [increasedappetite],
    "indigestion": [indigestion],
    "inflammatorynails": [inflammatorynails],
    "internalitching": [internalitching],
    "irregularsugarlevel": [irregularsugarlevel],
    "irritability": [irritability],
    "irritationinanus": [irritationinanus],
    "itching": [itching],
    "jointpain": [jointpain],
    "kneepain": [kneepain],
    "lackofconcentration": [lackofconcentration],
    "lethargy": [lethargy],
    "lossofappetite": [lossofappetite],
    "lossofbalance": [lossofbalance],
    "lossofsmell": [lossofsmell],
    "malaise": [malaise],
    "mildfever": [mildfever],
    "moodswings": [moodswings],
    "movementstiffness": [movementstiffness],
    "mucoidsputum": [mucoidsputum],
    "musclepain": [musclepain],
    "musclewasting": [musclewasting],
    "muscleweakness": [muscleweakness],
    "nausea": [nausea],
    "neckpain": [neckpain],
    "nodalskineruptions": [nodalskineruptions],
    "obesity": [obesity],
    "painbehindtheeyes": [painbehindtheeyes],
    "painduringbowelmovements": [painduringbowelmovements],
    "painfulwalking": [painfulwalking],
    "paininanalregion": [paininanalregion],
    "palpitations": [palpitations],
    "passageofgases": [passageofgases],
    "patchesinthroat": [patchesinthroat],
    "phlegm": [phlegm],
    "polyuria": [polyuria],
    "prominentveinsoncalf": [prominentveinsoncalf],
    "puffyfaceandeyes": [puffyfaceandeyes],
    "pusfilledpimples": [pusfilledpimples],
    "receivingbloodtransfusion": [receivingbloodtransfusion],
    "receivingunsterileinjections": [receivingunsterileinjections],
    "rednessofeyes": [rednessofeyes],
    "redsorearoundnose": [redsorearoundnose],
    "redspotsoverbody": [redspotsoverbody],
    "restlessness": [restlessness],
    "runnynose": [runnynose],
    "rustysputum": [rustysputum],
    "scurring": [scurring],
    "shivering": [shivering],
    "silverlikedusting": [silverlikedusting],
    "sinuspressure": [sinuspressure],
    "skinpeeling": [skinpeeling],
    "skinrash": [skinrash],
    "slurredspeech": [slurredspeech],
    "smalldentsinnails": [smalldentsinnails],
    "spinningmovements": [spinningmovements],
    "spotting urination": [spotting_urination],
    "stiffneck": [stiffneck],
    "stomachbleeding": [stomachbleeding],
    "stomachpain": [stomachpain],
    "sunkeneyes": [sunkeneyes],
    "sweating": [sweating],
    "swelledlymphnodes": [swelledlymphnodes],
    "swellingjoints": [swellingjoints],
    "swellingofstomach": [swellingofstomach],
    "swollenbloodvessels": [swollenbloodvessels],
    "swollenextremeties": [swollenextremeties],
    "swollenlegs": [swollenlegs],
    "throatirritation": [throatirritation],
    "toxiclook(typhos)": [toxiclook_typhos],
    "ulcersontongue": [ulcersontongue],
    "unsteadiness": [unsteadiness],
    "visualdisturbances": [visualdisturbances],
    "vomiting": [vomiting],
    "wateringfromeyes": [wateringfromeyes],
    "weaknessinlimbs": [weaknessinlimbs],
    "weaknessofonebodyside": [weaknessofonebodyside],
    "weightgain": [weightgain],
    "weightloss": [weightloss],
    "yellowcrustooze": [yellowcrustooze],
    "yellowingofeyes": [yellowingofeyes],
    "yellowishskin": [yellowishskin],
    "yellowurine": [yellowurine]
}
   


# Collect user inputs in a DataFrame
input_data = pd.DataFrame(user_inputs)

# Make predictions using the loaded model
probs = loaded_model.predict_proba(input_data)

# Create a dictionary to store class labels and their corresponding probabilities
class_probabilities = dict(zip(loaded_model.classes_, probs[0]))

# Display the diseases with probabilities greater than 0.0
st.write("Top Disease related to your :")
for disease, probability in class_probabilities.items():
    if probability > 0.0:
        st.write(f"{disease}: Probability {probability:.2f}")