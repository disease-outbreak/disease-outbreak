# National Disease Outbreak Detection Project

## Abstract

The **Disease Outbreak Detection Project** is a data-driven initiative aimed at early identification and monitoring of disease outbreaks on a global scale. Leveraging data science and machine learning, we strive to improve disease surveillance and response strategies to mitigate health risks.

## Goal

The primary goal of this project is to develop a predictive model that can detect disease outbreaks with high accuracy and facilitate timely response measures. By analyzing diverse datasets related to health and epidemiology, we aim to create a reliable tool for public health organizations and authorities.

## Data Dictionary

| Disease                               | Definition                                           | Datatype |
|---------------------------------------|-----------------------------------------------------|----------|
| (vertigo) Paroymsal Positional Vertigo | an inner ear disorder that causes vertigo when you move your head                                 | object   |
| AIDS                                  | a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV)                                    | object   |
| Acne                                  | an inflammatory disorder of the skin                                    | object   |
| Alcoholic hepatitis                   | inflammation of the liver caused by drinking alcohol                     | object   |
| Allergy                               |  reaction to substances in the environment that are harmless to most people                                 | object   |
| Arthritis                             | for Arthritis Disease                                | object   |
| Bronchial Asthma                      | for Bronchial Asthma Disease                         | object   |
| Cervical spondylosis                  | for Cervical Spondylosis Disease                     | object   |
| Chicken pox                           | for Chicken Pox Disease                             | object   |
| Chronic cholestasis                   | for Chronic Cholestasis Disease                     | object   |
| Common Cold                           | for Common Cold Disease                             | object   |
| Dengue                                | for Dengue Disease                                  | object   |
| Diabetes                              | for Diabetes Disease                                | object   |
| Dimorphic hemmorhoids(piles)          | for Dimorphic Hemorrhoids (Piles) Disease           | object   |
| Drug Reaction                         | for Drug Reaction Disease                            | object   |
| Fungal infection                      | for Fungal Infection Disease                         | object   |
| GERD                                  | for GERD Disease                                     | object   |
| Gastroenteritis                       | for Gastroenteritis Disease                          | object   |
| Heart attack                          | for Heart Attack Disease                             | object   |
| Hepatitis B                           | for Hepatitis B Disease                             | object   |
| Hepatitis C                           | for Hepatitis C Disease                             | object   |
| Hepatitis D                           | for Hepatitis D Disease                             | object   |
| Hepatitis E                           | for Hepatitis E Disease                             | object   |
| Hypertension                          | for Hypertension Disease                            | object   |
| Hyperthyroidism                       | for Hyperthyroidism Disease                         | object   |
| Hypoglycemia                          | for Hypoglycemia Disease                            | object   |
| Hypothyroidism                        | for Hypothyroidism Disease                          | object   |
| Impetigo                             | for Impetigo Disease                               | object   |
| Jaundice                              | for Jaundice Disease                                | object   |
| Malaria                               | for Malaria Disease                                 | object   |
| Migraine                              | for Migraine Disease                                | object   |
| Osteoarthristis                       | for Osteoarthritis Disease                          | object   |
| Paralysis (brain hemorrhage)          | for Paralysis (Brain Hemorrhage) Disease           | object   |
| Peptic ulcer diseae                   | for Peptic Ulcer Disease                            | object   |
| Pneumonia                             | for Pneumonia Disease                               | object   |
| Psoriasis                             | for Psoriasis Disease                               | object   |
| Tuberculosis                          | for Tuberculosis Disease                            | object   |
| Typhoid                               | for Typhoid Disease                                 | object   |
| Urinary tract infection               | for Urinary Tract Infection Disease                  | object   |
| Varicose veins                         | for Varicose Veins Disease                          | object   |
| hepatitis A                           | for Hepatitis A Disease                            | object   |



| Disease                              |                                       |      |       |       |       |        |      |      |      |     |      |         |      |          |         |              |
|--------------------------------------|---------------------------------------|------|-------|-------|-------|--------|------|------|------|-----|------|---------|------|----------|---------|--------------|
| Acidity                              | Back Pain                           | Bladder Discomfort | Breathlessness | Burning Micturition | Chest Pain | Chills  | Constipation | Continuous Sneezing | Cough | Cramps | Fatigue | Headache | High Fever | Indigestion | Joint Pain | Mood Swings |
| Muscle Wasting                  | Muscle Weakness               | Neck Pain | Pain During Bowel Movements | Patches in Throat | Pus-filled Pimples | Shivering | Skin Rash | Stiff Neck | Stomach Pain | Sunken Eyes | Vomiting | Weakness in Limbs | Weight Gain | Weight Loss | Yellowish Skin | Itching |
| Abdominal Pain                    | Acidity                                | Anxiety | Blackheads | Bladder Discomfort | Blister | Breathlessness | Bruising | Chest Pain | Chills | Cold Hands and Feet | Cough | Cramps | Dehydration | Dizziness | Fatigue |
| Foul Smell of Urine          | Headache | High Fever | Indigestion | Joint Pain | Knee Pain | Lethargy | Loss of Appetite | Mood Swings | Nausea | Neck Pain | Nodal Skin Eruptions | Pain During Bowel Movements | Pain in Anal Region | Patches in Throat |
| Pus-filled Pimples | Restlessness | Shivering | Skin Peeling | Skin Rash | Stiff Neck | Stomach Pain | Sunken Eyes | Sweating | Swelling Joints | Ulcers on Tongue | Vomiting | Weakness in Limbs | Weakness of One Body Side | Weight Gain | Weight Loss | Yellowish Skin |
| Abdominal Pain                      | Altered Sensorium                  | Anxiety | Blackheads | Blister | Bloody Stool | Blurred and Distorted Vision | Breathlessness | Bruising | Burning Micturition | Chest Pain | Chills | Cold Hands and Feet | Continuous Feel of Urine |
| Cough | Dark Urine | Dehydration | Diarrhea | Dischromic Patches | Dizziness | Extramarital Contacts | Fatigue | Foul Smell of Urine | Headache | High Fever | Hip Joint Pain | Joint Pain | Knee Pain | Lethargy | Loss of Appetite | Loss of Balance | Mood Swings |
| Movement Stiffness | Nausea | Neck Pain | Nodal Skin Eruptions | Obesity | Painful Walking | Passage of Gases | Red Sore Around Nose | Restlessness | Scarring | Silver-Like Dusting | Skin Peeling | Spinning Movements | Stomach Pain | Sweating | Swelling Joints |
| Swelling of Stomach | Ulcers on Tongue | Vomiting | Watering from Eyes | Weakness of One Body Side | Weight Loss | Yellowish Skin | Abdominal Pain | Altered Sensorium | Anxiety | Blackheads | Blister | Bloody Stool | Blurred and Distorted Vision | Breathlessness |
| Burning Micturition | Chest Pain | Continuous Feel of Urine | Cough | Dark Urine | Diarrhea | Dischromic Patches | Distention of Abdomen | Dizziness | Excessive Hunger | Extramarital Contacts | Family History | Fatigue | Headache | High Fever | Hip Joint Pain |
| Irregular Sugar Level | Irritation in Anus | Lack of Concentration | Lethargy | Loss of Appetite | Loss of Balance | Mood Swings | Movement Stiffness | Nausea | Obesity | Painful Walking | Passage of Gases | Red Sore Around Nose | Restlessness | Scarring |
| Silver-Like Dusting | Skin Peeling | Spinning Movements | Spotting Urination | Sweating | Swelling Joints | Swelling of Stomach | Swollen Legs | Vomiting | Watering from Eyes | Weight Loss | Yellow Crust Ooze | Yellowing of Eyes | Yellowish Skin | Abdominal Pain | Brittlenails |
| Chest Pain | Diarrhea | Drying and Tingling Lips | Enlarged Thyroid | Excessive Hunger | Increased Appetite | Irritability | Loss of Appetite | Malaise | Mild Fever | Muscle Pain | Muscle Weakness | Nausea | Phlegm | Sweating | Swelling Joints | Swelling of Stomach |
| Swollen Legs | Unsteadiness | Yellow Crust Ooze | Yellowing of Eyes | Yellowish Skin | Abdominal Pain | Brittlenails | Chest Pain | Diarrhea | Drying and Tingling Lips | Fast Heart Rate | Increased Appetite | Irritability | Loss of Appetite | Malaise | Mild Fever |
| Muscle Weakness | Pain Behind the Eyes | Phlegm | Polyuria | Slurred Speech | Swollen Legs | Throat Irritation | Toxic Look (Typhos) | Visual Disturbances | Yellowing of Eyes | Abnormal Menstruation | Acute Liver Failure | Back Pain | Belly Pain | Depression |
| Fast Heart Rate | Irritability | Malaise | Mild Fever | Muscle Pain | Pain Behind the Eyes | Polyuria | Receiving Blood Transfusion | Red Spots Over Body | Redness of Eyes | Rusty Sputum | Slurred Speech | Swollen Legs | Throat Irritation | Toxic Look (Typhos) |
| Yellowing of Eyes | Abnormal Menstruation | Acute Liver Failure | Back Pain | Belly Pain | Coma | Depression | Irritability | Malaise | Muscle Pain | Palpitations | Receiving Blood Transfusion | Receiving Unsterile Injections | Red Spots Over Body | Redness of Eyes |
| Rusty Sputum | Sinus Pressure | Swollen Lymph Nodes | Yellowing of Eyes | Abnormal Menstruation | Coma | Irritability | Malaise | Muscle Pain | Palpitations | Receiving Unsterile Injections | Runny Nose | Sinus Pressure | Stomach Bleeding | Swollen Lymph Nodes |
| Abnormal Menstruation | Congestion | Malaise | Muscle Pain | Phlegm | Red Spots Over Body | Runny Nose | Stomach Bleeding | Chest Pain | Congestion | Phlegm | Red Spots Over Body | Blood in Sputum | Chest Pain | Loss of Smell | Blood in Sputum |
| Loss of Smell | Muscle Pain | Muscle Pain |



## Team

- **Project Lead:** Theodore Quansah
- **Data Engineer:** Nisha
- **Data Analyst:** Adam
- **Data Scientist:** Marcelino

## How to Duplicate This Project

To replicate this project on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash on Windows
   git clone https://github.com/yourusername/disease-outbreak-project.git
   cd disease-outbreak-project
2. **Install Dependencies:**
   ```bash on Windows
   set seed = 42
   
   

