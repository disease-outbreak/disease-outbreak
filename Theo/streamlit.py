import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

# Load your dataset
df = pd.read_csv('combined_disease_df_final.csv')

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


#=============================================================================================
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

    # Assuming disease_probabilities is your DataFrame with columns 'Disease' and 'Probability'
    # Display the top 5 probabilities in a bar chart
    top5_diseases = disease_probabilities.nlargest(5, 'Probability')

    # Reverse the order of the DataFrame to have the highest probability at the top
    top5_diseases = top5_diseases[::-1]

    # Lightseagreen color
    bar_color = 'lightseagreen'

    # Streamlit app
    st.title('Top 5 Disease Probabilities')

    # Create a horizontal bar chart with lightseagreen color
    fig, ax = plt.subplots()
    bars = ax.barh(top5_diseases['Disease'], top5_diseases['Probability'], color=bar_color)
    ax.set_ylabel('Predicted Diseases')
    ax.set_xlabel('Probability (%)')

    # Display the numbers on the right of each bar as whole percentages
    for bar in bars:
        xval = bar.get_width()
        percentage_label = f'{int(xval)}%'
        plt.text(xval, bar.get_y() + bar.get_height()/2, percentage_label, ha='left', va='center', fontsize=10)

    # Display the chart using Streamlit's st.pyplot
    st.pyplot(fig)



#=============================================================================================






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

