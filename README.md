<a name="top"></a>
![name of photo](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%209.10.45%20AM.png?raw=true)

***
[[Description](#project_description)]
[[Planning](#planning)]
[[Acquire & Prep](#acquire_and_prep)]
[[Exploration](#explore)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
[[Recreate](#recreate)]
[[Meet the Team](#team)]
[[Interact](#interact)]
___


## <a name="project_description"></a>
![desc](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%209.47.31%20AM.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Description
    
Diagnosis of diseases often requires expertise and thorough medical examination. However, for common ailments or as a preliminary diagnostic tool, we aim to utilize symptom data to predict potential diseases. This tool can assist healthcare professionals as a reference, guide patients in understanding their conditions, or even help in telemedicine where immediate physical diagnosis isn't feasible.

### Goals
    
The primary goal of this project is to develop a predictive model that can predict a disease with high accuracy. By analyzing data acquired from the World Health Orginization related to disease symptoms in the U.S., we aim to create a reliable and user-friendly tool for individuals and public health organizations.
    
### Data Source
    
- Data was gathered from "The World Health Organization" and Columbia University website
- Other data from the following website to create a dashboard of mortality rates.

### Data Dictionary
|  Column  | Definition                                              | Data Type       |    
|----------|---------------------------------------------------------|-----------------|
| Disease  | Different type of diseases patients have in our dataset | Target variable |
| Symptoms | Different type of symptoms patients have in our dataset | object          |

***
</details>

## <a name="planning"></a> 
![plan](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%2011.14.17%20AM.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Project Outline:
    
- Initial Questions:
    - How do we plan to accomplish this project based off the goal?
    - How will it be used?
    - Where and how we will acquire data?
    - What specific features to move forward with?
    - What model will we use?   
  
- Acquisiton: 
    - Read data into python
    - Summarize data
    
- Prepare and clean: 
    - Potentially Drop features
    - Handle null values
    - Adjust data types
    - Rename columns
  
- Exploratory analysis:
    - Ask questions about our data
    - Make a hypothesis
    - Create visuals
    - Run statistical test
    
- Modeling:
    - Create multiple models
    - Choose the best model
    - Run a test
    - Conclude results
    - Make recommendations

### Target variable
- 'Disease'

***
</details>

## <a name="acquire_and_prep"></a> 
![acquire_prep](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%209.52.48%20AM.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Acquire Data:
- Data was gathered from The World Health Organization and Columbia University website
- Other data from the following website to create a dashboard of mortality rates.
    
The dataset comprises various diseases and their associated symptoms. Each disease can have multiple symptoms, and each symptom can be associated with multiple diseases
    
- Dataset Structure
    - Disease and symptoms(1):
        - Source: W.H.O.
        - Rows: 4920
        - Columns: 18
    - Symptoms and Severity:
        - Source: W.H.O.
        - Rows: 133
        - Columns:2
    - Disease and symptoms(2):
        - Source: Columbia.edu
        - Rows: 1866
        - Columns: 3
    
### Prepare Data
- Cleaned and preprocessed disease_df; result stored in processed_df.
- Transformed symptom data in processed_df to one-hot encoding; result in encoded_df.
- Aggregated preprocessed data and encoded symptoms into aggregated_df.
- Identified and resolved missing symptoms between aggregated_df and severity_df.
- Standardized and corrected column names in aggregated_df to align with severity_df.
- Removed UMLS codes and preprocessed scraped_df for analysis readiness.
- Eliminated duplicate columns in scraped_df to maintain data integrity.
- Converted numerical string columns to numeric types in scraped_df, except 'Disease'.
- Reshaped scraped_df into pivot format for enhanced analysis capability.
- Merged scraped_df with disease_df and conducted further preprocessing.
- Compared symptom presence between scraped_df and severity_df to spot discrepancies.
- Aligned scraped_df column names with severity_df, addressing missing data.
- Split combined dataset into stratified train, validation, and test sets; sizes outputted.

### Split data:
- Training set (60%)
- Validation set (20%)
- Test set (20%)
***

</details>

## <a name="explore"></a> 
![dict](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%209.58.00%20AM.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Statistical Analysis of Symptoms and Conditions:

- A strong statistical relationship exists between "yellowing of the eyes" and conditions such as Hepatitis (P less than 0.05), underscored by the frequent occurrence of this symptom alongside "yellowish skin" in the tri-gram frequency analysis.
"High fever" paired with "cough" in the tri-gram frequency chart indicates a significant correlation with respiratory conditions, highlighting the need for further clinical investigation when these symptoms are present together.

### Clinical Insights from Statistical Findings:
- The presence of "yellowing of the eyes" and "yellowish skin" should prompt clinical consideration for liver-related conditions, including Hepatitis and bile duct disorders.
The pairing of "high fever" with symptoms like "cough" and "headache" in the tri-gram analysis suggests a potential link to infectious diseases or inflammatory conditions that warrant a thorough clinical evaluation.

### Symptom Frequency Analysis:
- The symptom "fatigue" tops the frequency chart, suggesting it is a ubiquitous symptom across a multitude of conditions.
Other symptoms like "headache" and "nausea" are also prominent, indicating common issues affecting the nervous and digestive systems, respectively.
The frequency of "abdominal pain" and "vomiting" underscores their importance as symptoms in the dataset, pointing to a range of potential gastrointestinal or systemic disorders.

### N-Gram Analysis Insights:
- The bi-gram "loss of appetite" is highly frequent, indicating that this symptom is often reported and may be significant in the diagnostic process of various conditions.
The tri-gram "loss of appetite" shows how symptom combinations can provide more specific indications of health issues, possibly related to digestive health or metabolic disorders.

***   
</details>    

## <a name="model"></a> 
![model](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%2010.27.47%20AM.png?raw=true)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

The purpose of this modeling was to predict diseases based on the given symptoms. We evaluated several models, including Random Forest, Logistic Regression, and KNN, against a baseline model. Here's a summary of the results:

### Baseline Model:
- Accuracy: 0.0208
This low accuracy is expected since the baseline model predicts diseases based on the most frequent class without any true learning.

### Random Forest:
- Training Accuracy: 1.0000
- Validation Accuracy: 0.9583
- Test Accuracy: 0.8958
The Random Forest model performed remarkably well on the training dataset, achieving perfect accuracy. This indicates that it could potentially overfit to the training data. However, its high validation accuracy demonstrates that it generalizes fairly well to unseen data. The test accuracy further validates its robustness.

### Logistic Regression:
- Training Accuracy: 1.0000
- Validation Accuracy: 0.9583
Similar to the Random Forest model, the Logistic Regression model also showed perfect accuracy on the training data and impressive performance on the validation set. This suggests that the model might have identified clear linear boundaries among the features.

### KNN:
- Training Accuracy: 0.8472
- Validation Accuracy: 0.3542
The KNN model demonstrated satisfactory performance on the training data but showed a significant drop in accuracy on the validation set. This could imply that KNN isn't the best model for this type of data or the chosen hyperparameters are not optimal

***

</details>  

## <a name="conclusion"></a> 
![conclusion](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%2010.00.09%20AM.png?raw=true)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

The Disease Symptoms Prediction Model project aimed to leverage symptom data to predict potential diseases, assisting healthcare professionals and patients in preliminary diagnostics. This objective was approached through the development of machine learning models, utilizing a dataset encompassing a variety of diseases and their associated symptoms.

### Achievement of Goals:
- We successfully developed a machine learning model, with the Random Forest classifier emerging as the most accurate, significantly outperforming the baseline model.
The relationship between symptoms and diseases was elucidated through statistical analysis, confirming that certain symptoms such as abdominal pain and vomiting are strong indicators of specific conditions like Alcoholic Hepatitis and Chronic Cholestasis, respectively.
- A baseline model was established, providing a reference point for evaluating the effectiveness of more sophisticated predictive models.

### Key Findings:
- Statistical significance was identified between certain symptoms and diseases, validating the model's capability to capture these relationships.
- Symptom frequency analysis and N-gram visualizations provided deeper insights into common and distinctive symptom patterns.
The Random Forest model, with a test accuracy of 89.58%, was identified as the most promising predictive model in our trials.

### Recommendations:
- Due to its high validation and test accuracies, the Random Forest model is recommended for initial deployment in a controlled environment to gauge real-world efficacy.
- Collaboration with medical professionals is advised to interpret the model's predictions and to incorporate their feedback for refinement.

### Next Steps:
- Integration into User-Friendly Platforms: The next phase involves creating a user interface for the model, making it accessible to end-users who can report symptoms and receive disease predictions.
- Dataset Expansion: To improve the model's comprehensiveness and accuracy, we plan to include a broader range of diseases, especially rare conditions, to enhance predictive capabilities.
- Continuous Model Improvement: We aim to continuously refine the model by incorporating medical professional feedback and adjusting it according to the latest medical research and data.

### "If I Had More Time, I Would...":
- Explore Advanced Models: Experiment with deep learning and ensemble methods to potentially uncover complex patterns in symptom-disease relationships that simpler models might miss.
- Conduct a Thorough Hyperparameter Tuning: Allocate more time to fine-tune the models, especially KNN, to ensure that we are not overlooking a potentially suitable model due to suboptimal parameters.
- Implement a Feedback Loop: Develop a system to collect user and professional feedback on the model’s predictions to facilitate ongoing learning and improvement.
- Focus on Interpretability: Devote efforts to make the model's decision process more transparent, aiding healthcare professionals in understanding the rationale behind predictions, which is crucial for medical applications.

***
</details>  


## <a name="recreate"></a> 
![recreate](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%2010.53.45%20AM.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### 1. Getting started
- Start by cloning the github repository on the terminal, type: 
git clone git@github.com:disease-outbreak/disease-outbreak.git

- Install Conda, Python, VS Code or Jupyter Notebook
</details>
    
## <a name="team"></a>
![meet](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%2010.52.03%20AM.png?raw=true)

The Dream Team:

![team](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%201.29.42%20PM.png?raw=true)


>>>>>>>>>>>>>>>
.

</details>  


## <a name="interact"></a> 
![interact](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-09%20at%202.35.07%20PM.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

Interact with our model and data!
    
### Streamlit Application
- [Streamlit Application](https://hitzuman-disease-outbreak2-codeupcapstonestreamlit-sqfsjj.streamlit.app)

### Tableau Dashboard
- [Tableau Dashboard](https://public.tableau.com/app/profile/nisha.bista/viz/DiseaseMortality/Dashboard12?publish=yes)
