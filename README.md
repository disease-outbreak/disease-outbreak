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
    
- Data was gathered from "The World Health Organization" website
    - https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7
- Other data from the following website to create a dashboard of mortality rates.
    - https://sa2020.org/city-council-profiles


### Data Dictionary
    
| Attribute | Definition | Data Type |
| ----- | ----- | ----- | 
| call_reason | The department division within the City deaprtment to whom the case is assigned. | object |
| case_status | The status of a case which is either open or closed. | object |
| case_type | The service request type name for the issue being reported. Examples include stray animals, potholes, overgrown yards, junk vehicles, traffic signal malfunctions, etc. | object |
| closed_date | The date and time that the case/request was was closed. If blank, the request has not been closed as of the Report Ending Date. | object |
| council_district | The Council District number from where the issue was reported. | int64 |
| days_before_or_after_due | How long before or after the due date were the cases closed | float64 |
| days_open | The number of days between a case being opened and closed. | float64 |
| dept | The City department to whom the case is assigned. | object |
| due_date | Every service request type has a due date assigned to the request, based on the request type name. The SLA Date is the due date and time for the request type based on the service level agreement (SLA). Each service request type has a timeframe in which it is scheduled to be addressed. | object |
| is_late | This indicates whether the case has surpassed its Service Level Agreement due date for the specific service request. | object |
| open_date | The date and time that a case was submitted. | object |
| open_month | Month of the year the case was made | int64 | 
| open_week | Week of the year the case was made | int64 | 
| open_year | The year the case was made | int64 | 
| pct_time_of_used | How much of the resolution_days_due was the case open? | float64 | 
| resolution_days_due | The number of days between a case being opened and due. | float64 |
| source_id | The source id is the method of input from which the case was received. | object |
    
\*  Indicates the target feature in this City of San Antonio data.

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
    - Formulate hypothesis
    - Where and how we will acquire data?
    - What specific features to move forward with?
    - Read the data into python environment
    - Save data in a file    
  
- Acquisiton of data:
    - Download CSV from the City of San Antonio website.
        - https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7 
    - Read data into python
    - Summarize data
    
- Prepare and clean data with python - Jupyter Labs: 
    - Potentially Drop features
    - Handle null values
    - Adjust data types
    - Rename columns
  
- Explore data:
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
- Data was gathered from "The World Health Organization" website
    - https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7
- Other data from the following website to create a dashboard of mortality rates.
    - https://sa2020.org/city-council-profiles
    
The dataset comprises various diseases and their associated symptoms. Each disease can have multiple symptoms, and each symptom can be associated with multiple diseases
    
- Dataset Structure
    - Disease and symptoms:
        - Rows: 4920
        - Columns: 18
    - Symptoms and Severity:
        - Rows: 133
        - Columns:2
    
### Prepare Data
- Convert symptom columns to consistent data types.
- One-hot encode each symptom column.
- Aggregate the one-hot encoded columns to remove duplicates.
- Construct the final dataset combining diseases and their respective symptoms.

***

</details>

## <a name="explore"></a> 
![dict](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%209.58.00%20AM.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>
    
### Split data:
- Training set (60%)
- Validation set (20%)
- Test set (20%)

### Statistical Analysis of Symptoms and Conditions:
  - A high statistical significance was found between abdominal pain and Alcoholic Hepatitis P less than .05 with a T-statistic of 21.049 and a P-value of approximately 2.996 x 10^-94, indicating a robust correlation.
  - Vomiting showed a strong statistical association with Chronic Cholestasis, evidenced by a T-statistic of 12.975 and a P-value around 7.03 x 10^-38, also significantly surpassing the standard significance threshold.

### Clinical Insights from Statistical Findings:
  - Abdominal pain as a symptom warrants consideration for Alcoholic Hepatitis diagnosis.
  - Vomiting is a significant indicator for Chronic Cholestasis and should be factored into diagnostic processes for this condition.

### Symptom Frequency Analysis:
  - "Fever" emerged as the most common symptom, highlighting its prevalence across various diseases.
  - Symptoms associated with respiratory issues and general malaise such as "cough," "headache," and "fatigue" are frequently reported.
  - Liver-related symptoms like "jaundice" were noted, suggesting liver issues are represented within the dataset.
  - Gastrointestinal symptoms such as "vomiting" and "diarrhea" are commonly reported, indicating the importance of digestive health in the dataset's scope.

### N-Gram Analysis Insights:
  - Bi-grams such as "loss of" and "high fever" are prevalent, reflecting common symptom descriptions or pairings.
  - Tri-grams like "loss of appetite" provide a more detailed picture of symptom patterns, with this particular tri-gram being the most frequent.
  - The bi-gram and tri-gram visualizations reveal common co-occurring symptoms, which can aid in symptom pattern recognition and possibly hint at underlying conditions.

### Overall Summary:
  - The analysis underlines the significance of fever as a common presenting symptom, which may be of interest for broader epidemiological studies.
  - The data underscores the importance of considering symptom patterns, such as abdominal pain and vomiting, in the clinical assessment and potential diagnosis of liver diseases.
  - The use of bi-grams and tri-grams has proven effective in identifying common symptom pairs and clusters within the dataset, offering valuable insights for healthcare professionals to refine their diagnostic criteria and for researchers to understand symptomatology better.

***   
</details>    

## <a name="model"></a> 
![model](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%2010.27.47%20AM.png?raw=true)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

Summary of modeling choices...
        
### Models Made:
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- Ridge Classifier
- SGD Classifier

### Baseline Accuracy  
- 57.199%
      
| Model | Accuracy with Train | Accuracy with Validate |
| ---- | ----| ---- | 
| Logistic Regression | 61.1% | 61% |
| Decision Tree | 68% | 68% |
| Random Forest | 66.6% | 66.4% |
| KNN | 57%  | 57% |
| Ridge Classifier | 59% | 59% |
| SGD Classifier | 56% | 56% |
    
    
## Selecting the Best Model:

- Decision Tree

- Why did we choose this model?
    - This model ran the best accross train and validate.
    
- What does this model do?
    - Decision trees are flexible models that don’t increase their number of parameters as we add more features (if we build them correctly). At each node of a decision tree, one of the features of our data is evaluated in order to make an specific data point follow a certain path when making a prediction.

### Model on All Data Sets

| Best Model | Accuracy with Train | Accuracy with Validate | Accuracy with Test|
| ---- | ----| ---- | ---- |
| Decision Tree | 68% | 68% | 68% |


***

</details>  

## <a name="conclusion"></a> 
![conclusion](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%2010.00.09%20AM.png?raw=true)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

**We found....**

- Each department is better in certain areas about being on time/early and late in others.
- The more calls a department had the better they were at getting issues resolved on time.
- Internal requests were generally late in comparison to other forms of reporting.
- When an issue was reported via the app, there were no extremely late responses.
- Customer Service generally got issues resolved late or very late. 
- Animal Services usually only gave a day to complete a case and those cases usually took months to close.
- Winter months tend to have the longest average days open time, while Autumn months have the shortest.

**With further time...**

- Overall extremely late responses are spread out throughout the city. There is a significant delay within calls listed as on time. Therefore, we would like to evaluate the amount of time between districts for calls that were considered on time. 
- Analyze the data further through time series analysis. Some questions that we would like to investigate are:
    - Do days of the week effect when the case was done?
    - Are Mondays the slowest days because of the weekend backlog?
    - Do minor holidays affect response time?
- Obtain census data to gain insight more into zip codes, neighborhoods, and demographics beyond just the large districts.
- Determine priority level for each call as a feature based on the number of days given and department to explore if there is a correlation with the level of delay.

**We recommend...**
  
- The City of San Antonio should create standardized timelines for each department to follow when solving cases.
- Animal Care Services and Customer Service should both have a thorough review of their cases and timelines to rectify latency issues.
- Late and extremely late cases should be investigated through all departments.
- The classification in the raw data set for whether a case was completed late or not needs to be re-made. This is due to an issue where this feature classifies cases as being late when they were completed as late. For example if a case was due in fifteen days but was completed a day before its due date, it would be classified as late.


</details>  


## <a name="recreate"></a> 
![recreate](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%2010.53.45%20AM.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### 1. Getting started
- Start by cloning the github repository on your From your terminal command line, type: 
git clone git@github.com:3-1-1-Codeup/project.git

- Download .CSV of Data from the link below and name it as service-calls.csv in your working directory:
https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7

- Use the wrangle.py, explore.py, and model.py to follow the processes we used.
    
Good luck I hope you enjoy your project!

</details>
    
## <a name="team"></a>
![meet](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%2010.52.03%20AM.png?raw=true)

The Dream Team:

![team](https://github.com/disease-outbreak/disease-outbreak/blob/main/Marcelino/Screenshot%202023-11-06%20at%201.29.42%20PM.png?raw=true)


>>>>>>>>>>>>>>>
.