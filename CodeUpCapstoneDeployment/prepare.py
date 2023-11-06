import pandas as pd

from sklearn.model_selection import train_test_split


def process_dataframe(df):
    """
    Process the dataframe by:
    - Converting all column names to lowercase.
    - Converting symptom columns to string type for consistent data type.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The processed dataframe.
    """
    
    # Convert column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Convert symptom columns to string type for consistent data type
    df[df.columns[1:]] = df[df.columns[1:]].astype(str)
    
    return df


def one_hot_encode_symptoms(df):
    """
    One-hot encodes the symptom columns of the input DataFrame.

    Parameters:
    - df (pd.DataFrame): Input dataframe with symptom columns starting from the second column.

    Returns:
    - pd.DataFrame: A dataframe with one-hot encoded symptom columns.
    """
    
    # One-hot encode each symptom column
    encoded_list = [pd.get_dummies(df[col], prefix="", prefix_sep="") for col in df.columns[1:]]
    encoded_df = pd.concat(encoded_list, axis=1)
    
    return encoded_df



def aggregate_encoded_columns(df, encoded_df):
    """
    Aggregates the one-hot encoded columns to remove duplicates, then concatenates the 
    Disease column with the aggregated symptoms. Finally, it drops the 'nan' column.

    Parameters:
    - df (pd.DataFrame): The original dataframe containing the 'disease' column.
    - encoded_df (pd.DataFrame): The one-hot encoded dataframe.

    Returns:
    - pd.DataFrame: A dataframe with aggregated symptom columns and the 'disease' column.
    """
    
    # Aggregate the one-hot encoded columns to remove duplicates
    encoded_df = encoded_df.groupby(encoded_df.columns, axis=1).sum().clip(upper=1)

    # Concatenate the Disease column with the aggregated symptoms
    df_final = pd.concat([df["disease"], encoded_df], axis=1)

    # Drop the 'nan' column that resulted from the NaN values in the original dataset
    df_final = df_final.drop(columns='nan')
    
    return df_final



def compare_symptoms(df_final, df1):
    """
    Compares symptoms present in the final dataframe 'df_final' and the given dataframe 'df1'.
    It prints the symptoms that are missing in either dataframe.

    Parameters:
    - df_final (pd.DataFrame): The final dataframe containing aggregated symptoms.
    - df1 (pd.DataFrame): The reference dataframe to compare against.

    Returns:
    - tuple: Two sets containing missing symptoms in df1 and df_final respectively.
    """
    
    # Normalize the columns by stripping spaces and converting to lowercase
    df_final.columns = [col.strip().lower() for col in df_final.columns]

    # Convert the lists to sets
    set_symptoms_df_final = set(list(df_final.columns.sort_values()))
    set_symptoms_df1 = set(list(df1.Symptom.sort_values()))

    # Find symptoms present in df_final but not in df1
    missing_in_df1 = set_symptoms_df_final - set_symptoms_df1
    if missing_in_df1:
        print("Symptoms present in df_final but not in df1:", missing_in_df1)

    # Find symptoms present in df1 but not in df_final
    missing_in_df_final = set_symptoms_df1 - set_symptoms_df_final
    if missing_in_df_final:
        print("Symptoms present in df1 but not in df_final:", missing_in_df_final)
    
    return missing_in_df1, missing_in_df_final



def correct_column_names(df_final, df1, missing_in_df1, missing_in_df_final):
    """
    Corrects column names in df_final based on the disparities between missing_in_df1 and missing_in_df_final,
    and replaces 1s in df_final columns with the corresponding weights.
    
    Args:
    - df_final (pd.DataFrame): DataFrame whose columns need to be corrected.
    - df1 (pd.DataFrame): Reference DataFrame containing Symptom-Weight pairs.
    - missing_in_df1 (set): Set of column names present in df_final but not in df1.
    - missing_in_df_final (set): Set of column names present in df1 but not in df_final.
    
    Returns:
    - pd.DataFrame: DataFrame with corrected column names and weights applied.
    """
    
    # Remove 'disease' from missing_in_df1 and 'prognosis' from missing_in_df_final
    if 'disease' in missing_in_df1:
        missing_in_df1.remove('disease')
    if 'prognosis' in missing_in_df_final:
        missing_in_df_final.remove('prognosis')

    # Zip together the two sets and create a dictionary from the pairs
    disparity_mapping = dict(zip(missing_in_df1, missing_in_df_final))

    # Correct column names in df_final using the disparity_mapping
    df_final.columns = [disparity_mapping[col] if col in disparity_mapping else col for col in df_final.columns]
    
    # Convert Symptom-Weight pairs in df1 to dictionary
    weights_dict = df1.set_index('Symptom')['weight'].to_dict()

    # Replace 1s in df_final columns with the corresponding weights
    for symptom, weight in weights_dict.items():
        if symptom in df_final.columns:
            df_final[symptom] = df_final[symptom].replace(1, weight)
    
    return df_final




def stratified_data_split(df, test_size=0.2, validation_size=0.2, random_state=42):
    """
    Split the data into training, validation, and test sets using stratification.
    
    Args:
    - df (pd.DataFrame): The input DataFrame.
    - test_size (float): Proportion of data to include in the test set.
    - validation_size (float): Proportion of data to include in the validation set.
    - random_state (int): The seed used by the random number generator.
    
    Returns:
    - tuple: (train, val, test) DataFrames.
    """
    
    # Calculate intermediate test size for correct proportions
    intermediate_test_size = test_size / (test_size + validation_size)
    
    # Split the data into training (remainder) and a temporary set (test_size + validation_size)
    train, temp = train_test_split(df, test_size=test_size + validation_size, stratify=df['disease'], random_state=random_state)

    # Split the temporary set into validation and test sets with stratification
    val, test = train_test_split(temp, test_size=intermediate_test_size, stratify=temp['disease'], random_state=random_state)

    return train, val, test