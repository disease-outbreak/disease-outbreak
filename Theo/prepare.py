import re
import numpy as np
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




def stratified_data_split(combined_disease_df, test_size=0.2, validation_size=0.2, random_state=42):
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
    df = combined_disease_df.groupby('disease').filter(lambda x: len(x) > 1)
    
    # Calculate intermediate test size for correct proportions
    intermediate_test_size = test_size / (test_size + validation_size)
    
    # Split the data into training (remainder) and a temporary set (test_size + validation_size)
    train, temp = train_test_split(df, test_size=test_size + validation_size, stratify=df['disease'], random_state=random_state)

    # Split the temporary set into validation and test sets with stratification
    val, test = train_test_split(temp, test_size=intermediate_test_size, stratify=temp['disease'], random_state=random_state)

    return train, val, test


def clean_umls_codes(df):
    disease_col='Disease' 
    symptoms_col='Symptoms'
    def remove_umls_codes(s):
        if isinstance(s, str):  # Only apply the regex if s is a string
            return re.sub(r'UMLS:[A-Za-z0-9_]+_', '', s)
        else:
            return s  # Return as is if not a string

    # Apply the function to the 'Disease' column, ensure it's a string first
    df[disease_col] = df[disease_col].astype(str).apply(remove_umls_codes)

    # Apply the function to each symptom in the 'Symptoms' column, ensure they are strings
    df[symptoms_col] = df[symptoms_col].apply(lambda x: ', '.join(remove_umls_codes(str(symptom)) for symptom in str(x).split('\n')))

    return df



def clean_disease_name(name):
    """
    Cleans up disease names by normalizing white spaces and formatting possessive forms.

    Parameters:
    name (str): The disease name to be cleaned.

    Returns:
    str: The cleaned disease name.
    """
    name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces/newlines with a single space
    name = re.sub(r"(\w)'s\b", r"\1's_", name)  # Add an underscore after possessives
    name = re.sub(r'\b_', '', name)  # Remove any underscores at the beginning of a word
    name = re.sub(r'_\b', '', name)  # Remove any underscores at the end of a word
    return name.strip()

def process_dataframes(df):
    """
    Processes a DataFrame by cleaning disease names, exploding symptoms into separate rows,
    creating a pivot table, and ensuring column names are properly formatted.

    Parameters:
    df (DataFrame): The input DataFrame with a 'Disease' column and a 'Symptoms' column.

    Returns:
    DataFrame: A processed DataFrame with diseases as index and symptoms spread across columns.
    """
    # Clean the 'Disease' column
    df['Disease'] = df['Disease'].apply(clean_disease_name)

    # Explode the 'Symptoms' column
    df = df.explode('Symptoms')

    # Create a pivot table
    pivot_df = df.pivot_table(index=['Disease', 'Count of Disease Occurrence'], 
                              columns='Symptoms', 
                              aggfunc=lambda x: 'Present', fill_value=0)

    # Flatten the MultiIndex in columns if necessary
    if isinstance(pivot_df.columns, pd.MultiIndex):
        pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    else:
        pivot_df.columns = pivot_df.columns.str.replace(' ', '_')

    # Reset index to make 'Disease' and 'Count of Disease Occurrence' into columns again
    pivot_df.reset_index(inplace=True)

    # Ensure the column names are properly formatted
    pivot_df.columns = [clean_disease_name(col) for col in pivot_df.columns]
    
    return pivot_df



def remove_duplicate_columns(df):
    """
    Removes duplicate columns from a DataFrame by cleaning up the column names and
    keeping only the unique ones.

    Parameters:
    df (DataFrame): The input DataFrame with potentially duplicated columns due to punctuation.

    Returns:
    DataFrame: A DataFrame with unique columns.
    """
    original_columns = df.columns

    # Step 1: Create a mapping of columns without punctuation to the original column names
    column_mapping = {}
    for col in original_columns:
        cleaned_col = col.replace(',', '').replace('_', ' ')
        if cleaned_col not in column_mapping:
            column_mapping[cleaned_col] = [col]
        else:
            column_mapping[cleaned_col].append(col)

    # Step 2: From each list of duplicates, keep only the first occurrence
    unique_columns = []
    for cleaned_col, cols in column_mapping.items():
        # Sort the list to ensure consistency, for example, if you want to keep the one without punctuation as priority
        cols_sorted = sorted(cols, key=lambda x: (x.count('_') + x.count(','), x))
        unique_columns.append(cols_sorted[0])  # append the first occurrence after sorting

    # Step 3: Reindex the dataframe with the unique columns only
    pivot_df_unique = df[unique_columns]

    return pivot_df_unique



def convert_columns_to_numeric(df, exclude_column):
    """
    Converts columns in a DataFrame to numeric types, excluding a specified column.

    Parameters:
    df (DataFrame): The DataFrame whose columns are to be converted.
    exclude_column (str): The name of the column to exclude from conversion.

    Returns:
    DataFrame: The DataFrame with the specified columns converted to numeric types.
    """
    # Get a list of column names, excluding the specified column
    numeric_columns = [col for col in df.columns if col != exclude_column]
    
    # Perform the numeric conversion on the selected columns
    converted_df = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # If the shapes match, proceed with assignment
    if converted_df.shape == df[numeric_columns].shape:
        # Assign the converted numeric values back to the original dataframe
        df.loc[:, numeric_columns] = converted_df
        print("Conversion successful.")
    else:
        # If shapes are not the same, something went wrong with the conversion
        print("The DataFrames do not match in shape. Cannot assign.")

    return df



def prepare_pivot_df(pivot_df):
    """
    Processes the pivot DataFrame by splitting the 'Disease' entries, converting column names to lowercase,
    and filling NaNs.
    
    Parameters:
    pivot_df (DataFrame): The DataFrame to be processed.
    
    Returns:
    DataFrame: A processed DataFrame with the specified transformations applied.
    """
    pivot_df.columns = pivot_df.columns.str.lower()
    # Only apply split if the entry is a string, otherwise keep the original value.
    pivot_df['disease'] = pivot_df['disease'].apply(lambda x: x.split('^')[-1] if isinstance(x, str) else x)
    
    # Convert column names to lowercase
    pivot_df.columns = pivot_df.columns.str.lower()
    
    # Fill NaN values with 0
    pivot_df = pivot_df.fillna(0)
    
    # Ensure the 'Count of Disease Occurrence' column is of type int, but do not repeat indices
    pivot_df['count of disease occurrence'] = pivot_df['count of disease occurrence'].astype(int)
    
    # Simply return the pivot_df without repeating rows
    return pivot_df



def merge_and_prepare_dfs(repeated_df, df_final):
    """
    Renames columns in the first DataFrame according to a mapping dictionary, converts data types,
    ensures index consistency, merges with the second DataFrame on 'disease', and handles duplicate columns.
    
    Parameters:
    repeated_df (DataFrame): The first DataFrame to be merged.
    df_final (DataFrame): The second DataFrame to be merged.
    columns_mapping (dict): A dictionary mapping old column names to new ones for the first DataFrame.
    
    Returns:
    DataFrame: The merged DataFrame with combined information and cleaned-up columns.
    """

    # Dictionary to map pivot_df columns to df_final columns based on your suggestions
    columns_mapping = {
    'diarrhea': 'diarrhoea',              # Spelling difference (American vs. British)
    'haemorrhage': 'bleeding',            # Clinical term vs. common term
    'shortness_of_breath': 'dyspnea',     # Synonyms
    'pain_abdominal': 'abdominal_pain',   # Reversal of words
    'vomiting': 'nausea_and_vomiting',               # 'vomiting' and 'nausea' might not be exact synonyms
    'haematuria': 'blood_in_urine',       # British spelling and clinical term vs. common term
    'fever': 'high_fever',                # General term vs. specific term
    'arthralgia': 'joint_pain', 
    'chill': 'chills',
    'fever': 'high_fever',
    'shortness,__of_breath': 'breathlessness',
    'dyspnea': 'breathlessness',
    'worry': 'anxiety'
    }
    # Rename the columns in repeated_df according to the mapping provided
    repeated_df = repeated_df.rename(columns=columns_mapping)

    # Convert data types
    repeated_df = repeated_df.convert_dtypes()
    df_final = df_final.convert_dtypes()

    # Ensure the indices are of the same type
    repeated_df.index = repeated_df.index.astype(int)
    df_final.index = df_final.index.astype(int)

    # Merge the DataFrames
    merged_df = pd.merge(repeated_df, df_final, on='disease', how='outer', suffixes=('', '_df_final'))
    
    # Drop the specific '_df_final' column as mentioned
    merged_df.drop(columns=['breathlessness_df_final'], inplace=True)

    # Handle duplicate columns by summing their values and dropping the duplicates
    for column in merged_df.columns:
        if column.endswith('_df_final'):
            original_column = column.replace('_df_final', '')
            # Ensure that you're operating on numerical columns only.
            if pd.api.types.is_numeric_dtype(merged_df[original_column]) and pd.api.types.is_numeric_dtype(merged_df[column]):
                merged_df[original_column] = merged_df[original_column].fillna(0) + merged_df[column].fillna(0)
                # Drop the _df_final column after summing
                merged_df.drop(columns=[column], inplace=True)
            else:
                print(f"Non-numeric column found: {column}")

    merged_df = merged_df.fillna(0)
    
    return merged_df