import pandas as pd

def load_and_preview_datasets(dataset_path, symptom_severity_path):
    """
    Loads two datasets and previews their shapes and first few rows.

    :param dataset_path: The file path to the first dataset.
    :param symptom_severity_path: The file path to the second dataset.
    :return: A tuple containing dataframes loaded from the two datasets, 
             their shapes, and the head of the dataframes.
    """

    # Load the datasets
    df = pd.read_csv(dataset_path)
    df1 = pd.read_csv(symptom_severity_path)

    # Return the dataframes, their shapes, and their first few rows
    return (df, df1)