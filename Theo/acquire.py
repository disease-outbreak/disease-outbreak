import os
import requests
import pandas as pd

from bs4 import BeautifulSoup


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




def scrape_disease_symptoms(url):
    """
    Scrape the disease-symptom data from the given URL, if a CSV with the data doesn't already exist.

    Parameters:
    url (str): The webpage URL to scrape the data from.
    csv_filename (str): The name of the CSV file to check or save the data to.

    Returns:
    DataFrame: A pandas DataFrame containing the scraped data.
    """
    csv_filename = 'disease_symptoms.csv'
    # Check if the CSV file already exists
    if os.path.exists(csv_filename):
        return pd.read_csv(csv_filename)

    # Send a GET request to the webpage
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table you want to scrape
    table = soup.find('table')
    if not table:
        print("Could not find the table in the webpage.")
        return None

    # Initialize lists to store the scraped data
    diseases = []
    counts = []
    symptoms = []

    # Iterate over each row in the table, skipping the header row
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if len(cells) >= 3:
            disease = cells[0].text.strip()
            count = cells[1].text.strip()
            symptom = cells[2].text.strip()

            diseases.append(disease)
            counts.append(count)
            symptoms.append(symptom)

    # Store the data in a DataFrame
    data = pd.DataFrame({
        'Disease': diseases,
        'Count of Disease Occurrence': counts,
        'Symptoms': symptoms
    })

    # Save the data to a CSV file
    data.to_csv(csv_filename, index=False)
    
    return data


