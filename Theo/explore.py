from itertools import chain
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.stats import ttest_ind
import seaborn as sns
import re

def disease_symptom_correlation(df, disease, symptom, alpha=0.05):
    """
    Checks the correlation between a specific disease and symptom using independent T-test.
    
    Args:
    - df (pd.DataFrame): The input DataFrame.
    - disease (str): The specific disease to be analyzed.
    - symptom (str): The symptom to be correlated with the disease.
    - alpha (float): Significance level for the test. Default is 0.05.
    
    Returns:
    - str: Result interpretation of the T-test.
    """
    
    # Filter the data into two groups: the specified 'disease' and 'Other Diseases'
    disease_group = df[df['disease'] == disease]
    other_diseases_group = df[df['disease'] != disease]

    # Perform an independent T-test
    t_statistic, p_value = ttest_ind(disease_group[symptom], other_diseases_group[symptom])

    # Display the results
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)

    # Interpret the results
    if p_value < alpha:
        return "We reject the null hypothesis, they have a significant correlation."
    else:
        return "We fail to reject the null hypothesis, they don't have a significant correlation."



def check_correlation(df, disease, symptom, alpha=0.05):
    """
    Checks the correlation between a specific disease and symptom using independent T-test.
    
    Args:
    - df (pd.DataFrame): The input DataFrame.
    - disease (str): The specific disease to be analyzed.
    - symptom (str): The symptom to be correlated with the disease.
    - alpha (float): Significance level for the test. Default is 0.05.
    
    Returns:
    - str: Result interpretation of the T-test.
    """
    
    # Filter the data into two groups: the specified 'disease' and 'Other Diseases'
    disease_group = df[df['disease'] == disease]
    other_diseases_group = df[df['disease'] != disease]

    # Perform an independent T-test
    t_statistic, p_value = ttest_ind(disease_group[symptom], other_diseases_group[symptom])

    # Display the results
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)

    # Interpret the results
    if p_value < alpha:
        return f"We reject the null hypothesis. '{disease}' and '{symptom}' have a significant correlation."
    else:
        return f"We fail to reject the null hypothesis. '{disease}' and '{symptom}' don't have a significant correlation."

    
    
def format_y_label(label):
    # Replace underscores with spaces and capitalize the first letter
    formatted_label = label.replace("_", " ").capitalize()
    return formatted_label



def plot_disease_counts(df, n=10):
    """
    Plots the count of the top n diseases in the DataFrame.
    
    Args:
    - df (pd.DataFrame): The input DataFrame containing disease data.
    - n (int): The number of top diseases to be displayed. Default is 20.
    
    Returns:
    - None: Displays the bar plot.
    """
    
    disease_counts = df['disease'].value_counts().head(n)
    plt.figure(figsize=(12, 6))
    color = 'lightseagreen'

    bars = disease_counts.plot(kind='barh', color=color)

    plt.title("Disease Distribution")
    plt.ylabel("")
    plt.xlabel("")

    # Add values at the end of each bar
    for bar in bars.patches:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{int(width):.0f}', ha='left', va='center')
    
    sns.despine(left=True, bottom=True)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.tick_params(axis='x', which='both', bottom=False)
    plt.xticks([])
    plt.show()



def plot_symptom_frequency(df, n=10):
    """
    Plots the frequency of the top n symptoms in the DataFrame.
    
    Args:
    - df (pd.DataFrame): The input DataFrame containing symptom data.
    - n (int): The number of top symptoms to be displayed. Default is 20.
    
    Returns:
    - None: Displays the bar plot.
    """
    
    # Convert all non-zero entries to 1
    binary_df = df.drop(columns='disease').applymap(lambda x: 1 if x != 0 else 0)

    # Sum occurrences of each symptom
    symptom_counts = binary_df.sum()
    
    # Format the y-axis labels and store them in a list
    formatted_labels = [format_y_label(label) for label in symptom_counts.sort_values(ascending=True).tail(n).index]

    # Plot the top n symptoms based on their frequency
    plt.figure(figsize=(10, 6))
    color = 'lightseagreen'

    # Create the horizontal bar chart
    bars = symptom_counts.sort_values(ascending=True).tail(n).plot(kind='barh', color=color)

    plt.title("Symptom Frequency")
    plt.xlabel("")
    plt.ylabel("")

    # Add values at the end of each bar with spaces added
    for bar in bars.patches:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{int(width):.0f}', ha='left', va='center')

    # Set the formatted y-axis labels
    plt.yticks(range(n), formatted_labels)

    sns.despine(left=True, bottom=True)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.tick_params(axis='x', which='both', bottom=False)
    plt.xticks([])

    plt.show()

    return symptom_counts



def generate_wordcloud_from_symptoms(symptom_counts):
    """
    Generates and displays a word cloud based on symptom frequencies.
    
    Args:
    - symptom_counts (pd.Series): A series where the index is the symptom names and the values are their counts.
    
    Returns:
    - None: Displays the word cloud.
    """
    
    # Generate the word cloud using the symptom counts
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(symptom_counts)

    # Plot the word cloud image
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()



def generate_ngrams_from_df(df):
    """
    Generate and count bi-grams and tri-grams from a DataFrame containing symptom words.
    
    Args:
    - df (pd.DataFrame): A DataFrame containing symptom words. One of the columns must be named 'disease'.
    
    Returns:
    - bi_gram_counts (Counter): A counter object with bi-gram frequencies.
    - tri_gram_counts (Counter): A counter object with tri-gram frequencies.
    """
    
    # Drop the 'Disease' column and flatten the DataFrame to get a list of all symptom words
    all_symptoms = df.drop(columns=['disease']).values.flatten()

    # Convert all values to strings and filter out the string representation of NaN (to clean the data)
    all_symptoms = [str(s) for s in all_symptoms if str(s) != 'nan' and str(s) != 'NaN']

    # Tokenize the symptoms: split each symptom at underscores to get individual words
    tokens = list(chain.from_iterable([symptom.split('_') for symptom in all_symptoms]))

    # Generate bi-grams (2 word combinations) and tri-grams (3 word combinations) from the token list
    bi_grams = list(ngrams(tokens, 2))
    tri_grams = list(ngrams(tokens, 3))

    # Count frequencies of each bi-gram and tri-gram to identify most common combinations
    bi_gram_counts = Counter(bi_grams)
    tri_gram_counts = Counter(tri_grams)
    
    return bi_gram_counts, tri_gram_counts




def plot_top_n_bi_grams(bi_gram_counts, n=10):
    """
    This function sorts the bi-gram counts dictionary by values in descending order,
    and plots the top N bi-grams and their frequencies.

    :param bi_gram_counts: A dictionary with bi-grams as keys and their counts as values.
    :param n: The number of top bi-grams to plot.
    """

    # Sort the bi_gram_counts dictionary by values (frequencies) in descending order and take top N
    top_n_bi_grams = dict(sorted(bi_gram_counts.items(), key=lambda item: item[1], reverse=True)[:n])

    # Visualization of Top N Bi-grams
    plt.figure(figsize=(10, 5))
    plt.barh(['_'.join(bg) for bg in top_n_bi_grams.keys()], top_n_bi_grams.values())
    plt.xlabel('Bi-grams')
    plt.ylabel('Frequency')
    plt.title(f'Top {n} Bi-gram Frequencies')
    plt.tight_layout()
    plt.show()




def plot_top_n_tri_grams(tri_gram_counts, n=10):
    """
    This function sorts the tri-gram counts dictionary by values in descending order,
    and plots the top N tri-grams and their frequencies using a horizontal bar chart.

    :param tri_gram_counts: A dictionary with tri-grams as keys and their counts as values.
    :param n: The number of top tri-grams to plot, default is 20.
    """

    # Sort the tri_gram_counts dictionary by values (frequencies) in descending order and take top N
    top_n_tri_grams = dict(sorted(tri_gram_counts.items(), key=lambda item: item[1], reverse=True)[:n])
    
    formatted_labels = [format_y_label('_'.join(tg)) for tg in list(reversed(list(top_n_tri_grams.keys())))]

    # Visualization of Top N Tri-grams (sorted from biggest to smallest) from top to bottom
    plt.figure(figsize=(10, 6))
    bars = plt.barh(['_'.join(tg) for tg in list(reversed(list(top_n_tri_grams.keys())))],
                   list(reversed(list(top_n_tri_grams.values()))), height = .5, color='lightseagreen')
    plt.xlabel('')
    plt.ylabel('')
    # Set the formatted y-axis labels
    plt.yticks(range(n), formatted_labels)
    sns.despine(left=True, bottom=True)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.tick_params(axis='x', which='both', bottom=False)
    plt.xticks([])
    plt.title(f'Top {n} Tri-gram Frequencies')
    plt.tight_layout()  # This will often make labels less likely to be cut off

    # Add values at the end of each bar
    for bar, value in zip(bars, list(reversed(list(top_n_tri_grams.values())))):
        plt.text(value, bar.get_y() + bar.get_height() / 2, f'{value}', ha='left', va='center')

    plt.show()













