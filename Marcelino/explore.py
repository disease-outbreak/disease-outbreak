import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def symptoms(df, target ):
    
    df2 = df.drop(columns= 'disease')

    symptoms = []  

    for c in df2.columns:

        s = df[df['disease'] == target][c].values

        symptoms.append(s)
    
    s_list = [element for subarray in symptoms for element in subarray]
    
    return pd.Series(s_list)


def symptom_dist(df, target):
    # Create the bar plot with value counts
    s = symptoms(df, target)
    value_counts = s.value_counts()
    
    # Plot the horizontal bar chart
    ax = value_counts.plot(kind='barh', color='lightseagreen')
    
    # Add the total count on the right of each bar
    for i, v in enumerate(value_counts):
        ax.text(v + 0.2, i, str(v), ha='left', va='center')
    
    plt.title(f'Value Counts of Symptoms for {target}')
    plt.ylabel('')
    plt.xticks([])
    plt.xlabel('')
    plt.yticks(rotation=0)
    sns.despine(left=True, bottom=True)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    
    plt.show()