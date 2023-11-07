import pandas as pd

from sklearn.utils import resample
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import joblib  # Import joblib



def encode_disease_column(train_df, val_df, test_df, column_name):
    """
    Encodes the specified column in the training data and applies the encoding to
    the validation and test data.

    :param train_df: DataFrame containing the training data.
    :param val_df: DataFrame containing the validation data.
    :param test_df: DataFrame containing the test data.
    :param column_name: The name of the column to be encoded.
    :return: A tuple of the modified DataFrames and the fitted LabelEncoder.
    """

    # Initialize the Label Encoder
    encoder = LabelEncoder()

    # Fit the encoder on the column of the training data
    train_df[column_name + '_encoded'] = encoder.fit_transform(train_df[column_name])

    # Transform the column of validation and test data using the fitted encoder
    val_df[column_name + '_encoded'] = encoder.transform(val_df[column_name])
    test_df[column_name + '_encoded'] = encoder.transform(test_df[column_name])

    # Return the modified DataFrames and the fitted LabelEncoder
    return train_df, val_df, test_df, encoder


def balance_data(data):
    # Identify the majority class
    majority_class = data['disease_encoded'].value_counts().idxmax()
    
    # Separate majority and minority classes
    data_majority = data[data.disease_encoded == majority_class]
    data_minority = data[data.disease_encoded != majority_class]
    
    # Upsample minority class
    data_minority_upsampled = resample(data_minority, 
                                       replace=True, 
                                       n_samples=len(data_majority),
                                       random_state=42)
    
    # Combine majority class with upsampled minority class
    data_balanced = pd.concat([data_majority, data_minority_upsampled])
    
    return data_balanced


def split_features_labels(balanced_train, balanced_val, balanced_test, label_col):
    """
    Splits the balanced datasets into features and labels.

    :param balanced_train: DataFrame containing the balanced training data.
    :param balanced_val: DataFrame containing the balanced validation data.
    :param balanced_test: DataFrame containing the balanced test data.
    :param label_col: The name of the column containing the labels.
    :return: A tuple of the features and labels for the train, validation, and test sets.
    """

    # Splitting balanced train dataset
    X_train = balanced_train.drop([label_col, label_col + '_encoded'], axis=1)
    y_train = balanced_train[label_col + '_encoded']

    # Splitting balanced validation dataset
    X_val = balanced_val.drop([label_col, label_col + '_encoded'], axis=1)
    y_val = balanced_val[label_col + '_encoded']

    # Splitting balanced test dataset
    X_test = balanced_test.drop([label_col, label_col + '_encoded'], axis=1)
    y_test = balanced_test[label_col + '_encoded']

    return X_train, y_train, X_val, y_val, X_test, y_test



def train_evaluate_baseline(X_train, y_train, X_val, y_val, strategy="most_frequent", random_state=42):
    """
    Trains a baseline classifier on the training data and evaluates it on the validation data.

    :param X_train: Feature matrix for the training data.
    :param y_train: Labels vector for the training data.
    :param X_val: Feature matrix for the validation data.
    :param y_val: Labels vector for the validation data.
    :param strategy: Strategy to use for the DummyClassifier. Default is "most_frequent".
    :param random_state: The random state for reproducibility. Default is 42.
    :return: The trained baseline classifier and the accuracy on the validation set.
    """

    # Initialize the baseline classifier with the specified strategy and random state
    clf_baseline = DummyClassifier(strategy=strategy, random_state=random_state)

    # Train the classifier on the training data
    clf_baseline.fit(X_train, y_train)

    # Predict on the validation set
    y_pred_baseline_val = clf_baseline.predict(X_val)

    # Compute and print the accuracy on the validation set
    accuracy_baseline = accuracy_score(y_val, y_pred_baseline_val)
    print(f"Baseline Accuracy: {accuracy_baseline:.4f}")

    # Return the classifier and accuracy
    return clf_baseline, accuracy_baseline




def train_evaluate_random_forest(X_train, y_train, X_val, y_val, random_state=42):
    """
    Trains a Random Forest classifier on the training data and evaluates it
    on both the training and validation data.

    :param X_train: Feature matrix for the training data.
    :param y_train: Labels vector for the training data.
    :param X_val: Feature matrix for the validation data.
    :param y_val: Labels vector for the validation data.
    :param random_state: The random state for reproducibility.
    :return: A dictionary with trained classifier and accuracies on training and validation sets.
    """

    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(random_state=random_state)

    # Fit the model to the training data
    clf.fit(X_train, y_train)

    # Predict and calculate accuracy on the training set
    y_pred_train = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"Training Accuracy (Random Forest): {train_accuracy:.4f}")

    # Predict and calculate accuracy on the validation set
    y_pred_val = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy (Random Forest): {val_accuracy:.4f}")

    # Return the trained classifier and accuracies
    return {
        'classifier': clf,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    }




def train_evaluate_logistic_regression(X_train, y_train, X_val, y_val, max_iter=10000, random_state=42):
    """
    Trains a Logistic Regression classifier on the training data and evaluates
    it on both the training and validation data.

    :param X_train: Feature matrix for the training data.
    :param y_train: Labels vector for the training data.
    :param X_val: Feature matrix for the validation data.
    :param y_val: Labels vector for the validation data.
    :param max_iter: Maximum number of iterations for solver to converge.
    :param random_state: The random state for reproducibility.
    :return: A dictionary with trained classifier, training accuracy, and validation accuracy.
    """

    # Initialize the Logistic Regression classifier with given parameters
    clf_lr = LogisticRegression(max_iter=max_iter, random_state=random_state)

    # Fit the model to the training data
    clf_lr.fit(X_train, y_train)

    # Predict and calculate accuracy on the training set
    y_pred_train_lr = clf_lr.predict(X_train)
    train_accuracy_lr = accuracy_score(y_train, y_pred_train_lr)
    print(f"Training Accuracy (Logistic Regression): {train_accuracy_lr:.4f}")

    # Predict and calculate accuracy on the validation set
    y_pred_val_lr = clf_lr.predict(X_val)
    val_accuracy_lr = accuracy_score(y_val, y_pred_val_lr)
    print(f"Validation Accuracy (Logistic Regression): {val_accuracy_lr:.4f}")

    # Return the trained classifier and accuracies
    return {
        'classifier': clf_lr,
        'train_accuracy': train_accuracy_lr,
        'val_accuracy': val_accuracy_lr
    }





def train_evaluate_knn(X_train, y_train, X_val, y_val, n_neighbors=5):
    """
    Trains a K-Nearest Neighbors classifier on the training data and evaluates
    it on both the training and validation data.

    :param X_train: Feature matrix for the training data.
    :param y_train: Labels vector for the training data.
    :param X_val: Feature matrix for the validation data.
    :param y_val: Labels vector for the validation data.
    :param n_neighbors: Number of neighbors to use by default for k-neighbors queries.
    :return: A dictionary with trained classifier, training accuracy, and validation accuracy.
    """

    # Initialize the KNN classifier with the specified number of neighbors
    clf_knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the classifier to the training data
    clf_knn.fit(X_train, y_train)

    # Predict and calculate accuracy on the training set
    y_pred_train_knn = clf_knn.predict(X_train)
    train_accuracy_knn = accuracy_score(y_train, y_pred_train_knn)
    print(f"Training Accuracy (KNN): {train_accuracy_knn:.4f}")

    # Predict and calculate accuracy on the validation set
    y_pred_val_knn = clf_knn.predict(X_val)
    val_accuracy_knn = accuracy_score(y_val, y_pred_val_knn)
    print(f"Validation Accuracy (KNN): {val_accuracy_knn:.4f}")

    # Return the trained classifier and accuracies
    return {
        'classifier': clf_knn,
        'train_accuracy': train_accuracy_knn,
        'val_accuracy': val_accuracy_knn
    }




def train_evaluate_random_forest_on_test(X_train, y_train, X_test, y_test, random_state=42):
    """
    Trains a Random Forest classifier on the training data and evaluates it on the test data.

    :param X_train: Feature matrix for the training data.
    :param y_train: Labels vector for the training data.
    :param X_test: Feature matrix for the test data.
    :param y_test: Labels vector for the test data.
    :param random_state: The random state for reproducibility.
    :return: A dictionary with the trained classifier and test accuracy.
    """

    # Initialize the Random Forest classifier with the given random state
    clf = RandomForestClassifier(random_state=random_state)

    # Fit the model to the training data
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred_test = clf.predict(X_test)
    
    # Calculate accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy (Random Forest): {test_accuracy:.4f}")
    
    #Save the trained classifier to a joblib file
    joblib.dump(clf, 'rf_model.sav')

    # Return the trained classifier and test accuracy
    return {
        'classifier': clf,
        'test_accuracy': test_accuracy
    }