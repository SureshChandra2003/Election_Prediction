# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Basic cleaning (remove missing values, if any)
    data.dropna(inplace=True)

    # Encode sentiment labels to numeric
    label_encoder = LabelEncoder()
    data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

    # Split data into features and labels
    X = data['text']
    y = data['sentiment']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test