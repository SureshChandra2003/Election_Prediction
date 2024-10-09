# naive_bayes_model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from data_preprocessing import preprocess_data

def train_naive_bayes(X_train, y_train, X_test, y_test):
    # Convert text data to feature vectors
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Initialize and train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    # Predict on test data
    y_pred = model.predict(X_test_vectorized)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Naive Bayes Model Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data('social_media_data.csv')
    train_naive_bayes(X_train, y_train, X_test, y_test)