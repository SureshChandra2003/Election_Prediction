# lstm_model.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from data_preprocessing import preprocess_data

def train_lstm(X_train, y_train, X_test, y_test):
    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Pad sequences
    max_length = max(len(seq) for seq in X_train_seq)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

    # Build the LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_pad, y_train, epochs=5, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f'LSTM Model Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data('social_media_data.csv')
    train_lstm(X_train, y_train, X_test, y_test)