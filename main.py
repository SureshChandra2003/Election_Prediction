from naive_bayes_model import train_naive_bayes
from lstm_model import train_lstm
from data_preprocessing import preprocess_data  # Import the preprocess_data function

# Main function to run both models
if __name__ == "__main__":
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data('social_media_data.csv')
    
    print("Running Naive Bayes Model:")
    train_naive_bayes(X_train, y_train, X_test, y_test)  # Pass the data to the function
    
    print("\nRunning LSTM Model:")
    train_lstm(X_train, y_train, X_test, y_test)  # Pass the data to the function