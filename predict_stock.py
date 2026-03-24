import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime

# 1. Configuration
TICKER = 'AAPL' # Yahoo Finance ticker
START_DATE = '2015-01-01'
END_DATE = datetime.date.today().strftime('%Y-%m-%d')
SEQUENCE_LENGTH = 60 # Number of past days to use for predicting the next day
EPOCHS = 20
BATCH_SIZE = 32

def load_data(ticker, start, end):
    print(f"Fetching data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end)
    # yfinance sometimes returns a multi-level column index. We flatten it or extract 'Close'.
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    else:
        df = df[['Close']]
    
    # Drop any potential NaN values
    df.dropna(inplace=True)
    return df

def preprocess_data(df, sequence_length):
    # We only care about the close price
    data = df.values
    
    # Split into training and testing before scaling to prevent data leakage (80% for training)
    train_size = int(len(data) * 0.8)
    
    # Scale the data to be between 0 and 1, fitting ONLY on the training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_size, :])
    
    # Transform the entire data based on the training data's min/max
    scaled_data = scaler.transform(data)
    
    train_data = scaled_data[:train_size, :]
    test_data = scaled_data[train_size - sequence_length:, :] # Include past sequence_length days for test set
    
    # Create sequences
    x_train, y_train = [], []
    for i in range(sequence_length, len(train_data)):
        x_train.append(train_data[i-sequence_length:i, 0])
        y_train.append(train_data[i, 0])
        
    x_test, y_test = [], []
    for i in range(sequence_length, len(test_data)):
        x_test.append(test_data[i-sequence_length:i, 0])
        y_test.append(test_data[i, 0]) # The actual scaled value
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # Reshape specifically for LSTM [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_train, y_train, x_test, y_test, scaler, train_size

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # 1. Load Data
    df_raw = load_data(TICKER, START_DATE, END_DATE)
    print(f"Data loaded successfully. Shape: {df_raw.shape}")
    
    # 2. Preprocess Data
    x_train, y_train, x_test, y_test, scaler, train_size = preprocess_data(df_raw, SEQUENCE_LENGTH)
    
    # 3. Build Model
    model = build_model((x_train.shape[1], 1))
    model.summary()
    
    # 4. Train Model
    print("Training model...")
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    # 5. Evaluate and Predict
    print("Making predictions...")
    predictions = model.predict(x_test)
    
    # Inverse transform to get actual prices instead of scaled values
    predictions = scaler.inverse_transform(predictions)
    # The actual test prices needed an unsqueeze basically to inverse transform
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(((predictions - y_test_unscaled) ** 2)))
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    
    # 6. Visualize the Results
    train = df_raw.iloc[:train_size].copy()
    valid = df_raw.iloc[train_size:].copy()
    valid['Predictions'] = predictions
    
    plt.figure(figsize=(16,8))
    plt.title(f'{TICKER} Stock Price Prediction using LSTM')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train.index, train.values, label='Training Data')
    plt.plot(valid.index, valid.values, label='Actual Price (Test)')
    plt.plot(valid.index, valid['Predictions'], label='Predicted Price')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    print("Saving plot to prediction_plot.png")
    plt.savefig("prediction_plot.png")
    
    # Also show the plot if running interactively
    # plt.show()
    print("Process Complete. Check prediction_plot.png.")

if __name__ == "__main__":
    main()
