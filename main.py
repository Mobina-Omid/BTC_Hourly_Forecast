import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def get_timestamp_ms(date_str):
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return int(dt.timestamp() * 1000)

def get_btc_prices_binance(start_date, end_date):
    start_timestamp = get_timestamp_ms(start_date)
    end_timestamp = get_timestamp_ms(end_date)

    url = "https://api.binance.com/api/v3/klines"

    params = {
        'symbol': 'BTCUSDT',
        'interval': '1d',
        'startTime': start_timestamp,
        'endTime': end_timestamp
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)


def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    non_zero_actual = actual != 0
    mape = np.mean(np.abs((actual[non_zero_actual] - predicted[non_zero_actual]) / actual[non_zero_actual])) * 100
    return mape

start_date = '2024-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

btc_prices = get_btc_prices_binance(start_date, end_date)

if btc_prices:
    btc_prices = np.array([[float(price[4])] for price in btc_prices])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(btc_prices)

    look_back = 1

    train_size = int(len(scaled_data) * 0.8)
    train, test = scaled_data[:train_size], scaled_data[train_size:]

    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    if X_test.size > 0:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    else:
        print("X_test is empty. Adjust your look_back or check your data.")

    model = Sequential()
    model.add(Input(shape=(look_back, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    if X_train.size > 0:
        model.fit(X_train, Y_train, epochs=20, batch_size=1, verbose=2)
    else:
        print("Not enough training data to fit the model.")

    test_loss = model.evaluate(X_test, Y_test, verbose=2)
    print(f"Test Loss: {test_loss}")

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    Y_train_inv = scaler.inverse_transform([Y_train])
    Y_test_inv = scaler.inverse_transform([Y_test])

    train_dates = pd.date_range(start=start_date, periods=len(Y_train_inv[0]), freq='D')

    test_dates = pd.date_range(start=train_dates[-1] + pd.Timedelta(days=1), periods=len(Y_test_inv[0]), freq='D')

    train_diffs = train_predict.flatten() - Y_train_inv[0]
    test_diffs = test_predict.flatten() - Y_test_inv[0]

    plt.figure(figsize=(12, 6))
    plt.plot(train_dates, Y_train_inv[0], label='Actual Prices (Train)', color='blue')
    plt.plot(train_dates, train_predict, label='Predicted Prices (Train)', color='orange')
    plt.plot(test_dates, Y_test_inv[0], label='Actual Prices (Test)', color='green')
    plt.plot(test_dates, test_predict, label='Predicted Prices (Test)', color='red')

    for i in range(len(test_diffs)):
        if i % 10 == 0:
            plt.text(test_dates[i], test_predict[i], f"{test_diffs[i]:.2f}", fontsize=8, ha='center')

    for i in range(len(train_diffs)):
        if i % 10 == 0:
            plt.text(train_dates[i], train_predict[i], f"{train_diffs[i]:.2f}", fontsize=8, ha='center')

    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.title('Bitcoin Price Prediction')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    print("Differences between Actual and Predicted Prices:")
    print("Train Differences:")
    for date, actual, predicted, diff in zip(train_dates, Y_train_inv[0], train_predict.flatten(), train_diffs):
        print(f"{date.date()}: Actual = {actual:.2f}, Predicted = {predicted:.2f}, Difference = {diff:.2f}")

    print("\nTest Differences:")
    for date, actual, predicted, diff in zip(test_dates, Y_test_inv[0], test_predict.flatten(), test_diffs):
        print(f"{date.date()}: Actual = {actual:.2f}, Predicted = {predicted:.2f}, Difference = {diff:.2f}")

    mape_train = calculate_mape(Y_train_inv[0], train_predict.flatten())
    mape_test = calculate_mape(Y_test_inv[0], test_predict.flatten())
    print(f"\nTraining MAPE: {mape_train:.2f}%")
    print(f"Testing MAPE: {mape_test:.2f}%")

else:
    print("No data retrieved from the API.")
