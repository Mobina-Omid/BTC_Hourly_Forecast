# Bitcoin Price Prediction Using LSTM

This project demonstrates a Bitcoin price prediction model using Long Short-Term Memory (LSTM) neural networks. It retrieves historical Bitcoin data from Binance, pre-processes it, and trains an LSTM model to predict future prices. The model is evaluated based on Mean Absolute Percentage Error (MAPE) and plotted to compare actual vs predicted prices.

## Project Overview

The primary goal of this project is to predict Bitcoin prices using an LSTM-based deep learning model. The script performs the following key tasks:
- Retrieves historical price data from Binance's API.
- Pre-processes the data by normalizing it and splitting it into training and testing datasets.
- Builds an LSTM model using TensorFlow/Keras to predict future Bitcoin prices.
- Evaluates the model's performance based on testing data and calculates Mean Absolute Percentage Error (MAPE).
- Plots the actual vs predicted prices and shows the differences between them.

## Results
The model was trained and tested on historical Bitcoin price data, and its performance was evaluated using Mean Absolute Percentage Error (MAPE).

Training MAPE: 2.25%

Testing MAPE: 1.74%

Below is a sample plot showing the comparison between actual and predicted Bitcoin prices:

  ![BTC_Forecast](https://github.com/user-attachments/assets/6a0a62dc-fb21-4cf9-97dc-3e16e4f5b64c)


## Requirements

- Python 3.x

## Tech Stack:
**Programming Language:** Python

### Model Evaluation:
- Accuracy comparison between raw data and percentage-based data.

## How to Use:
1. **Run the application:** The project includes a Python script to launch the GUI.
2. **Enter passenger details:** Input features like age, fare, sex, pclass, etc.
3. **Get Prediction:** The model predicts the probability of survival based on user input.

