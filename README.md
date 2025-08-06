# Stock Market Prediction and Analysis   

A comprehensive machine learning project for analyzing and predicting stock prices of major tech companies using time series forecasting techniques.  

### Project Overview  

This project performs detailed analysis and prediction on stock market data for Apple (AAPL), Google (GOOG), Microsoft (MSFT), and Amazon (AMZN) using two different approaches:
- **ARIMA** for time series forecasting 
- **LSTM Neural Networks** for deep learning-based prediction

###  Analysis Components

 1. **Exploratory Data Analysis**
- Closing price trends and patterns
- Daily return calculations and distributions
- Moving averages (10, 20, 50-day periods)
- Volume analysis and trading patterns

 2. **Risk Assessment**
- Risk-return tradeoff analysis
- Correlation matrix between different stocks
- Volatility measurements using standard deviation
- Investment risk profiling

 3. **Time Series Forecasting (ARIMA)**
- Seasonal decomposition of Google stock prices
- ARIMA model implementation with parameters (10,1,2)
- Statistical validation and residual analysis
- 20-day forward price predictions

 4. **Deep Learning Prediction (LSTM)**
- LSTM neural network for Amazon stock prediction
- Feature scaling and data preprocessing
- 60-day lookback window for pattern recognition
- Model validation with RMSE evaluation

###  Technologies Used

- **Python Libraries**: pandas, numpy, matplotlib, seaborn
- **Data Source**: Yahoo Finance API (yfinance)
- **Statistical Modeling**: statsmodels (ARIMA)
- **Deep Learning**: TensorFlow/Keras (LSTM)
- **Preprocessing**: scikit-learn (MinMaxScaler)

###  Key Findings

- **Risk Analysis**: MSFT shows lower risk with stable returns, ideal for conservative investors
- **Correlation**: Strong positive correlation observed between tech stocks
- **ARIMA Performance**: Successfully captures trend and seasonal patterns in Google stock
- **LSTM Results**: Amazon stock predictions with good trend following

##  Model Performance

- **ARIMA Model**: Effective for short-term trend forecasting with statistical significance
- **LSTM Model**: Captures complex patterns with decreasing accuracy over longer prediction horizons
- Both models successfully identify general market direction and underlying trends

##  Future Improvements

- Implement ensemble methods combining ARIMA and LSTM
- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Extend analysis to more stocks and sectors
- Incorporate external factors (news sentiment, economic indicators)


