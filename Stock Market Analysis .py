#!/usr/bin/env python
# coding: utf-8

# **STOCK MARKET PREDICTION AND ANALYSIS** 
# 
# 1. Stocks from Apple, Amazon, Google, and Microsoft are explored (closing prices, daily return, moving average). 
# 2. Correlation between stocks is observed. 
# 3. Risk of investing in a particular stock is measured.  
# 4. Time Series forecasting is done using ARIMA for Google Stocks.
# 5. Future stock prices are predicted through Long Short Term Memory (LSTM) method. 

# In[1]:


get_ipython().system('pip install yfinance pandas_datareader')


# In[2]:


get_ipython().system('pip install --upgrade yfinance')


# In[3]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')

# Reading stock data from Yahoo Finance
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime

stock_data = {}

# Stocks used for this analysis
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']


end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)


for stock in tech_list:
    stock_data[stock] = yf.download(stock, start=start, end=end)


AAPL = stock_data['AAPL']
GOOG = stock_data['GOOG']
MSFT = stock_data['MSFT']
AMZN = stock_data['AMZN']


company_list = [AAPL, GOOG, MSFT, AMZN]
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]


for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name

df = pd.concat(company_list, axis=0)

print(df.tail(10))


# In[4]:


df.head(10)


# In[5]:


# checking if data is downloaded correctly
for ticker in tech_list:
    print(f"{ticker} data:\n", stock_data[ticker].head(), "\n")


# In[6]:


# Checking if 'Adj Close' exists
for ticker in tech_list:
    print(f"{ticker} columns:\n", stock_data[ticker].columns, "\n")


# **Closing Price :**  
# The closing price is also referred to as “close”. Essentially it is the final traded price of a financial asset at the end of a trading day or a trading session. 

# In[7]:


#Closing Price
plt.figure(figsize=(10, 10))

stock_data['AAPL']['Adj Close'].plot()
plt.title("AAPL Adjusted Close")
plt.show()

stock_data['GOOG']['Adj Close'].plot()
plt.title("GOOG Adjusted Close")
plt.show()

stock_data['MSFT']['Adj Close'].plot()
plt.title("MSFT Adjusted Close")
plt.show()

stock_data['AMZN']['Adj Close'].plot()
plt.title("AMZN Adjusted Close")
plt.show()



# In[8]:


closing_df = pd.DataFrame()

for stock in tech_list:
    closing_df[stock] = stock_data[stock]['Adj Close']

tech_rets = closing_df.pct_change()

tech_rets.head()


# In[9]:


rets = tech_rets.dropna()

area = np.pi * 20

plt.figure(figsize=(8, 8))

for label in rets.columns:
    plt.scatter(rets[label].mean(), rets[label].std(), s=area, label=label)

plt.xlabel('Expected Return')
plt.ylabel('Risk (Standard Deviation)')

plt.legend(title='Stocks')

plt.show()


# Risk-Return Tradeoff : Higher is expected return, more is the risk for the stocks.
# MSFT shows low risks and potentially low returns ideal for risk averse investors. 

# In[10]:


# compare the daily percentage return of two stocks to check correlation
sns.jointplot(x='AMZN', y='GOOG', data=tech_rets, kind='scatter', color='purple')

# Comparison Analysis for all combinations
sns.pairplot(tech_rets, kind='reg')


# 1. Each histogram shows rougly a bell curved shape, while AMZN stocks are normally distributed. 
# 2. A positive correlation is observed amongst most pairs. Slightly weaker correlations may exist for certain pairs, but none show negative or no correlation.
# 3. The regression lines in the scatter plots indicate linear relationships between the pairs of stocks. This suggests that when one stock’s return increases, the others tend to increase as well.
# 4. Stocks like GOOG and AMZN may exhibit higher dispersion (greater volatility) compared to AAPL and MSFT.

# In[11]:


#Volume of Sales
plt.figure(figsize=(10, 10))

company['Volume'].plot()
plt.ylabel('Volume')
plt.xlabel(None)
plt.title(f"Sales Volume for {AAPL} ")
    
plt.tight_layout()


# **Moving average** is calculated to analyze data points by creating a series of averages from different subsets of the full data set. In finance, it is commonly used to smooth out short-term fluctuations in stock prices or other data to reveal long-term trends. 
# 
# Simple moving averages (SMAs) use a simple arithmetic average of prices over some timespan, while exponential moving averages (EMAs) place greater weight on more recent prices than older ones over the time period.

# In[12]:


#Moving Average
ma_day = [10, 20, 50]

for ma in ma_day:
    for company in company_list:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Adj Close'].rolling(ma).mean()
        

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0])
axes[0,0].set_title('APPLE')

GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1])
axes[0,1].set_title('GOOGLE')

MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0])
axes[1,0].set_title('MICROSOFT')

AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1])
axes[1,1].set_title('AMAZON')


# In[13]:


#daily return for stocks
for company in company_list:
    company['Daily Return'] = company['Adj Close'].pct_change()

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

AAPL['Daily Return'].plot(ax=axes[0,0], legend=True, linestyle='--', marker='o')
axes[0,0].set_title('APPLE')

GOOG['Daily Return'].plot(ax=axes[0,1], legend=True, linestyle='--', marker='o')
axes[0,1].set_title('GOOGLE')

MSFT['Daily Return'].plot(ax=axes[1,0], legend=True, linestyle='--', marker='o')
axes[1,0].set_title('MICROSOFT')

AMZN['Daily Return'].plot(ax=axes[1,1], legend=True, linestyle='--', marker='o')
axes[1,1].set_title('AMAZON')


# In[14]:


plt.figure(figsize=(12, 9))

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Daily Return'].hist(bins=50, color='green')
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(f'{company_name[i - 1]}')
    
plt.tight_layout()


# In[15]:


plt.figure(figsize=(12, 10))

#correlation of stock return
plt.subplot(2, 2, 1)
sns.heatmap(tech_rets.corr(), annot=True, cmap='ocean')
plt.title('Correlation of stock return')

#correlation of stock closing price
plt.subplot(2, 2, 2)
sns.heatmap(closing_df.corr(), annot=True, cmap='ocean')
plt.title('Correlation of stock closing price')


# **TIME SERIES FORECASTING USING ARIMA FOR GOOGLE STOCK PRICES**

# In[16]:


import datetime
from datetime import date, timedelta
today = date.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('GOOG', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
print(data.tail())


# In[17]:


data = data[["Date", "Close"]]
print(data.head())


# In[18]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(15, 10))
plt.plot(data["Date"], data["Close"])


# In[19]:


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data["Close"], 
                            model='multiplicative', period = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(15, 10)


# 1. The overall price has been increasing over time.
# 2. There are recurring cyclical patterns in the price, likely due to daily, weekly, or monthly factors.
# 3. There might be additional factors influencing the price that are not captured by the trend or seasonality components.

# In[20]:


pd.plotting.autocorrelation_plot(data["Close"])


# In[21]:


#Since the curve is moving down after the 10th line of the first boundary, therefore p = 10


# In[22]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data["Close"], lags = 100)


# In[23]:


# 2 points are far away from others, therefore q=2 and since data is seasonal , d = 1


# In[24]:


import statsmodels.api as sm
import matplotlib.pyplot as plt
p, d, q = 10, 1, 2

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data["Close"], order=(p, d, q))
fitted = model.fit()

print(fitted.summary())


# In[25]:


predictions = fitted.predict()
print(predictions)


# In[26]:


import warnings
model=sm.tsa.statespace.SARIMAX(data['Close'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())


# 1. Residuals show no significant autocorrelation or heteroskedasticity however they deviate significantly from normality. 
# 2. Despite similar fit metrics, neither model clearly outperforms the other based on information criteria alone, as both are complex and might overfit.
# 3. Both models include terms with high p-values (insignificant terms) that suggest potential model simplification.

# In[27]:


predictions = model.predict(len(data), len(data)+20)
print(predictions)


# In[28]:


data["Close"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")


# **PREDICTING CLOSING STOCK PRICE FOR AMZN USING LSTM**

# In[29]:


df = yf.download('AMZN', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
df["Date"] = df.index
df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
df.reset_index(drop=True, inplace=True)
print(df.tail())


# In[30]:


plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()


# In[31]:


df = df["Close"]
print(df.head())


# In[32]:


dataset = df.values
training_data_len = int(np.ceil( len(dataset) * .95 ))



# In[33]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[34]:


train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[35]:


get_ipython().system('pip install tensorflow')


# In[36]:


from keras.models import Sequential
from keras.layers import Dense, LSTM

# LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[37]:


test_data = scaled_data[training_data_len - 60: , :]

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse


# In[38]:


train = df.iloc[:training_data_len]
valid = df.iloc[training_data_len:]
valid['Predictions'] = predictions
valid


# In[39]:


# Visualizing the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['AMZN'])
plt.plot(valid[['AMZN', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# **Results** : 
# The model has learned the underlying patterns in the historical data and is able to capture the general direction of the time series. However, towards the end of the prediction period, the predictions deviate from the actual data. This suggests that the model's accuracy might decrease as the prediction horizon increases.
# 
# Possible Reasons for Prediction Deviation can be model complexity, data variability or bias. 
# 
