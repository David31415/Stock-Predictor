# This program predicts stock prices by using machine learning models

# Install the dependencies

import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#Get the stock data

stock_name = "AEVA"

df = yf.download(stock_name, start='1970-01-01', end='2025-04-10') # data frame
print(df)

# First, let's check what columns are actually available

forecast_out = 1

close = df["Close"]
open = df["Open"]
low = df["Low"]
high = df["High"]

df['Close_Prediction'] = close.shift(-forecast_out)
df['Open_Prediction'] = open.shift(-forecast_out)
df['Low_Prediction'] = low.shift(-forecast_out)
df['High_Prediction'] = high.shift(-forecast_out)

################################################################ Close

### Create the independent data set (X)
X_close = np.array(df["Close"])
#Remove the last 'n' rows
X_close = X_close[:-forecast_out]
print("X_close: ", X_close)

# Create the dependent data set (Y)
y_close = np.array(df['Close_Prediction'])
# Get all of the y values except the last "n" rows. 
y_close = y_close[:-forecast_out]
print("y_close: ", y_close)

################################################################ Open

### Create the independent data set (X)
X_open = np.array(df["Open"])
#Remove the last 'n' rows
X_open = X_open[:-forecast_out]
print("X_open: ", X_open)

# Create the dependent data set (Y)
y_open = np.array(df['Open_Prediction'])
# Get all of the y values except the last "n" rows. 
y_open = y_open[:-forecast_out]
print("y_open: ", y_open)

################################################################ Low

### Create the independent data set (X)
X_low = np.array(df["Low"])
#Remove the last 'n' rows
X_low = X_low[:-forecast_out]
print("X_low: ", X_low)

# Create the dependent data set (Y)
y_low = np.array(df['Low_Prediction'])
# Get all of the y values except the last "n" rows. 
y_low = y_low[:-forecast_out]
print("y_low: ", y_low)

################################################################ High

### Create the independent data set (X)
X_high = np.array(df["High"])
#Remove the last 'n' rows
X_high = X_high[:-forecast_out]
print("X_high: ", X_high)

# Create the dependent data set (Y)
y_high = np.array(df['High_Prediction'])
# Get all of the y values except the last "n" rows. 
y_high = y_high[:-forecast_out]
print("y_high: ", y_high)
# split data into 80% training and 20% testing

################################################################ TEST - ################################################################
################################################################ TEST - CLOSE
################################################################ TEST - CLOSE
################################################################ TEST - ################################################################

x_train_close, x_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, test_size = 0.2)
svr_rbf = SVR(kernel = "rbf", C = 1e3, gamma=0.1)
svr_rbf.fit(x_train_close, y_train_close)
svm_confidence_close = svr_rbf.score(x_test_close, y_test_close)
print("svm confidence_close",  svm_confidence_close)

lr = LinearRegression()
lr.fit(x_train_close, y_train_close)

lr_confidence_close = lr.score(x_test_close, y_test_close)
print("lr confidence_close",  lr_confidence_close)

# x_forecast_close = np.array(df["Close_Prediction"])[-forecast_out:]

x_forecast_close = X_close[-forecast_out:]
lr_prediction_close = lr.predict([x_forecast_close])
svm_prediction_close = svr_rbf.predict([x_forecast_close])

print(lr_prediction_close)
print(svm_prediction_close)

################################################################# TEST - OPEN

x_train_open, x_test_open, y_train_open, y_test_open = train_test_split(X_open, y_open, test_size = 0.2)
svr_rbf = SVR(kernel = "rbf", C = 1e3, gamma=0.1)
svr_rbf.fit(x_train_open, y_train_open)
svm_confidence_open = svr_rbf.score(x_test_open, y_test_open)
print("svm confidence_open",  svm_confidence_open)

lr = LinearRegression()
lr.fit(x_train_open, y_train_open)

lr_confidence_open = lr.score(x_test_open, y_test_open)
print("lr confidence_open",  lr_confidence_open)

x_forecast_open = np.array(df["Open_Prediction"])[-forecast_out:]
lr_prediction_open = lr.predict([x_forecast_open])
svm_prediction_open = svr_rbf.predict([x_forecast_open])

################################################################# TEST - HIGH

x_train_high, x_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, test_size = 0.2)
svr_rbf = SVR(kernel = "rbf", C = 1e3, gamma=0.1)
svr_rbf.fit(x_train_high, y_train_high)
svm_confidence_high = svr_rbf.score(x_test_high, y_test_high)
print("svm confidence_high",  svm_confidence_high)

lr = LinearRegression()
lr.fit(x_train_high, y_train_high)

lr_confidence_high = lr.score(x_test_high, y_test_high)
print("lr confidence_high",  lr_confidence_high)

x_forecast_high = np.array(df["High_Prediction"])[-forecast_out:]
lr_prediction_high = lr.predict([x_forecast_high])
svm_prediction_high = svr_rbf.predict([x_forecast_high])

################################################################# TEST - LOW

x_train_low, x_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, test_size = 0.2)
svr_rbf = SVR(kernel = "rbf", C = 1e3, gamma=0.1)
svr_rbf.fit(x_train_low, y_train_low)
svm_confidence_low = svr_rbf.score(x_test_low, y_test_low)
print("svm confidence_low",  svm_confidence_low)

lr = LinearRegression()
lr.fit(x_train_low, y_train_low)

lr_confidence_low = lr.score(x_test_low, y_test_low)
print("lr confidence_low",  lr_confidence_low)

x_forecast_low = np.array(df["Low_Prediction"])[-forecast_out:]
lr_prediction_low = lr.predict([x_forecast_low])
svm_prediction_low = svr_rbf.predict([x_forecast_low])

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
####################################################### - RESULTS

print(stock_name)
print("AVG OPEN: ", (lr_prediction_open))
print("AVG HIGH: ", (lr_prediction_high))
print("AVG LOW: ", (lr_prediction_low))
print("AVG CLOSE: ", (lr_prediction_close))