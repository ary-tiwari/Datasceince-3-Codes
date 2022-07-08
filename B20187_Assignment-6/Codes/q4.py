# Aryan Tiwari
# B20187
# 8982562898


#importing useful libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg as AR



#reading time-series data from csv
series = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'],
                      index_col=['Date'],
                      sep=',')


#slicing testing and training data                      
test_size = 0.35                    # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
temp = pd.Series(X[:len(X)-tst_sz, 0])


l, r = 1, len(train)

#calculating optimal lag from heuristics
while l <= r:
	m = int(l+(r-l)/2)
	if temp.autocorr(lag = m) > 2/np.sqrt(len(train)):
		l = m+1
	else:
		r = m-1

print('The heuristic value for the optimal number of lags is',r)

#training AR model
window = r                             # The lag=5
model = AR(train, lags=window)
model_fit = model.fit()                 # fit/train the model
coef = model_fit.params                 # Get the coefficients of AR model


history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = []        #list to store predictions

for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window, length)]
	pred = coef[0]
	for d in range(window):
        #prediction of future values using AR coefficients
		pred += coef[d+1]*lag[window-d-1]           
	obs = test[t]
	predictions.append(pred)
	history.append(obs)


#function to calculate RMSE %
def rmse(test, pred):
	return np.sqrt(np.sum(np.square(test-pred))/(len(test)))/np.mean(test)*100

#function to calculate Mean absolute percentage error
def mape(test, pred):
	s = 0
	for i in range(len(test)):
		s += abs(test[i][0]-pred[i])/test[i]
	return (s*100/len(test))[0]


#Printing RMSE and MAPE values for heuristic parameters
print('RMSE :',rmse(test, np.array(predictions)))
print('MAPE :',mape(test, predictions))

