# Aryan Tiwari
# B20187
# 8982562898


#importing useful libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg as AR


series = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'],
                      index_col=['Date'],
                      sep=',')


#slicing testing and training data                      
test_size = 0.35                    # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]


#function to calculate RMSE %
def rmse(test, pred):
	return np.sqrt(np.sum(np.square(test-pred))/(len(test)))/np.mean(test)*100

#function to calculate Mean absolute percentage error
def mape(test, pred):
	s = 0
	for i in range(len(test)):
		s += abs(test[i][0]-pred[i])/test[i]
	return (s*100/len(test))[0]


lags = [1,5,10,15,25]                #list of different lag values
rmse_arr, mape_arr = [], []
for lag in lags:
    model = AR(train, lags=lag)
    model_fit = model.fit()                 # fit/train the model
    coef = model_fit.params
    history = train[len(train)-lag:]
    history = [history[i] for i in range(len(history))]
    predictions = []        #list to store predictions

    for t in range(len(test)):
        length = len(history)
        lag_series = [history[i] for i in range(length-lag, length)]
        pred = coef[0]
        for d in range(lag):
    #prediction of future values using AR coefficients
            pred += coef[d+1]*lag_series[lag-d-1] 

        obs = test[t]
        predictions.append(pred)
        history.append(obs)
    #printing RMSE and MAPE
    print('Lag:', lag)
    print('RMSE in prediction:',rmse(test, np.array(predictions)))
    print('Mean absolute percentage error',mape(test, predictions))
    print('\n')

    rmse_arr.append(rmse(test, np.array(predictions)))    #appending RMSE value for different lags
    mape_arr.append(mape(test, predictions))              #appending MAPE value for different lags

#bar chart showing RMSE (%) on the y-axis and lagged values on the x-axis
plt.bar(lags, rmse_arr, width = 3)
plt.xlabel('lags')
plt.ylabel('RMSE %')
plt.show()

#bar chart showing MAPE on the y-axis and lagged values on the x-axis
plt.bar(lags, mape_arr, width = 3)
plt.xlabel('lags')
plt.ylabel('MAPE')
plt.show()