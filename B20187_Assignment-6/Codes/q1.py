# Aryan Tiwari
# B20187
# 8982562898


#importing useful libraries
from matplotlib import markers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.graphics.tsaplots import plot_acf

#reading time-series data from csv
series = pd.read_csv('daily_covid_cases.csv')

#q1_a
series.plot.line('Date', 'new_cases')   #plotting new-cases per day
plt.xlabel('Dates')
plt.ylabel('Number of new cases')
plt.show()


#q1_b
lag1 = series.shift(1)              #creating lagged series

#calculating autocorrelation for lag = 1
print(series['new_cases'].corr(lag1['new_cases']))

#plotting scatter plot of original and lgged series
plt.scatter(series['new_cases'], lag1['new_cases'])
plt.show()

lags = np.arange(1,7,1)
coef = []

for lag in lags:
    lag_series = series.shift(lag)
    corr_coef = series['new_cases'].corr(lag_series['new_cases'])
    #printing auto-correlation coefficients 
    print('The correlation coefficient for lag:', lag, 'is', corr_coef)
    coef.append(corr_coef)


#line plot between obtained correlation coefficients (y-axis) and lagged values (x-axis)
plt.plot(lags, coef, marker='o')
plt.show()


#plotting a correlogram
plot_acf(series['new_cases'])
plt.show()