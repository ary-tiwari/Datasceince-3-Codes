# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


import pandas as pd
import numpy as np

df = pd.read_csv('landslide_data3_miss.csv')


print('Number of missing values in each attribute before step 2 \n',df.isnull().sum())

#Step from q2
df.dropna(subset=['stationid'], inplace=True)
df.dropna(thresh= 7, inplace=True)

print('Number of missing values in each attribute after step 2 \n', df.isnull().sum())





