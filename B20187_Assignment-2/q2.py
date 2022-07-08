# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


import pandas as pd
import numpy as np

df = pd.read_csv('landslide_data3_miss.csv')

len0 = len(df)

print("Length of old dataframe: ", len0)

df.dropna(subset=['stationid'], inplace=True)

len1 = len(df)

print("Length of new dataframe after removing missing values of stationid: ", len1)
print("Number of deleted rows in (a)= ", len0-len1)

df.dropna(thresh= 7, inplace=True)

len2 = len(df)

print('Rows deleted in (b)=', len1-len2)