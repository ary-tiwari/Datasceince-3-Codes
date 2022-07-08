# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('landslide_data3_miss.csv')
df_copy = df.copy()
df_orig = pd.read_csv('landslide_data3_original.csv')

#filling data using linear interpolation
for i in df.columns:
    if i != 'dates' and i != 'stationid':
        df.interpolate(method='linear', inplace=True)

#Stats for landslide_missing after interpolation
print('Stats for landslide_missing csv after interpolation')
print('\n')
for i in df.columns:
    if i != 'dates' and i != 'stationid':
        print("Stats for attribute ", i)
        print(df[i].describe())
        print('Mode: ', df[i].mode())
        print('\n')
print('\n')


def RMSE(attribute):
    Na = df_copy[attribute].isnull().sum()
    sum = 0
    for j in range(len(df)):
        sum = sum + (df[attribute][j] - df_orig[attribute][j])**2
    rmse = (sum/Na)**(0.5)
    return rmse

rmse_array = []

for attribute in df.columns:
    if attribute != 'dates' and attribute != 'stationid':
        a = RMSE(attribute)
        print('RMSE of', attribute, 'is', a )
        rmse_array.append(a)
    else:
        rmse_array.append(0)


plt.bar(df.columns, rmse_array)
plt.title('RMSE after interolation')
plt.xlabel('Attributes')
plt.ylabel("RMSE")
plt.show()