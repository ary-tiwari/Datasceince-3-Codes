# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.dtypes.missing import isna, isnull



df = pd.read_csv('landslide_data3_miss.csv')
A = df.columns
B = np.zeros(len(A))
for i in range(len(A)):
    for j in df[A[i]]:
        if (isna(j)):
            B[i] = B[i] + 1


print(df.isnull().sum())


plt.bar(A, B)
plt.show()
