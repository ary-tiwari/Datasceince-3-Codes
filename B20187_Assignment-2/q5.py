# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


df = pd.read_csv('landslide_data3_miss.csv')

#filling data using linear interpolation
for i in df.columns:
    if i != 'dates' and i != 'stationid':
        df.interpolate(method='linear', inplace=True)


def outlier(attribute):
    q1=np.percentile(df[attribute],25)     #1st quartile
    q3=np.percentile(df[attribute],75)     #3rd quartile
    max=q3 + 1.5*(q3 - q1)                 #Maximum
    min=q1 - 1.5*(q3 - q1)                 #Minimum
    oq1=df[df[attribute]<min]              #defining outliers lesser than Minimum
    oq3=df[df[attribute]>max]              #defining outliers lesser  than Maximum
    print('IQR: ', q3-q1,"{}".format(attribute))
    return oq1, oq3

def replace(attribute):              #replace the outliers with median
    oq1, oq3 = outlier(attribute)
    for j in oq1[attribute].index:        
        df[attribute].loc[j]=df[attribute].median()
    for k in oq3[attribute].index:         
        df[attribute].loc[k]=df[attribute].median()


#Before replacement

#temperature
print('The outliers for temperature are as follows \n', outlier('temperature'))
plt.boxplot(df['temperature'])
plt.title("Temp before replacement")
plt.show()

#rain
print("The outliers for rain are as follows \n", outlier('rain'))
plt.boxplot(df['rain'])
plt.title("Rain before replacement")
plt.show()

#After replacement
replace('temperature')
#temperature
plt.boxplot(df['temperature'])
plt.title("Temp after replacement")
plt.show()

#rain
replace('rain')
plt.boxplot(df['rain'])
plt.title("Rain after replacement")
plt.show()

