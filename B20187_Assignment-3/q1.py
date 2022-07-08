# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def outlier(attribute):                     #function to define outliers
    q1=np.percentile(df[attribute],25)     #1st quartile
    q3=np.percentile(df[attribute],75)     #3rd quartile
    max=q3 + 1.5*(q3 - q1)                 #Maximum
    min=q1 - 1.5*(q3 - q1)                 #Minimum
    oq1=df[df[attribute]<min]              #defining outliers lesser than Minimum
    oq3=df[df[attribute]>max]              #defining outliers more than Maximum
    # print('IQR: ', q3-q1,"{}".format(attribute))
    return oq1, oq3

def replace(attribute):              #function to replace the outliers with median
    oq1, oq3 = outlier(attribute)
    median = df[attribute].median()
    for j in oq1[attribute].index:        
        df[attribute].loc[j]= median
    for k in oq3[attribute].index:        
        df[attribute].loc[k]= median



df = pd.read_csv('pima-indians-diabetes.csv')
for i in df.columns:                #replacing outliers
    if (i != 'class'):
        replace(i)

#Part-a: Min-Max Normalization
def normalization(df, attribute, new_min, new_max):
    df[attribute] = (df[attribute] - df[attribute].min())/(df[attribute].max() - df[attribute].min())*(new_max - new_min) + (new_min)

df_normalized = df.copy()

print('\n Normalization \n')
for i in df_normalized.columns:
    if (i != 'class'):
        normalization(df_normalized, i, 5, 12)
        print('Old min and max for {} are:'.format(i), df[i].min(), 'and', df[i].max())
        print('New min and max for {} are:'.format(i), df_normalized[i].min(), 'and', df_normalized[i].max())
        print('\n')

#Part-b: Standardization
def standardization(df, attribute):
    df[attribute] = (df[attribute] - df[attribute].mean() )/df[attribute].std()
    
df_standardized = df.copy()

print('\n Standardization \n')
for i in df_standardized.columns:
    if (i != 'class'):
        standardization(df_standardized, i)
        print('Old mean and standard deviation for {} are:'.format(i), df[i].mean(), 'and', df[i].std())
        print('New mean and standard deviation for {} are:'.format(i), round(df_standardized[i].mean(), 5), 'and', df_standardized[i].std())
        print('\n')