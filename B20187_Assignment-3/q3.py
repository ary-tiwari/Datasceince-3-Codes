# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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



df = pd.read_csv('pima-indians-diabetes.csv')       #reading csv
df.drop(['class'],axis=1,  inplace=True)            #dropping class attribute
for i in df.columns:                #replacing outliers
    replace(i)

def standardization(df, attribute):     #function to standardize data
    df[attribute] = (df[attribute] - df[attribute].mean() )/df[attribute].std()


for i in df.columns:
        standardization(df, i)



#Part-a

value, vector = np.linalg.eig(np.cov(df.T))     # eigen-values and eigen-vectors of dataframe
value.sort()
value = value[::-1]     #sorting eigen values in descending order

#applying PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df)      #array with 2 principal components

principal_df = pd.DataFrame(df_pca, columns = ['PC1', 'PC2'])

#plotting 2-dimensional (reduced) dataframe
plt.scatter(principal_df['PC1'], principal_df['PC2'])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title("Scatter plot of dimensionally reduced  data")
plt.show()

# Comparing variance of reduced data with eigen values
print('Part-a')
print('\nComparing variance of reduced data with eigen values\n')

for i in range(2):
    print('The variance of PC-{} is :'.format(i+1), pca.explained_variance_[i] )
    print('The corresponding eigen-value is:', value[i], '\n')


#Part-b         
#Plotting eigen values
plt.plot(range(1,9),value, marker='o')
plt.xlabel('Index')
plt.ylabel('Eigen Values')
plt.title("Plot of Eigen-values")
plt.show()


#Part-c
recon_error=[]
l = []
for i in range(2,9):
    l.append(i)
    pca=PCA(n_components=i)         #specifying no. of principal components
    pca_array = pca.fit_transform(df)      #Data with Reduced Dimension
    pca_df = pd.DataFrame(pca_array, columns = list(range(i)))
    recon_data = pca.inverse_transform(pca_array)   #Reconstructed Data
    rmse = (((np.subtract(df.values, recon_data))**2).mean())**0.5      #Euclidean error in reconstructed data
    recon_error.append(rmse)
    cov_matrix = pca_df.cov()       #creating covariance matrix
    print('\nThe covariance of dimension {} is\n'.format(i), cov_matrix.to_csv(), '\n')

plt.plot(l, recon_error)
plt.xlabel('Dimension of PCA')
plt.ylabel('Euclidean error in reconstruction')
plt.show()


#Part-d

print('Covariance matrix of original csv is : \n')
print(df.cov().to_csv())        #printing Covariance matrix of original csv
print('\n')
print('Covariance matrix of Reconstucted df with l=8 is:  \n')
print(cov_matrix.to_csv())


