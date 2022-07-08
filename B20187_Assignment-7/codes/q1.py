# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898

#importing useful libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


#reading data from csv
df = pd.read_csv('Iris.csv')

#seperating attributes and labels
data = df.drop('Species', axis=1)
label = df['Species']

#performing PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(data)      #array with 2 principal components

#reduced dataframe with 2 attributes
df_pca = pd.DataFrame(df_pca, columns = ['PC1', 'PC2'])

#plot of 2D data obtained from PCA
df_pca.plot.scatter(x = 'PC1', y = 'PC2')
plt.show()


value, vector = np.linalg.eig(np.cov(data.T))     # eigen-values and eigen-vectors of dataframe
value.sort()
value = value[::-1]     #sorting eigen values in descending order

#Plotting eigen values
plt.plot(range(1,5),value, marker='o')
plt.xlabel('Components')
plt.ylabel('Eigen Values')
plt.title("Plot of Eigen-values")
plt.show()




