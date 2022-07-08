# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898

#importing useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn import metrics, decomposition
from sklearn.mixture import GaussianMixture as GMM

# reading csv file
df = pd.read_csv('Iris.csv') 

#seperating data and label
df_data = df.iloc[:,:-1]
df_label = df['Species']
df_pca = decomposition.PCA(n_components=2).fit_transform(df_data)
df_pca = pd.DataFrame(df_pca, columns=['D1', 'D2'])

#function to calculate purity score
def purity_score(y_true, y_pred): 
    # compute contingency matrix (also called confusion matrix) 
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    # print('Contingency matrix/Confusion matrix:',contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels 
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy 
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

K = [2, 3, 4, 5, 6, 7] # number of clusters
log_likelihood = []
purity = []
for i in K:
    gmm = GMM(n_components=i) #declaring the model
    gmm.fit(df_pca) #fitting the data to the model
    log_likelihood.append(gmm.score_samples(df_pca).sum()) # log likelihood
    gmm_prediction = gmm.predict(df_pca) # the class labels assigned to the data after GMM clustering
    p = purity_score(df_label, gmm_prediction) # finding the purity score
    purity.append(p) # appending the purity score to a list


# plotting K vs log likelihood
plt.plot(K, log_likelihood, marker='o')
plt.xlabel('K')
plt.ylabel('Total Log Likelihood')
plt.title('Elbow method to find the optimal number of clusters')
plt.show()


# printing the purity score
print('The value of purity score for different values of K are : ')
purity_table = pd.DataFrame(list(zip(K, purity)),
               columns =['K', 'Purity Score'])
print(purity_table)