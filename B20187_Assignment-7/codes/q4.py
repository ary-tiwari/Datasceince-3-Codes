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
df_data = df.iloc[:,:-1]
df_label = df['Species']
df_pca = decomposition.PCA(n_components=2).fit_transform(df_data)
df_pca = pd.DataFrame(df_pca, columns=['D1', 'D2'])

def purity_score(y_true, y_pred): 
    # compute contingency matrix (also called confusion matrix) 
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    # print('Contingency matrix/Confusion matrix:',contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels 
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy 
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

#building a GMM classifier
K = 3
gmm = GMM(n_components = K)
gmm.fit(df_pca)
GMM_prediction = gmm.predict(df_pca) # the class labels assigned to the data after GMM clustering
clustercentre_gmm = pd.DataFrame(gmm.means_) # locating the cluster centres
labels = np.unique(GMM_prediction) # getting the unique labels assigned
class_labels = pd.DataFrame(GMM_prediction, columns=['Class'])
combined = pd.concat([df_pca, class_labels], axis=1)

# q4_a (plot for clustered data)
for i in labels:
    label = 'Class' + str(i)
    plt.scatter(combined.loc[combined['Class'] == i]['D1'], combined.loc[combined['Class'] == i]['D2'], label=label)
plt.scatter(clustercentre_gmm[0], clustercentre_gmm[1], s=70, marker='*', color='red',label = 'Centroids')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('GMM Clustering for Reduced Data')
plt.legend()
plt.show()

# q4_b printing Distortion Measure
print('Total Log Likelihood for K=3 clusters =', round(gmm.score_samples(df_pca).sum(), 2))

# q4_c printing Purity Score
print('Purity score for data =', round(purity_score(df_label, GMM_prediction), 2))