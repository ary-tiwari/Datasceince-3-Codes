# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898

#importing useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn import  decomposition, metrics
from sklearn.cluster import KMeans

# reading csv file
df = pd.read_csv('Iris.csv') 

#seperating data and label
df_data = df.iloc[:,:-1]
df_label = df['Species']
df_pca = decomposition.PCA(n_components=2).fit_transform(df_data)
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])

K = 3 
kmeans=KMeans(n_clusters=K) 
kmeans.fit(df_pca)
kmeans_prediction = kmeans.predict(df_pca) # getting the assigned class labels after clustering
clustercentre_kmeans = pd.DataFrame(kmeans.cluster_centers_) # locating the cluster centers
labels = np.unique(kmeans_prediction) # getting the unique class labels
class_labels = pd.DataFrame(kmeans_prediction, columns=['Class'])
combined = pd.concat([df_pca, class_labels], axis=1)


# q2_a (plot for clustered data)
for i in labels:
    label = 'Class' + str(i)
    plt.scatter(combined.loc[combined['Class'] == i]['PC1'], combined.loc[combined['Class'] == i]['PC2'], label=label)
plt.legend()
plt.scatter(clustercentre_kmeans[0], clustercentre_kmeans[1], s=70, marker='*', color='red',label = 'Centroids')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Kmeans Clustering for Reduced Data')
plt.legend()
plt.show()


#function to calculate purity score
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true,y_pred)
    # Find optimal one-to-one mapping between cluster labels and true labels  
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy  
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)  

# q2_b (Distortion Measure)
print(f'Distortion Measure for {K} clusters =', round(kmeans.inertia_, 2))

# q2_c (Purity Score)
print('Purity score for training data =', round(purity_score(df_label, kmeans_prediction), 2))