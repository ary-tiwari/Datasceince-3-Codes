# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


#importing useful libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#Reading csv and storing data in Dataframe
df_train=pd.read_csv('SteelPlateFaults-train.csv')
df_test=pd.read_csv('SteelPlateFaults-test.csv')


#Function for Min-Max Normalization
def normalization(df, attribute, new_min, new_max):
    df[attribute] = (df[attribute] - df[attribute].min())/(df[attribute].max() - df[attribute].min())*(new_max - new_min) + (new_min)

df_train_normalized = df_train.copy()
df_test_normalized = df_test.copy()

#Normalisation
for i in df_train_normalized.columns:
    if (i != 'class'):
        normalization(df_train_normalized, i, 0, 1)
        normalization(df_test_normalized, i, 0, 1)

X_train = df_train_normalized.drop('Class', axis=1)         #Training Feature
X_test = df_test_normalized.drop('Class', axis=1)           #testing feature
X_label_train = df_train_normalized['Class']                #training labels
X_label_test = df_test_normalized['Class']                  #test labels


#Exporting split data as csv
X_train.to_csv('SteelPlateFaults-train-Normalised.csv')
X_test.to_csv('SteelPlateFaults-test-normalised.csv')


#K-nearest neighbor (KNN) classification
def KNN(k):
    neigh = KNeighborsClassifier(n_neighbors=k)     #setting number of nearest neighbours
    neigh.fit(X_train, X_label_train)               #training model by knn method
    prediction = neigh.predict(X_test)              #predicting labels of testing data
    return prediction


k = [1, 3, 5]       #k-values


for k in k:
    prediction = KNN(k)
    print('The confusion matrix for k={} is:'.format(k))
    print(confusion_matrix(X_label_test, prediction))       # printing confusion-matrix
    #printing accuracy of knn predictions
    print('The accuracy for k={} is:'.format(k), accuracy_score(X_label_test, prediction), '\n')