# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


#importing useful libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#Reading csv and storing data in Dataframe
df=pd.read_csv('SteelPlateFaults-2class.csv') 
X = df.copy().drop('Class', axis=1)                       #Feature
X_label = df['Class']               #Target attribute


#splitting dataframe in testing and training parts
[X_train, X_test, X_label_train, X_label_test] = train_test_split(X, X_label, test_size=0.3, random_state=42, shuffle=True)

#Exporting data as csv
X_train =X_train.assign(Class=X_label_train)
X_train.to_csv('SteelPlateFaults-train.csv', index=False)
X_test =X_test.assign(Class=X_label_test)
X_test.to_csv('SteelPlateFaults-test.csv', index=False)


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
