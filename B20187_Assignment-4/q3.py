# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


#importing useful libraries
import pandas as pd
import numpy as np
from numpy.linalg import det, inv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Log of probability function of Unimodal Gaussian Distribution
def prob(x, cov_mat, u_mat, prior):
    diff = x-u_mat
    inv_cov = inv(cov_mat)
    prod0 = np.dot(diff.T, inv_cov)
    prod = np.dot(prod0, diff)
    return np.log(prior)+0.5*np.log(det(inv_cov))-11.5*np.log(2*np.pi)-0.5*prod


X_train = pd.read_csv('SteelPlateFaults-train.csv')
X_test = pd.read_csv('SteelPlateFaults-test.csv')


# removing attributes that make the covariance matrix singular
delete = ["TypeOfSteel_A300", "TypeOfSteel_A400", "X_Minimum", "Y_Minimum"]
for i in delete:
    X_train.drop(i, axis=1, inplace=True)
    X_test.drop(i, axis=1, inplace=True)


#grouping training data by class
class0 = X_train.groupby('Class').get_group(0)
class0.drop('Class',axis=1 ,inplace=True)
class1 = X_train.groupby('Class').get_group(1)
class1.drop('Class',axis=1 ,inplace=True)


#mean vector calculation
u0 = class0.mean()
u1 = class1.mean()

#covariance matrix calculation
c0 = class0.cov()
c1 = class1.cov()

#exporting mean matrix and covariance matrix
c0.to_csv('c0.csv', index=False)
c1.to_csv('c1.csv', index=False)
u0.to_csv('u0.csv', index=False)
u1.to_csv('u1.csv', index=False)

#bayes classifier
prediction = []
for i in X_test.index:          #iterating through every tuple
    p1 = prob(X_test.iloc[i, :23], c1, u1, 509/(509+273))
    p0 = prob(X_test.iloc[i, :23], c0, u0, 273/(509+273))
    if p1 > p0:
        prediction.append(1)
    else:
        prediction.append(0)

#printing confusion matrix and accuracy scores
print("Confusion matrix:")
print(confusion_matrix(X_test["Class"], prediction))
print("Accuracy score:")
print(accuracy_score(X_test["Class"], prediction))