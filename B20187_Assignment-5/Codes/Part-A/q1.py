# Aryan Tiwari
# B20187
# 8982562898


#importing needed libraries
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

#reading data from csv
train = pd.read_csv('SteelPlateFaults-train.csv')
test = pd.read_csv('SteelPlateFaults-test.csv')


# removing attributes that make the covariance matrix singular
delete = ["TypeOfSteel_A300", "TypeOfSteel_A400", "X_Minimum", "Y_Minimum"]
for i in delete:
    train.drop(i, axis=1, inplace=True)
    test.drop(i, axis=1, inplace=True)


x_train = train.drop('Class', axis= 1)
y_train = train['Class']
x_test = test.drop('Class', axis= 1)
y_test = test['Class']

#grouping training data by class
class0 = train.groupby('Class').get_group(0)
class0.drop('Class',axis=1 ,inplace=True)
class1 = train.groupby('Class').get_group(1)
class1.drop('Class',axis=1 ,inplace=True)


Q = [2,4,8,16]

for q in Q:
    GMM = GaussianMixture(n_components=q, covariance_type='full', random_state= 42)
    #GMM for class 0
    GMM.fit(class0)
    p0 = GMM.score_samples(x_test) + np.log(len(class0)/len(train))
    

    #GMM for class 1
    GMM.fit(class1)
    p1 = GMM.score_samples(x_test) + np.log(len(class1)/len(train))
    
    #predicting values using gmm model
    pred = []
    for i in range(len(x_test)):
        if p0[i]>p1[i]:
            pred.append(0)
        else:
            pred.append(1)
    
    #printing confusion matrix and accuracy score
    print('The confusion matrix for Q =',q, 'is:')
    print(confusion_matrix(y_test, pred))
    print('The accuracy score for Q =',q, 'is:', accuracy_score(y_test, pred),'\n\n')

    




