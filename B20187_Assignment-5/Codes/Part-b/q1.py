# Aryan Tiwari
# B20187
# 8982562898


#importing needed libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('abalone.csv')

X = df.copy().drop('Rings', axis=1)    #Feature
X_label = df['Rings']                  #Target attribute


#splitting dataframe in testing and training parts
[X_train, X_test, X_label_train, X_label_test] = train_test_split(X, X_label, test_size=0.3, random_state=42, shuffle=True)

#Exporting data as csv
X_train =X_train.assign(Rings=X_label_train)
X_train.to_csv('abalone-train.csv', index=False)
X_test =X_test.assign(Rings=X_label_test)
X_test.to_csv('abalone-test.csv', index=False)


#finding attribute with highest coefficient
max_corr=-1000
for col in X.columns:
    cor = X[col].corr(X_label, method='pearson')
    if cor>max_corr:
        max_corr = cor
        max_col = col

# Training linear regression model
reg = LinearRegression().fit(pd.DataFrame(X_train[max_col]),pd.DataFrame(X_label_train))


#q1_a
#plotting chosen attribute with target attribute
input = sorted(X_train[max_col])
pred = reg.predict(pd.DataFrame(input))
plt.plot(input, pred, c = 'red')                #Best Fit Line plot
plt.scatter(X_train[max_col], X_label_train)    #scatter plot of max corr attribute with Rings
plt.xlabel(max_col)
plt.ylabel("Rings")
plt.title("Best Fit Line ")

plt.show()

#q1_b and q1_c
#Function to calculate Rmse
def rmse(pred, actual):
	return np.sqrt(np.sum(np.square(pred-actual))/len(pred))

#Prediction from training dataset
pred_train = reg.predict(pd.DataFrame(X_train[max_col]))
print('\nThe Rms error of prediction from training data is: ', rmse(pred_train, pd.DataFrame(X_label_train)), '\n')

#Prediction from test dataset
pred_test = reg.predict(pd.DataFrame(X_test[max_col]))
print('\nThe Rms error of prediction from testing data is: ', rmse(pred_test, pd.DataFrame(X_label_test)), '\n')

#q1_d
#plotting predicted data against actual data
plt.scatter(X_test["Rings"], pred_test)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Predicted vs Actual Rings")
plt.show()




