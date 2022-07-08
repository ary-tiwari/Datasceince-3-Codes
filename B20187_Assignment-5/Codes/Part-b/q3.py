# Aryan Tiwari
# B20187
# 8982562898


#importing needed libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


df = pd.read_csv('abalone.csv')

X = df.copy().drop('Rings', axis=1)    #Feature
X_label = df['Rings']                  #Target attribute


#splitting dataframe in testing and training parts
[X_train, X_test, Y_train, Y_test] = train_test_split(X, X_label, test_size=0.3, random_state=42, shuffle=True)

#Exporting data as csv
X_train =X_train.assign(Rings=Y_train)
X_train.to_csv('abalone-train.csv', index=False)
X_test =X_test.assign(Rings=Y_test)
X_test.to_csv('abalone-test.csv', index=False)


#finding attribute with highest correlation coefficient
max_corr=-1000
for col in X.columns:
    cor = X[col].corr(X_label, method='pearson')
    if cor>max_corr:
        max_corr = cor
        max_col = col


#Function to calculate Rmse
def rmse(pred, actual):
	return np.sqrt(np.sum(np.square(pred-actual))/len(pred))


P = [2, 3, 4, 5]            #degrees of the polynomial


#polynomial regression with training dataset
X = pd.DataFrame(X_train[max_col])
Y = X_train["Rings"]
RMSE = []
for p in P:
	poly_features = PolynomialFeatures(p)
	poly_input = poly_features.fit_transform(X)
	# perform linear regression on the transformed input vectors to perform polynomial regression
	reg = LinearRegression()
	reg.fit(poly_input, Y)
	# prediction of training data
	train_pred = reg.predict(poly_input)
    #printing rmse
	print('RMSE of training data prediction with P =',p,'is',rmse(train_pred, Y))
	RMSE.append(rmse(train_pred, Y))
print('\n')


# bar plot of rmse of training prediction
plt.bar(P, RMSE)
plt.title('Rmse of Training predicted data')
plt.show()


#polynomial regression with testing dataset
RMSE = []
for p in P:
	poly_features = PolynomialFeatures(p)
	poly_input = poly_features.fit_transform(X)
	#performing polynomial regression
	reg = LinearRegression()
	reg.fit(poly_input, Y)
	# prediction for test dataset
	test_poly = poly_features.fit_transform(pd.DataFrame(X_test[max_col]))
	test_pred = reg.predict(test_poly)
    #rmse calculation
	print('RMSE of training data prediction with P =',p,'is',rmse(test_pred, Y_test))
	RMSE.append(rmse(test_pred, Y_test))
print('\n')


# bar plot of rmse values of testing prediction
plt.bar(P, RMSE)
plt.title('Rmse of Testing predicted data')
plt.show()


#q3_c
# Choosing the best fit curve
best_fit = P[np.argmin(RMSE)]
print('Degree of polynomial with best fit is:',best_fit)
#poly regression with best fit
poly_features = PolynomialFeatures(best_fit)
regressor = LinearRegression().fit(poly_features.fit_transform(X), Y)

# plot of best fit curve
inp = X_train[max_col]
plt.plot(sorted(inp), regressor.predict(poly_features.fit_transform(pd.DataFrame(sorted(inp)))), c = "red")
plt.scatter(inp, Y)
plt.xlabel(max_col)
plt.ylabel('Rings')
plt.title('Best fit poly curve')
plt.show()

#q3_d
# Scatter plot of actual vs predicted values
plt.scatter(Y_test, regressor.predict(poly_features.fit_transform(pd.DataFrame(X_test[max_col]))))
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.show()