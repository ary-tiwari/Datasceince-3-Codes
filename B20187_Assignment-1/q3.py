# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


import pandas as pd

data = pd.read_csv("pima-indians-diabetes.csv")

pregs = data['pregs']
plas = data['plas']
pres = data['pres']
skin = data['skin']
test = data['test']
BMI = data['BMI']
pedi = data['pedi']
Age = data['Age']


# Correlation coefficient of Age with other attributes
s = 'Correlation coefficient of Age with'

print(s, 'pregs is', Age.corr(pregs), '\n')
print(s, 'plas is', Age.corr(plas), '\n')
print(s, 'pres is', Age.corr(pres), '\n')
print(s, 'skin is', Age.corr(skin), '\n')
print(s, 'test is', Age.corr(test), '\n')
print(s, 'BMI is', Age.corr(BMI), '\n')
print(s, 'pedi', Age.corr(pedi), '\n \n')

# Correlation coefficient of BMI with other 
s1 = 'Correlation coefficient of BMI with'

print(s1, 'pregs is', BMI.corr(pregs), '\n')
print(s1, 'plas is', BMI.corr(plas), '\n')
print(s1, 'pres is', BMI.corr(pres), '\n')
print(s1, 'skin is', BMI.corr(skin), '\n')
print(s1, 'test is', BMI.corr(test), '\n')
print(s1, 'Age is', BMI.corr(Age), '\n')
print(s1, 'pedi', BMI.corr(pedi))