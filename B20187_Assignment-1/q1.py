# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898




from os import set_blocking
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


# Number of times pregnant
print("Number of times pregnant")
print('Mean = ', pregs.mean())
print('Median = ', pregs.median())
print('Mode = ', pregs.mode())
print('Minimum = ', pregs.min())
print('Maximum = ', pregs.max())
print('Standard Deviation = ', pregs.std())
print('\n')

# Plasma glucose concentration 2 hours in an oral glucose tolerance test
print("Plasma glucose concentration 2 hours in an oral glucose tolerance test")
print('Mean = ', plas.mean())
print('Median = ', plas.median())
print('Mode = ', plas.mode())
print('Minimum = ', plas.min())
print('Maximum = ', plas.max())
print('Standard Deviation = ', plas.std())
print('\n')


# Diastolic blood pressure (mm Hg)
print(" Diastolic blood pressure (mm Hg) ")
print('Mean = ', pres.mean())
print('Median = ', pres.median())
print('Mode = ', pres.mode())
print('Minimum = ', pres.min())
print('Maximum = ', pres.max())
print('Standard Deviation = ', pres.std())
print('\n')


# Triceps skin fold thickness (mm)
print('Triceps skin fold thickness (mm)')
print('Mean = ', skin.mean())
print('Median = ', skin.median())
print('Mode = ', skin.mode())
print('Minimum = ', skin.min())
print('Maximum = ', skin.max())
print('Standard Deviation = ', skin.std())
print('\n')


# 2-Hour serum insulin (mu U/mL)
print('2-Hour serum insulin (mu U/mL)')
print('Mean = ', test.mean())
print('Median = ', test.median())
print('Mode = ', test.mode())
print('Minimum = ', test.min())
print('Maximum = ', test.max())
print('Standard Deviation = ', test.std())
print('\n')


# Body mass index (weight in kg/(height in m)^2)
print('Body mass index (weight in kg/(height in m)^2)')
print('Mean = ', BMI.mean())
print('Median = ', BMI.median())
print('Mode = ', BMI.mode())
print('Minimum = ', BMI.min())
print('Maximum = ', BMI.max())
print('Standard Deviation = ', BMI.std())
print('\n')


# Diabetes pedigree function
print('Diabetes pedigree function')
print('Mean = ', pedi.mean())
print('Median = ', pedi.median())
print('Mode = ', pedi.mode())
print('Minimum = ', pedi.min())
print('Maximum = ', pedi.max())
print('Standard Deviation = ', pedi.std())
print('\n')


# Age
print('Age')
print('Mean = ', Age.mean())
print('Median = ', Age.median())
print('Mode = ', Age.mode())
print('Minimum = ', Age.min())
print('Maximum = ', Age.max())
print('Standard Deviation = ', Age.std())