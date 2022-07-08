# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("pima-indians-diabetes.csv")

# Scatter plots of Age with other attributes

data.plot(kind = 'scatter', x = 'Age', y = 'pregs')
plt.ylabel('Number of times pregnant')


data.plot(kind = 'scatter', x = 'Age', y = 'plas')
plt.ylabel('Plasma glucose concentration 2 hours in an oral glucose tolerance test')


data.plot(kind = 'scatter', x = 'Age', y = 'pres')
plt.ylabel('Diastolic blood pressure (mm Hg)')


data.plot(kind = 'scatter', x = 'Age', y = 'skin')
plt.ylabel('Triceps skin fold thickness (mm)')


data.plot(kind = 'scatter', x = 'Age', y = 'test')
plt.ylabel('2-Hour serum insulin (mu U/mL)')


data.plot(kind = 'scatter', x = 'Age', y = 'BMI')
plt.ylabel('Body mass index (weight in kg/(height in m)^2)')


data.plot(kind = 'scatter', x = 'Age', y = 'pedi')
plt.ylabel('Diabetes pedigree function')



plt.show()








