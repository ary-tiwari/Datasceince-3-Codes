# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("pima-indians-diabetes.csv")

# Scatter plots of BMI with other attributes

data.plot(kind = 'scatter', x = 'BMI', y = 'pregs')
plt.ylabel('Number of times pregnant')


data.plot(kind = 'scatter', x = 'BMI', y = 'plas')
plt.ylabel('Plasma glucose concentration 2 hours in an oral glucose tolerance test')


data.plot(kind = 'scatter', x = 'BMI', y = 'pres')
plt.ylabel('Diastolic blood pressure (mm Hg)')


data.plot(kind = 'scatter', x = 'BMI', y = 'skin')
plt.ylabel('Triceps skin fold thickness (mm)')


data.plot(kind = 'scatter', x = 'BMI', y = 'test')
plt.ylabel('2-Hour serum insulin (mu U/mL)')


data.plot(kind = 'scatter', x = 'BMI', y = 'Age')
plt.ylabel('Age(years)')


data.plot(kind = 'scatter', x = 'BMI', y = 'pedi')
plt.ylabel('Diabetes pedigree function')



plt.show()








