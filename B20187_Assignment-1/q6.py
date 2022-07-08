# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("pima-indians-diabetes.csv")


data.boxplot('pregs')
plt.show()

data.boxplot('plas')
plt.show()

data.boxplot('pres')
plt.show()

data.boxplot('skin')
plt.show()

data.boxplot('test')
plt.show()

data.boxplot('BMI')
plt.show()

data.boxplot('pedi')
plt.show()

data.boxplot('Age')
plt.show()