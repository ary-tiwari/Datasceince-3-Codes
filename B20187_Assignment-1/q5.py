# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("pima-indians-diabetes.csv")

df = data.groupby('class')

class0 = df.get_group(0)
class1 = df.get_group(1)

class0.hist('pregs')
plt.title('Class 0')
class1.hist('pregs')
plt.title('Class 1')
plt.show()

