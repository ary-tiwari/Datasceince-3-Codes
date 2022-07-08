# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("pima-indians-diabetes.csv")

# Histogram of preg
data["pregs"].plot(kind = 'hist')
plt.xlabel('Number of times pregnant')
plt.show()

# Histogram of skin
data["skin"].plot(kind = 'hist')
plt.xlabel('Triceps skin fold thickness (mm)')

plt.show()
