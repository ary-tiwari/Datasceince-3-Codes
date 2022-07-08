# Name - Aryan Tiwari
# B20187
# Contact number - 8982562898


from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt


mean = np.array([0, 0])    #mean of distribution
covariance = np.array([[13, -3], [-3, 5]])      #covariance matrix of distribution
data = np.random.multivariate_normal(mean, covariance, 1000)        #synthetic data generation

#Part-A
#Plotting synthetic data
x = [data[: , 0]]
y = [data[:, 1]]
plt.scatter(x, y)
plt.title('Scatter plot of synthetic data')  #Scatter plot of synthetic data
plt.axis('equal')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()



#Part-B
value, vector = np.linalg.eig(covariance)    #numpy function to calculate eigen-values and eigen-vectors
plt.scatter(x, y)       #Scatter plot of synthetic data
plt.quiver([0,0], [0,0], vector[0], vector[1], scale=3 )      #plotting eigen vectors
plt.axis('equal')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Eigen vectors on scatter plot of synthetic data')
plt.show()



#Part-C
projection = np.dot(data, vector)

#Projection on 1st Eigen Vector
plt.scatter(x, y)       #Scatter plot of synthetic data
plt.quiver([0,0], [0,0], vector[0], vector[1], scale=3 )      #plotting eigen vectors
plt.scatter(projection*vector[0][0], projection*vector[1][0], marker='+', color='r')
plt.axis('equal')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Projection on 1st Eigen-Vector')
plt.show()

#Projection on 2nd Eigen Vector
plt.scatter(x, y)       #Scatter plot of synthetic data
plt.quiver([0,0], [0,0], vector[0], vector[1], scale=3 )      #plotting eigen vectors
plt.scatter(projection*vector[0][1], projection*vector[1][1], marker='+', color='r')
plt.axis('equal')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Projection on 2nd Eigen-Vector')
plt.show()

#Part-d
recon = np.dot(projection, vector.T)
Rmse = (((np.subtract(data, recon))**2).mean())**0.5
print("Euclidean error in reconstructed data is: ",round(Rmse, 3))