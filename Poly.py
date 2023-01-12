import numpy as np                                          # will help to do the math from the data
import matplotlib.pyplot as plt                             # will help to plot the graph 
from sklearn.linear_model import LinearRegression           # will help target prediction value based on independent variables for the regression model
from sklearn.preprocessing import PolynomialFeatures        # will help plot features with degree less than or equal to the specified degree

# the training set
x_train = [[5], [10], [12], [15], [18]]
y_train = [[3], [7], [10], [15], [22]]

# the testing set
x_test = [[5], [7], [11], [21]]
y_test = [[9], [14], [15], [23]]

# train the linear regression model and plot a prediction
regress = LinearRegression()
regress.fit(x_train, y_train)

# creating the line of best fit
xx = np.linspace(0, 26, 100)
yy = regress.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# setting the degree of the polynomial regression model
quadratic_feature = PolynomialFeatures(degree=2)

# the preprocessor transforms an input data matrix into a new data matrix of a given degree
# change X_train to x_train
X_train_quadratic = quadratic_feature.fit_transform(x_train)
X_test_quadratic = quadratic_feature.transform(x_test) 

# training and testing the regressor quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_feature.transform(xx.reshape(xx.shape[0], 1))

# Plotting the graph with the x-axis title and y-axis title
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')

# title for the graph
plt.title('Pressure Against Tempreture:')

# title for the x label axis
plt.xlabel('Temperature')

# title for the y label axis
plt.ylabel('Pressure')

# creating the vertical and horizontal scale
plt.axis([0, 25, 0, 25])

# displaying the grid
plt.grid(True)
plt.scatter(x_train, y_train)

# displaying the graph with the training and testing data with the quadratic model
plt.show()
print(x_train)
print(X_train_quadratic)
print(x_test)
print(X_test_quadratic)