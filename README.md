# Machine-learning-Poly
This script uses the libraries numpy, matplotlib, and scikit-learn to create a linear regression model and a quadratic regression model for a given set of training data. The training data consists of two lists, x_train and y_train, representing the independent and dependent variables respectively. The script also uses a testing set, x_test and y_test, to test the accuracy of the models.

The script first fits a linear regression model using the LinearRegression() class from scikit-learn and plots the line of best fit on a graph. It then uses the PolynomialFeatures() class from scikit-learn to transform the input data into a new data matrix of a given degree (2 in this case). The script then fits a quadratic regression model using this new data and plots the line of best fit on the same graph, along with the scatter plot of the training data.

The script also includes various plotting options such as setting axis labels and titles, creating a grid, and adjusting the scale of the graph. The script will display the final graph, and will print out the original data set, transformed training dataset, and transformed test dataset.



