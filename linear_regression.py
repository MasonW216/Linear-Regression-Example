import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loss_function(m,c,points):
    total_loss = 0
    for i in range(len(points)):
        x = points.iloc[i,0]
        y = points.iloc[i,1]
        total_loss += (y - (m*x + c)) ** 2
    return total_loss / float(len(points))
# this function calculates the mean squared error for an assumed inital m and c and returns the total loss

def gradient_descent_m(m,c,points,learning_rate):
    gradient_descent = 0
    for i in range(len(points)):
        x = points.iloc[i,0]
        y = points.iloc[i,1]
        gradient_descent += -2*x*(y-(m*x+c))/float(len(points))
    m -= learning_rate * gradient_descent
    return m
# this function uses the partial derivative of the loss function with respect to m to find the gradient descent and updates m accordingly

def gradient_descent_c(m,c,points,learning_rate):
    gradient_descent = 0
    for i in range(len(points)):
        x = points.iloc[i,0]
        y = points.iloc[i,1]
        gradient_descent += -2*(y-(m*x+c))/float(len(points))
    c -= learning_rate * gradient_descent
    return c
# this function uses the partial derivative of the loss function with respect to c to find the gradient descent and updates c accordingly

data = pd.read_csv('d:\VS Code Projects\Linear Regression Example\linear_regression_practice.csv')

m=0
c=0
x = np.array(data.iloc[:,0])
y = np.array(data.iloc[:,1])
# initialising

for i in range(1000):
    m = gradient_descent_m(m,c,data,0.0001)
    c = gradient_descent_c(m,c,data,0.0001)
# iterating 1000 times to update m and c using gradient descent

print(m,c)
# printing the final values of m and c

plt.scatter(x,y,color='blue')
plt.plot(x,m*x+c,color='red')
plt.show()
# plotting the data points and the regression line