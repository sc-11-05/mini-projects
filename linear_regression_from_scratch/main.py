import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('CardioGoodFitness.csv')

# checking for the loss (MSE)
def loss_function(x, true_y, n, m, b):
    Total_error = 0

    for i in range(n):
        Total_error += (true_y[i] - (m * x[i] + b)) ** 2

    Total_error = Total_error / n 

    return Total_error

# decreasing the loss
def gradient_descent(L, current_m, current_b, x, true_y, n):

    m_gradient = 0
    b_gradient = 0

    for i in range (n):
        m_gradient += x[i] * (true_y[i] - (current_m * x[i] + current_b))
        b_gradient += (true_y[i] - (current_m * x[i] + current_b))
    
    m = current_m - L * (-(2/n) * m_gradient)
    b = current_b - L * (-(2/n) * b_gradient)

    return m, b

# Predicting miles based on age
x = df['Age']
y = df['Miles']

m = 0 
b = 0
L = 0.0001

n = len(df) # getting the length of dataframe

epochs = 1000

print("MSE Before",round(loss_function(x, y, n, m, b),2))

for i in range(epochs):
    if i % 100 ==0:
        print("Epoch:", i)
    m, b = gradient_descent(L, m, b, x, y, n) 

print("MSE After",round(loss_function(x, y, n, m, b),2))



#plotting the graph
plt.scatter(x, y, color= "black")
plt.plot(x, m * x + b , color = "red")
plt.show()
