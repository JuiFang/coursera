import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5.0, 5.0, 0.1)

y = 2 * (x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise

plt.plot(x, ydata, 'bo')
plt.plot(x, y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

y = 1*(x**3) + 1*x+3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata, 'bo')
plt.plot(x, y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata, 'bo')
plt.plot(x, y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

Y = np.exp(x)
plt.plot(x, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

Y = np.log(x)
plt.plot(x, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

Y = 1-4/(1+np.power(3, x-2))
plt.plot(x,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

import pandas as pd

df = pd.read_csv("china_gdp.csv")
print(df.head(10))

plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.xlabel('Year')
plt.ylabel('CDP')
plt.show()

Y = 1.0 / (1.0 + np.exp(-x))
plt.plot(x,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1+np.exp(-Beta_1*(x-Beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0

Y_pred = sigmoid(x_data, beta_1, beta_2)
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')
plt.show()

xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

from scipy.optimize import curve_fit

popt, pcov = curve_fit(sigmoid, xdata, ydata)
print("beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
