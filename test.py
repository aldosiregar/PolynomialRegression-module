import numpy as np
import matplotlib.pyplot as plt
import PolynomialRegression
from sklearn.model_selection import train_test_split

np.random.seed(553)
x = np.linspace(-5,5,1000)
y = 3 * np.square(x) + 2 * x + 7 + np.random.normal(0, 5, 1000)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=43)

x_plot = np.arange(np.min(x), np.max(x).astype(np.int8),0.1)

"""polgress = PolynomialRegression.PolynomialRegression(x_train=x_train, y_train=y_train)
polgress.fitForEachLossFunction(x_plot=x_plot, iter=500)"""

polgress = PolynomialRegression.PolynomialRegression(x_train=x_train, y_train=y_train, alpha=0.00001,degree=2)

loss_function_list = ["MAE", "MSE", "RMSE", "R-Squared"]

a = PolynomialRegression.PolynomialRegression(
            x_train=x_train, y_train=y_train, alpha=0.001,degree=2
        )

b = PolynomialRegression.PolynomialRegression(
            x_train=x_train, y_train=y_train, alpha=0.00001,degree=2
        )

c = PolynomialRegression.PolynomialRegression(
            x_train=x_train, y_train=y_train, alpha=0.001,degree=2
        )

d = PolynomialRegression.PolynomialRegression(
            x_train=x_train, y_train=y_train, alpha=0.001,degree=2
        )

loss_functions = [a,b,c,d]

index = 0 
for i in loss_functions:
    i.fit(loss_function_type=loss_function_list[index], iter=500)
    index += 1

fig, axes = plt.subplots(2,2)
fig.set_size_inches((15,15))
color = ["k", "r", "y", "g"]
index = 0
for i in range(2):
    for j in range(2):
        axes[i,j].scatter(x=x, y=y)
        axes[i,j].plot(x_plot, 
            loss_functions[index].predict(
                                x_plot
                            ), color=color[index])
        axes[i,j].set_title(loss_function_list[index])
        index += 1

plt.show()

del polgress