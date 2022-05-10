import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-10,10,num =100)

fx = []
for i in range(len(x)):
    fx.append(5*x[i]**5+18*x[i]**4+31*x[i]**3-14*x[i]**2+7*x[i]+19)
    ## 5th oder polynomial 5x^5 + 18x^4 + 31x^3 - 14x^2 + 7x - 19

plt.plot(x,fx)
plt.grid()
plt.axvline()
plt.axhline()
plt.show()