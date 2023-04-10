import numpy as np
import matplotlib.pyplot as plt

#we generate 1000 values with a standard deviation of 1.0 and an average of 5.0
x = np.random.normal(5.0, 1.0, 1000)

#we generate 1000 values with a standard deviatin of 2.0 and an average of 10.0
y = np.random.normal(10.0, 2.0,1000)

#we display them in xy pair in the order of the sets
plt.scatter(x, y, color="magenta")
plt.show()
