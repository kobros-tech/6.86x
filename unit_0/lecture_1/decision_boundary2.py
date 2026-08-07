import matplotlib.pyplot as plt
import numpy as np

positive = [(4,4),(5,3),(3,5)]
negative = [(1,1),(2,1),(1,2)]

for x, y in positive:
    plt.scatter(x, y, marker="+", s=100)

for x, y in negative:
    plt.scatter(x, y, marker="_", s=100)

# Decision boundary: x + y = 5
x = np.linspace(0, 6, 100)
y = 5 - x

plt.plot(x, y)

plt.xlim(0, 6)
plt.ylim(0, 6)

plt.show()
