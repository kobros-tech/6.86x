"""
Visualizing the Decision Boundary
"""

import matplotlib.pyplot as plt

positive = [(4,4),(5,3),(3,5)]
negative = [(1,1),(2,1),(1,2)]

for x,y in positive:
    plt.scatter(x,y,marker="+",s=100)

for x,y in negative:
    plt.scatter(x,y,marker="_",s=100)

plt.show()
