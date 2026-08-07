"""
The lecture talks about:
h(x)=sign(w⋅x+b)
"""

import numpy as np

w = np.array([-2, -1, 3, 1])
b = 0

def classify(x):
    score = np.dot(w, x) + b

    if score >= 0:
        return 1
    else:
        return -1

movie = np.array([0, 1, 1, 0])

print(classify(movie))
