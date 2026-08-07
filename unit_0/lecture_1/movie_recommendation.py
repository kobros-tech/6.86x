"""
The goal:
Predict whether you like a movie (+1) or dislike it (-1).

Features:
Action?
Comedy?
SciFi?
Drama?

Label:
+1 = like
-1 = dislike
"""

from sklearn.linear_model import Perceptron

# Training Data
movies = [
    # Action, Comedy, SciFi, Drama, Label
    [1, 0, 0, 0, -1],  # Fury
    [0, 0, 0, 1, -1],  # Gone Girl
    [0, 0, 1, 0, 1],   # Interstellar
    [0, 0, 1, 1, 1]   # The Martian
]

# Features
X = [movie[:-1] for movie in movies]  # first 4 columns

# Labels
y = [movie[-1] for movie in movies]   # last column

model = Perceptron()

model.fit(X, y)

new_movie = [[0, 1, 1, 0]]

prediction = model.predict(new_movie)

print(prediction)

"""
Meaning:
Action=0
Comedy=1
SciFi=1
Drama=0

Predicted: Like (+1)
"""
