"""
Training examples:
"This product is amazing"      -> Positive
"I love it"                    -> Positive
"Waste of money"               -> Negative
"Terrible quality"             -> Negative

Convert text into feature vectors.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron

reviews = [
    "This product is amazing",
    "I love it",
    "Waste of money",
    "Terrible quality"
]

labels = [1, 1, -1, -1]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(reviews)

model = Perceptron()

model.fit(X, labels)

# To test the prediction of already known
for review in reviews:
    X_test = vectorizer.transform([review])
    print(review, model.predict(X_test))

new_review = ["Amazing quality product"]

X_new = vectorizer.transform(new_review)

prediction = model.predict(X_new)

print(prediction)
