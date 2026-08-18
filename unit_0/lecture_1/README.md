# Lecture 1 — Introduction and Linear Classification

Lecture 1 introduces the central machine-learning problem of learning a classifier from labeled examples. It then restricts the hypothesis class to linear classifiers and introduces the perceptron update rule.

The lecture directory is organized as **one executable notebook + one study README**:

- `lecture_1_linear_classification.ipynb` — the executable learning artifact
- `README.md` — the mathematical study guide

The notebook is designed to be run from top to bottom in Jupyter or VS Code. It keeps the calculations, small examples, visualizations, and perceptron implementation visible instead of hiding the mechanism behind a library.

---

## 1. The machine-learning problem

A supervised classification problem starts with labeled training examples

$$
(x_1,y_1),\ldots,(x_n,y_n),
$$

where each input $x_i$ is represented by a vector and, for binary classification,

$$
y_i \in \{-1,+1\}.
$$

We want to learn a classifier

$$
f : \mathbb{R}^d \rightarrow \{-1,+1\}
$$

using the training set so that it also works well on **new, unseen examples**.

This distinction is fundamental:

- fitting the training examples is not the whole goal;
- we want a rule that **generalizes** beyond the training set.

### Why not memorize the training set?

With enough freedom, it is possible to construct a rule that memorizes every training example. Such a rule can have zero training error while behaving unpredictably on examples that were never seen during training.

This motivates a central machine-learning idea: instead of considering every possible function, choose a useful **hypothesis class** and learn within that class.

---

## 2. Model selection

The hypothesis class should be neither unnecessarily large nor unnecessarily restrictive.

A very large class can contain functions that fit the training data extremely well but generalize poorly. A class that is too small may not contain a useful classifier at all.

Lecture 1 therefore makes an explicit modeling choice: start with the class of **linear classifiers**.

This is the beginning of the model-selection perspective that will become more important later in the course.

---

## 3. Linear classifier through the origin

For the first linear classifier, the score is

$$
\theta^T x,
$$

where

$$
\theta =
\begin{bmatrix}
\theta_1\\
\vdots\\
\theta_d
\end{bmatrix}.
$$

The classifier is

$$
f(x;\theta)=\operatorname{sign}(\theta^T x).
$$

Equivalently,

$$
f(x;\theta)=
\begin{cases}
+1 & \text{if } \theta^T x \ge 0,\\
-1 & \text{if } \theta^T x < 0.
\end{cases}
$$

The important point is that the **score** is real-valued, while the final prediction is binary.

---

## 4. Decision boundary

The prediction changes when the score crosses zero. Therefore the decision boundary is

$$
\theta^T x = 0.
$$

In two dimensions,

$$
\theta_1x_1+\theta_2x_2=0,
$$

which is a line through the origin.

In $d$ dimensions, the boundary is a $(d-1)$-dimensional hyperplane.

The two regions are determined by the sign of the score:

- $\theta^T x > 0$ gives class $+1$;
- $\theta^T x < 0$ gives class $-1$;
- $\theta^T x = 0$ lies on the decision boundary.

---

## 5. The geometry of $\theta$

The parameter vector $\theta$ is **normal (perpendicular)** to the decision boundary.

This follows directly from the equation

$$
\theta^T x=0.
$$

In two dimensions, if the boundary is

$$
\theta_1x_1+\theta_2x_2=0,
$$

then $[\theta_1,\theta_2]^T$ points perpendicular to the line.

The dot product also tells us how the score changes as we move through feature space. Moving in the direction of $\theta$ increases $\theta^Tx$ most rapidly.

The notebook visualizes this relationship with a simple two-dimensional dataset.

---

## 6. A limitation of linear classifiers

A linear classifier sees an input as a vector of numbers and combines those numbers using a weighted sum.

For image data, for example, this means that the classifier does not automatically understand spatial relationships between neighboring pixels. If the same permutation of pixel positions is applied consistently to every example, the linear model can still operate on the reordered vectors without knowing that the pixels used to be neighbors.

This is an important modeling lesson:

> A model can only use structure that is represented in its input representation and hypothesis class.

Linear classifiers are useful and mathematically simple, but they do not automatically capture every kind of structure in the data.

---

## 7. Training error

Once a hypothesis class has been chosen, we still need to select a particular classifier from that class.

For a classifier $f$ and training examples $(x_i,y_i)$, the zero-one training error is

$$
\widehat{E}(f)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}\left[f(x_i)\ne y_i\right].
$$

Here,

$$
\mathbf{1}[\text{condition}]
$$

is $1$ when the condition is true and $0$ otherwise.

So the training error is simply the fraction of training examples classified incorrectly.

The notebook computes this directly for a small dataset so that the definition is connected to actual predictions.

---

## 8. General loss functions

Zero-one loss treats every mistake equally. More generally, we can define a loss function

$$
L(y,f(x))
$$

that measures how undesirable a prediction is.

The empirical objective can then be written as

$$
\frac{1}{n}
\sum_{i=1}^{n}
L\left(y_i,f(x_i)\right).
$$

Later lectures will replace the discontinuous zero-one classification error with optimization-friendly losses and regularization. Lecture 1 gives the basic idea: **define what counts as a bad prediction, then choose parameters that reduce the objective**.

---

## 9. Perceptron learning

The perceptron is a simple algorithm for selecting the parameters of a linear classifier.

Start with some parameter vector $\theta$. Visit the training examples one at a time.

For a training example $(x_i,y_i)$:

1. compute the prediction $f(x_i;\theta)$;
2. if it is correct, leave $\theta$ unchanged;
3. if it is wrong, update $\theta$.

The update is

$$
\theta \leftarrow \theta + y_i x_i.
$$

The update is made **only when the example is misclassified**.

---

## 10. Why does the perceptron update help?

Define the agreement between the classifier and the true label as

$$
 y_i\theta^Tx_i.
$$

If the example is classified correctly, this quantity is positive. If it is misclassified, it is negative.

Suppose $(x_i,y_i)$ is currently misclassified. After the update,

$$
\theta' = \theta + y_i x_i.
$$

The new agreement is

$$
\begin{aligned}
y_i(\theta')^Tx_i
&=y_i(\theta+y_ix_i)^Tx_i\\
&=y_i\theta^Tx_i+y_i^2x_i^Tx_i\\
&=y_i\theta^Tx_i+\lVert x_i\rVert^2.
\end{aligned}
$$

Because

$$
\lVert x_i\rVert^2 \ge 0,
$$

and is positive for a nonzero example, the agreement for the current example increases after the update.

This does **not** mean that every update improves every training example. An update can make other examples worse. The important fact is that the current mistake is pushed in the correct direction.

---

## 11. Perceptron algorithm

The learning loop can be summarized as:

```text
initialize theta
      |
      v
visit a training example
      |
      v
compute theta^T x
      |
      v
is the prediction correct?
   /          \\
 yes            no
  |              |
  v              v
keep theta   theta <- theta + y x
   \\          /
    v          v
  continue through the data
          |
          v
      repeat epochs
```

The notebook implements this algorithm directly with NumPy.

No scikit-learn perceptron is required for the core implementation. The goal is to make the update rule and its effect on the parameters explicit.

---

## 12. A small two-dimensional example

Consider the training data

$$
X=
\begin{bmatrix}
4&4\\
5&3\\
3&5\\
1&1\\
2&1\\
1&2
\end{bmatrix}
$$

with labels

$$
 y=
\begin{bmatrix}
+1\\
+1\\
+1\\
-1\\
-1\\
-1
\end{bmatrix}.
$$

These points can be separated by a line. The notebook uses them to show how the parameter vector changes after mistakes and how the decision boundary changes during training.

For a two-feature classifier, the boundary has the form

$$
\theta_1x_1+\theta_2x_2=0.
$$

---

## 13. Bias and the augmented-vector idea

The first linear classifier in this lecture is constrained to have a boundary through the origin.

A more flexible classifier includes a bias term:

$$
 f(x)=\operatorname{sign}(\theta^Tx+\theta_0).
$$

Then the decision boundary becomes

$$
\theta^Tx+\theta_0=0,
$$

which can move away from the origin.

The bias can be incorporated into an augmented vector. Define

$$
\widetilde{x}=
\begin{bmatrix}
x\\1
\end{bmatrix},
\qquad
\widetilde{\theta}=
\begin{bmatrix}
\theta\\\theta_0
\end{bmatrix}.
$$

Then

$$
\widetilde{\theta}^{T}\widetilde{x}
=
\theta^Tx+\theta_0.
$$

This formulation is useful because it lets the bias be handled as another coordinate.

The expanded treatment of the bias and augmented representation appears in Lecture 2, where the perceptron implementation uses it explicitly.

---

## 14. Perceptron convergence — preview of Lecture 2

If the training data are linearly separable, the perceptron will eventually find a separating parameter vector under the standard assumptions.

Lecture 1 introduces the algorithm and the intuition behind the update. Lecture 2 develops the convergence result and builds a fuller perceptron implementation around it.

So the progression is:

```text
Lecture 1
problem formulation
      ↓
linear hypothesis class
      ↓
decision boundary
      ↓
training error
      ↓
perceptron update
      ↓
Lecture 2
perceptron convergence + implementation
```

---

## 15. From Lecture 1 to the later lectures

The first four lectures form a deliberate sequence:

```text
Lecture 1
choose a model family + learn a linear classifier
        ↓
Lecture 2
perceptron + convergence + practical implementation
        ↓
Lecture 3
loss + regularization + gradient-based optimization
        ↓
Lecture 4
validation + cross-validation + hyperparameter selection
```

The distinction between **parameters** and **hyperparameters** becomes important later:

- parameters such as $\theta$ are learned while fitting a model;
- hyperparameters such as a regularization strength $\alpha$ are selected by a model-selection procedure.

This is why Lecture 1 begins with the hypothesis-class question rather than treating parameter fitting as the entire machine-learning problem.

---

## 16. Implementation philosophy

The repository follows a learning-first progression across the lectures:

1. Start with the mathematical definition.
2. Implement the mechanism directly with Python/NumPy.
3. Inspect intermediate calculations and parameter updates.
4. Visualize the geometry when possible.
5. Use higher-level libraries only after understanding the underlying operation.

For Lecture 1, this means the notebook focuses on the classifier, score, decision boundary, training error, and perceptron update rather than hiding the ideas behind scikit-learn.

---

## 17. What to study

You should be able to explain these relationships without looking them up:

### Classification

$$
 f : \mathbb{R}^d\rightarrow\{-1,+1\}
$$

### Linear score

$$
 s(x)=\theta^Tx
$$

### Prediction

$$
 f(x;\theta)=\operatorname{sign}(\theta^Tx)
$$

### Decision boundary

$$
\theta^Tx=0
$$

### Training error

$$
\widehat{E}(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}\left[f(x_i;\theta)\ne y_i\right]
$$

### Agreement

$$
 y_i\theta^Tx_i
$$

### Perceptron update on a mistake

$$
\theta\leftarrow\theta+y_ix_i
$$

And you should understand **why** the update increases the agreement on the current misclassified example.

---

## Running the notebook

Open `lecture_1_linear_classification.ipynb` in Jupyter or VS Code and run the cells from top to bottom.

The notebook is self-contained and uses Python, NumPy, and Matplotlib for the core demonstrations. It generates the calculations and plots when executed.

### Suggested environment

```bash
python -m pip install numpy matplotlib jupyter
jupyter notebook
```

Then open:

```text
unit_0/lecture_1/lecture_1_linear_classification.ipynb
```

The notebook is the primary executable artifact; this README is the compact mathematical reference for studying the lecture.

---

## Source alignment

The material follows the Lecture 1 progression of the course: supervised binary classification, generalization and model selection, linear classifiers through the origin, the geometry of the decision boundary, training error, and the perceptron update rule. The repository's examples are rewritten as executable learning demonstrations rather than copies of lecture text.
