# Lecture 1 — Introduction and Linear Classification

Lecture 1 introduces the machine-learning problem of learning a classifier from labeled examples. It then restricts the hypothesis class to linear classifiers and introduces the perceptron update rule.

Lecture 1 follows the same **notebook + study README** structure used throughout this repository:

- `lecture_1_linear_classification.ipynb` — the executable learning artifact
- `README.md` — the mathematical study guide

The notebook is the primary learning artifact. Open it in Jupyter or VS Code and run it from top to bottom. It contains the calculations, NumPy implementations, visualizations, and experiments.

---

## 1. The machine-learning problem

A supervised classification problem starts with labeled training examples

$$
(x_1,y_1),\ldots,(x_n,y_n)
$$

where each input $x_i$ is represented by a feature vector and, for binary classification,

$$
y_i\in\{-1,+1\}
$$

The goal is to learn a classifier

$$
f:\mathbb{R}^d\rightarrow\{-1,+1\}
$$

that performs well not only on the training examples but also on **new, unseen examples**.

This is the idea of **generalization**.

A model that simply memorizes the training set may achieve very low training error without learning a useful rule for new data. Machine learning therefore asks us to choose a useful **hypothesis class** and learn an appropriate function within that class.

---

## 2. Hypothesis classes and model selection

A hypothesis class is a collection of candidate functions that the learning algorithm is allowed to consider.

A class that is too restrictive may not contain a good classifier. A class that is unnecessarily flexible may fit the training data extremely well while generalizing poorly.

Lecture 1 therefore makes a concrete modeling choice: begin with the class of **linear classifiers**.

This distinction is important:

- the **hypothesis class** describes the family of models we allow;
- the **parameters** determine one particular model inside that family.

Later lectures build on this distinction when they introduce regularization, optimization, and hyperparameter selection.

---

## 3. Linear classifier through the origin

For the first linear classifier, define the score:

$$
s(x)=\theta^{T}x
$$

where the parameter vector is

$$
\theta=
\begin{bmatrix}
\theta_1\\
\theta_2\\
\vdots\\
\theta_d
\end{bmatrix}
$$

The classifier predicts according to the sign of the score:

$$
f(x;\theta)=\mathrm{sign}\left(\theta^{T}x\right)
$$

Equivalently,

$$
f(x;\theta)=
\begin{cases}
+1 & \text{if } \theta^{T}x\geq 0,\\
-1 & \text{if } \theta^{T}x<0
\end{cases}
$$

The score $\theta^{T}x$ is a real number. The prediction $f(x;\theta)$ is one of the two class labels, $-1$ or $+1$.

---

## 4. Decision boundary

The prediction changes when the score crosses zero. Therefore the decision boundary is

$$
\theta^{T}x=0
$$

For two features,

$$
\theta_1x_1+\theta_2x_2=0
$$

which is a straight line through the origin.

In $d$ dimensions, the boundary is a $(d-1)$-dimensional hyperplane.

The two sides of the boundary are determined by the score:

- $\theta^{T}x>0$ gives class $+1$;
- $\theta^{T}x<0$ gives class $-1$;
- $\theta^{T}x=0$ lies on the decision boundary.

---

## 5. Geometry of the parameter vector

The vector $\theta$ is **normal**, or perpendicular, to the decision boundary.

For two features, the boundary is

$$
\theta_1x_1+\theta_2x_2=0
$$

and the normal vector is

$$
\begin{bmatrix}
\theta_1\\
\theta_2
\end{bmatrix}
$$

The dot product also explains the direction of increasing score. Moving in the direction of $\theta$ increases $\theta^{T}x$ most rapidly.

The notebook visualizes both the decision boundary and the parameter vector so that the algebra and geometry can be connected directly.

---

## 6. A limitation of linear classifiers

A linear classifier combines the input features through a weighted sum. It therefore depends strongly on how the data are represented and on the structure allowed by the hypothesis class.

For example, a linear classifier applied to image pixels does not automatically understand that neighboring pixels are spatially related. The model receives a vector of numbers; any useful structure must be represented through the features or through a richer hypothesis class.

The general lesson is:

> A model can only exploit structure that is represented in its input and allowed by its hypothesis class.

This is why choosing the model family is an important part of machine learning.

---

## 7. Training error

After choosing a hypothesis class, we need to select a particular classifier from that class.

For a classifier $f$ and training examples $(x_i,y_i)$, the zero-one training error is

$$
\widehat{E}(f)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\left[f(x_i)\ne y_i\right]
$$

where the indicator is defined by

$$
\mathbf{1}[A]=
\begin{cases}
1 & \text{if } A \text{ is true},\\
0 & \text{if } A \text{ is false}
\end{cases}
$$

Therefore, training error is simply the fraction of training examples that are classified incorrectly.

The notebook calculates this quantity directly from predictions and labels.

---

## 8. Loss functions

Zero-one loss treats every classification mistake equally. More generally, we can define a loss function

$$
L(y,f(x))
$$

that measures how undesirable a prediction is.

The average loss over the training set is

$$
\frac{1}{n}\sum_{i=1}^{n}L\left(y_i,f(x_i)\right)
$$

The zero-one classification error is useful for measuring performance, but it is difficult to optimize directly because the sign function creates a discontinuous objective.

Later lectures introduce optimization-friendly losses and regularization. Lecture 1 provides the conceptual starting point: define what constitutes a bad prediction, then learn parameters that improve the objective.

---

## 9. Perceptron learning

The perceptron is a simple algorithm for learning the parameters of a linear classifier.

Start with a parameter vector $\theta$ and examine training examples one at a time.

For a training example $(x_i,y_i)$:

1. compute the prediction;
2. if the prediction is correct, leave $\theta$ unchanged;
3. if the prediction is wrong, update $\theta$.

The update on a misclassified example is

$$
\theta\leftarrow\theta+y_i x_i
$$

The update is performed **only when the example is misclassified**.

---

## 10. Why does the perceptron update help?

Define the agreement between the classifier and the true label as

$$
z_i=y_i\theta^{T}x_i
$$

If the example is correctly classified, then $z_i>0$. If it is misclassified, then $z_i<0$.

Suppose $(x_i,y_i)$ is currently misclassified and we make the update

$$
\theta'=\theta+y_i x_i
$$

The new agreement is

$$
\begin{aligned}
y_i(\theta')^{T}x_i
&=y_i(\theta+y_i x_i)^{T}x_i\\
&=y_i\theta^{T}x_i+y_i^2x_i^{T}x_i\\
&=y_i\theta^{T}x_i+\lVert x_i\rVert^2
\end{aligned}
$$

Because

$$
\lVert x_i\rVert^2\geq 0
$$

and is positive whenever $x_i\ne 0$, the agreement for the current example increases after the update.

This does **not** mean that every update improves every training example. Changing $\theta$ can make other examples better or worse. The important point is that the update pushes the current mistake toward the correct side of the boundary.

---

## 11. Perceptron learning loop

The learning process can be summarized as:

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

The notebook implements this algorithm directly with NumPy. The core perceptron demonstration does not depend on scikit-learn.

The purpose is to make the parameter update, its effect on the score, and the resulting decision boundary visible.

---

## 12. Two-dimensional example

The notebook uses the following small linearly separable dataset:

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
\end{bmatrix}
$$

For two features, a linear decision boundary has the form

$$
\theta_1x_1+\theta_2x_2=0
$$

The notebook uses this dataset to show how mistakes cause parameter updates and how the learned boundary changes during training.

---

## 13. Bias and the augmented-vector formulation

The first classifier in this lecture has a decision boundary constrained to pass through the origin.

A more flexible linear classifier includes a bias term:

$$
f(x)=\mathrm{sign}\left(\theta^{T}x+\theta_0\right)
$$

Its decision boundary is

$$
\theta^{T}x+\theta_0=0
$$

The bias allows the boundary to move away from the origin.

The bias can be incorporated into an augmented feature vector. Define

$$
\widetilde{x}=
\begin{bmatrix}
x\\
1
\end{bmatrix}
$$

and the augmented parameter vector

$$
\widetilde{\theta}=
\begin{bmatrix}
\theta\\
\theta_0
\end{bmatrix}
$$

Then the score becomes

$$
\widetilde{\theta}^{T}\widetilde{x}
=\theta^{T}x+\theta_0
$$

The bias has not disappeared. It is represented by the final coordinate of the augmented vectors.

Lecture 2 develops this representation more fully and uses it in the perceptron implementation.

---

## 14. Perceptron convergence — preview of Lecture 2

For linearly separable training data, the perceptron convergence theorem guarantees that the perceptron eventually finds a separating parameter vector under the standard assumptions.

For non-linearly-separable data, there is no separating parameter vector for the perceptron to find.

Lecture 1 introduces the algorithm and explains the update. Lecture 2 develops the convergence result and builds a fuller implementation around it.

The progression is:

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

The first four lectures form a deliberate progression:

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

A key distinction becomes important later:

- **parameters**, such as $\theta$, are learned while fitting a model;
- **hyperparameters**, such as a regularization strength $\alpha$, are selected by a model-selection procedure.

This is why Lecture 1 begins with the hypothesis-class question instead of treating parameter fitting as the entire machine-learning problem.

---

## 16. Implementation philosophy

The repository follows the same learning-first progression throughout the lectures:

1. Start with the mathematical definition.
2. Implement the mechanism directly with Python and NumPy.
3. Inspect intermediate calculations and parameter updates.
4. Visualize the geometry when possible.
5. Compare with higher-level libraries only after understanding the underlying operation.

For Lecture 1, the notebook therefore focuses on the classifier, score, decision boundary, training error, agreement, and perceptron update rather than hiding the mechanism behind a library implementation.

---

## 17. What to study

You should be able to explain these relationships without looking them up.

### Binary classification

$$
f:\mathbb{R}^d\rightarrow\{-1,+1\}
$$

### Linear score

$$
s(x)=\theta^{T}x
$$

### Prediction

$$
f(x;\theta)=\mathrm{sign}\left(\theta^{T}x\right)
$$

### Decision boundary

$$
\theta^{T}x=0
$$

### Training error

$$
\widehat{E}(\theta)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\left[f(x_i;\theta)\ne y_i\right]
$$

### Agreement

$$
z_i=y_i\theta^{T}x_i
$$

### Perceptron update

$$
\theta\leftarrow\theta+y_i x_i
$$

### Bias form

$$
f(x)=\mathrm{sign}\left(\theta^{T}x+\theta_0\right)
$$

### Augmented representation

$$
\widetilde{x}=
\begin{bmatrix}
x\\
1
\end{bmatrix}
,\qquad
\widetilde{\theta}=
\begin{bmatrix}
\theta\\
\theta_0
\end{bmatrix}
$$

### Augmented score

$$
\widetilde{\theta}^{T}\widetilde{x}
=\theta^{T}x+\theta_0
$$

Most importantly, you should be able to explain **why** the perceptron update increases the agreement on the current misclassified example.

---

## Running the notebook

Open `lecture_1_linear_classification.ipynb` in Jupyter or VS Code and run the cells from top to bottom.

The notebook is self-contained and uses Python, NumPy, and Matplotlib for the core demonstrations. It generates the calculations, training results, and plots when executed.

### Suggested environment

```bash
python -m pip install numpy matplotlib jupyter
jupyter notebook
```

Then open:

```text
unit_0/lecture_1/lecture_1_linear_classification.ipynb
```

The notebook is the primary executable artifact; this README is the mathematical reference for studying the lecture.

---

## Equation-rendering safeguard

The equations in this README intentionally follow the same conservative GitHub MathJax style used in Lecture 2 and the later lecture READMEs:

- display equations use `$$` on separate lines;
- inline mathematics uses `$...$`;
- transposes use the explicit form `^{T}`;
- the sign function uses `\mathrm{sign}(...)`;
- multiline matrices use explicit `\\` row separators;
- `cases` and `aligned` environments remain entirely inside a display block;
- equations are never placed inside Markdown code blocks;
- no single-backslash matrix row separators are used.

When editing this file, preserve these conventions and verify the **rendered GitHub page**, not only the raw Markdown source.

---

## Source alignment

The material follows the Lecture 1 progression of the course: supervised binary classification, generalization and hypothesis classes, linear classifiers through the origin, decision-boundary geometry, training error, loss, and the perceptron update rule. The repository examples are executable learning demonstrations rather than copies of lecture text.
