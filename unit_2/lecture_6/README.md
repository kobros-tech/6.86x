# Lecture 6 — Nonlinear Classification

This directory contains the Lecture 6 study guide and executable demonstrations for **nonlinear classification**.

The lecture develops one central idea in stages:

```text
nonlinearly transform the input
        |
        v
linear classification in feature space
        |
        v
nonlinear decision function in the original space
        |
        v
high-dimensional feature spaces become expensive
        |
        v
kernel functions avoid explicit feature construction
        |
        v
kernel perceptron
```

---

## 1. Why nonlinear classification?

A linear classifier can only separate examples using a linear decision boundary in the original feature space.

Some datasets cannot be separated this way. A simple one-dimensional example is

$$
x=-1 \rightarrow +1,\qquad x=0 \rightarrow -1,\qquad x=1 \rightarrow +1.
$$

No threshold of a linear function of $x$ can classify all three examples correctly.

The lecture's solution is to change the representation rather than abandon the linear learning machinery.

---

## 2. Feature maps

A **feature map** transforms an input $x$ into a new representation:

$$
\phi:x\mapsto\phi(x).
$$

For a scalar input, a simple example is

$$
\phi(x)=
\begin{bmatrix}
 x\\
 x^2
\end{bmatrix}.
$$

A linear classifier in the transformed coordinates is

$$
f(x)=\mathrm{sign}\left(\theta^T\phi(x)+\theta_0\right).
$$

Expanding the dot product gives

$$
f(x)=\mathrm{sign}\left(\theta_1x+\theta_2x^2+\theta_0\right).
$$

This is a **nonlinear classifier in the original $x$ space**, even though it is linear in the transformed feature vector.

### Key principle

$$
\text{linear in feature space} \;\Longrightarrow\; \text{possibly nonlinear in input space}
$$

The lecture emphasizes retaining the original coordinates when adding features so that the transformed representation does not discard the information available before transformation.

---

## 3. Geometry: lifting the data

Consider the points

$$
(-1,+1),\qquad(0,-1),\qquad(1,+1).
$$

Under

$$
\phi(x)=(x,x^2),
$$

they become

$$
(-1,1),\qquad(0,0),\qquad(1,1).
$$

The points are now linearly separable in feature space. For example, a horizontal boundary can separate the point at the origin from the two points with second coordinate $1$.

In the original space, that same classifier corresponds to thresholding a quadratic function.

For a two-dimensional input, the lecture gives another example:

$$
\phi(x_1,x_2)=
\begin{bmatrix}
 x_1\\
 x_2\\
 x_1x_2
\end{bmatrix}.
$$

The additional coordinate can make a previously inseparable dataset linearly separable after the data are lifted into three dimensions.

---

## 4. More features, more expressive models

For a scalar input we can keep adding polynomial coordinates:

$$
\phi(x)=
\begin{bmatrix}
 x\\
 x^2\\
 x^3\\
 \vdots\\
 x^p
\end{bmatrix}.
$$

A linear predictor in this representation becomes a polynomial predictor in the original variable.

For a two-dimensional input, a second-order expansion contains the original coordinates, squared coordinates, and cross terms such as

$$
x_1^2,\qquad x_2^2,\qquad x_1x_2.
$$

The number of coordinates grows rapidly as the input dimension and polynomial order increase.

This creates the central computational problem of the lecture:

> We would like to work with rich, high-dimensional feature representations without explicitly constructing enormous feature vectors.

---

## 5. Model complexity and validation

Adding more features increases the expressive power of the model. A more expressive model can fit training data increasingly well, but training performance alone does not tell us which representation will generalize best.

The lecture therefore connects feature-map complexity to **validation** and **leave-one-out cross-validation**.

For leave-one-out cross-validation, each training example is held out in turn:

1. remove one example;
2. train on the remaining examples;
3. predict the held-out example;
4. record the error;
5. repeat for every example;
6. average the results.

This provides a way to compare different feature representations without choosing the most complicated representation merely because it fits the training set best.

---

## 6. The computational cost of explicit feature maps

Suppose $x\in\mathbb{R}^d$. A polynomial expansion introduces many products of the original coordinates.

Even second-order features already require on the order of $d^2$ terms if all coordinate products are considered. Higher orders grow still faster.

So explicitly storing and manipulating

$$
\phi(x)\in\mathbb{R}^D
$$

can be expensive when $D$ is very large.

The lecture's key observation is that many linear algorithms use feature vectors primarily through **inner products**.

That observation leads to kernels.

---

## 7. Kernel functions

Given a feature map $\phi$, define the kernel

$$
K(x,x')=\phi(x)^T\phi(x').
$$

A kernel therefore represents an inner product in feature space while taking only the original examples as arguments.

For suitable feature maps, $K(x,x')$ can be computed much more cheaply than explicitly constructing both feature vectors.

For example, polynomial feature maps lead to polynomial kernels. A common form discussed by the lecture is

$$
K(x,x')=(1+x^Tx')^p.
$$

The important computational idea is:

$$
\phi(x)^T\phi(x')
\quad\text{can sometimes be computed directly as}\quad
K(x,x')
$$

without explicitly constructing $\phi(x)$ or $\phi(x')$.

This is the **kernel trick**.

---

## 8. Why kernels help linear methods

Suppose a linear method in feature space needs an expression involving

$$
\theta^T\phi(x).
$$

If the learned parameter vector can itself be expressed using training feature vectors, then predictions can be rewritten using inner products between training examples and the new example.

Those inner products can then be replaced by kernel evaluations.

This lets us run a linear algorithm **implicitly** in a high-dimensional feature space while computing only scalar kernel values.

---

## 9. Kernel Perceptron

The lecture derives this idea for the Perceptron.

In feature space, the ordinary Perceptron update is

$$
\theta\leftarrow\theta+y_i\phi(x_i)
$$

whenever example $i$ is misclassified.

Starting from $\theta=0$, the final parameter vector can be written as

$$
\theta=\sum_{j=1}^{n}\alpha_j y_j \phi(x_j)
$$

where $\alpha_j$ counts how many times the Perceptron has updated on training example $j$.

Now consider a new example $x$. Its score is obtained by substituting the expansion of $\theta$ into the inner product:

$$
\theta^T\phi(x) = \left(\sum_{j=1}^{n}\alpha_j y_j \phi(x_j)\right)^T\phi(x)
$$

By linearity of the inner product,

$$
\theta^T\phi(x) = \sum_{j=1}^{n}\alpha_j y_j \phi(x_j)^T\phi(x)
$$

Using the kernel definition,

$$
\theta^T\phi(x) = \sum_{j=1}^{n}\alpha_j y_j K(x_j,x)
$$

So we no longer need to store the high-dimensional $\theta$ explicitly.

### Kernel Perceptron update

Initialize

$$
\alpha_j=0\qquad\text{for all }j
$$

For training example $i$, compute the score

$$
s_i=\sum_{j=1}^{n}\alpha_j y_j K(x_j,x_i)
$$

If

$$
y_i s_i\le 0
$$

make a mistake and update

$$
\alpha_i\leftarrow\alpha_i+1
$$

The classifier can therefore be implemented entirely in terms of the kernel function and the training examples.

---

## 10. Kernel values as similarity

The lecture gives a useful interpretation of the kernel computation.

The value

$$
K(x_j,x_i)
$$

measures how strongly the feature representation of $x_j$ aligns with the feature representation of $x_i$.

The prediction combines:

- how important training example $j$ became, through $\alpha_j$;
- its label $y_j$;
- its kernel similarity to the example being classified.

Thus the kernel Perceptron can be viewed as a classifier whose decision is built from weighted similarities to training examples.

---

## 11. Kernel composition rules

The lecture also shows that new kernels can be constructed from existing kernels.

If $K_1$ and $K_2$ are valid kernels, then their sum is a valid kernel:

$$
K(x,x')=K_1(x,x')+K_2(x,x').
$$

Multiplying a kernel by suitable scalar functions gives another valid kernel. The lecture also discusses products of kernels.

These rules are useful because they let us construct richer feature representations indirectly, without explicitly writing down every coordinate of the corresponding feature map.

---

## 12. What to remember

The core chain of ideas is:

1. Some datasets are not linearly separable in the original representation.
2. A feature map $\phi(x)$ can transform the data into a space where a linear classifier works.
3. A linear classifier in feature space can represent a nonlinear decision function in the original space.
4. Rich feature maps can become computationally expensive to construct explicitly.
5. Kernels compute feature-space inner products without explicitly constructing the feature vectors.
6. Many linear algorithms can therefore be converted into kernel methods.
7. For the Perceptron,

$$
\theta=\sum_j\alpha_j y_j\phi(x_j)
$$

and prediction becomes

$$
f(x)=\mathrm{sign}\left(\sum_j\alpha_j y_j K(x_j,x)\right)
$$

8. The kernel Perceptron updates only the coefficient associated with a misclassified training example.

---

## 13. Demos

The notebooks in this directory follow the lecture's progression rather than hiding the mathematics behind a library implementation.

### `01_feature_maps_and_nonlinear_classification.ipynb`

Visualizes a dataset that is not linearly separable in the original space, applies a feature map, and shows the resulting linear separation in feature space.

### `02_kernel_trick.ipynb`

Constructs polynomial feature vectors for small examples and verifies that their explicit inner product agrees with the corresponding polynomial kernel. It then demonstrates why the kernel computation avoids explicitly constructing the expanded representation.

### `03_kernel_perceptron.ipynb`

Implements the Perceptron in feature space and then derives and implements the equivalent kernel Perceptron using only kernel evaluations and $\alpha$ coefficients.

The goal of these demos is **conceptual transparency**: every important transformation should be visible in the notebook output.

---

## 14. Connection to earlier lectures

Lecture 6 directly builds on the linear-classification work from Unit 1.

```text
Linear classifier
      |
      v
feature map phi(x)
      |
      v
linear classifier in feature space
      |
      v
nonlinear classifier in original space
      |
      v
high-dimensional feature space
      |
      v
kernel function K(x, x')
      |
      v
kernelized linear method
      |
      v
kernel Perceptron
```

The most important conceptual shift is that **the algorithm does not have to know the coordinates of the feature space explicitly**. If it only needs inner products, a kernel can provide those inner products directly.
