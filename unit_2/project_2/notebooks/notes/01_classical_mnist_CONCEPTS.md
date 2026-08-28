# Notebook 01 — Concepts Not Covered in Units 1–2

This notebook compares linear regression, a linear SVM, and Softmax regression on MNIST. Linear regression and the binary-margin idea behind the SVM were covered in Lecture 5 and Unit 1. Two things in this notebook go beyond that material: **multiclass SVM via one-vs-rest**, and **Softmax regression**. Both are explained here.

---

## 1. From binary to multiclass: one-vs-rest

Unit 1 only defines a **binary** linear classifier: a single hyperplane separating two classes with hinge loss and a margin.

MNIST has ten classes (digits `0`–`9`). The **one-vs-rest** (also called one-vs-all) strategy turns a binary method into a multiclass method without changing the underlying binary algorithm:

1. For each digit $k \in \{0,\dots,9\}$, train one binary classifier that treats class $k$ as the positive class and every other digit as negative.
2. This produces ten separate binary decision functions $f_0,\dots,f_9$, each with its own weight vector.
3. To classify a new example $x$, evaluate all ten classifiers and predict the class whose classifier is most confident:

$$
\hat y = \arg\max_{k} \; f_k(x).
$$

So a ten-class problem is reduced to ten independent binary-margin problems, each one exactly the kind of hinge-loss SVM already covered in Unit 1. `classical/svm.py` uses `sklearn`'s built-in `multi_class="ovr"` option, which performs this training procedure automatically.

### Trade-off to be aware of

Because each binary classifier is trained independently, nothing forces the ten decision functions to be on a comparable numerical scale, so "most confident" is only an approximate notion of confidence, not a calibrated probability. Softmax regression, described next, is one way to get an explicitly probabilistic multiclass model instead.

---

## 2. Softmax regression

Softmax regression (also called multinomial logistic regression) is a **direct multiclass model**: it does not train ten separate classifiers, but rather one model that outputs a full probability distribution over all ten classes at once.

### 2.1 From scores to probabilities

The model keeps one parameter vector $\theta_k$ per class $k$, exactly the way the multiclass SVM does. For an input $x$, each class gets a raw score

$$
z_k = \theta_k^{T} x.
$$

Unlike the SVM, Softmax converts these scores into a genuine probability distribution using the **softmax function**:

$$
p_k = \frac{\exp(z_k)}{\displaystyle\sum_{j=0}^{9}\exp(z_j)}.
$$

Two properties make this a valid probability distribution:

- every $p_k > 0$, because $\exp(\cdot)$ is always positive;
- $\sum_k p_k = 1$, because the denominator is exactly the sum of all the numerators.

The predicted class is the one with the largest probability:

$$
\hat y = \arg\max_k p_k.
$$

Because $\exp$ is monotonically increasing, this is equivalent to picking the class with the largest raw score $z_k$ — the probabilities are a reinterpretation of the scores, not a different decision rule.

### 2.2 The temperature parameter

`classical/softmax.py` divides every score by a **temperature** $\tau$ before applying softmax:

$$
p_k = \frac{\exp(z_k/\tau)}{\displaystyle\sum_{j=0}^{9}\exp(z_j/\tau)}.
$$

Temperature does not change which class is predicted (it does not change the arg-max), but it changes how *sharply peaked* the probability distribution is:

- as $\tau \to 0^+$, the distribution approaches a one-hot vector on the arg-max class (very confident);
- as $\tau \to \infty$, the distribution approaches a uniform distribution over all ten classes (very unsure);
- $\tau = 1$ recovers ordinary softmax.

The notebook uses $\tau$ to control how confidently the model reports its predictions, independent of the decision itself.

### 2.3 The training objective: negative log-likelihood

Unit 1 trains classifiers with hinge loss; Lecture 5 trains regression with squared loss. Softmax regression uses a different loss, appropriate for a probability output: the **negative log-likelihood** of the true label.

For one example with true label $y$, the loss is

$$
L_i(\theta) = -\log p_{y}.
$$

This penalizes the model heavily when it assigns low probability to the correct class, and only lightly when it assigns high probability to the correct class ($-\log(1) = 0$). Averaged over $n$ training examples and combined with an $L_2$ regularization term on $\theta$ (the same kind of penalty from Lecture 5), the empirical objective implemented in `compute_cost_function` is

$$
J(\theta) = -\frac{1}{n}\sum_{i=1}^{n}\log p^{(i)}_{y_i} \;+\; \frac{\lambda}{2}\sum_k \|\theta_k\|^2.
$$

This is minimized with gradient descent, the same optimization pattern from Lecture 5 — only the loss function and the model's output (a probability vector instead of a single score) are new.

---

## 3. What to remember

1. One-vs-rest reduces a $k$-class problem to $k$ independent binary hinge-loss problems, reusing Unit 1's SVM directly.
2. Softmax regression instead trains one model that outputs a full probability distribution over all classes.
3. Softmax converts raw per-class scores into probabilities that are positive and sum to 1.
4. Temperature $\tau$ rescales the scores before softmax; it changes the confidence of the output distribution but not the predicted class.
5. Softmax regression is trained by minimizing the negative log-likelihood of the true labels (plus $L_2$ regularization), using the same gradient-descent machinery introduced in Lecture 5.
