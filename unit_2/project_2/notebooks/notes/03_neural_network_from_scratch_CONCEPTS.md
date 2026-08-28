# Notebook 03 — Concepts Not Covered in Units 1–2

**Neural networks are formally covered in Unit 3.** This notebook is a hands-on preview: it builds one tiny network from scratch so that forward propagation and backpropagation are concrete before Unit 3 develops the general theory. This note defines just enough vocabulary to read the code and follow the notebook — the full mathematical treatment (general architectures, the chain rule in general form, choices of activation and loss functions, and why these choices matter) belongs to Unit 3 and is intentionally not duplicated here.

---

## 1. What a neural network is, informally

Every model studied so far — linear regression, linear classifiers, Softmax regression — computes a prediction from a **single** linear transformation of the input (possibly followed by a fixed nonlinearity like softmax). A neural network instead **chains several transformations together**, alternating linear steps with a nonlinear function applied elementwise. Each such step is called a **layer**.

The network implemented in `neural_networks/from_scratch.py` has the following architecture:

$$
\text{2 inputs} \;\to\; \text{3 hidden ReLU units} \;\to\; \text{1 linear output}.
$$

Each of the 3 "hidden units" is just a single number, computed the same way a linear model's score is computed, but then passed through a nonlinear function before being used by the next layer.

## 2. ReLU: a new nonlinearity

The **rectified linear unit (ReLU)** is defined elementwise as

$$
\mathrm{ReLU}(x) = \max(0, x).
$$

It passes positive values through unchanged and clips negative values to zero. Its derivative is

$$
\frac{d}{dx}\mathrm{ReLU}(x) =
\begin{cases}
1 & x > 0 \\
0 & x < 0,
\end{cases}
$$

with the value at $x=0$ handled by convention (the code uses $0$). This derivative is what makes the hidden layer's contribution to the gradient either "pass through" or "block," depending on whether that unit was active for a given input — this is the mechanism `rectified_linear_unit_derivative` implements.

## 3. Forward propagation

**Forward propagation** simply means computing the network's output by applying each layer in order: linear step, then ReLU, then the next linear step. In the code, this is `NeuralNetwork.forward`, which records every intermediate value (`z1`, `a1`, `z2`, `y_hat`) because those intermediate values are needed again in the next step.

## 4. Backpropagation, at the level needed to read this notebook

**Backpropagation** is the procedure for computing how much each parameter contributed to the final loss, by working backward through the layers one at a time and repeatedly applying the chain rule. Lecture 5 already computed one gradient by hand — the gradient of the squared-error objective with respect to a single linear model's parameters. Backpropagation is the same chain-rule idea, applied through *multiple* layers instead of one.

Concretely, for the squared loss $L = \tfrac12(\hat y - y)^2$ used here, the gradient with respect to the output layer is computed first, and then that gradient is propagated backward through each preceding layer:

$$
\frac{\partial L}{\partial \hat y} \;\longrightarrow\; \frac{\partial L}{\partial z_2} \;\longrightarrow\; \frac{\partial L}{\partial W_2},\,\frac{\partial L}{\partial b_2} \;\longrightarrow\; \frac{\partial L}{\partial a_1} \;\longrightarrow\; \frac{\partial L}{\partial z_1} \;\longrightarrow\; \frac{\partial L}{\partial W_1},\,\frac{\partial L}{\partial b_1}.
$$

`NeuralNetwork.backward` implements exactly this chain, one line per arrow above. The general derivation of backpropagation for arbitrary networks — why this always works, and how it scales to many layers — is developed properly in Unit 3; this notebook's purpose is only to make one small, concrete instance of it fully visible.

## 5. Gradient checking

`numerical_gradient_check` is a debugging technique, not a new modeling idea: it estimates a derivative using a **finite difference**,

$$
\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i+\epsilon) - L(\theta_i-\epsilon)}{2\epsilon},
$$

and compares that numerical estimate to the analytic gradient produced by `backward`. If the two agree closely, it's good evidence the backpropagation code is correct. This is a general-purpose sanity check that applies to any differentiable model, not something specific to neural networks.

---

## 6. What to remember (operational, for this notebook only)

1. A neural network chains linear steps with a nonlinear activation function between them.
2. ReLU$(x) = \max(0,x)$ is the nonlinearity used here; it passes positive inputs through and zeroes out negative inputs.
3. Forward propagation computes the output layer by layer, saving intermediate values.
4. Backpropagation computes gradients layer by layer, working backward from the loss, using the chain rule at each step — the same principle as the Lecture 5 gradient derivation, extended through multiple layers.
5. A numerical gradient check compares the analytic backprop gradient against a finite-difference estimate as a correctness check.

The full theory of neural networks — general architectures, other activation and loss functions, and why backpropagation generalizes to arbitrary networks — is covered in Unit 3.
