# Lecture 1 README Rendering Safeguard

This file documents the Markdown/LaTeX conventions used in the Lecture 1 study guide.

## Equation rules

Use GitHub-compatible display math with a blank line before and after every equation:

$$
f(x)=\operatorname{sign}(\theta^T x)
$$

For multiline equations, use explicit row separators `\\`:

$$
\begin{bmatrix}
\theta_1\\
\theta_2\\
\vdots\\
\theta_d
\end{bmatrix}
$$

For cases, keep the complete environment inside one display block:

$$
\hat{y}=\begin{cases}
+1 & \text{if } f(x)\ge 0,\\
-1 & \text{if } f(x)<0
\end{cases}
$$

For aligned derivations, use `aligned` and explicit `\\` row separators:

$$
\begin{aligned}
z_i'
&=y_i(\theta+y_i x_i)^T x_i\\
&=y_i\theta^T x_i+y_i^2x_i^T x_i\\
&=y_i\theta^T x_i+\lVert x_i\rVert^2
\end{aligned}
$$

## Important safeguards

- Do not put raw LaTeX commands such as `\\theta`, `\\begin{bmatrix}`, or `\\operatorname{sign}` inside inline code when the intention is to render mathematics.
- Do not replace matrix row separators with a single backslash. Matrix rows require `\\`.
- Keep display equations between `$$` delimiters rather than mixing display and inline delimiters.
- Use `$...$` only for short expressions inside prose.
- Keep equations on separate Markdown lines.
- Escape backslashes correctly when generating the README programmatically.
- Prefer simple MathJax supported by GitHub over unnecessary custom LaTeX environments.
- After editing, inspect the rendered GitHub page rather than relying only on the raw Markdown source.

## Canonical Lecture 1 equations

### Linear score

$$
s(x)=\theta^T x
$$

### Prediction

$$
f(x;\theta)=\operatorname{sign}(\theta^T x)
$$

### Decision boundary

$$
\theta^T x=0
$$

### Training error

$$
\widehat{E}(\theta)
=\frac{1}{n}\sum_{i=1}^{n}
\mathbf{1}\left[f(x_i;\theta)\ne y_i\right]
$$

### Agreement

$$
y_i\theta^T x_i
$$

### Perceptron update

$$
\theta\leftarrow\theta+y_i x_i
$$

### Bias form

$$
f(x)=\operatorname{sign}(\theta^T x+\theta_0)
$$

### Augmented vectors

$$
\widetilde{x}
=\begin{bmatrix}
x\\
1
\end{bmatrix}
,\qquad
\widetilde{\theta}
=\begin{bmatrix}
\theta\\
\theta_0
\end{bmatrix}
$$

### Augmented score

$$
\widetilde{\theta}^T\widetilde{x}
=\theta^T x+\theta_0
$$

This safeguard is intentionally separate from the study README. It is a maintenance reference for future edits so that Lecture 1 follows the same reliable equation-rendering conventions used throughout Lectures 2–4.
