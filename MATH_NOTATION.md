# Mathematical Notation and Markdown Conventions

This file defines the mathematical-writing conventions used throughout the repository.

The goal is simple: keep equations readable, consistent, and reliably rendered by GitHub MathJax. Individual lecture and project READMEs should focus on the machine-learning ideas themselves rather than repeating these rendering rules.

## 1. Display equations

Use `$$` on separate lines for displayed mathematics:

```markdown
$$
f(x)=\theta^{T}x+b
$$
```

Do not place display equations inside Markdown code fences unless the code fence is specifically showing Markdown syntax, as in this guide.

## 2. Inline mathematics

Use `$...$` for mathematical expressions inside normal prose:

```markdown
The parameter vector is $\theta$.
```

Avoid mixing raw Unicode mathematical symbols such as `θ`, `α`, or `λ` with LaTeX notation in the same document. Use LaTeX consistently: `$\theta$`, `$\alpha$`, `$\lambda$`.

## 3. Vectors and matrices

Use bold notation only when the distinction is mathematically useful. Otherwise, the repository generally uses ordinary symbols such as $x$ and $\theta$ for vectors.

For column vectors:

```markdown
$$
\theta=
\begin{bmatrix}
\theta_1\\
\theta_2\\
\vdots\\
\theta_d
\end{bmatrix}
$$
```

Use `\\` for row breaks inside `bmatrix`, `pmatrix`, `aligned`, and similar multiline environments.

## 4. Transposes

Always brace the transpose superscript, regardless of which symbol is being transposed:

```markdown
$\theta^{T}x$
$x^{T}z$
$X^{T}X$
$\phi(x)^{T}\phi(x')$
```

rather than `$\theta^T x$`, `$x^Tz$`, or similar unbraced forms. This applies even though `T` is a single character and the general rule in Section 9 would not otherwise require braces here — transposes are an explicit exception, kept consistent across the repository regardless of what is being transposed. Keep spacing consistent and avoid unnecessary visual variations.

## 5. Common operators and functions

Use standard LaTeX operators for named mathematical functions:

- sign: `$\mathrm{sign}(x)$`
- argmin: `$\arg\min$`
- argmax: `$\arg\max$`
- exp: `$\exp(x)$`
- log: `$\log(x)$`
- max: `$\max(x)$`
- min: `$\min(x)$`
- trace: `$\operatorname{tr}(A)$` when needed

Use `\mathrm{sign}` rather than plain `sign` when the sign function is written in an equation.

## 6. Norms and absolute values

Use standard delimiters:

```markdown
$\lVert x\rVert_2$
```

for an $L_2$ norm, and

```markdown
$|x|$
```

for an absolute value when no ambiguity exists.

## 7. Sets and domains

Use LaTeX for number sets:

- `$\mathbb{R}$` for the real numbers
- `$\mathbb{R}^d$` for a $d$-dimensional real vector space
- `$\mathbb{N}$` for natural numbers when needed

For example:

```markdown
$f:\mathbb{R}^d\rightarrow\{-1,+1\}$
```

## 8. Greek parameters

Write Greek symbols with LaTeX commands:

| Meaning | Standard notation |
| --- | --- |
| parameter vector | $\theta$ |
| bias | $\theta_0$ or $b$ |
| regularization strength | $\alpha$ |
| learning rate | $\eta$ |
| matrix-factorization rank | $k$ |
| matrix-factorization regularization | $\lambda$ |
| loss/objective | $L$, $J$, or the symbol defined by the lecture |

Do not alternate between forms such as `alpha`, `α`, and `$\alpha$` when referring to the same mathematical quantity. In prose, write `the regularization parameter $\alpha$`.

## 9. Subscripts and superscripts

Use braces whenever a subscript or superscript contains more than one character:

- `$\theta_i$`
- `$\theta_0$`
- `$\theta_{\mathrm{best}}$`
- `$x_i^{(k)}$`
- `$\theta^{T}$`

Use `\mathrm{...}` for textual labels inside mathematical notation, such as `$\theta_{\mathrm{best}}$` or `$J_{\mathrm{train}}$`.

## 10. Piecewise definitions

Keep the entire `cases` environment inside one display block:

```markdown
$$
f(x)=
\begin{cases}
+1 & \text{if } s(x)\geq 0,\\
-1 & \text{if } s(x)<0
\end{cases}
$$
```

## 11. Multiline derivations

Use `aligned` when several lines form one derivation:

```markdown
$$
\begin{aligned}
J(\theta)
&=L(\theta)+\alpha R(\theta)\\
&=\frac{1}{n}\sum_{i=1}^{n}L_i+\alpha R(\theta)
\end{aligned}
$$
```

Keep alignment markers `&` and row separators `\\` inside the display block.

## 12. Fractions and sums

Prefer semantic LaTeX over slash notation in displayed mathematics:

```markdown
$$
\frac{1}{n}\sum_{i=1}^{n}L_i
$$
```

rather than writing a long mathematical fraction with `/`.

## 13. Probability notation

Use `$P(Y=y\mid X=x)$` for conditional probability and `$p(y\mid x)$` for a conditional probability/density when the context defines the convention.

For a probability vector, use a symbol such as $p$ and make its components explicit, for example $p_k$.

For Softmax:

```markdown
$$
p_k=\frac{\exp(z_k)}{\sum_{j=0}^{C-1}\exp(z_j)}
$$
```

Use `$\log$` for logarithms and avoid informal forms such as `ln` unless the distinction is relevant.

## 14. Equality and optimization symbols

Use the mathematical symbol that expresses the intended relationship:

- `$=$` equality
- `$\approx$` approximation
- `$\propto$` proportionality
- `$\leq$`, `$\geq$` inequalities
- `$\rightarrow$` mapping or limit/direction when appropriate
- `$\leftarrow$` parameter update when appropriate
- `$\arg\min$` minimization argument
- `$\arg\max$` maximization argument

For an iterative update, prefer:

```markdown
$$
\theta\leftarrow\theta-\eta\nabla_{\theta}J
$$
```

## 15. Naming model-selection quantities

Keep parameter optimization and hyperparameter selection visibly distinct.

For example:

```markdown
$$
\theta^*(\alpha)=\arg\min_{\theta}J(\theta;\alpha)
$$

$$
\alpha^*=\arg\max_{\alpha}S(\alpha)
$$
```

If a starred quantity is used, define what the star means. Avoid introducing `best`, `optimal`, `star`, and other names interchangeably for the same quantity.

## 16. Code and mathematical notation

Keep Python identifiers in code formatting and mathematical quantities in LaTeX.

For example:

> The regularization strength is represented mathematically by $\alpha$ and in the Python implementation by `alpha`.

This is preferable to writing `alpha (α)` throughout the document.

## 17. GitHub rendering checklist

Before considering a mathematical README complete:

1. Check every display equation for matching `$$` delimiters.
2. Check every multiline environment for `\\` row separators.
3. Check every `\begin{...}` has the corresponding `\end{...}`.
4. Check braces and parentheses are balanced.
5. Keep `cases` and `aligned` completely inside display blocks.
6. Use LaTeX commands for Greek symbols rather than raw Unicode symbols.
7. Keep notation consistent with the lecture's definitions.
8. Inspect the **rendered GitHub page**, not only the raw Markdown.

## 18. Scope of this guide

This file is the repository-wide reference for mathematical Markdown and LaTeX style. Lecture and project READMEs should not repeat this checklist unless a particular document needs a local clarification.

The mathematical notation itself remains part of each lecture's content: a lecture README should define the symbols it introduces and explain what they mean.