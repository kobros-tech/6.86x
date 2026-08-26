# Lecture 7 — Matrix Factorization and Collaborative Filtering

This lecture studies how to predict missing user-movie ratings by assuming that the complete ratings matrix has a low-rank structure.

The key distinction is:

- $Y$ contains the ratings we observe.
- $X$ is the unknown complete ratings matrix.
- $\hat X$ is the model's predicted complete matrix.
- $U$ and $V$ are learned latent-factor matrices.

---

## 1. The rating matrix

Suppose there are $n$ users and $m$ movies. The complete ratings matrix is

$$
X\in\mathbb{R}^{n\times m}.
$$

Entry $X_{ai}$ is the rating of user $a$ for movie $i$.

Let $\Omega$ be the set of observed user-movie pairs:

$$
\Omega=\{(a,i):Y_{ai}\text{ is observed}\}.
$$

For example, an observed-rating matrix can contain missing entries:

$$
Y=
\begin{bmatrix}
5 & \text{missing} & 7\\
1 & 2 & \text{missing}
\end{bmatrix}.
$$

The missing entries are unknown ratings, not zeros.

---

## 2. Low-rank factorization

The central assumption is that users and movies can be represented using a small number of latent factors.

For rank $1$,

$$
X\approx uv^T,
$$

where $u$ contains one scalar latent coordinate for each user and $v$ contains one scalar latent coordinate for each movie.

For rank $k$,

$$
X\approx UV^T,
$$

where

$$
U\in\mathbb{R}^{n\times k},
\qquad
V\in\mathbb{R}^{m\times k}.
$$

Each row of $U$ is a $k$-dimensional latent vector for one user. Each row of $V$ is a $k$-dimensional latent vector for one movie.

The factorization uses

$$
nk+mk=k(n+m)
$$

latent parameters instead of $nm$ independent entries.

When $k$ is much smaller than $n$ and $m$, this is a compact representation of the rating matrix.

---

## 3. Latent factors and predicted ratings

A latent factor is a learned representation. It is **not itself a rating**.

For rank $1$, $u_a$ and $v_i$ are scalars, so the predicted rating is

$$
\hat X_{ai}=u_av_i.
$$

For rank $k$, $u_a$ and $v_i$ are $k$-dimensional vectors, so the predicted rating is their inner product:

$$
\hat X_{ai}=u_a^Tv_i.
$$

The transpose is needed here because the factors are vectors. Their inner product produces one scalar. In coordinates,

$$
\hat X_{ai}=\sum_{r=1}^{k}U_{ar}V_{ir}.
$$

For $k=1$, this inner product reduces to ordinary scalar multiplication, which is why the rank-1 expression is simply $u_av_i$.

The matrix form is

$$
\hat X=UV^T.
$$

Its dimensions are

$$
(n\times k)(k\times m)=n\times m.
$$

Thus $UV^T$ produces one prediction for every user-movie pair.

---

## 4. Example of latent-factor values

Suppose the current movie-factor vector for a rank-1 model is

$$
v=
\begin{bmatrix}
2\\
7\\
8
\end{bmatrix}.
$$

The values $2$, $7$, and $8$ are latent-factor values. They are not copied from the observed ratings matrix. They may be initial values and are subsequently updated by the algorithm.

For user 1, suppose the observed ratings are

$$
Y_{11}=5,
\qquad
Y_{13}=7.
$$

Movie 1 has current factor $v_1=2$, while movie 3 has current factor $v_3=8$. Therefore,

$$
\hat X_{11}=2u_1,
\qquad
\hat X_{13}=8u_1.
$$

The corresponding observed-rating errors are

$$
(5-2u_1)^2
$$

and

$$
(7-8u_1)^2.
$$

Here $5$ and $7$ are observed ratings, while $2$ and $8$ are latent movie factors.

---

## 5. From latent factors to the prediction matrix

For two users and three movies in a rank-1 model, write

$$
u=(u_1,u_2)^T,
\qquad
v=(v_1,v_2,v_3)^T.
$$

Their outer product is

$$
uv^T=
\begin{bmatrix}
 u_1v_1 & u_1v_2 & u_1v_3\\
 u_2v_1 & u_2v_2 & u_2v_3
\end{bmatrix}.
$$

Each entry is one predicted rating. For example,

$$
\hat X_{12}=u_1v_2.
$$

For rank $k$,

$$
\hat X=UV^T.
$$

The model therefore learns the latent factors and uses them to generate the complete prediction matrix.

---

## 6. The regularized objective

For rank $1$, the objective is

$$
J(u,v)=\frac{1}{2}\sum_{(a,i)\in\Omega}\left(Y_{ai}-u_av_i\right)^2+\frac{\lambda}{2}\left(\sum_{a=1}^{n}u_a^2+\sum_{i=1}^{m}v_i^2\right).
$$

The first term measures reconstruction error on **observed ratings only**. The second term regularizes the latent factors.

For rank $k$,

$$
J(U,V)=\frac{1}{2}\sum_{(a,i)\in\Omega}\left(Y_{ai}-u_a^Tv_i\right)^2+\frac{\lambda}{2}\left(\|U\|_F^2+\|V\|_F^2\right).
$$

The missing ratings do not appear in the data-fitting term because their true values are unknown.

---

## 7. Alternating minimization and rank selection

With the movie factors fixed, update one user factor at a time. For user $a$,

$$
u_a=\frac{\sum_{i:(a,i)\in\Omega}Y_{ai}v_i}{\lambda+\sum_{i:(a,i)\in\Omega}v_i^2}.
$$

This keeps the movie factors fixed and chooses the value of $u_a$ that minimizes the regularized squared error over the movies that user $a$ has actually rated.

For user 1 and $v=(2,7,8)$, the observed ratings are $Y_{11}=5$ and $Y_{13}=7$. Therefore,

$$\nu_1=\frac{5(2)+7(8)}{\lambda+2^2+8^2}=\frac{66}{\lambda+68}.
$$

For user 2, the observed ratings are $Y_{21}=1$ and $Y_{22}=2$. Therefore,

$$\nu_2=\frac{1(2)+2(7)}{\lambda+2^2+7^2}=\frac{16}{\lambda+53}.
$$

The observed rating `1` is used in the update for $u_2$; a missing rating is not used.

With the user factors fixed, update each movie factor:

$$
v_i=\frac{\sum_{a:(a,i)\in\Omega}Y_{ai}u_a}{\lambda+\sum_{a:(a,i)\in\Omega}u_a^2}.
$$

The algorithm alternates:

```text
initialize v
    |
    v
update u while v is fixed
    |
    v
update v while u is fixed
    |
    v
repeat until convergence
```

The joint objective is non-convex, so different initializations can lead to different local optima.

For rank $k$,

$$
\hat X_{ai}=u_a^Tv_i=\sum_{r=1}^{k}U_{ar}V_{ir}.
$$

The value $k$ is the number of latent dimensions. It is a hyperparameter, so candidate values can be compared using validation performance. If larger validation score is better,

$$
k^*=\arg\max_k S_{\mathrm{val}}(k).
$$

---

## 8. What to remember

1. $Y$ contains observed ratings.
2. $X$ is the unknown complete ratings matrix.
3. $\hat X$ is the model's predicted complete matrix.
4. For rank $1$, $\hat X=uv^T$.
5. For rank $k$, $\hat X=UV^T$.
6. $U$ stores one latent vector for each user, and $V$ stores one latent vector for each movie.
7. A latent factor is a learned representation, not a rating.
8. For rank $1$, a prediction is $u_av_i$ because the factors are scalars.
9. For rank $k$, a prediction is $u_a^Tv_i$ because the factors are vectors.
10. The loss uses only $(a,i)\in\Omega$ because missing ratings are unknown.
11. Alternating minimization updates one factor block while holding the other fixed.
12. $k$ is the number of latent dimensions and is selected as a hyperparameter.
13. Missing ratings are predicted from the learned $\hat X$.

---

## README writing and math-rendering clues

These notes document the Markdown/LaTeX conventions that work reliably for the GitHub README files in this repository.

### 1. Use one consistent style for mathematical examples

Use plain code fences for every Markdown/LaTeX source example in this section. Do not add a language label such as `markdown` or `text`, because syntax highlighting can make examples appear inconsistently colored.

```
$u_a$
$Y_{ai}$
$\sum_{i=1}^{m}v_i^2$
```

### 2. Keep headings and equations separate

Do not write a heading and a display equation on the same line:

```
# $$ J(u,v)
```

Write the heading normally, then put the equation in its own display block:

```
## The regularized objective

$$
J(u,v)=...
$$
```

### 3. Use `$$` for display equations

For equations that should appear on their own line, use a standalone opening and closing `$$`:

```
$$
X\approx UV^T.
$$
```

For short expressions inside normal text, use single `$` delimiters.

### 4. Do not escape underscores inside LaTeX

Inside math delimiters, write `_` normally:

```
$u_a$
$Y_{ai}$
$\sum_{i=1}^{m}v_i^2$
```

Do not write `\_` inside a LaTeX math expression.

### 5. Be careful when generating README content

When generating Markdown through another programming or tool layer, special character sequences can be interpreted before GitHub sees them. Inspect the raw Markdown source after writing it, especially around LaTeX commands and newlines.

### 6. Keep matrix equations in a simple `bmatrix` block

Use this style for matrices:

```
$$
Y=
\begin{bmatrix}
5 & \text{missing} & 7\\
1 & 2 & \text{missing}
\end{bmatrix}.
$$
```

Use `\\` for row breaks and `&` between columns.

### 7. Check the raw Markdown, not only the rendered page

When a formula looks wrong on GitHub, inspect the actual README source. Look especially for heading markers attached to equations, escaped underscores inside math, missing `$$` delimiters, malformed `\left` / `\right` pairs, and incorrect matrix row breaks.

### 8. Copy the style from the earlier Unit 2 READMEs

Before adding new mathematics, compare the formatting with the existing Lecture 5 and Lecture 6 README files. Keeping the same conventions is safer than introducing a new Markdown/LaTeX style.

### Final rule

**Keep Markdown structure and LaTeX structure separate.** Headings, lists, code fences, and paragraphs belong to Markdown; mathematical notation belongs inside `$...$` or `$$...$$`.
