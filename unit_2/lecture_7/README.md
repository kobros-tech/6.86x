# Lecture 7 — Matrix Factorization and Collaborative Filtering

This lecture studies how to predict missing user-movie ratings by assuming that the complete ratings matrix has a low-rank structure.

The main workflow is:

```text
observed ratings
      ↓
learn latent factors U and V
      ↓
reconstruct X̂ = UVᵀ
      ↓
compare candidate ranks k
      ↓
select the best k
      ↓
fit the final model
      ↓
predict genuinely missing ratings
```

The key distinction is:

- $Y$ contains the ratings we observe.
- $X$ is the unknown complete ratings matrix.
- $\hat X$ is the model's predicted complete matrix.
- $U$ and $V$ are learned latent-factor matrices.
- $k$ is the number of latent dimensions and is a hyperparameter.

---

## 1. Notebooks

The notebooks build the idea step by step.

### `01_matrix_factorization.ipynb`

Introduces low-rank matrix factorization and alternating minimization. It shows how the observed ratings are used to learn $U$ and $V$, and how their product produces the predicted matrix:

$$
\hat X=UV^T.
$$

### `02_rank_selection.ipynb`

Introduces rank selection. Several candidate values of $k$ are fitted and compared using reconstruction error on observed and held-out entries. Because the experiment uses a synthetic matrix whose complete values are known, the hidden entries can be compared with the truth.

The experiment selects the value of $k$ with the lowest held-out RMSE.

In a real recommender system, the analogous model-selection step uses a validation split of ratings that are known but temporarily hidden from training.

### `03_final_matrix_completion.ipynb`

Continues from the selected rank. It demonstrates the next practical step: fit one final model with the selected $k$, reconstruct the complete prediction matrix, and use the predicted values to fill genuinely missing entries.

The notebook also makes the important distinction between temporarily hidden known ratings, which can be evaluated, and genuinely missing ratings, whose true values are unknown.

---

## 2. The rating matrix

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

## 3. Low-rank factorization

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

## 4. Latent factors and predicted ratings

A latent factor is a learned representation. It is **not itself a rating**.

For rank $1$, $u_a$ and $v_i$ are scalars, so the predicted rating is

$$
\hat X_{ai}=u_av_i.
$$

For rank $k$, $u_a$ and $v_i$ are $k$-dimensional vectors, so the predicted rating is their inner product:

$$
\hat X_{ai}=u_a^Tv_i.
$$

In coordinates,

$$
\hat X_{ai}=\sum_{r=1}^{k}U_{ar}V_{ir}.
$$

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

## 5. The regularized objective

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

## 6. Alternating minimization

With the movie factors fixed, update one user factor at a time. For user $a$,

$$
u_a=\frac{\sum_{i:(a,i)\in\Omega}Y_{ai}v_i}{\lambda+\sum_{i:(a,i)\in\Omega}v_i^2}.
$$

This keeps the movie factors fixed and chooses the value of $u_a$ that minimizes the regularized squared error over the movies that user $a$ has actually rated.

With the user factors fixed, update each movie factor:

$$
v_i=\frac{\sum_{a:(a,i)\in\Omega}Y_{ai}u_a}{\lambda+\sum_{a:(a,i)\in\Omega}u_a^2}.
$$

The algorithm alternates:

```text
initialize V
    |
    v
update U while V is fixed
    |
    v
update V while U is fixed
    |
    v
repeat until convergence
```

The joint objective is non-convex, so different initializations can lead to different local optima.

---

## 7. Selecting the latent rank $k$

For rank $k$,

$$
\hat X_{ai}=u_a^Tv_i=\sum_{r=1}^{k}U_{ar}V_{ir}.
$$

The value $k$ is the number of latent dimensions. It is a hyperparameter, so we do not learn it as an entry of $U$ or $V$.

Instead, we try several candidate values, such as

$$
k\in\{1,2,3,4,5\},
$$

and evaluate each candidate on held-out known ratings.

For RMSE, lower is better:

$$
RMSE=\sqrt{\frac{1}{N}\sum_{j=1}^{N}(y_j-\hat y_j)^2}.
$$

In the synthetic experiment, the complete matrix is known, so the entries hidden from training can be compared with their true values. The selected rank is

$$
k^*=\arg\min_k RMSE_{\mathrm{val}}(k).
$$

This is the role of `02_rank_selection.ipynb`.

A larger $k$ can reduce reconstruction error on observed ratings while making predictions on unseen ratings worse. Comparing held-out error helps detect this form of overfitting.

---

## 8. What happens after selecting $k$?

Selecting $k^*$ is **model selection**, not the end of the modeling process.

Once $k^*$ has been selected, we fit one final factorization using that rank:

$$
U^*,V^*\leftarrow\text{factorize}(Y,k^*).
$$

Then reconstruct the complete prediction matrix:

$$
\hat X=U^*{V^*}^T.
$$

For an observed rating, the original value remains available. For a genuinely missing rating, the corresponding entry of $\hat X$ is the model's prediction.

For example,

$$
Y=
\begin{bmatrix}
5 & \text{missing} & 2\\
\text{missing} & 4 & \text{missing}\\
3 & \text{missing} & 5
\end{bmatrix}
$$

might lead to

$$
\hat X=
\begin{bmatrix}
5.1 & 3.8 & 2.1\\
3.4 & 4.0 & 3.2\\
3.0 & 3.5 & 5.0
\end{bmatrix}.
$$

The completed matrix keeps the observed ratings and uses the predictions only at missing positions:

$$
\begin{bmatrix}
5 & 3.8 & 2\\
3.4 & 4 & 3.2\\
3 & 3.5 & 5
\end{bmatrix}.
$$

This is the purpose of `03_final_matrix_completion.ipynb`.

---

## 9. Synthetic experiment versus the real world

The notebooks use a synthetic complete matrix because it lets us know the truth.

### Synthetic experiment

We can generate

$$
X_{\mathrm{true}}=U_{\mathrm{true}}V_{\mathrm{true}}^T,
$$

hide some entries, fit the model using the observed entries, and compare the predictions with the hidden truth.

### Real recommender system

The complete matrix is not available. We only have observed ratings:

$$
Y=
\begin{bmatrix}
5 & ? & 2\\
? & 4 & ?\\
3 & ? & 5
\end{bmatrix}.
$$

To evaluate the model, we temporarily hide some **known** ratings and use them as validation/test data. After selecting $k$, we refit the final model with the data available for final training and then predict the ratings that are genuinely missing.

The important distinction is:

- **Validation/test entry:** the true rating is known; we hide it temporarily so that we can measure prediction error.
- **Genuinely missing entry:** the true rating is unknown; we can predict it, but we cannot directly calculate its RMSE.

---

## 10. Complete workflow

Putting the notebooks together:

```text
Observed ratings Y
       |
       v
01_matrix_factorization.ipynb
       |
       | learn U and V for a chosen k
       v
Predicted matrix X̂
       |
       v
02_rank_selection.ipynb
       |
       | compare candidate k values
       v
Best rank k*
       |
       v
03_final_matrix_completion.ipynb
       |
       | fit final U*, V* using k*
       v
X̂ = U*V*ᵀ
       |
       v
Use X̂ only where ratings are genuinely missing
```

When a separate test set is available, it should remain untouched while $k$ and other modeling choices are selected. It is used only for the final evaluation of the selected modeling procedure.

---

## 11. What to remember

1. $Y$ contains observed ratings.
2. $X$ is the unknown complete ratings matrix.
3. $\hat X$ is the model's predicted complete matrix.
4. For rank $1$, $\hat X=uv^T$.
5. For rank $k$, $\hat X=UV^T$.
6. $U$ stores one latent vector for each user, and $V$ stores one latent vector for each movie.
7. A latent factor is a learned representation, not a rating.
8. $k$ is the number of latent dimensions and is selected as a hyperparameter.
9. RMSE measures prediction error; lower RMSE is better.
10. Held-out known ratings are used to select $k$ and evaluate generalization.
11. After selecting $k^*$, fit the final model with that rank.
12. The final reconstruction $\hat X=U^*{V^*}^T$ provides predictions for missing entries.
13. Genuinely missing ratings cannot be evaluated directly because their true values are unknown.
