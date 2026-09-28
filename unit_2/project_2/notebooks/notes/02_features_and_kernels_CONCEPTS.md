# Notebook 02 — Concepts Not Covered in Units 1–2

This notebook builds on Lecture 6 (feature maps, the kernel trick, polynomial kernels). Two techniques used here were not introduced in any earlier lecture: **Principal Component Analysis (PCA)** and the **Gaussian / RBF kernel**. Both are explained below.

---

## 1. Principal Component Analysis (PCA)

### 1.1 This is not the matrix factorization from Lecture 7

Lecture 7 also involves a matrix decomposition ($X \approx UV^{T}$), but for a different purpose: filling in *missing* ratings from a small number of latent factors. PCA has a different goal: given **fully observed** data (every MNIST pixel is known), find the directions along which the data varies the most, so that each example can be summarized with far fewer numbers while losing as little information as possible.

### 1.2 The problem PCA solves

Each MNIST image is a point $x \in \mathbb{R}^{784}$. Working directly with 784 numbers per image is expensive and most of those dimensions are redundant (neighboring pixels are highly correlated). PCA finds a small set of new coordinate directions — the **principal components** — that capture as much of the spread (variance) of the data as possible.

### 1.3 Centering the data

PCA first centers the data so that each feature has mean zero:

$$
\tilde x_i = x_i - \bar x, \qquad \bar x = \frac{1}{n}\sum_{i=1}^n x_i.
$$

This is what `center_data` computes. Centering matters because PCA is about *directions of variation around the mean*, not about the raw pixel intensities.

### 1.4 The scatter (covariance) matrix and its eigenvectors

Once the data is centered, form the $d\times d$ scatter matrix

$$
S = \tilde X^{T} \tilde X,
$$

where $\tilde X$ stacks the centered examples as rows (this is the same $X^{T}X$ structure that appeared in the closed-form linear regression solution in Lecture 5, but used here for a different purpose). $S$ is proportional to the covariance matrix of the data.

The **principal components** are the eigenvectors of $S$:

$$
S v_j = \lambda_j v_j.
$$

Each eigenvector $v_j$ is a direction in the original 784-dimensional pixel space. Its eigenvalue $\lambda_j$ measures how much the data varies along that direction — a larger eigenvalue means the data spreads out more along $v_j$. `principal_components` sorts the eigenvectors by decreasing eigenvalue, so the first column is the single direction of greatest variance, the second column is the next-greatest direction (orthogonal to the first), and so on.

### 1.5 Projecting onto the top components

To reduce an example to $m$ dimensions, project it onto the top $m$ principal components:

$$
x_{\text{pca}} = V_m^{T} \tilde x, \qquad V_m = [v_1, \dots, v_m].
$$

This is what `project_onto_PC` computes. The result is an $m$-dimensional summary of the image that preserves as much of the original variance as possible for that choice of $m$.

### 1.6 Reconstruction

Because the principal components are orthonormal directions, a projected point can be mapped back into the original 784-dimensional pixel space:

$$
\hat x = V_m x_{\text{pca}} + \bar x.
$$

`reconstruct_PC` implements this. Reconstructing with $m \ll 784$ components and comparing $\hat x$ to the original image is a direct, visual way to see how much information a small number of components preserves.

### 1.7 Why this matters for the notebook

PCA is used here as a **dimensionality-reduction preprocessing step**: instead of feeding raw 784-pixel vectors into a classifier, the notebook can feed in a much smaller PCA representation and compare accuracy versus dimensionality, or simply visualize the digits in 2 dimensions (`plot_PC`) to see how well-separated the classes already look before any classifier is trained.

---

## 2. The Gaussian / RBF kernel

Lecture 6 introduces the kernel trick and works through one example in detail: the **polynomial kernel**,

$$
K(x,x') = (x^{T}x' + c)^p.
$$

This notebook introduces a second, very different kernel: the **radial basis function (RBF)**, also called the **Gaussian kernel**:

$$
K(x,x') = \exp\left(-\gamma \lVert x - x' \rVert^2\right).
$$

### 2.1 What it measures

Unlike the polynomial kernel, which is built from the inner product $x^{T}x'$, the RBF kernel is built from the **squared Euclidean distance** $\lVert x-x'\rVert^2$ between two examples. Two points that are close together get a kernel value near $1$; two points that are far apart get a kernel value near $0$. So $K(x,x')$ behaves like a **similarity score that decays smoothly with distance**, in contrast to the polynomial kernel's similarity based on directional alignment.

### 2.2 The role of gamma

The scale parameter $\gamma > 0$ controls how quickly that similarity decays:

- a **large** $\gamma$ means the kernel value drops to near zero even for nearby points — the model treats only very close points as similar (a narrow, locally sensitive kernel);
- a **small** $\gamma$ means the kernel value stays large even for distant points — the model treats a wide neighborhood of points as similar.

$\gamma$ is a hyperparameter, so — following the same experimental discipline used throughout this repository — it must be chosen using validation data, never the test set (see Lecture 6, Section 5, on validation and feature-map complexity).

### 2.3 Why it still avoids constructing an explicit feature map

Like the polynomial kernel, the RBF kernel corresponds to an inner product in some (much higher-dimensional, in this case infinite-dimensional) feature space $\phi$:

$$
K(x,x') = \phi(x)^{T}\phi(x').
$$

The notebook never needs to construct $\phi(x)$ explicitly — exactly the computational motivation for kernels established in Lecture 6, Sections 6–7. `rbf_kernel` computes $K$ directly from pairwise squared distances, without ever forming the underlying (infinite-dimensional) feature vectors.

---

## 3. What to remember

1. PCA finds orthogonal directions (principal components) of maximum variance in the data by taking the eigenvectors of the centered scatter matrix $X^{T}X$, sorted by eigenvalue.
2. Projecting onto the top $m$ components gives a compact $m$-dimensional summary of each example; reconstruction maps that summary back to the original pixel space.
3. PCA is unrelated to the $UV^{T}$ factorization in Lecture 7 — that method fills in missing entries of a partially observed matrix, while PCA reduces the dimensionality of fully observed data.
4. The RBF kernel $K(x,x') = \exp(-\gamma\lVert x-x'\rVert^2)$ measures similarity by distance rather than by direction, unlike the polynomial kernel.
5. $\gamma$ controls how narrow or wide that similarity neighborhood is, and should be tuned on validation data.
6. Like the polynomial kernel, the RBF kernel avoids ever constructing its (here, infinite-dimensional) feature map explicitly.
