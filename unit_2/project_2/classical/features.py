"""
Feature-representation utilities: PCA and the explicit cubic feature map
phi(x) (Notebook 02, sections 2-3).
"""
import numpy as np


def center_data(X):
    """
    Returns a centered version of X, where each feature has mean 0.

    Returns:
        X_centered - (n, d) NumPy array
        feature_means - (d,) NumPy array of the original column means
    """
    feature_means = X.mean(axis=0)
    return (X - feature_means), feature_means


def principal_components(centered_data):
    """
    Returns the principal-component directions of centered_data, as the
    columns of a (d, d) matrix, sorted by decreasing eigenvalue of the
    (unnormalized) covariance/scatter matrix X^T X.
    """
    scatter_matrix = centered_data.T @ centered_data
    eigen_values, eigen_vectors = np.linalg.eigh(scatter_matrix)
    order = np.argsort(eigen_values)[::-1]
    return eigen_vectors[:, order]


def project_onto_PC(X, pcs, n_components, feature_means):
    """
    Centers X using feature_means, then projects onto the first
    n_components columns of pcs (the leading principal directions).

    Returns:
        an (n, n_components) NumPy array.
    """
    X_centered = X - feature_means
    return X_centered @ pcs[:, :n_components]


def cubic_features(X):
    """
    Explicit cubic feature map phi(x): returns, for every row of X, all
    monomials of total degree exactly 3 in the (d + 1) augmented
    coordinates (X plus a constant 1), scaled so that
        phi(x) . phi(z) == (x . z + 1)^3.

    Only used on small illustrative examples in the notebook: for the full
    784-dimensional MNIST vectors, the output dimension grows as O(d^3)
    and is intractable to construct explicitly -- which is exactly the
    motivation for the polynomial *kernel* introduced right afterwards.
    """
    n, d = X.shape
    X_withones = np.ones((n, d + 1))
    X_withones[:, :-1] = X
    new_d = int((d + 1) * (d + 2) * (d + 3) / 6)

    new_data = np.zeros((n, new_d))
    for x_i in range(n):
        X_i = X[x_i]
        X_i = X_i.reshape(1, X_i.size)
        col_index = 0

        if d > 2:
            comb_2 = X_i.T @ X_i
            unique_2 = comb_2[np.triu_indices(d, 1)]
            unique_2 = unique_2.reshape(unique_2.size, 1)
            comb_3 = unique_2 @ X_i
            keep_m = np.zeros(comb_3.shape)
            index = 0
            for i in range(d - 1):
                keep_m[index + np.arange(d - 1 - i), i] = 0
                tri_keep = np.triu_indices(d - 1 - i, 1)
                correct_0 = tri_keep[0] + index
                correct_1 = tri_keep[1] + i + 1
                keep_m[correct_0, correct_1] = 1
                index += d - 1 - i

            unique_3 = np.sqrt(6) * comb_3[np.nonzero(keep_m)]
            new_data[x_i, np.arange(unique_3.size)] = unique_3
            col_index = unique_3.size

        newdata_colindex = col_index
        for j in range(d + 1):
            new_data[x_i, newdata_colindex] = X_withones[x_i, j] ** 3
            newdata_colindex += 1
            for k in range(j + 1, d + 1):
                new_data[x_i, newdata_colindex] = (
                    X_withones[x_i, j] ** 2 * X_withones[x_i, k] * (3 ** 0.5)
                )
                newdata_colindex += 1
                new_data[x_i, newdata_colindex] = (
                    X_withones[x_i, j] * X_withones[x_i, k] ** 2 * (3 ** 0.5)
                )
                newdata_colindex += 1
                if k < d:
                    new_data[x_i, newdata_colindex] = (
                        X_withones[x_i, j] * X_withones[x_i, k] * (6 ** 0.5)
                    )
                    newdata_colindex += 1

    return new_data


def plot_PC(X, pcs, labels, feature_means, ax=None):
    """
    Projects X onto the first two principal components and scatter-plots
    each point labeled by its digit.
    """
    import matplotlib.pyplot as plt

    pc_data = project_onto_PC(X, pcs, n_components=2, feature_means=feature_means)
    text_labels = [str(z) for z in labels.tolist()]
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(pc_data[:, 0], pc_data[:, 1], alpha=0, marker=".")
    for i, txt in enumerate(text_labels):
        ax.annotate(txt, (pc_data[i, 0], pc_data[i, 1]))
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    return ax


def reconstruct_PC(x_pca, pcs, n_components, feature_means):
    """Reconstructs an image (or batch of images) from its PCA representation."""
    return x_pca @ pcs[:, :n_components].T + feature_means
