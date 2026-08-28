"""
Linear SVM classifiers (Notebook 01, section 4).

The binary SVM connects directly to the hinge-loss / max-margin idea from
the Unit 1-2 lectures. The multiclass version applies a one-vs-rest
strategy so digit classification can be expressed as ten binary margin
problems, one per digit.
"""
import numpy as np
from sklearn.svm import LinearSVC


def one_vs_rest_svm(train_x, train_y, test_x, C=0.1, random_state=0):
    """
    Trains a single binary linear SVM (hinge loss, max-margin separator).

    Args:
        train_x - (n, d) NumPy array
        train_y - (n,) NumPy array of 0/1 labels
        test_x - (m, d) NumPy array

    Returns:
        pred_test_y - (m,) NumPy array of predicted 0/1 labels
    """
    clf = LinearSVC(C=C, random_state=random_state, max_iter=10000)
    clf.fit(train_x, train_y)
    return clf.predict(test_x)


def multi_class_svm(train_x, train_y, test_x, C=0.1, random_state=0):
    """
    Trains a multiclass linear SVM using scikit-learn's built-in one-vs-rest
    strategy: it fits one binary hinge-loss classifier per digit class and
    predicts the class whose classifier is most confident.

    Args:
        train_x - (n, d) NumPy array
        train_y - (n,) NumPy array of integer labels (0-9)
        test_x - (m, d) NumPy array

    Returns:
        pred_test_y - (m,) NumPy array of predicted integer labels
    """
    clf = LinearSVC(C=C, random_state=random_state, max_iter=10000, multi_class="ovr")
    clf.fit(train_x, train_y)
    return clf.predict(test_x)


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)
