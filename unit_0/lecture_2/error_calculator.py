import numpy as np


def training_error(predictions, labels):
    """
    Calculate training error.

    E_n(h) =
    (# misclassified examples) / n
    """

    if len(predictions) != len(labels):
        raise ValueError(
            "predictions and labels must have same length"
        )

    mistakes = 0

    for prediction, label in zip(predictions, labels):

        if prediction != label:
            mistakes += 1

    return mistakes / len(labels)


def accuracy(predictions, labels):
    """
    Accuracy =
    (# correct predictions) / n
    """

    if len(predictions) != len(labels):
        raise ValueError(
            "predictions and labels must have same length"
        )

    correct = 0

    for prediction, label in zip(predictions, labels):

        if prediction == label:
            correct += 1

    return correct / len(labels)


def confusion_counts(predictions, labels):
    """
    Return TP, TN, FP, FN
    """

    tp = tn = fp = fn = 0

    for prediction, label in zip(predictions, labels):

        if prediction == 1 and label == 1:
            tp += 1

        elif prediction == -1 and label == -1:
            tn += 1

        elif prediction == 1 and label == -1:
            fp += 1

        elif prediction == -1 and label == 1:
            fn += 1

    return tp, tn, fp, fn


def main():

    labels = np.array([
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1
    ])

    predictions = np.array([
        1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        -1
    ])

    error = training_error(
        predictions,
        labels
    )

    acc = accuracy(
        predictions,
        labels
    )

    tp, tn, fp, fn = confusion_counts(
        predictions,
        labels
    )

    print("===== RESULTS =====")
    print(f"Training Error : {error:.2f}")
    print(f"Accuracy       : {acc:.2f}")
    print()
    print("Confusion Counts")
    print(f"TP = {tp}")
    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")


if __name__ == "__main__":
    main()
