# Project 1 — Automatic Review Analyzer

This project applies the linear-classification ideas from Unit 1 to sentiment analysis of product reviews.

The project uses Python files for the reusable implementation and Jupyter notebooks for experimentation and visualization.

## 1. Project goal

Given a product review represented as text, predict whether the review expresses a positive or negative sentiment.

The project connects several Unit 1 ideas:

- binary linear classification;
- bag-of-words feature representation;
- the perceptron algorithm;
- the average perceptron;
- hinge-loss-based SVM learning;
- regularization;
- stochastic sub-gradient optimization;
- training and validation error.

The sentiment labels are represented as:

$$
y \in \{-1,+1\}
$$

A review is converted into a feature vector $x$, and a linear classifier predicts

$$
\hat{y}=\mathrm{sign}\left(\theta^{T}x\right)
$$

For a classifier with a bias term, we can augment the feature vector with a constant one, so the same expression can represent an affine decision boundary.

## 2. Project structure

The project uses Python files for the reusable implementation and Jupyter notebooks for experimentation and visualization.

```text
project_1/
├── README.md
├── review_analyzer.py
├── demo.py
├── test_review_analyzer.py
├── automatic_review_analyzer.ipynb
└── word_weight_comparison.ipynb
```

The Python module contains the reusable learning algorithms, feature extraction, data parsing, and evaluation functions.

`automatic_review_analyzer.ipynb` is the main experimental notebook. It demonstrates the three classifiers, explores the toy review dataset, plots training behavior, examines Pegasos regularization, visualizes learned feature weights, and runs the automated tests.

`word_weight_comparison.ipynb` is a focused follow-up experiment comparing the word weights learned by Perceptron, Average Perceptron, and Pegasos on the same vocabulary. It makes explicit that word weights are not unique to Pegasos: all three are linear classifiers and therefore learn a weight for each vocabulary feature.

The notebooks import the implementation from `review_analyzer.py` rather than duplicating the learning algorithms.

## 3. Text representation: bag of words

A simple representation for a review is a bag of words. We build a dictionary of the words occurring in the training corpus and represent each review by the words it contains.

For a vocabulary

$$
V=\{w_1,w_2,\ldots,w_d\}
$$

a review becomes a vector

$$
x\in\mathbb{R}^d
$$

This project uses a sparse dictionary representation rather than constructing a dense matrix for every review. A feature is stored only when its value is nonzero.

The default feature extractor is a binary bag of words: a word contributes `1` when it occurs in a review. This keeps the connection between text and the linear classifier transparent.

## 4. Binary linear classification

For a review feature vector $x$ and parameter vector $\theta$:

$$
f(x;\theta)=\theta^{T}x
$$

The prediction is

$$
\hat{y}=\mathrm{sign}\left(f(x;\theta)\right)
$$

The sign of the score determines the predicted sentiment.

### Classification error

For a dataset $S$, the classification error of a parameter vector $\theta$ is

$$
\mathrm{error}(\theta;S)=\frac{1}{|S|}\sum_{(x,y)\in S}\mathbf{1}\left[\mathrm{sign}\left(\theta^{T}x\right)\neq y\right]
$$

## 5. Perceptron

The perceptron changes the parameters when an example is misclassified.

For an example $(x_i,y_i)$, the update is

$$
\theta\leftarrow\theta+y_i x_i
$$

when

$$
y_i\theta^{T}x_i\leq 0
$$

The quantity $y_i\theta^{T}x_i$ is useful because it combines the true label and the classifier score. A positive value means the prediction has the correct sign; a non-positive value means the example is misclassified or lies exactly on the decision boundary.

## 6. Average perceptron

The average perceptron records the parameter vectors encountered during training and uses their average for prediction.

If the sequence of learned vectors is

$$
\theta^{(1)},\theta^{(2)},\ldots,\theta^{(T)}
$$

the averaged parameter vector is conceptually

$$
\bar{\theta}=\frac{1}{T}\sum_{t=1}^{T}\theta^{(t)}
$$

Averaging can make the classifier less sensitive to the exact parameter vector obtained at the end of the training trajectory.

## 7. Pegasos

The project also implements the Pegasos algorithm for a regularized linear SVM objective.

For a training example $(x,y)$, the hinge loss is

$$
\ell(\theta;(x,y))=\max\{0,1-y\theta^{T}x\}
$$

For a dataset of $m$ examples, the regularized objective is

$$
J(\theta)=\frac{\lambda}{2}\|\theta\|^2+\frac{1}{m}\sum_{i=1}^{m}\max\{0,1-y_i\theta^{T}x_i\}
$$

Pegasos uses a decreasing learning rate

$$
\eta_t=\frac{1}{\lambda t}
$$

and a stochastic or mini-batch sub-gradient update followed by projection.

For a mini-batch $B_t$ of size $r$, let

$$
A_t=\left\{i\in B_t:y_i\theta_t^{T}x_i<1\right\}
$$

be the active examples whose hinge loss has a nonzero sub-gradient. The update before projection is

$$
\theta_{t+1/2}=(1-\eta_t\lambda)\theta_t+\frac{\eta_t}{r}\sum_{i\in A_t}y_i x_i
$$

The denominator is the **full batch size $r$**, not the number of active examples. Inactive examples contribute zero to the sum, but they remain part of the batch average.

When `batch_size=1`, this reduces to the familiar single-example Pegasos update:

$$
\theta_{t+1/2}=(1-\eta_t\lambda)\theta_t+\eta_t y_t x_t
$$

when the example violates the margin. If it satisfies the margin, the hinge-loss contribution is zero and only the regularization shrinkage remains.

The implementation then projects the parameter vector onto the ball required by the Pegasos algorithm:

$$
\|\theta\|\leq\frac{1}{\sqrt{\lambda}}
$$

This projection is separate from the per-step regularization shrinkage: shrinkage is part of every update, while projection enforces the norm constraint after the update.

### Why this paper matters here

The paper by Shai Shalev-Shwartz, Yoram Singer, Nathan Srebro, and Andrew Cotter presents Pegasos as a stochastic sub-gradient method for the primal SVM objective. Its analysis gives an approximately $\tilde{O}(1/\epsilon)$ iteration requirement for an $\epsilon$-accurate solution, with each iteration operating on a training example. The paper also emphasizes its suitability for large text-classification problems, which makes it a natural research connection for this project.

Reference:

> Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011). *Pegasos: Primal estimated sub-gradient solver for SVM*. Mathematical Programming, 127(1), 3–30. DOI: 10.1007/s10107-010-0420-4.

## 8. Perceptron vs. Pegasos

The two algorithms illustrate an important progression in Unit 1.

| Algorithm | Main idea | Loss / objective | Regularization |
| --- | --- | --- | --- |
| Perceptron | Correct mistakes | Perceptron mistake rule | No explicit regularizer |
| Average perceptron | Average training trajectory | Perceptron mistake rule | No explicit regularizer |
| Pegasos | Optimize regularized SVM objective | Hinge loss | L2 regularization |

The project therefore moves from a mistake-driven learning rule to an optimization-based regularized classifier.

## 9. Learned word weights

Because the review representation is a bag of words, each word corresponds to one component of the learned parameter vector. If the vocabulary is $V=\{w_1,\ldots,w_d\}$, then the classifier learns weights $\theta_1,\ldots,\theta_d$ associated with those words.

For a review vector $x$, the linear score is

$$
f(x;\theta)=\theta^{T}x
$$

so a word's weight contributes to the score when that word is present in the review. A positive weight pushes the score toward the positive class, while a negative weight pushes it toward the negative class. The magnitude indicates the strength of that feature's contribution within the learned model.

These weights are **not specific to Pegasos**. Perceptron, Average Perceptron, and Pegasos all learn a linear weight vector over the same vocabulary. Their training rules and objectives differ, so the numerical weights can differ even though they represent the same kind of model.

The focused `word_weight_comparison.ipynb` notebook visualizes the three learned weight vectors on the same toy dataset. This makes it possible to ask whether the algorithms agree about which words are associated with positive or negative sentiment and how regularization changes the magnitude of Pegasos weights.

This is an interpretability experiment on a deliberately small dataset, not evidence that individual words universally determine sentiment.

## 10. Data format

The implementation accepts simple text files in which each non-empty line contains a label followed by the review text.

Examples:

```text
+1 this product is excellent and useful
-1 this product is disappointing and broken
+1 fast delivery and great quality
-1 poor quality and terrible experience
```

A tab-separated format is also accepted:

```text
+1\tthis product is excellent
-1\tthis product is disappointing
```

Labels must be `+1` or `-1`.

The project does not commit the course's review dataset to the repository. This keeps the repository lightweight and avoids redistributing course-provided data. The code is designed so a compatible local dataset can be supplied explicitly.

## 11. Running the demo

From the project directory:

```bash
python3 demo.py
```

The demo creates a small synthetic review dataset, builds a vocabulary, trains the three classifiers, and prints their training accuracy. This makes the project executable without downloading external data.

For a real review dataset, use the functions in `review_analyzer.py` to load the data, build the vocabulary from the training split, transform both training and validation reviews, train a classifier, and evaluate it.

## 12. Running the research notebooks

From the project directory:

```bash
jupyter notebook automatic_review_analyzer.ipynb
```

For the focused word-weight experiment:

```bash
jupyter notebook word_weight_comparison.ipynb
```

The notebooks import the reusable implementation from `review_analyzer.py`. If the Python module is edited while a notebook kernel is running, restart the kernel or use IPython's autoreload support so that the notebook executes the updated module.

## 13. Running tests

From the project directory:

```bash
python3 -m unittest -v
```

The tests cover tokenization, parsing, deterministic vocabulary construction, sparse feature extraction, the three learning algorithms, deterministic Pegasos behavior for a fixed seed, the Pegasos projection constraint, mini-batch normalization, and an end-to-end training/validation path.
