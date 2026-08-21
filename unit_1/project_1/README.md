# Project 1 — Automatic Review Analyzer

This project applies the Unit 1 linear-classification ideas to sentiment analysis of product reviews.

The project separates **general-purpose learning algorithms** from **review-specific text representation**:

- `linear_classification.py` contains sparse Perceptron, Average Perceptron, and Pegasos implementations.
- `automatic_review_analyzer.ipynb` demonstrates how review strings become sparse `(x, y)` vectors, trains all three classifiers, selects Pegasos `lambda`, compares learned word weights, and visualizes three decision boundaries together.
- `word_weight_comparison.ipynb` focuses on comparing the word weights learned by the three classifiers.
- `demo.py` is a small dependency-free demonstration of the general classifier module.
- `test_linear_classification.py` tests the reusable algorithms.

## 1. Project goal

Given a product review represented as text, predict whether the review expresses positive or negative sentiment.

The sentiment labels are

$$
y\in\{-1,+1\}
$$

and a review is converted into a feature vector

$$
x\in\mathbb{R}^d.
$$

A linear classifier uses

$$
f(x;\theta)=\theta^T x
$$

and predicts

$$
\hat{y}=\mathrm{sign}\left(\theta^T x\right).
$$

For a classifier with a bias term, the feature vector can be augmented with a constant one so that the same expression represents an affine decision boundary.

## 2. Project structure

```text
project_1/
├── README.md
├── linear_classification.py
├── demo.py
├── test_linear_classification.py
├── automatic_review_analyzer.ipynb
└── word_weight_comparison.ipynb
```

The important architectural boundary is:

```text
review text
    |
    v
notebook: tokenize + vocabulary + vectorize
    |
    v
sparse labeled vectors (x, y)
    |
    v
linear_classification.py
    |
    +--> Perceptron
    +--> Average Perceptron
    +--> Pegasos
```

The learning module does not depend on product reviews. It can be reused with any sparse binary-classification data represented as labeled feature mappings.

## 3. Text representation: bag of words

For a vocabulary

$$
V=\{w_1,w_2,\ldots,w_d\}
$$

a review becomes a vector

$$
x\in\mathbb{R}^d.
$$

The notebook demonstrates a binary bag-of-words representation: a vocabulary word contributes `1` when it occurs in a review. The feature representation is sparse, so only nonzero features are stored.

The vocabulary is built from training reviews only. Validation and test reviews are transformed using that existing vocabulary. This avoids leaking information from validation or test data into the representation.

The conversion functions are intentionally kept in the notebook because they are part of the **application-specific text pipeline**, not the general-purpose learning module.

## 4. General-purpose linear classifiers

`linear_classification.py` contains the shared sparse-vector operations:

- `dot()` — sparse dot product;
- `predict()` — linear prediction;
- `accuracy()` — classification accuracy.

It also contains the three Unit 1 classifiers:

- `perceptron()`;
- `average_perceptron()`;
- `pegasos()`.

### Perceptron

The perceptron changes the parameters when an example is misclassified:

$$
\theta\leftarrow\theta+y_i x_i
$$

when

$$
y_i\theta^T x_i\leq 0.
$$

The quantity $y_i\theta^T x_i$ combines the true label with the classifier score. A positive value means the prediction has the correct sign; a non-positive value means that the example is misclassified or lies on the decision boundary.

### Average Perceptron

The average perceptron averages the parameter trajectory. Conceptually,

$$
\bar{\theta}=\frac{1}{T}\sum_{t=1}^{T}\theta^{(t)}.
$$

The implementation uses lazy timestamp accumulation so that the averaging remains efficient for sparse vectors.

### Pegasos

Pegasos optimizes a regularized linear SVM objective. For a training example $(x,y)$, the hinge loss is

$$
\ell(\theta;(x,y))=\max\{0,1-y\theta^T x\}.
$$

The regularized objective is

$$
J(\theta)=\frac{\lambda}{2}\|\theta\|^2+\frac{1}{m}\sum_{i=1}^{m}\max\{0,1-y_i\theta^T x_i\}.
$$

The learning rate is

$$
\eta_t=\frac{1}{\lambda t}.
$$

The project uses the mini-batch form of Pegasos. For a batch $B_t$, define the active subset

$$
A_t=\{i\in B_t:y_i\theta_t^T x_i<1\}.
$$

The hinge-loss contribution is averaged over the **full batch size** $|B_t|$:

$$
\theta_{t+1/2}=(1-\eta_t\lambda)\theta_t+\frac{\eta_t}{|B_t|}\sum_{i\in A_t}y_i x_i.
$$

Inactive examples contribute zero to the sum, but remain part of the batch average.

The implementation then projects the parameter vector onto the Pegasos ball:

$$
\|\theta\|\leq\frac{1}{\sqrt{\lambda}}.
$$

Shrinkage and projection have different roles: shrinkage is part of every update, while projection enforces the norm constraint.

## 5. Hyperparameter selection

The project notebook explicitly selects the Pegasos regularization parameter rather than fixing it arbitrarily.

For candidate values of $\lambda$, the notebook measures validation accuracy and selects

$$
\lambda^*=\arg\max_{\lambda}\mathrm{ValidationAccuracy}(\lambda).
$$

The important phrase is **the lambda that produces maximum validation accuracy**, not the numerically largest lambda.

Lecture 4 uses the symbol $\alpha$ for its regularization coefficient. The project keeps this notation distinct: Pegasos uses `lambda_`, while the Lecture 4 SVM implementation uses `alpha`. They should not be silently treated as the same variable merely because both control regularization.

Perceptron and Average Perceptron do not have this explicit regularization parameter, so the project does not invent an `alpha` or `lambda` for them.

For real experiments, the test set should remain untouched while selecting the algorithm and hyperparameters.

## 6. Experimental notebook

`automatic_review_analyzer.ipynb` is the main project experiment. It displays the complete workflow:

1. define the toy product-review data;
2. tokenize review strings;
3. construct the training vocabulary;
4. convert reviews into sparse feature vectors and labels;
5. train Perceptron, Average Perceptron, and Pegasos;
6. select the Pegasos `lambda` with validation accuracy;
7. compare final training and validation accuracy;
8. inspect learned word weights;
9. plot the three classifiers' learned decision boundaries together with review points;
10. connect the implementation to the Pegasos research paper.

### Decision-boundary visualization

The complete bag-of-words representation can have many dimensions, so the notebook also provides a deliberately small two-dimensional visualization.

Two selected words are used as coordinates:

- `excellent` → $x_1$;
- `terrible` → $x_2$.

The points remain review examples, but only these two word features are shown. For a two-dimensional classifier,

$$
\theta_1x_1+\theta_2x_2=0
$$

and, when $\theta_2\neq0$,

$$
x_2=-\frac{\theta_1}{\theta_2}x_1.
$$

The notebook overlays the Perceptron, Average Perceptron, and Pegasos boundaries in one coordinate system. This provides geometric intuition without pretending that the full review problem is only two-dimensional.

## 7. Learned word weights

Because the review representation is a bag of words, every vocabulary word corresponds to one component of the learned parameter vector.

A positive weight pushes the score toward the positive class; a negative weight pushes it toward the negative class. The magnitude describes the strength of that feature's contribution within the learned model.

These weights are **not specific to Pegasos**. Perceptron, Average Perceptron, and Pegasos are all linear classifiers over the same vocabulary. Their different training rules and objectives can produce different numerical weights.

`word_weight_comparison.ipynb` visualizes these weights on the same toy vocabulary. This is an interpretability experiment, not evidence that individual words universally determine sentiment.

## 8. Research connection

The Pegasos implementation is connected to:

> Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011). *Pegasos: Primal estimated sub-gradient solver for SVM*. Mathematical Programming, 127(1), 3–30. DOI: 10.1007/s10107-010-0420-4.

The paper presents Pegasos as a stochastic sub-gradient method for the primal SVM objective and discusses its suitability for large-scale learning. Sparse bag-of-words features make this connection particularly natural for text classification.

The project experiments are study-oriented demonstrations rather than claims about general sentiment performance.

## 9. Data format

The original project implementation accepted simple labeled review files. The review-specific parsing and vectorization logic is now demonstrated directly in the notebook so the general-purpose module does not depend on a review file format.

A compatible review dataset can use lines such as:

```text
+1 this product is excellent and useful
-1 this product is disappointing and broken
+1 fast delivery and great quality
-1 poor quality and terrible experience
```

Labels are represented as `+1` and `-1`.

The course review dataset is not committed to this repository.

## 10. Running the demo

From the project directory:

```bash
python3 demo.py
```

The demo uses sparse labeled vectors directly and trains all three classifiers. It demonstrates that the learning module is independent of the review application.

## 11. Running the notebooks

```bash
jupyter notebook automatic_review_analyzer.ipynb
```

For the focused word-weight experiment:

```bash
jupyter notebook word_weight_comparison.ipynb
```

## 12. Running the tests

```bash
python3 -m unittest -v test_linear_classification.py
```

The tests cover:

- Perceptron learning;
- Average Perceptron learning;
- Pegasos learning;
- deterministic Pegasos behavior for a fixed seed;
- the Pegasos projection constraint;
- mini-batch normalization;
- basic input validation.

## 13. Experimental workflow

For a real experiment:

```text
reviews
   |
   +-------------------+
   |                   |
training             test
   |
   v
build vocabulary
   |
   v
vectorize
   |
   v
validation / cross-validation
   |
   v
choose algorithm + hyperparameters
   |
   v
retrain on all training data
   |
   v
final test evaluation
```

The test set must not influence model selection. The vocabulary must also be learned from training data only.

## 14. Relation to Unit 1

The project is the practical synthesis of the unit:

```text
Lecture 1
Binary classification
       |
       v
Lecture 2
Linear classifiers + perceptron
       |
       v
Lecture 3
Hinge loss + margins + regularization
       |
       v
Lecture 4
Generalization + validation + hyperparameters
       |
       v
Project 1
Automatic Review Analyzer
       |
       +--> bag-of-words representation
       +--> Perceptron
       +--> Average Perceptron
       +--> Pegasos / regularized SVM
       +--> validation and hyperparameter selection
       +--> word-weight interpretability
       +--> decision-boundary visualization
```

The project should therefore be studied as an application of the mathematical ideas in the lectures, not as an unrelated NLP exercise.

## Equation-rendering safeguard

The equations in this README follow the conservative GitHub MathJax style used throughout the Unit 1 documentation:

- display equations use `$$` on separate lines;
- inline mathematics uses `$...$`;
- transposes use the explicit form `^{T}`;
- the sign function uses `\mathrm{sign}(...)`;
- equations are not placed inside Markdown code blocks;
- no `\left` or `\right` delimiters are used unless they have a matching pair;
- multiline expressions remain entirely inside a display block;
- Markdown code blocks contain only literal code, not mathematical LaTeX.

When editing this file, verify the **rendered GitHub page**, not only the raw Markdown source.
