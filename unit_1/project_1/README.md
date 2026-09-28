# Project 1 — Automatic Review Analyzer

This project applies the Unit 1 linear-classification ideas to sentiment analysis of product reviews.

The project separates **general-purpose learning algorithms** from **review-specific data preparation**:

- `linear_classification.py` contains sparse Perceptron, Average Perceptron, and Pegasos implementations.
- `review_data.py` contains the UCI review loading, reproducible stratified split, tokenization, vocabulary construction, and vectorization pipeline shared by the notebooks.
- `automatic_review_analyzer.ipynb` is the main experiment: it builds the sparse representation, performs a controlled comparison, selects Pegasos `lambda`, evaluates the final models, and examines word weights.
- `word_weight_comparison.ipynb` focuses on the final learned word weights using the same split, vocabulary, training budget, and selected Pegasos `lambda` as the main experiment.
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
f(x;\theta)=\theta^{T} x
$$

and predicts

$$
\hat{y}=\mathrm{sign}\left(\theta^{T} x\right).
$$

For a classifier with a bias term, the feature vector can be augmented with a constant one so that the same expression represents an affine decision function.

## 2. Project structure

```text
project_1/
├── README.md
├── linear_classification.py
├── review_data.py
├── demo.py
├── test_linear_classification.py
├── automatic_review_analyzer.ipynb
└── word_weight_comparison.ipynb
```

The architectural boundary is:

```text
review text
    |
    v
review_data.py
    |
    +--> tokenize
    +--> vocabulary
    +--> vectorize
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

The project uses a binary bag-of-words representation: a vocabulary word contributes `1` when it occurs in a review. The feature representation is sparse, so only nonzero features are stored.

The resulting feature space is **high-dimensional**: each vocabulary word corresponds to one dimension. In the current 3,000-review experiment, the training vocabulary contains approximately 1,766 words, so each review is represented as a sparse vector in approximately $\mathbb{R}^{1766}$. The classifiers operate on this full feature space during training and evaluation; there is no 2-D projection involved in the actual classification experiment.

The vocabulary is built from training reviews only. Validation and test reviews are transformed using that existing vocabulary. This avoids leaking information from validation or test data into the representation.

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
y_i\theta^{T} x_i\leq 0.
$$

### Average Perceptron

The average perceptron averages the parameter trajectory. Conceptually,

$$
\bar{\theta}=\frac{1}{T}\sum_{t=1}^{T}\theta^{(t)}.
$$

The implementation uses lazy timestamp accumulation so that the averaging remains efficient for sparse vectors.

### Pegasos

Pegasos optimizes a regularized linear SVM objective. For a training example $(x,y)$, the hinge loss is

$$
\ell(\theta;(x,y))=\max\{0,1-y\theta^{T} x\}.
$$

The regularized objective is

$$
J(\theta)=\frac{\lambda}{2}\|\theta\|^2+\frac{1}{m}\sum_{i=1}^{m}\max\{0,1-y_i\theta^{T} x_i\}.
$$

The learning rate is

$$
\eta_t=\frac{1}{\lambda t}.
$$

The project uses the mini-batch form of Pegasos. For a batch $B_t$, define the active subset

$$
A_t=\{i\in B_t:y_i\theta_t^{T} x_i<1\}.
$$

The hinge-loss contribution is averaged over the **full batch size** $|B_t|$:

$$
\theta_{t+1/2}=(1-\eta_t\lambda)\theta_t+\frac{\eta_t}{|B_t|}\sum_{i\in A_t}y_i x_i.
$$

The implementation then projects the parameter vector onto the Pegasos ball:

$$
\|\theta\|\leq\frac{1}{\sqrt{\lambda}}.
$$

## 5. Hyperparameter selection

The main notebook explicitly selects the Pegasos regularization parameter rather than fixing it arbitrarily.

For candidate values of $\lambda$, validation accuracy is measured and the best value is selected:

$$
\lambda^*=\arg\max_{\lambda}\mathrm{ValidationAccuracy}(\lambda).
$$

If multiple values have the same maximum validation accuracy, the project selects the smallest one.

Lecture 4 uses the symbol $\alpha$ for its regularization coefficient. The project keeps this notation distinct: Pegasos uses `lambda_`, while the Lecture 4 SVM implementation uses `alpha`.

The test set must remain untouched while selecting the algorithm and hyperparameters.

## 6. Experimental workflow

`automatic_review_analyzer.ipynb` uses the complete 3,000-review UCI dataset and demonstrates:

1. reproducible stratified train/validation/test splitting;
2. sparse bag-of-words feature construction;
3. a **controlled comparison** using the same training budget and a fixed Pegasos `lambda`;
4. a separate **Pegasos hyperparameter selection** experiment using validation accuracy;
5. a **final model comparison** after retraining on train + validation;
6. final test evaluation on the untouched test set;
7. learned word-weight analysis.

### Controlled vs tuned comparison

These are intentionally different experiments.

**Controlled comparison:**

- same training data;
- same epoch budget;
- fixed Pegasos `lambda_=1e-3`;
- compare the learning rules directly.

**Tuned final comparison:**

- select Pegasos `lambda*` using validation accuracy;
- retrain the final models on train + validation;
- evaluate the final models on the test set.

This distinction prevents a hyperparameter-selection result from being confused with a controlled algorithm comparison.

## 7. Learned word weights

Because the review representation is a bag of words, every vocabulary word corresponds to one component of the learned parameter vector.

A positive weight pushes the score toward the positive class; a negative weight pushes it toward the negative class. The actual contribution of feature $j$ is

$$
\theta_jx_j.
$$

With binary bag-of-words, a present word has $x_j=1$, so its weight is directly its contribution to the score.

These weights are **not specific to Pegasos**. Perceptron, Average Perceptron, and Pegasos are all linear classifiers over the same vocabulary. Their different training rules and objectives can produce different numerical weights.

`word_weight_comparison.ipynb` analyzes the **final models** using the same split and vocabulary as the main notebook and the selected Pegasos `lambda* = 5e-3` found by the experiment.

The word-weight analysis is an interpretability experiment, not evidence that individual words universally determine sentiment.

## 8. Research connection

The Pegasos implementation is connected to:

> Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011). *Pegasos: Primal estimated sub-gradient solver for SVM*. Mathematical Programming, 127(1), 3–30. DOI: 10.1007/s10107-010-0420-4.

The project experiments are study-oriented demonstrations rather than claims about general sentiment performance.

## 9. Data format

The review-specific data preparation is isolated from the general-purpose learning algorithms. A compatible labeled review file can use lines such as:

```text
+1 this product is excellent and useful
-1 this product is disappointing and broken
+1 fast delivery and great quality
-1 poor quality and terrible experience
```

Labels are represented as `+1` and `-1`.

## 10. Running the demo and notebooks

```bash
python3 demo.py
```

```bash
jupyter notebook automatic_review_analyzer.ipynb
jupyter notebook word_weight_comparison.ipynb
```

The notebooks download the UCI dataset if it is not already present under `data/sentiment_labelled_sentences/`.

## 11. Running the tests

```bash
python3 -m unittest -v test_linear_classification.py
```

The tests cover Perceptron, Average Perceptron, Pegasos, deterministic Pegasos behavior for a fixed seed, the Pegasos projection constraint, mini-batch normalization, and basic input validation.

## 12. Experimental safeguards

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

The test set must not influence model selection. The vocabulary must also be learned from training data only. A fixed random seed is used for the Project 1 split and Pegasos mini-batch shuffling so that the experiments are reproducible.

## 13. Relation to Unit 1

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
```

The project should therefore be studied as an application of the mathematical ideas in the lectures, not as an unrelated NLP exercise.

For repository-wide Markdown and LaTeX conventions, see [MATH_NOTATION.md](../../MATH_NOTATION.md).
