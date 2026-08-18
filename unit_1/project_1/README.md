# Project 1 — Automatic Review Analyzer

This project applies the linear-classification ideas from Unit 1 to sentiment analysis of product reviews.

The implementation is intentionally **Python-first** rather than notebook-first. A project is easier to test, reuse, and extend when the learning algorithms and feature extraction live in ordinary Python modules. A notebook can be added later as a presentation layer, but it should not be the source of truth for the implementation.

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

## 2. Why Python files instead of a notebook?

The lectures use Jupyter notebooks because notebooks are excellent for demonstrating one concept at a time. This project is different: it contains reusable algorithms, feature extraction, data parsing, and tests.

Therefore the primary artifacts are Python files:

```text
project_1/
├── README.md
├── review_analyzer.py
├── demo.py
└── test_review_analyzer.py
```

This structure keeps the mathematical implementation inspectable while allowing the code to be executed from the command line and tested automatically.

If a visual walkthrough is useful later, a notebook can import these functions without duplicating the implementation.

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
\mathrm{error}(\theta;S)
=
\frac{1}{|S|}
\sum_{(x,y)\in S}
\mathbf{1}\left[\mathrm{sign}\left(\theta^{T}x\right)\neq y\right]
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
\bar{\theta}
=\frac{1}{T}\sum_{t=1}^{T}\theta^{(t)}
$$

Averaging can make the classifier less sensitive to the exact parameter vector obtained at the end of the training trajectory.

## 7. Pegasos

The project also implements the Pegasos algorithm for a regularized linear SVM objective.

For a training example $(x,y)$, the hinge loss is

$$
\ell(\theta;(x,y))
=\max\{0,1-y\theta^{T}x\}
$$

The regularized objective is

$$
J(\theta)
=\frac{\lambda}{2}\|\theta\|^2
+\frac{1}{m}\sum_{i=1}^{m}
\max\{0,1-y_i\theta^{T}x_i\}
$$

Pegasos performs stochastic sub-gradient updates and a projection step. With a single example and learning rate

$$
\eta_t=\frac{1}{\lambda t}
$$

the update has the form

$$
\theta_{t+1/2}
=(1-\eta_t\lambda)\theta_t
+\eta_t y_t x_t
$$

when the example violates the margin condition

$$
y_t\theta_t^{T}x_t<1
$$

If the example satisfies the margin, the loss contribution has zero sub-gradient and only the regularization shrinkage remains.

The implementation then projects the parameter vector onto the ball required by the Pegasos algorithm.

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

## 9. Data format

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

## 10. Running the demo

From the project directory:

```bash
python3 demo.py
```

The demo creates a small synthetic review dataset, builds a vocabulary, trains the three classifiers, and prints their training accuracy. This makes the project executable without downloading external data.

For a real review dataset, use the functions in `review_analyzer.py` to load the data, build the vocabulary from the training split, transform both training and validation reviews, train a classifier, and evaluate it.

## 11. Running the tests

```bash
python3 -m unittest -v test_review_analyzer.py
```

The tests cover:

- review parsing;
- vocabulary construction;
- sparse feature extraction;
- prediction and accuracy;
- perceptron learning;
- average perceptron learning;
- Pegasos learning;
- a small end-to-end sentiment-classification example.

## 12. Correct experimental workflow

For a real experiment, keep the test set untouched while making model choices:

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
train classifier
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

In particular, the vocabulary should be learned from the training data rather than from the complete dataset. Otherwise information from validation or test examples can leak into the representation.

## 13. Important Pegasos hyperparameters

The implementation exposes:

- `lambda_` — regularization strength;
- `epochs` — number of passes through the training examples;
- `batch_size` — number of examples in each stochastic update;
- `seed` — random seed used for shuffling.

These are **hyperparameters**, not learned model parameters.

For Pegasos, the regularization parameter is especially important because the learning rate is tied to it:

$$
\eta_t=\frac{1}{\lambda t}
$$

Changing $\lambda$ therefore changes both the regularization strength and the optimization schedule.

## 14. Study questions

1. Why is $y\theta^{T}x$ more useful for the perceptron update than looking at $\theta^{T}x$ alone?
2. What does the hinge-loss margin condition $y\theta^{T}x<1$ mean?
3. Why does Pegasos include a projection step?
4. How does L2 regularization affect the parameter vector?
5. Why should the vocabulary be constructed from training data only?
6. Why can averaging perceptron parameters improve stability?
7. How does the Pegasos update connect Lecture 3's hinge loss to stochastic optimization in Lecture 4?
8. Why is Pegasos particularly appropriate for sparse text features?

## 15. Relation to Unit 1

This project is the practical synthesis of the unit:

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
       +--> bag-of-words text representation
       +--> perceptron
       +--> average perceptron
       +--> Pegasos / regularized SVM
       +--> validation and final evaluation
```

The project should be studied as an application of the mathematical ideas in the lectures, not as an unrelated NLP exercise.

## 16. Attribution

The project is an original study-oriented implementation inspired by the MIT 6.86x course material and the Pegasos research paper. It is not a copy of the course's proprietary starter code or solution code.

## Equation-rendering safeguard

The equations in this README intentionally follow the same conservative GitHub MathJax style used in Lecture 1 and the later lecture READMEs:

- display equations use `$$` on separate lines;
- inline mathematics uses `$...$`;
- transposes use the explicit form `^{T}`;
- the sign function uses `\mathrm{sign}(...)`;
- multiline matrices use explicit `\\` row separators;
- `cases` and `aligned` environments remain entirely inside a display block;
- equations are never placed inside Markdown code blocks;
- no single-backslash matrix row separators are used.

When editing this file, preserve these conventions and verify the **rendered GitHub page**, not only the raw Markdown source.
