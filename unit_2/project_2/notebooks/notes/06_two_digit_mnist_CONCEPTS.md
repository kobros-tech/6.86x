# Notebook 06 — Concepts Not Covered in Units 1–2

**Neural networks are formally covered in Unit 3.** This notebook extends Notebooks 04–05 to an image containing two digits, which needs a new structural idea: **multiple output heads sharing one internal representation**. This note is a vocabulary preview, not the full treatment.

---

## 1. Why one ten-class output is not enough

Every classifier so far — including the fully connected and convolutional networks in Notebooks 04–05 — produces a single prediction from a fixed set of classes (one digit, 0–9). The two-digit MNIST task needs **two** independent digit labels from a single image (one 42×28 image containing two digits side by side). A single ten-way output cannot represent two labels at once, so the network needs two separate predictions.

## 2. Shared representation, separate heads

Rather than training two completely independent networks, `two_digit/mlp.py` and `two_digit/conv.py` use one **shared trunk** — the same flatten/linear/convolution layers used in Notebooks 04–05 — to compute one internal representation of the image, and then attach **two separate output layers** ("heads") on top of it:

$$
\text{image} \;\to\; \text{shared layers} \;\to\;
\begin{cases}
\text{head 1} \to \text{10 scores for the first digit} \\
\text{head 2} \to \text{10 scores for the second digit.}
\end{cases}
$$

In code, `self.head_digit1` and `self.head_digit2` are two separate `nn.Linear` layers, both applied to the same `shared_repr` output. The intuition for sharing the trunk is that useful low-level features for recognizing *a* digit (edges, strokes, loops) are useful for recognizing *either* digit in the image, so learning them once and reusing them for both predictions is more efficient than learning everything twice.

## 3. The combined loss

Each head produces its own cross-entropy loss (the same loss from Notebook 04, `F.cross_entropy`), one per digit. The two per-digit losses are combined by **averaging**, not summing:

$$
L = \frac{L_1 + L_2}{2}.
$$

Because both losses are computed from the same forward pass through the shared trunk, minimizing $L$ updates the shared layers using information from *both* prediction tasks at once, in addition to updating each head separately using only its own loss.

### 3.1 Why averaging, not summing

An earlier version of this notebook combined the two losses by summing them, $L = L_1 + L_2$, instead of averaging. With the CNN model, that produced a real, observable training failure: the training loss climbed to roughly $\ln(10) \approx 2.30$ — the loss value produced by guessing uniformly among the 10 possible digit classes — and test accuracy collapsed to about 10%, i.e. the model stopped learning partway through training instead of improving.

The reason is that summing two independent cross-entropy losses roughly doubles the size of the gradient flowing back into the shared trunk compared to using either loss alone, since gradients add the same way the losses do. That larger effective step size was enough to destabilize training for this particular optimizer and learning rate. Dividing by 2 rescales the combined gradient back down to roughly the same magnitude a single-task loss would produce, and training became stable again — the CNN in this notebook reaches roughly 90% accuracy on each digit and roughly 80% exact-match accuracy once both digits must be correct simultaneously.

This is a useful general lesson for any multi-task loss, not just this notebook: **how multiple losses are combined (sum vs. average vs. a weighted combination) directly affects the effective gradient magnitude, and can be the difference between stable training and divergence** — independent of whether the model architecture itself is correct. When a multi-output model trains unexpectedly poorly, the way its losses are combined is worth checking before concluding the architecture is at fault.

## 4. Evaluation metrics for two labels

With two predictions per example, there is more than one way to report accuracy:

- **first-digit accuracy** — how often head 1's prediction matches the true first digit;
- **second-digit accuracy** — how often head 2's prediction matches the true second digit;
- **exact-match accuracy** — how often *both* predictions are simultaneously correct on the same example.

Exact-match accuracy is always less than or equal to either individual digit accuracy, since it requires both heads to be right at once.

---

## 5. What to remember (operational, for this notebook only)

1. Multiple prediction tasks can share one internal representation ("trunk") while having separate output layers ("heads") for each task.
2. The combined loss $L=(L_1+L_2)/2$ lets the shared layers learn from both tasks simultaneously.
3. How multiple losses are combined (sum vs. average) changes the effective gradient magnitude and can affect training stability — summing caused this notebook's CNN to diverge; averaging fixed it.
4. With multiple outputs, report per-output accuracy separately as well as exact-match accuracy across all outputs together.

Full coverage of multi-task and multi-output network design is in Unit 3.
