# Notebook 04 — Concepts Not Covered in Units 1–2

**Neural networks are formally covered in Unit 3.** This notebook hands the mechanics that were done by hand in Notebook 03 over to PyTorch. This note only bridges the vocabulary — it does not re-derive anything Unit 3 will cover properly.

---

## 1. From by-hand backprop to autograd

Notebook 03 computed forward propagation and backpropagation manually, one line of code per derivative. PyTorch instead tracks every operation performed on its tensors and computes all gradients automatically when `loss.backward()` is called — this is called **automatic differentiation (autograd)**. The `NeuralNetwork.backward` method from Notebook 03 and PyTorch's `loss.backward()` compute the *same kind* of thing (gradients via the chain rule); PyTorch just does it generically for any network you build, instead of by hand for one specific architecture.

## 2. PyTorch building blocks used here

- **`nn.Sequential`** — chains layers together in order, exactly like the layer-by-layer structure in Notebook 03, but declared rather than hand-coded.
- **`nn.Linear(in, out)`** — a linear layer: computes $Wx + b$ for a learnable weight matrix $W$ and bias $b$, the same computation as one layer of the from-scratch network.
- **`nn.ReLU()`** — the same ReLU nonlinearity from Notebook 03, as a reusable layer.
- **`Flatten`** — reshapes a 28×28 image into a single 784-length vector, since `nn.Linear` expects a flat vector, not a 2-D grid. Flattening this way discards the image's spatial layout, which motivates the convolutional network in Notebook 05.

## 3. `F.cross_entropy`

Notebook 01 computed the Softmax negative-log-likelihood loss by hand (softmax the scores, then take $-\log$ of the true class's probability). `F.cross_entropy(out, y)` performs exactly that computation in one call: it applies softmax to the raw output scores (**logits**) internally and returns the negative log-likelihood of the true label. The model in `fully_connected.py` therefore outputs raw scores, not probabilities — the probability conversion happens inside the loss function, not the model.

## 4. Optimizer, momentum, epochs, and batches

- **Optimizer** — the object that performs the parameter update step ($\theta \leftarrow \theta - \eta\nabla J(\theta)$, from Lecture 5) automatically, given the gradients autograd computed.
- **Momentum** — an optional modification to plain gradient descent that accumulates a running average of past gradients before updating, which can smooth out the optimization path. The mechanics of momentum are covered properly in Unit 3; for this notebook, it is enough to know it's a variant of the gradient-descent update rule already introduced in Lecture 5.
- **Epoch** — one full pass through the entire training set.
- **Mini-batch** — instead of computing the gradient over the full training set at once (as Lecture 5's demo does), the gradient is computed and applied on small subsets ("batches") of the data at a time. `batchify_data` and `run_epoch` implement this; it does not change the underlying gradient-descent idea, only how much data is used per update step.

---

## 5. What to remember (operational, for this notebook only)

1. PyTorch's autograd computes the same gradients Notebook 03 computed by hand, automatically, for any network built from its layers.
2. `nn.Linear` and `nn.ReLU` are reusable versions of the linear step and ReLU nonlinearity from Notebook 03.
3. `F.cross_entropy` combines softmax and negative-log-likelihood into one call, matching the by-hand Softmax loss from Notebook 01.
4. Mini-batch gradient descent applies the same update rule from Lecture 5, but computed on small batches of data rather than the whole training set at once.

Full coverage of network architectures, optimizers, and training dynamics is in Unit 3.
