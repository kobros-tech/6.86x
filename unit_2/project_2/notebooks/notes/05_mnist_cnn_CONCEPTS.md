# Notebook 05 — Concepts Not Covered in Units 1–2

**Neural networks are formally covered in Unit 3.** This notebook introduces convolution as an image-specific alternative to the flattened fully connected network from Notebook 04. This note is a vocabulary preview, not the full treatment.

---

## 1. Why flattening is a limitation

Notebook 04's fully connected network flattened each 28×28 image into a 784-length vector before doing anything else. That throws away the fact that nearby pixels are spatially related — pixel $(5,5)$ and pixel $(5,6)$ are next to each other in the image, but after flattening they might end up far apart in the vector and are treated by the linear layer as unrelated inputs. A **convolutional layer** is built to use that spatial structure instead of discarding it.

## 2. A word of caution: "kernel" means something different here

Lecture 6 uses the word **kernel** for a similarity function $K(x,x')$ used to avoid constructing an explicit feature map. In convolutional networks, **kernel** (also called a **filter**) means something unrelated: a small grid of learnable numbers slid across the image. Both uses of the word are standard in the field, but they are different concepts — keep them separate.

## 3. Convolution, informally

A convolutional filter is a small weight grid (here, $3\times3$) that is applied to every local patch of the image. At each position, the filter multiplies its weights against the pixels underneath it and sums the result, producing one output value per position. Sliding the same filter across the whole image produces a **feature map** — one output value per position the filter visited.

Two properties follow directly from this construction:

- **Locality** — each output value only depends on a small local patch of the input (its **receptive field**), not the whole image.
- **Weight/filter sharing** — the *same* filter weights are reused at every position, rather than learning separate weights for every pixel location. This is why a convolutional layer has far fewer parameters than a fully connected layer covering the same input.

## 4. Pooling

**Max pooling** slides a small window (here, $2\times2$) across a feature map and keeps only the largest value in each window, discarding the rest. This shrinks the spatial size of the representation and makes the resulting features slightly less sensitive to the exact pixel position of a feature (a digit shifted by one pixel still activates roughly the same pooled output).

## 5. Tracking spatial dimensions through the network

Each convolution without padding shrinks the spatial size by (filter size − 1); each $2\times2$ max-pool halves it. `build_cnn_model`'s docstring tracks this for the $28\times28$ MNIST input:

$$
28\times28 \;\xrightarrow{\text{Conv }3\times3}\; 26\times26 \;\xrightarrow{\text{Pool }2\times2}\; 13\times13 \;\xrightarrow{\text{Conv }3\times3}\; 11\times11 \;\xrightarrow{\text{Pool }2\times2}\; 5\times5.
$$

After the second pooling stage there are 16 channels of $5\times5$ feature maps, which is why the final linear layer expects $16\times5\times5=400$ input features once flattened.

---

## 6. What to remember (operational, for this notebook only)

1. A convolutional filter is a small, learnable weight grid applied identically at every position of the image (weight sharing), so each output only depends on a local patch (locality/receptive field).
2. "Kernel" here means a convolutional filter, not the similarity function $K(x,x')$ from Lecture 6 — same word, different concept.
3. Max pooling shrinks a feature map by keeping only the largest value in each small window.
4. Spatial dimensions shrink through the network according to the filter size (no padding) and pooling window size; tracking these sizes is necessary to know the input size of the final linear layer.

Full coverage of convolutional architectures and why they generalize well for images is in Unit 3.
