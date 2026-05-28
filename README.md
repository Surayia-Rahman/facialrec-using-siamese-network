
# Facial Verification Sandbox using Siamese Neural Networks

This repository implements a lightweight, from-scratch **Facial Verification Engine** in Python using **TensorFlow** and **Keras**. The core network relies on a **Siamese Neural Network architecture** that processes pairs of images (twin pipelines) through shared weights to calculate a localized L1 similarity distance metric. It includes an end-to-end data pipeline from custom face collection to custom model-layer compilation and live webcam verification loops.

---

## 🏗️ Project Framework & Pipeline

The system is executed within a continuous machine learning cycle containing six main stages:

```text
 [1. Video Stream Ingestion] ➔ [2. Image Pipeline Preprocessing] ➔ [3. Shared Embedding Net]
                                                                            │
 [6. Live OpenCV GUI Feed]  🦿   [5. Logistic Sigmoid Class] 🦿  [4. L1 Distance Computation]

```

1. **System Dependency Provisioning:** Sets up the execution environment using precise module hooks to stabilize the PyTorch/Keras network graph.
2. **Dynamic Workspace Instantiation:** Automates file directory structures to organize input data fragments into *Positive*, *Negative*, and *Anchor* streams.
3. **Multi-Class Image Ingestion:** Generates customized local data distributions using unique hardware identifiers (`uuid`), downloading public adversarial benchmarks to enforce negative contrast classes.
4. **Tensor Normalization & Preprocessing:** Streamlines data through an optimization pipeline that standardizes variable dimensions and rescales float profiles.
5. **Siamese Network Model Engineering:** Architectures a twin convolutional feature extractor coupled with a custom absolute difference layer to classify binary face inputs.
6. **Live Execution & Metric Validation:** Deploys a live camera interface leveraging a double-threshold scoring loop (`Detection` vs `Verification`) to verify user identities in real-time.

---

## 🛠️ Deep Architectural Details

### 1. Data Pipe Setup & Structural Classes

* **Dataset Management:** Partitions data pairs into logical tensor slices wrapped inside memory-optimized `tf.data.Dataset` pipelines.
* **Anchor Stream:** Reference target photos captured of the system user.
* **Positive Stream:** Dynamic alternative validation captures of the user under different angles or lighting conditions.
* **Negative Stream:** Background baseline benchmark imagery extracted from the open-source *Labelled Faces in the Wild (LFW)* dataset.


* **Pre-processing Transforms:** Input paths flow through an isolated decoding routine (`tf.io.decode_jpeg`), rescaling resolution grids to a standardized **100×100×3** shape, and normalizing intensity arrays to floating bounds $[0, 1]$.
* **Dataset Splitting:** Combines vectorized records into structural test sets utilizing a **70% / 30% train-to-test partition split ratio**, managing mini-batches at a uniform batch density scale of 16.

---

### 2. Neural Graph Architecture (Siamese Neural Network)

The framework structures a shared **Embedding Neural Network** that reduces image features into high-dimensional geometric vectors. Two image streams flow simultaneously through these shared weights before calculating spatial divergence metrics:

```text
Image Input A ➔ [ Shared Embedding Convolutional Neural Network ] ➔ Vector A ┐
                                                                             ├──➔ [ L1 Distance Layer ] ➔ [ Dense Sigmoid Output ]
Image Input B ➔ [ Shared Embedding Convolutional Neural Network ] ➔ Vector B ┘

```

#### 🔹 The Shared Embedding Core

* **Layer 1:** 2D Convolutional (`Conv2D`: 64 filters, $10\times10$ kernel, ReLU) $\rightarrow$ MaxPooling (`MaxPooling2D`: $2\times2$, padding=`same`).
* **Layer 2:** 2D Convolutional (`Conv2D`: 128 filters, $7\times7$ kernel, ReLU) $\rightarrow$ MaxPooling (`MaxPooling2D`: $2\times2$, padding=`same`).
* **Layer 3:** 2D Convolutional (`Conv2D`: 128 filters, $4\times4$ kernel, ReLU) $\rightarrow$ MaxPooling (`MaxPooling2D`: $2\times2$, padding=`same`).
* **Layer 4:** 2D Convolutional (`Conv2D`: 256 filters, $4\times4$ kernel, ReLU) $\rightarrow$ `Flatten` Layer.
* **Output Embedding Projection:** Dense Layer mapping **4,096 scalar properties** running on a Sigmoid activation profile.

#### 🔹 The Similarity Distance Matching Layer (`L1Dist`)

A custom-built Keras layer written from scratch subclassing `tf.keras.layers.Layer`. It evaluates the absolute differences between the anchor feature embedding matrix and the validation feature embedding matrix element-wise:

```python
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
       
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

```

* **Classifier Head:** Feeds the resultant L1 vector into a final singular output node (`Dense(1)`) running a standard `sigmoid` transfer function to predict a similarity confidence interval between $0$ (Complete Anomaly) and $1$ (Identical Match).

---

### 3. Optimizer Configuration & Training Loop

* **Objective Loss Function:** Binary Cross-Entropy (`tf.losses.BinaryCrossentropy`), evaluating verification error outputs.
* **Optimizer Algorithm:** `Adam` optimizer configured with a fine learning rate index parameter of `1e-4` ($0.0001$).
* **Custom Backpropagation Step (`train_step`):** Uses an explicit gradient-tracking scope (`tf.GradientTape`) to calculate operational weight differentials across trainable variables manually and apply parameter changes dynamically.
* **Metrics Performance Validation:** After a 50-epoch training sequence, the system converges to high performance metrics across the test matrix:
* **Precision Score:** **1.0** ($100\%$)
* **Recall Score:** **1.0** ($100\%$)



---

## 🖥️ Live OpenCV Verification Pipeline

The real-time authentication interface relies on a localized dual-threshold testing architecture:

1. **Detection Threshold (Set to 0.9):** The target image input captured from the webcam is paired against 50 sequential verification images stored locally inside `application_data/verification_images`. Individual pairs must exceed a prediction threshold of `0.9` to count as a positive candidate detection match.
2. **Verification Threshold (Set to 0.7):** For the application to grant verified system access, the percentage of total matches divided by the total sample size must surpass the authorization cutoff barrier of `70%`.

---

## 🚀 Installation & System Workspace Setup

### 1. Requirements & Core Packages

Install the mandatory machine learning, imaging math, and graph processing backend ecosystem dependencies:

```bash
pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python matplotlib numpy

```

### 2. Data Structure Organization

Execute the main file path initialization routines to create folders in your current workspace directory before loading files:

```python
import os
os.makedirs(os.path.join('data', 'positive'))
os.makedirs(os.path.join('data', 'negative'))
os.makedirs(os.path.join('data', 'anchor'))

```

### 3. Training & Live Ingestion Execution

* Press the **`a` key** on your keyboard inside the active GUI webcam window to record target face states into the `anchor` collection path.
* Press the **`p` key** to capture alternative valid verification files into the `positive` collection path.
* Press the **`v` key** to save your active camera frame and run the double-threshold verification calculation loop.
* Press the **`q` key** to gracefully close all background threads and release video device handles.

