# Neural Network Approach for Predicting Solar-Induced Fluorescence

> **Course:** Machine Learning for Modelling

## Overview

This project explores the use of neural networks to solve a critical challenge in remote sensing and climate science: the retrieval of Solar-Induced Fluorescence (SIF) from hyperspectral satellite data. SIF is a crucial indicator of plant photosynthetic activity and ecosystem health, but accurately disentangling its faint signal from atmospheric effects and surface reflectance is a complex, ill-posed inverse problem.

Our approach is grounded in a **self-supervised learning** framework, where a neural network is trained to decompose at-sensor radiance without requiring extensive ground-truth SIF measurements. The project's key novelty lies in using a **dual Radiative Transfer Model (RTM)** setup to generate a robust synthetic dataset for training and validation.

## Methodology

### 1. Synthetic Dataset Generation

Since direct, large-scale SIF measurements are scarce, we generated a comprehensive synthetic dataset to train our model in a controlled environment.

*   **Dual-RTM Approach:**
    1.  **SCOPE Model:** Used to simulate detailed plant-canopy processes, generating surface **Reflectance ($`R`$)** and top-of-canopy **Fluorescence ($`F`$)**.
    2.  **MODTRAN Model:** Used to simulate comprehensive atmospheric radiative transfer, generating the various **atmospheric transfer functions ($`t_x`$)**.
*   **Physical Model (LTOA):** The outputs from SCOPE and MODTRAN were combined using a physical model to generate the final **Top-of-Atmosphere ($`LTOA`$)** radiance—the signal that a satellite would "see".

### 2. Neural Network Architecture

The core of our solution is a multi-head encoder-decoder neural network designed for self-supervised learning.

*   **Goal:** The network takes the simulated $`LTOA`$ spectrum as input and learns to decompose it back into its constituent physical components ($`R`$, $`F`$, and atmospheric terms).
*   **Encoder Design:**
    *   A shared Multi-Layer Perceptron (MLP) backbone processes the high-dimensional hyperspectral $`LTOA`$ input and extracts a common set of meaningful features.
    *   **Multiple Output Heads:** A key architectural choice was to use a separate output head for each of the 11 physical variables ($`R`$, $`F`$, and 9 atmospheric terms, $`t_1...t_{11}`$). This allows each head to specialize in predicting its specific component.
*   **Self-Supervised Loop:**
    1.  The network predicts the physical components ($`\hat{R}`$, $`\hat{F}`$, $`\hat{t}_x`$).
    2.  These predicted components are fed back into the forward physical LTOA model to reconstruct the radiance ($`\widehat{LTOA}`$).
    3.  The **core self-supervised loss** is the Mean Squared Error (MSE) between the original input $`LTOA`$ and the reconstructed $`\widehat{LTOA}`$. This allows the network to learn the underlying physics without direct supervision on the components themselves.

### 3. Enhanced Loss Functions

Given the weak contribution of the SIF signal ($`F`$) to the total $`LTOA`$, the self-supervised loss alone provides a very weak gradient for $`F`$. To address this, we leveraged our synthetic ground truth to formulate enhanced, physics-regularized loss functions that added direct MSE supervision on the individual components ($`F`$, $`R`$, etc.), which proved crucial for diagnosing learning and improving performance.

## Key Findings

*   The network successfully learns to reconstruct the $`LTOA`$ spectrum from the predicted physical components.
*   However, the accurate retrieval of individual components, especially the faint Fluorescence ($`F`$) signal, remains highly challenging. The network exhibits a tendency to **overshoot** $`F`$ predictions, particularly when the true $`F`$ value is low.
*   This difficulty stems from the **ill-posed nature of the inverse problem**—multiple combinations of $`R`$, $`F`$, and $`t_x`$ can produce very similar $`LTOA`$ signals.

## Technologies Used

*   **Deep Learning:** Python, PyTorch
*   **Data Simulation:** Custom scripts interfacing with SCOPE and MODTRAN RTMs.
*   **Data Handling:** NumPy, Pandas, Matplotlib
