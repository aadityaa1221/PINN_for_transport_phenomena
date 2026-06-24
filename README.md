# Physics-Informed Neural Networks (PINNs) for Fluid Dynamics

In this repository, we implement Physics-Informed Neural Networks (PINNs) that allow us to solve both forward and inverse problems in the field of transport phenomena. 

Contrary to the traditional approach to Machine Learning, PINNs impose physics in the optimization process of the neural network. This imposes a rigid constraint to the loss function.

## 🧠 Project Overview

This project uses the automatic differentiation capabilities of PyTorch's Autograd engine to calculate the exact second order spatial derivative terms, thus imposing the laws of fluid dynamics and thermal diffusion on the network.

### Part 1: Forward Problem (1D Heat Equation)
Model thermal diffusion by minimizing Initial Condition (IC) error, Dirichlet boundary constraints, and PDE residuals.
* **Equation:** $u_t - \alpha u_{xx} = 0$

### Part 2: Inverse Problem (Parameter Discovery in Burgers' Equation)
The main advantage of PINNs is the ability to reverse engineer the unknown physical parameters from chaotic data. PINN automatically learns the value of the hidden kinematic viscosity ($\nu$) of the fluid from sensor data.
* **Equation:** $u_t + uu_x - \nu u_{xx} = 0$

## ✨ Key Architectural Features

* **Fourier Feature Embeddings:** Overcomes the spectral bias problem of standard# Physics-Informed Neural Networks (PINNs) for Fluid Dynamics

This repository contains a research-grade implementation of Physics-Informed Neural Networks (PINNs) designed to solve both forward and inverse problems in transport phenomena. 

Unlike purely data-driven machine learning models, PINNs integrate mathematical physics directly into the neural network's optimization loop, acting as a rigorous constraint on the loss landscape.

## 🧠 Project Overview

This project leverages PyTorch's Autograd engine to compute exact second-order spatial derivatives, forcing the network to obey fluid dynamics and thermal diffusion laws. 

### Part 1: Forward Problem (1D Heat Equation)
Models thermal diffusion by simultaneously minimizing Initial Condition (IC) error, Dirichlet boundary constraints, and Partial Differential Equation (PDE) residuals. 
* **Equation:** $u_t - \alpha u_{xx} = 0$

### Part 2: Inverse Problem (Parameter Discovery in Burgers' Equation)
The core strength of PINNs is reverse-engineering hidden physical parameters from chaotic data. This model autonomously discovers the hidden kinematic viscosity ($\nu$) of a fluid from noisy sensor data.
* **Equation:** $u_t + u u_x - \nu u_{xx} = 0$

## ✨ Key Architectural Features

* **Fourier Feature Embeddings:** Mitigates the "spectral bias" of standard MLPs, allowing the network to accurately approximate high-frequency wave structures and sharp gradients (shockwaves).
* **Adaptive Optimization Pipeline:** Utilizes a two-stage training methodology:
  1. **Adam Optimizer:** For robust initial exploration of the loss landscape.
  2. **L-BFGS Optimizer:** A second-order quasi-Newton method for high-precision mathematical convergence.
* **Latin Hypercube Sampling (LHS):** Implements dynamic, continuous domain resampling for collocation points, preventing the network from overfitting to a static grid.
* **Anchored Physics Constraints:** Mathematically anchors Initial and Boundary conditions to prevent the network from hallucinating non-physical wave histories.

## 📊 Quantitative Validation & Ablation Study

To prove the robustness of the architecture, an ablation study was conducted by injecting varying levels of Gaussian noise ($0\%$, $5\%$, $10\%$) into synthetic sensor data at $t=0.5$. 

* The true analytical fluid state was generated using the exact **Cole-Hopf transformation**.
* Despite heavily corrupted sensor data, the PINN successfully filtered the noise through its physics-loss penalty, achieving a **<5% relative error** in predicting the true hidden viscosity ($\nu \approx 0.00318$).


## 💻 Tech Stack
* **Deep Learning:** PyTorch (`torch.autograd`, `torch.nn`, `LBFGS`)
* **Scientific Computing:** SciPy (Quasi-Monte Carlo LHS), NumPy
* **Visualization:** Matplotlib

