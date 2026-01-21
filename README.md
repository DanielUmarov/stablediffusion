# Stable Diffusion (From Scratch)

This project is a **from-scratch implementation of a Stable Diffusion–style latent diffusion model**, built to understand, re-implement, and experiment with every major component of modern text-to-image diffusion systems.

Rather than treating Stable Diffusion as a black box, this repository reconstructs the full pipeline end-to-end and serves as a **learning, research, and experimentation framework** for diffusion models.

---

## Project Goals

* Build a complete **latent diffusion pipeline** from first principles
* Understand how architectural and algorithmic choices affect image generation
* Create a flexible codebase for experimenting with **samplers, UNet variants, and efficiency techniques**
* Bridge theory (papers) with practice (working code)

This project prioritizes **conceptual clarity, modularity, and research extensibility** over production-scale training.

---

## Implemented / In Progress Components

* **Variational Autoencoder (VAE)**
  Compresses images into a lower-dimensional latent space where diffusion is performed.

* **Diffusion UNet**
  Core denoising network with timestep conditioning and residual structure.

* **Synthetic Captioning / Conditioning**
  Text or synthetic conditioning signals to enable conditional generation.

* **Classifier-Free Guidance (CFG)**
  Implements conditional/unconditional training and guidance scaling at inference.

* **DDIM Sampling**
  Deterministic and accelerated sampling for faster inference and experimentation.

---

## What Is Diffusion? (A Physics-Inspired View)

At its core, a diffusion model is inspired by **physical diffusion processes** studied in statistical mechanics and stochastic dynamics.

### Forward Diffusion (Noising Process)

In physics, diffusion describes how structured information (e.g. particle concentration, temperature gradients) gradually disperses over time due to random fluctuations. In diffusion models, we **intentionally destroy structure** by adding Gaussian noise in small increments:

* Each timestep applies a small random perturbation
* Over many steps, data converges to an isotropic Gaussian distribution
* This process is *fixed* and does not require learning

Mathematically, this resembles a **discrete-time stochastic process** analogous to Brownian motion.

### Reverse Diffusion (Denoising Process)

The generative task is to learn the *reverse* of diffusion:

> Given a noisy state, predict how to remove just enough noise to recover structure.

This is analogous to learning a **time-reversed stochastic differential equation**, where the model estimates the score (gradient of the log probability density) of the data distribution.

The UNet is trained to predict the noise added at each timestep, enabling the system to iteratively refine pure noise into coherent structure.

---

### Visual Intuition

This repository includes a custom animation illustrating the **reverse diffusion process**, inspired by interactive diffusion visualizers used in research and education.

The animation shows:

* The transformation of noise into structured data
* The role of timestep-dependent denoising
* How different noise schedules influence sampling trajectories

This visualization is intended to build **physical and geometric intuition**, not just empirical understanding.

<p align="center">
  <img src="assets/swissroll_sde.gif" width="600"/>
</p>

<p align="center">
  <em><strong>Figure.</strong> Reverse diffusion on a low-dimensional toy data distribution.<br>
  Starting from isotropic Gaussian noise, the model iteratively removes noise to recover structure.</em>
</p>

---
While this example operates on a low-dimensional toy distribution
,the same reverse diffusion dynamics apply when denoising high-dimensional image or latent representations.

## Design Philosophy

* **End-to-end ownership**: Every major component is implemented explicitly rather than imported as a monolithic dependency.
* **Research-first structure**: Code is written to be readable, modifiable, and extensible for experimentation.
* **Sampler-aware thinking**: Sampling is treated as a first-class research object, not just an inference detail.
* **Compute-conscious**: The project is designed to explore efficiency-aware training and inference strategies.

---

## Planned Extensions / Research Directions

* Custom and hybrid **diffusion samplers**
* Sampler-aware or timestep-adaptive UNet training
* Conditional sparsity or selective weight activation
* Parameter-efficient fine-tuning (e.g. low-rank or localized updates)
* Improved latent representations and VAE ablations
* Training on small, curated datasets to study data efficiency

---

## Status

This repository is under active development. Components may be incomplete or intentionally simplified as part of the learning and research process.

The goal is not to replicate production Stable Diffusion performance, but to develop **deep mechanistic understanding and novel experimentation pathways**.

---

## Disclaimer

This project is intended for **educational and research purposes only** and is not affiliated with or endorsed by Stability AI or related organizations.

## References
Ho, J., Jain, A., & Abbeel, P. (2020).
Denoising Diffusion Probabilistic Models.
Advances in Neural Information Processing Systems (NeurIPS).

Song, Y., Meng, C., & Ermon, S. (2021).
Denoising Diffusion Implicit Models.
International Conference on Learning Representations (ICLR).

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
Score-Based Generative Modeling through Stochastic Differential Equations.
International Conference on Learning Representations (ICLR).

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).
High-Resolution Image Synthesis with Latent Diffusion Models.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Karras, T., Aittala, M., Aila, T., & Laine, S. (2022).
Elucidating the Design Space of Diffusion-Based Generative Models.
Advances in Neural Information Processing Systems (NeurIPS).
