# DCGAN vs. DDPM: A Comparative Study for Face Image Generation

A comparative study of **Deep Convolutional Generative Adversarial Networks (DCGAN)** and **Denoising Diffusion Probabilistic Models (DDPM)** for realistic face image generation using the **CelebA dataset**.

This academic project explores two generative modeling approaches, comparing their architectures, training processes, and image generation quality.

## Academic Information

- **Authors:** Khawla Chrifi Alaoui and Noura Ogbi
- **Supervisor:** Prof. Asmae Mourhir
- **Institution:** Al Akhawayn University, Morocco
- **Program:** MSc in Big Data Analytics
- **Status:** Academic research project (unpublished)

## Project Objectives

- Implement and train DCGAN and DDPM models using PyTorch.
- Generate synthetic facial images from the CelebA dataset.
- Evaluate image quality using **Fréchet Inception Distance (FID)**, **Inception Score (IS)**, and visual inspection.
- Analyze the strengths, limitations, and trade-offs of adversarial and diffusion-based generation.

## Dataset

**CelebA (CelebFaces Attributes Dataset)** contains 202,599 celebrity face images.

Images were preprocessed through center cropping, resizing to **64 × 64 pixels**, and normalization to the range **[-1, 1]**.

Dataset source: [CelebA on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

## Models and Implementation

### 1. DCGAN

- Convolutional generator and discriminator networks.
- A 100-dimensional latent noise vector used to generate RGB face images.
- Adversarial training through competition between the generator and discriminator.
- Training checkpoints and generated sample visualization.

### 2. DDPM

- U-Net-based denoising architecture with sinusoidal time embeddings and residual blocks.
- Forward diffusion process that gradually adds Gaussian noise.
- Reverse diffusion process that learns to recover images from noise.
- A diffusion schedule of **1,000 timesteps**, trained using the Adam optimizer.

## Evaluation and Findings

Both models were examined using **FID**, **Inception Score**, and qualitative image analysis.

The study observed that DCGAN generated recognizable facial structures but exhibited some blurriness and training instability. DDPM generally produced more visually realistic and coherent faces, although its iterative denoising process introduced greater computational cost.

The accompanying paper discusses the experimental comparison, observations, and limitations in more detail.

## Technologies

**Python · PyTorch · Torchvision · NumPy · Matplotlib · CelebA · Jupyter Notebook**

## Repository Contents

- `notebooks/DCGAN.ipynb` — DCGAN architecture, training, image generation, and evaluation.
- `notebooks/Diffusion_Model.ipynb` — DDPM implementation, diffusion processes, and training.
- `paper/` — Academic comparative study paper.
- `presentation/` — Final project presentation.
- `requirements.txt` — Project dependencies.

## Getting Started

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open the notebooks in Jupyter Notebook or a compatible GPU-enabled environment. The dataset is accessed using KaggleHub.

**Notes:** 
- Training both models is computationally intensive. Local dataset, output, and checkpoint paths may need to be configured for your environment.
- The notebooks were originally developed in a university GPU computing environment and contain hardcoded paths specific to that setup. These paths must be adjusted to match your local environment before execution.

