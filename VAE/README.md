DSBA intern - CV study

This project is a PyTorch-based implementation of a VAE for image data such as MNIST.  
You can train the VAE and evaluate its reconstruction/generation performance using the provided scripts.  

**New!**  
Vector Quantized VAE (VQ-VAE) is also implemented in this project.  
You can experiment with both continuous (VAE) and discrete (VQ-VAE) latent variable models.

---

## Project Structure

```
VAE/
├── app/                # Main code modules (data, model, trainer, train.py, eval.py, etc.)
│   ├── vq_model.py     # VQ-VAE model definition
│   ├── vq_trainer.py   # VQ-VAE training loop
│   ├── vq_loss.py      # VQ-VAE loss functions
│   ├── vq_data.py      # VQ-VAE data handling (if present)
│   └── ...             # Other VQ-VAE related modules
├── datasets/           # (Auto-downloaded) dataset storage
├── saved_model/        # Directory for saving trained model parameters
│   ├── vae_model.pth
│   └── vq_vae_model.pth   # (NEW) VQ-VAE model parameters (if saved separately)
├── pyproject.toml      # Project and dependency management file
├── README.md           # Project documentation (this file)
└── ...
```

---

## Package Management: uv & pyproject.toml

- This project uses **uv** for package management.
    - [uv documentation](https://github.com/astral-sh/uv)
- All dependencies are listed in **pyproject.toml**.
    - Main packages: `torch`, `torchvision`, `matplotlib`, `tqdm`, etc.

### Installation

```bash
# Install uv (if you haven't already)
pip install uv

# Install dependencies
uv sync
```

---

## Main Scripts and Modules

- **app/train.py**  
  Main script for training the VAE model.
- **app/eval.py**  
  Loads a saved model and evaluates it on the test dataset, including visualization of reconstructions.
- **app/data.py, model.py, loss.py, trainer.py**  
  Separated modules for data loading, model definition, loss functions, and training loop.

### VQ-VAE Implementation

- **app/vq_model.py**  
  Vector Quantized VAE (VQ-VAE) model definition.
- **app/vq_trainer.py**  
  Training loop and utilities for VQ-VAE.
- **app/vq_loss.py**  
  Loss functions specific to VQ-VAE.
- **app/vq_data.py** (if present)  
  Data loading and preprocessing for VQ-VAE experiments.
- You can extend or modify these modules for experiments with discrete latent representations.

---

## Model Saving

- After training, the model parameters are saved in **`saved_model/vae_model.pth`** at the project root.
- For VQ-VAE, model parameters may be saved as **`saved_model/vq_vae_model.pth`** (if implemented).
- You can load these files in the corresponding evaluation scripts for testing and image reconstruction.

---

## How to Run

### 1. Training

```bash
uv run python -m app.train
# For VQ-VAE (if you have a separate script, e.g., vq_train.py)
uv run python -m app.vq_train
```

### 2. Evaluation & Visualization

```bash
uv run python -m app.eval
# For VQ-VAE (if you have a separate script, e.g., vq_eval.py)
uv run python -m app.vq_eval
```

---

## Notes

- Python 3.12 or higher is recommended.
- The MNIST dataset will be automatically downloaded on first run.
- For further experiments or extensions (e.g., latent space visualization, using other datasets, or experimenting with VQ-VAE), refer to the modules in `app/`.
- The VQ-VAE implementation enables experiments with discrete latent spaces, which can be useful for generative modeling and compression tasks.

---

Let me know if you want to add more details or have it tailored for a specific audience!
