DSBA intern - CV study

This project is a PyTorch-based implementation of a VAE for image data such as MNIST.  
You can train the VAE and evaluate its reconstruction/generation performance using the provided scripts.  

---

## Project Structure

```
VAE/
├── app/                # Main code modules (data, model, trainer, train.py, eval.py, etc.)
├── datasets/           # (Auto-downloaded) dataset storage
├── saved_model/        # Directory for saving trained model parameters
│   └── vae_model.pth
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

---

## Model Saving

- After training, the model parameters are saved in **`saved_model/vae_model.pth`** at the project root.
- You can load this file in `eval.py` for testing and image reconstruction.

---

## How to Run

### 1. Training

```bash
uv run python -m app.train
```

### 2. Evaluation & Visualization

```bash
uv run python -m app.eval
```

---

## Notes

- Python 3.12 or higher is recommended.
- The MNIST dataset will be automatically downloaded on first run.
- For further experiments or extensions (e.g., latent space visualization, using other datasets), refer to the modules in `app/`.

---

Let me know if you want to add more details or have it tailored for a specific audience!
