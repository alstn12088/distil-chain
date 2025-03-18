# distil-chain

**distil-chain** is a project focused on **distilling MCMC** using neural network. 


## Installation

You can install the minimal required packages via **pip**:

```bash
pip install torch einops numpy ot matplotlib tqdm wandb seaborn
```

## Usage

Below is an example command to run **training**:

```bash
python train.py \
  --energy 9gmm \
  --batch_size 256 \
  --bootstrap_K 10 \
  --max_iter 2000
```

### Arguments

- **`--energy`**: Which energy function to use. Possible choices: `9gmm`, `25gmm`, `many_well`.
- **`--batch_size`**: Number of samples in each training batch.
- **`--bootstrap_K`**: Number of bootstrapped samples (for moment-based training).
- **`--max_iter`**: Theoretical maximum iteration of MCMC (for t = 1).

## Project Structure

- **`train.py`** – Main training script.  
- **`energies/`** – Defines energy functions like `9gmm`, `25gmm`, etc.  
- **`models/architectures.py`** – Contains various model architectures (distilled samplers).  
- **`utils.py`** – Helper functions (seeding, logging, etc.).  

