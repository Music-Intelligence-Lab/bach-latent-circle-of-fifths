# Circle of Fifths as Latent Geometry in Bach's Preludes and Fugues

## Overview

This project demonstrates that unsupervised deep learning can spontaneously recover fundamental music-theoretic structure. We trained a simple feedforward autoencoder on J.S. Bach's *Well-Tempered Clavier*, Book I, and discovered that the learned latent space organizes pieces hierarchically into keys arranged in circle-of-fifths geometry.

### Key Findings

- **Hierarchical Organization**: Sequences naturally cluster into pieces, which cluster into keys
- **Circle of Fifths Emerges**: Key centroids form a near-perfect circle (CV = 8.5%) in 2D PCA space
- **Harmonic Function Encoded**: Close-tonic pairs (e.g., C Major and D Minor) are closer than relative pairs (e.g., C Major and A Minor), despite having more accidentals
- **No Supervision Required**: All structure emerges purely from reconstruction loss

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bach-latent-space.git
cd bach-latent-space

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Pre-trained weights can be found on huggingface: https://huggingface.co/josephbakarji/bach-circle-of-fifths

### Generate Main Figure

To reproduce Figure 4 (hierarchical convergence):

```bash
python scripts/generate_figure4_convergence.py
```

This will:
1. Load pre-computed PCA coordinates from `data/2step_animation_data.json`
2. Generate 3-panel figure showing:
   - **Panel A**: Individual sequences (2,038 points) colored by key
   - **Panel B**: Piece-level centroids with relative key and close-tonic connections
   - **Panel C**: Key-level centroids with fitted circle overlay
3. Save output to `figures/fig4_combined_convergence.{pdf,png}`

Expected output statistics:
- Relative key distance: 1.35 +/- 0.62 PC units
- Close-tonic distance: 0.79 +/- 0.38 PC units
- Circle radius: 2.59 PC units (CV = 8.5%)

## Data Format

### `2step_animation_data.json`

This file contains pre-computed PCA projections and hierarchical aggregations:

```json
{
  "sequences": [
    {
      "piece": "0",
      "pc1": 1.234,
      "pc2": -0.567,
      "index": 0
    },
    ...
  ],
  "piece_centroids": [
    {
      "piece_id": 0,
      "key": "C",
      "mode": "Major",
      "avg_pc1": 1.697,
      "avg_pc2": 3.645
    },
    ...
  ],
  "key_centroids_no_mode": [
    {
      "key": "C",
      "avg_pc1": 0.741,
      "avg_pc2": 3.222
    },
    ...
  ]
}
```

### MIDI Files

- **Original Data** (`original_data/`): 24 MIDI files from Bach's Well-Tempered Clavier, Book I
  - BWV 0846-0869 (Preludes and Fugues 1-24)
  - Original tempos preserved

- **Normalized Data** (`normalized_data/`): Same pieces normalized to 60 BPM for consistent encoding

## Training Pipeline

### 1. MIDI Preprocessing

```bash
python process_normalized.py
```

This converts MIDI files to binary piano roll representations:
- **Sampling**: 4 frames/second (one frame per sixteenth note)
- **Pitch range**: 88 keys (MIDI 21-108, A0 to C8)
- **Sequence length**: 16 timesteps (overlapping windows, stride 1)
- **Output**: Flattened vectors of size 1408 (16 * 88)


**Training configuration** (`config.py`):
- Batch size: 1024
- Loss: Binary cross-entropy
- Optimizer: AdamW with weight decay
- Regularization: Dropout (0.1), L2 (0.001)
- Early stopping patience: 50 epochs

The provided weights (`epoch113_loss0.0291.weights.h5`) were trained for 113 epochs until convergence.

### 3. Dimensionality Reduction (Two-Step PCA)

The `two_step_pca_circle_analysis.py` script performs hierarchical PCA:

1. **Extract latent vectors** (128D) for all sequences
2. **Step 1**: Aggregate sequences into piece-level centroids
3. **Step 2**: Further aggregate pieces into key-level centroids (averaging Major/Minor pairs)
4. **PCA**: Reduce to 2D for visualization (PC1, PC2)
5. **Circle fitting**: Least-squares fit to key centroids

This two-step aggregation reveals the hierarchical structure.

## Reproducing Paper Results

### Figure 4: Hierarchical Convergence

```bash
python scripts/generate_figure4_convergence.py
```

Expected output matches paper Figure 4:
- Clear hierarchical clustering from sequences → pieces → keys
- Close-tonic pairs (purple dashed) closer than relative pairs (gray solid)
- Near-perfect circle of fifths (12 keys evenly spaced)
These statistics are computed during figure generation and printed to console.

## Model Weights

- **File**: `epoch113_loss0.0291.weights.h5`
- **Training loss**: 0.0291 (converged at epoch 113)
- **Latent dimension**: 128
- **Architecture**: Feedforward autoencoder (see `autoencoder.py`)

To load weights:

```python
from autoencoder import build_autoencoder
from config import *

encoder, decoder, autoencoder = build_autoencoder()
autoencoder.load_weights('epoch113_loss0.0291.weights.h5')
```


## License

This project is released under the MIT License. The MIDI files are from public domain recordings of J.S. Bach's *Well-Tempered Clavier*, Book I.

## Contact

- **Najla Sadek**: nss32@mail.aub.edu
- **Joseph Bakarji**: jb50@aub.edu.lb
- **Institution**: American University of Beirut, Lebanon

