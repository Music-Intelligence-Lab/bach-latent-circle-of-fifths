# Circle of Fifths as Latent Geometry in Bach's Preludes and Fugues

## Overview

This project demonstrates that unsupervised deep learning can spontaneously recover fundamental music-theoretic structure. We trained a simple feedforward autoencoder on J.S. Bach's *Well-Tempered Clavier*, Book I, and discovered that the learned latent space organizes pieces hierarchically into keys arranged in circle-of-fifths geometry.

### Interactive Demo

Explore the latent space interactively with MIDI playback:

**[Bach Latent Space Explorer](https://music-intelligence-lab.github.io/bach-latent-circle-of-fifths/)**

### Key Findings

- **Hierarchical Organization**: Sequences naturally cluster into pieces, which cluster into keys
- **Circle of Fifths Emerges**: Key centroids form a near-perfect circle (CV = 8.5%) in 2D PCA space
- **Harmonic Function Encoded**: Close-tonic pairs (e.g., C Major and D Minor) are closer than relative pairs (e.g., C Major and A Minor), despite having more accidentals
- **No Supervision Required**: All structure emerges purely from reconstruction loss

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Music-Intelligence-Lab/bach-latent-circle-of-fifths.git
cd bach-latent-circle-of-fifths

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Pre-trained weights can be found on huggingface: https://huggingface.co/josephbakarji/bach-circle-of-fifths

## Paper Plots

The `paper_plots/` folder contains scripts to reproduce key figures from the paper:

### Hierarchical Clustering (Figure 2)

```bash
cd paper_plots
python hierarchical_clustering_plot.py
```

Generates a 3-panel visualization showing:
- **A. Individual Sequences**: 2,038 sequences colored by tonic
- **B. Piece-Level Centroids**: 24 pieces with relative key (green) and Dorian (orange) connections
- **C. Circle-of-Fifths Fit**: 12 key centroids with fitted circle

Output: `hierarchical_clustering_3steps.png/pdf`

![Hierarchical Clustering](paper_plots/hierarchical_clustering_3steps.png)

### Classification of Unseen Pieces

```bash
cd paper_plots
python classified_pieces_plot.py
```

Visualizes 9 test pieces (red stars) with connections to their nearest training pieces, demonstrating the model's ability to classify unseen Bach compositions.

Output: `classified_pieces_visualization.png/pdf`

![Classification](paper_plots/classified_pieces_visualization.png)

### Paper Plots Data Files

The `paper_plots/` folder includes all necessary data:
- `processed_data/` - Sequence data (`all_sequences.npy`, indices, labels)
- `csv_data/centroids_pieces.csv` - 24 piece-level centroid coordinates
- `code/epoch113_loss0.0291.weights.h5` - Trained model weights
- `relative_key_distances.csv`, `wholetone_distances.csv` - Key relationship data

## MIDI Files

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

### 2. Dimensionality Reduction (Two-Step PCA)

The `two_step_pca_circle_analysis.py` script performs hierarchical PCA:

1. **Extract latent vectors** (128D) for all sequences
2. **Step 1**: Aggregate sequences into piece-level centroids
3. **Step 2**: Further aggregate pieces into key-level centroids (averaging Major/Minor pairs)
4. **PCA**: Reduce to 2D for visualization (PC1, PC2)
5. **Circle fitting**: Least-squares fit to key centroids

This two-step aggregation reveals the hierarchical structure.

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
