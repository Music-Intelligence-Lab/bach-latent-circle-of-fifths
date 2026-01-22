# Circle of Fifths as Latent Geometry in Bach's Well-Tempered Clavier

**Published at EAAI @ AAAI 2026, Singapore (PMLR)**

📄 **[Paper on OpenReview](https://openreview.net/forum?id=HXm5wUgRib)**

## Abstract

Can unsupervised deep learning methods encode fundamental music-theoretic features? We answer this question by training an autoencoder on J.S. Bach's Well-Tempered Clavier and analyzing its latent space via principal component analysis. Sequences in the first two principal components are clustered hierarchically into pieces and keys that spontaneously arrange in a circle-of-fifths geometry. Quantitatively, relative major-minor key pairs (sharing pitch collections) lie more than three times closer than non-relative pairs, and circle-of-fifths distance correlates strongly with learned distances. This structure emerges entirely from reconstruction loss, with no harmonic labels or supervision. Our results suggest that the circle of fifths is an intrinsic property of tonal relationships, demonstrating that unsupervised representation learning can recover harmonic principles that open the door for interpretable data-driven exploration of latent spaces across diverse musical traditions.

**Keywords**: Deep Learning, Music Information Retrieval, Autoencoders, Latent Space Analysis, Circle of Fifths, Bach

---

## 🎹 [Interactive Demo: Bach Latent Space Explorer](https://music-intelligence-lab.github.io/bach-latent-circle-of-fifths/)

**Explore the latent space interactively with MIDI playback!**

![Bach Latent Space Explorer](image.png)

---

## Key Findings

- **Hierarchical Organization**: Sequences naturally cluster into pieces, which cluster into keys
- **Circle of Fifths Emerges**: Key centroids form a near-perfect circle (CV = 8.5%) in 2D PCA space
- **Harmonic Function Encoded**: Close-tonic pairs (e.g., C Major and D Minor) are closer than relative pairs (e.g., C Major and A Minor), despite having more accidentals
- **No Supervision Required**: All structure emerges purely from reconstruction loss

## Installation

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

Pre-trained weights can be found on Huggingface: https://huggingface.co/josephbakarji/bach-circle-of-fifths

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

![Hierarchical Clustering](paper_plots/hierarchical_clustering_3steps.png)

### Classification of Unseen Pieces

```bash
cd paper_plots
python classified_pieces_plot.py
```

Visualizes 9 test pieces (red stars) with connections to their nearest training pieces, demonstrating the model's ability to classify unseen Bach compositions.

![Classification](paper_plots/classified_pieces_visualization.png)

### Data Files

The `paper_plots/` folder includes all necessary data to run the scripts:
- `processed_data/` - Sequence data (`all_sequences.npy`, indices, labels)
- `csv_data/centroids_pieces.csv` - 24 piece-level centroid coordinates
- `code/epoch113_loss0.0291.weights.h5` - Trained model weights
- `relative_key_distances.csv`, `wholetone_distances.csv` - Key relationship data

## MIDI Data

- **Original Data** (`original_data/`): 24 MIDI files from Bach's Well-Tempered Clavier, Book I (BWV 846-869)
- **Normalized Data** (`normalized_data/`): Same pieces normalized to 60 BPM for consistent encoding

## Model

- **Architecture**: Feedforward autoencoder (see `autoencoder.py`)
- **Latent dimension**: 128
- **Training loss**: 0.0291 (converged at epoch 113)
- **Weights**: `epoch113_loss0.0291.weights.h5`

```python
from autoencoder import build_autoencoder
from config import *

encoder, decoder, autoencoder = build_autoencoder()
autoencoder.load_weights('epoch113_loss0.0291.weights.h5')
```

## Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{sadek2026circle,
  title={Circle of Fifths as Latent Geometry in Bach's Well-Tempered Clavier},
  author={Sadek, Najla and Bakarji, Joseph},
  booktitle={EAAI @ AAAI 2026},
  year={2026},
  publisher={PMLR}
}
```

## License

MIT License. Bach's compositions are in the public domain.

## Contact

- **Najla Sadek**: nss32@mail.aub.edu
- **Joseph Bakarji**: jb50@aub.edu.lb
- **Institution**: American University of Beirut, Lebanon
