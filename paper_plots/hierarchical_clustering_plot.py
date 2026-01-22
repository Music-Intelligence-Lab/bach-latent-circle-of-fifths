"""
Hierarchical Clustering Visualization - 3 Steps
Reproduces Figure 2 from the paper showing emergence of circle-of-fifths geometry

Step 1: Individual sequences (2038 points) - colored by piece/tonic
Step 2: Piece-level centroids (24 points) - averaging sequences within each piece
Step 3: Key-level centroids (12 points) - averaging major/minor pairs with same tonic

IMPORTANT: This uses the autoencoder's 128D latent space, then applies PCA
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib.patheffects as pe

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "processed_data/all_sequences.npy"
INDICES_PATH = "processed_data/start_indices.txt"
LABELS_PATH = "processed_data/section_labels.txt"
WEIGHTS_PATH = "code/epoch113_loss0.0291.weights.h5"

# Model architecture
LATENT_DIM = 128
SEQUENCE_LENGTH = 16
INPUT_DIM = SEQUENCE_LENGTH * 88
ENCODER_LAYERS = [1024, 512, 256]
DROPOUT_RATE = 0.1

# 24 keys in circle-of-fifths order for the Well-Tempered Clavier Book I
PIECE_NAMES = [
    "1_C_Major", "2_C_Minor", "3_C#_Major", "4_C#_Minor",
    "5_D_Major", "6_D_Minor", "7_Eb_Major", "8_Eb_Minor",
    "9_E_Major", "10_E_Minor", "11_F_Major", "12_F_Minor",
    "13_F#_Major", "14_F#_Minor", "15_G_Major", "16_G_Minor",
    "17_Ab_Major", "18_Ab_Minor", "19_A_Major", "20_A_Minor",
    "21_Bb_Major", "22_Bb_Minor", "23_B_Major", "24_B_Minor"
]

# Extract key information
KEY_TONICS = ["C", "C", "C#", "C#", "D", "D", "Eb", "Eb", "E", "E", "F", "F",
              "F#", "F#", "G", "G", "Ab", "Ab", "A", "A", "Bb", "Bb", "B", "B"]

UNIQUE_TONICS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# ============================================================================
# COLOR SCHEME: one color per tonic, consistent across all figures
# ============================================================================
key_colors_base = [
    '#FF4444',  # C
    '#00CED1',  # C#
    '#FFD700',  # D
    '#40E0D0',  # Eb
    '#FF6347',  # E
    '#9370DB',  # F
    '#FF1493',  # F#
    '#32CD32',  # G
    '#FF8C00',  # Ab
    '#ADFF2F',  # A
    '#FF00FF',  # Bb
    '#1E90FF',  # B
]

KEY_COLOR_MAP = {tonic: key_colors_base[i] for i, tonic in enumerate(UNIQUE_TONICS)}

def get_piece_color(piece_idx: int):
    """Color for a piece index 0–23 based on its tonic (CM/Cm same color)."""
    tonic = KEY_TONICS[piece_idx]
    tonic_idx = UNIQUE_TONICS.index(tonic)
    return key_colors_base[tonic_idx]

print("=" * 70)
print("HIERARCHICAL CLUSTERING - 3 STEPS (WITH AUTOENCODER)")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\n[1/5] Loading data from: {DATA_PATH}")
all_sequences = np.load(DATA_PATH)
print(f"   Loaded shape: {all_sequences.shape}")

indices = np.loadtxt(INDICES_PATH, dtype=int).tolist()
indices.append(len(all_sequences))

sequences_flat = all_sequences.reshape(len(all_sequences), -1)

# ============================================================================
# BUILD AND LOAD AUTOENCODER
# ============================================================================
print(f"\n[2/5] Building autoencoder...")

encoder_input = Input(shape=(INPUT_DIM,), name='encoder_input')
x = encoder_input
for i, units in enumerate(ENCODER_LAYERS):
    x = Dense(units, activation='swish', name=f'encoder_dense_{i}')(x)
    x = BatchNormalization(name=f'encoder_bn_{i}')(x)
    x = Dropout(DROPOUT_RATE, name=f'encoder_dropout_{i}')(x)

latent = Dense(LATENT_DIM, activation='linear', name='latent')(x)
encoder = Model(encoder_input, latent, name='encoder')

# Decoder (needed to load weights)
decoder_input = Input(shape=(LATENT_DIM,), name='decoder_input')
x = decoder_input
for i, units in enumerate(reversed(ENCODER_LAYERS)):
    x = Dense(units, activation='swish', name=f'decoder_dense_{i}')(x)
    x = BatchNormalization(name=f'decoder_bn_{i}')(x)

decoder_output = Dense(INPUT_DIM, activation='sigmoid', name='decoder_output')(x)
decoder = Model(decoder_input, decoder_output, name='decoder')

# Full autoencoder
autoencoder_input = Input(shape=(INPUT_DIM,))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = Model(autoencoder_input, decoded, name='autoencoder')

print(f"   Loading weights from: {WEIGHTS_PATH}")
autoencoder.load_weights(WEIGHTS_PATH)
print(f"   [OK] Weights loaded successfully")

# ============================================================================
# ENCODE TO LATENT SPACE
# ============================================================================
print(f"\n[3/5] Encoding sequences to {LATENT_DIM}D latent space...")

batch_size = 512
latent_vectors = []
for i in range(0, len(sequences_flat), batch_size):
    batch = sequences_flat[i:i+batch_size]
    encoded_batch = encoder.predict(batch, verbose=0)
    latent_vectors.append(encoded_batch)

latent_vectors = np.vstack(latent_vectors)
print(f"   Latent vectors shape: {latent_vectors.shape}")

# ============================================================================
# PCA PROJECTION
# ============================================================================
print(f"\n[4/5] Applying PCA to {LATENT_DIM}D latent vectors")

pca = PCA(n_components=2)
sequences_pca = pca.fit_transform(latent_vectors)

print(f"   PC1 variance: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"   PC2 variance: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"   Total variance: {sum(pca.explained_variance_ratio_)*100:.2f}%")
print(f"   (Paper reports: ~8.68% total)")

# Print PCA statistics
seq_pc1 = sequences_pca[:, 0]
seq_pc2 = sequences_pca[:, 1]
print(f"\n   PC1 range: [{seq_pc1.min():.3f}, {seq_pc1.max():.3f}], mean={seq_pc1.mean():.3f}, std={seq_pc1.std():.3f}")
print(f"   PC2 range: [{seq_pc2.min():.3f}, {seq_pc2.max():.3f}], mean={seq_pc2.mean():.3f}, std={seq_pc2.std():.3f}")
print(f"   First 5 PC1 values: {seq_pc1[:5]}")
print(f"   First 5 PC2 values: {seq_pc2[:5]}")

# ============================================================================
# STEP 1: INDIVIDUAL SEQUENCES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Individual Sequences (2038 points)")
print("=" * 70)

sequence_labels = []
for i in range(len(indices) - 1):
    num_seq = indices[i + 1] - indices[i]
    sequence_labels.extend([i] * num_seq)

print(f"Total sequences: {len(sequence_labels)}")

# ============================================================================
# STEP 2: PIECE-LEVEL CENTROIDS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Piece-Level Centroids (24 points)")
print("=" * 70)

piece_centroids = []
for i in range(len(indices) - 1):
    start_idx = indices[i]
    end_idx = indices[i + 1]
    centroid = np.mean(sequences_pca[start_idx:end_idx], axis=0)
    piece_centroids.append(centroid)
    print(f"   Piece {i+1:2d} ({PIECE_NAMES[i]:15s}): centroid at ({centroid[0]:6.2f}, {centroid[1]:6.2f})")

piece_centroids = np.array(piece_centroids)

# ============================================================================
# STEP 3: KEY-LEVEL CENTROIDS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Key-Level Centroids (12 points)")
print("=" * 70)

key_centroids = []
key_labels = []

for tonic in UNIQUE_TONICS:
    tonic_indices = [i for i, t in enumerate(KEY_TONICS) if t == tonic]
    key_centroid = np.mean(piece_centroids[tonic_indices], axis=0)
    key_centroids.append(key_centroid)
    key_labels.append(tonic)

    piece_names_str = ", ".join([PIECE_NAMES[i] for i in tonic_indices])
    print(f"   Key {tonic:3s}: {piece_names_str}")

key_centroids = np.array(key_centroids)

# ============================================================================
# COMPUTE STATISTICS
# ============================================================================
key_distance_matrix = squareform(pdist(key_centroids, metric='euclidean'))

# Circle-of-fifths order
circle_of_fifths_order = ["C", "G", "D", "A", "E", "B", "F#", "C#", "Ab", "Eb", "Bb", "F"]
key_to_idx = {key: i for i, key in enumerate(UNIQUE_TONICS)}

# Whole-tone distances
wholetone_pairs = [
    ("C", "D"), ("D", "E"), ("E", "F#"), ("F#", "Ab"),
    ("Ab", "Bb"), ("Bb", "C"), ("C#", "Eb"), ("Eb", "F"),
    ("F", "G"), ("G", "A"), ("A", "B"), ("B", "C#")
]
wholetone_distances = []
for key1, key2 in wholetone_pairs:
    if key1 in key_to_idx and key2 in key_to_idx:
        idx1 = key_to_idx[key1]
        idx2 = key_to_idx[key2]
        wholetone_distances.append(key_distance_matrix[idx1, idx2])

mean_wholetone = np.mean(wholetone_distances)
std_wholetone = np.std(wholetone_distances)

# Circle-of-fifths correlation (Pearson)
cof_distances = []
euclidean_distances = []
for i in range(len(UNIQUE_TONICS)):
    for j in range(i+1, len(UNIQUE_TONICS)):
        key1 = UNIQUE_TONICS[i]
        key2 = UNIQUE_TONICS[j]

        if key1 in circle_of_fifths_order and key2 in circle_of_fifths_order:
            pos1 = circle_of_fifths_order.index(key1)
            pos2 = circle_of_fifths_order.index(key2)
            cof_dist = min(abs(pos1 - pos2), 12 - abs(pos1 - pos2))
            eucl_dist = key_distance_matrix[i, j]
            cof_distances.append(cof_dist)
            euclidean_distances.append(eucl_dist)

correlation, p_value = pearsonr(cof_distances, euclidean_distances)
print(f"\nCircle-of-fifths correlation: r={correlation:.3f}, p={p_value:.2e}")

# Circle fit
center = np.mean(key_centroids, axis=0)
distances = np.linalg.norm(key_centroids - center, axis=1)
radius = np.mean(distances)
cv = np.std(distances) / radius * 100

print(f"\n" + "=" * 70)
print("KEY STATISTICS (from paper)")
print("=" * 70)
print(f"\nWhole-tone apart keys:")
print(f"   Mean distance: {mean_wholetone:.2f} +/- {std_wholetone:.2f} PC units")
print(f"   (Paper: 0.79 +/- 0.36)")

print(f"\nCircle-of-fifths correlation:")
print(f"   Pearson r = {correlation:.2f}")
print(f"   (Paper: r = 0.71, p < 0.001)")

print(f"\nCircle fit:")
print(f"   Radius: {radius:.2f} PC units (CV: {cv:.2f}%)")
print(f"   (Paper: radius ~2.6, CV ~8.5%)")

# ============================================================================
# LOAD RELATIONSHIP DATA
# ============================================================================
print(f"\n[5/6] Loading key relationship data...")

try:
    relative_df = pd.read_csv("relative_key_distances.csv")
    wholetone_df = pd.read_csv("wholetone_distances.csv")
    print(f"   Loaded {len(relative_df)} relative key pairs")
    print(f"   Loaded {len(wholetone_df)} whole-tone (Dorian) pairs")
except FileNotFoundError as e:
    print(f"   Warning: Could not load relationship CSV files: {e}")
    relative_df = pd.DataFrame()
    wholetone_df = pd.DataFrame()

# ============================================================================
# PLOTTING
# ============================================================================
print(f"\n[6/6] Generating plots...")

# Different axis limits for each panel (as requested: 10, 4, 4)
all_pc1 = np.concatenate([sequences_pca[:, 0], piece_centroids[:, 0], key_centroids[:, 0]])
all_pc2 = np.concatenate([sequences_pca[:, 1], piece_centroids[:, 1], key_centroids[:, 1]])
data_x_range = (all_pc1.min() - 1, all_pc1.max() + 1)
data_y_range = (all_pc2.min() - 1, all_pc2.max() + 1)

# Use data-driven limits for plot 1 (sequences), fixed ±4 for plots 2&3
x_range_seq = data_x_range
y_range_seq = data_y_range
x_range_piece = (-4, 4)
y_range_piece = (-4, 4)
x_range_key = (-4, 4)
y_range_key = (-4, 4)

print(f"   Plot 1 axis limits: X={x_range_seq}, Y={y_range_seq}")
print(f"   Plot 2 axis limits: X={x_range_piece}, Y={y_range_piece}")
print(f"   Plot 3 axis limits: X={x_range_key}, Y={y_range_key}")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --------------------------------------------------------------------------
# PLOT 1: Individual Sequences
# --------------------------------------------------------------------------
ax = axes[0]
for i in range(len(indices) - 1):
    start_idx = indices[i]
    end_idx = indices[i + 1]
    ax.scatter(sequences_pca[start_idx:end_idx, 0],
               sequences_pca[start_idx:end_idx, 1],
               c=[get_piece_color(i)], alpha=0.4, s=20, edgecolors='none')

ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
ax.set_title('A. Individual Sequences\n(2,038 sequences)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.5)
ax.set_xlim(x_range_seq)
ax.set_ylim(y_range_seq)
ax.set_aspect('equal', adjustable='box')

# --------------------------------------------------------------------------
# PLOT 2: Piece-Level Centroids
# --------------------------------------------------------------------------
ax = axes[1]

for i, (centroid, name) in enumerate(zip(piece_centroids, PIECE_NAMES)):
    is_major = "Major" in name
    marker = 'o' if is_major else '^'

    ax.scatter(centroid[0], centroid[1],
               c=[get_piece_color(i)], s=200, marker=marker,
               edgecolors=None, linewidths=2,
               zorder=10, alpha=0.9)

    key = name.split('_')[1]
    label = f"{key}{'M' if is_major else 'm'}"

    txt = ax.annotate(label,
                      xy=(centroid[0], centroid[1]),
                      xytext=(0, 8),
                      textcoords='offset points',
                      fontsize=11, ha='center', fontweight='bold',
                      color='white', zorder=20)
    txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground='black')])

# Create mapping from key_mode to centroids
piece_centroid_dict = {}
for i, name in enumerate(PIECE_NAMES):
    parts = name.split('_')
    key = parts[1]
    mode = parts[2]
    key_mode = f"{key}_{mode}"
    piece_centroid_dict[key_mode] = piece_centroids[i]

# Draw relative key connections (green dashed)
if not relative_df.empty:
    for _, row in relative_df.iterrows():
        major_key = f"{row['major_key']}_Major"
        minor_key = f"{row['minor_key']}_Minor"
        if major_key in piece_centroid_dict and minor_key in piece_centroid_dict:
            maj_pos = piece_centroid_dict[major_key]
            min_pos = piece_centroid_dict[minor_key]
            ax.plot([maj_pos[0], min_pos[0]], [maj_pos[1], min_pos[1]],
                    'g--', linewidth=2, alpha=0.5, zorder=1)

# Draw whole-tone (Dorian) connections (orange solid)
if not wholetone_df.empty:
    for _, row in wholetone_df.iterrows():
        major_key = f"{row['major_key']}_Major"
        minor_key = f"{row['minor_key']}_Minor"
        if major_key in piece_centroid_dict and minor_key in piece_centroid_dict:
            maj_pos = piece_centroid_dict[major_key]
            min_pos = piece_centroid_dict[minor_key]
            ax.plot([maj_pos[0], min_pos[0]], [maj_pos[1], min_pos[1]],
                    'orange', linewidth=2.5, alpha=0.7, zorder=1)

ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
ax.set_title('B. Piece-Level Centroids\n(24 pairs)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.5)
ax.set_xlim(x_range_piece)
ax.set_ylim(y_range_piece)
ax.set_aspect('equal', adjustable='box')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=10, label='Major', markeredgecolor='black', markeredgewidth=1.5),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
           markersize=10, label='Minor', markeredgecolor='black', markeredgewidth=1.5),
    Line2D([0], [0], color='green', linestyle='--', linewidth=2, alpha=0.5,
           label='Relative keys'),
    Line2D([0], [0], color='orange', linestyle='-', linewidth=2.5, alpha=0.7,
           label='Dorian')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=6)

# --------------------------------------------------------------------------
# PLOT 3: Key-Level Centroids
# --------------------------------------------------------------------------
ax = axes[2]

for centroid, label in zip(key_centroids, key_labels):
    color = KEY_COLOR_MAP[label]

    # colored disc with black edge
    ax.scatter(centroid[0], centroid[1],
               c=[color], s=400,
               edgecolors=None, linewidths=2.5,
               zorder=10, alpha=0.95)

    # white letters with black outline
    txt = ax.annotate(label,
                      xy=(centroid[0], centroid[1]),
                      ha='center', va='center',
                      fontsize=18, fontweight='bold',
                      color='white', zorder=11)
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='black')])

# Draw fitted red dashed circle
circle = plt.Circle(center, radius, fill=False, color='red',
                    linewidth=3, linestyle='--',
                    label=f'Fitted circle')
ax.add_patch(circle)

ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
title_text = f'C. Circle-of-Fifths Fit \n(12 keys, radius={radius:.2f} PC units)'
ax.set_title(title_text, fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.5)
ax.set_xlim(x_range_key)
ax.set_ylim(y_range_key)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='upper right', fontsize=11)

# Save
plt.tight_layout()
output_path = "hierarchical_clustering_3steps.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n[SAVED] {output_path}")

output_path_pdf = "hierarchical_clustering_3steps.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"[SAVED] {output_path_pdf}")

plt.show()

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print("\nThis reproduces Figure 2 from the paper using the autoencoder's latent space!")
