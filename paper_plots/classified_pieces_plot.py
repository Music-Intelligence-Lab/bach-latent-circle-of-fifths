"""
Classified Pieces Visualization
Plots 10 classified pieces (excluding BWV 882) with their top 2 nearest training pieces
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
from scipy.spatial.distance import cdist
import matplotlib.patheffects as pe

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "processed_data/all_sequences.npy"
INDICES_PATH = "processed_data/start_indices.txt"
WEIGHTS_PATH = "code/epoch113_loss0.0291.weights.h5"

# Model architecture
LATENT_DIM = 128
SEQUENCE_LENGTH = 16
INPUT_DIM = SEQUENCE_LENGTH * 88
ENCODER_LAYERS = [1024, 512, 256]
DROPOUT_RATE = 0.1

# Training piece names
PIECE_NAMES = [
    "1_C_Major", "2_C_Minor", "3_C#_Major", "4_C#_Minor",
    "5_D_Major", "6_D_Minor", "7_Eb_Major", "8_Eb_Minor",
    "9_E_Major", "10_E_Minor", "11_F_Major", "12_F_Minor",
    "13_F#_Major", "14_F#_Minor", "15_G_Major", "16_G_Minor",
    "17_Ab_Major", "18_Ab_Minor", "19_A_Major", "20_A_Minor",
    "21_Bb_Major", "22_Bb_Minor", "23_B_Major", "24_B_Minor"
]

# 9 classified pieces (excluding BWV 882 - worst, and BWV 846 - not in to-test folder)
CLASSIFIED_PIECES = [
    ('French Suite No. 5', 'G Major', 3.565, 1.093),
    ('BWV 892 (Book II, #23)', 'B Major', -1.696, -3.656),
    ('Partita No. 1 (30 BPM)', 'Bb Major', -0.658, 2.568),
    ('Invention No. 1', 'C Major', 1.60, 3.70),
    ('Invention No. 8', 'F Major', 1.588, 4.349),
    ('Invention No. 13', 'A Minor', 3.652, 1.836),
    ('Invention No. 7', 'E Minor', 4.433, -0.782),
    ('BWV 883 (Book II, #14)', 'F# Minor', 1.340, -3.275),
    ('Invention No. 4', 'D Minor', 2.076, 2.456),
]

KEY_TONICS = ["C", "C", "C#", "C#", "D", "D", "Eb", "Eb", "E", "E", "F", "F",
              "F#", "F#", "G", "G", "Ab", "Ab", "A", "A", "Bb", "Bb", "B", "B"]

UNIQUE_TONICS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

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

def get_piece_color(piece_idx: int):
    tonic = KEY_TONICS[piece_idx]
    tonic_idx = UNIQUE_TONICS.index(tonic)
    return key_colors_base[tonic_idx]

print("=" * 70)
print("CLASSIFIED PIECES VISUALIZATION (9 pieces)")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\n[1/4] Loading data from: {DATA_PATH}")
all_sequences = np.load(DATA_PATH)
indices = np.loadtxt(INDICES_PATH, dtype=int).tolist()
indices.append(len(all_sequences))
sequences_flat = all_sequences.reshape(len(all_sequences), -1)

# ============================================================================
# BUILD AND LOAD AUTOENCODER
# ============================================================================
print(f"\n[2/4] Building autoencoder...")

encoder_input = Input(shape=(INPUT_DIM,), name='encoder_input')
x = encoder_input
for i, units in enumerate(ENCODER_LAYERS):
    x = Dense(units, activation='swish', name=f'encoder_dense_{i}')(x)
    x = BatchNormalization(name=f'encoder_bn_{i}')(x)
    x = Dropout(DROPOUT_RATE, name=f'encoder_dropout_{i}')(x)

latent = Dense(LATENT_DIM, activation='linear', name='latent')(x)
encoder = Model(encoder_input, latent, name='encoder')

decoder_input = Input(shape=(LATENT_DIM,), name='decoder_input')
x = decoder_input
for i, units in enumerate(reversed(ENCODER_LAYERS)):
    x = Dense(units, activation='swish', name=f'decoder_dense_{i}')(x)
    x = BatchNormalization(name=f'decoder_bn_{i}')(x)

decoder_output = Dense(INPUT_DIM, activation='sigmoid', name='decoder_output')(x)
decoder = Model(decoder_input, decoder_output, name='decoder')

autoencoder_input = Input(shape=(INPUT_DIM,))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = Model(autoencoder_input, decoded, name='autoencoder')

autoencoder.load_weights(WEIGHTS_PATH)
print(f"   [OK] Weights loaded")

# ============================================================================
# ENCODE AND PCA
# ============================================================================
print(f"\n[3/4] Encoding and PCA...")

batch_size = 512
latent_vectors = []
for i in range(0, len(sequences_flat), batch_size):
    batch = sequences_flat[i:i+batch_size]
    encoded_batch = encoder.predict(batch, verbose=0)
    latent_vectors.append(encoded_batch)

latent_vectors = np.vstack(latent_vectors)

pca = PCA(n_components=2)
sequences_pca = pca.fit_transform(latent_vectors)

# Compute piece centroids for training data
piece_centroids = []
for i in range(len(indices) - 1):
    start_idx = indices[i]
    end_idx = indices[i + 1]
    centroid = np.mean(sequences_pca[start_idx:end_idx], axis=0)
    piece_centroids.append(centroid)

piece_centroids = np.array(piece_centroids)

# Load centroids from CSV
pieces_df = pd.read_csv('csv_data/centroids_pieces.csv')

print(f"   Training piece centroids: {piece_centroids.shape}")
print(f"   Classified pieces: {len(CLASSIFIED_PIECES)}")

# ============================================================================
# PLOTTING
# ============================================================================
print(f"\n[4/4] Creating visualization...")

fig, ax = plt.subplots(figsize=(16, 12))

# Plot training piece centroids
for i, (centroid, name) in enumerate(zip(piece_centroids, PIECE_NAMES)):
    is_major = "Major" in name
    marker = 'o' if is_major else '^'

    ax.scatter(centroid[0], centroid[1],
               c=[get_piece_color(i)], s=250, marker=marker,
               edgecolors='black', linewidths=2,
               zorder=20, alpha=0.9)

    key = name.split('_')[1]
    label = f"{key}{'M' if is_major else 'm'}"

    txt = ax.annotate(label,
                      xy=(centroid[0], centroid[1]),
                      xytext=(0, 8),
                      textcoords='offset points',
                      fontsize=13, ha='center', fontweight='bold',
                      color='white', zorder=21)
    txt.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])

# Plot classified pieces as STARS with connections to top 2 nearest
latex_rows = []

for name, truth, pc1, pc2 in CLASSIFIED_PIECES:
    centroid = np.array([[pc1, pc2]])
    coords = pieces_df[['PC1', 'PC2']].values
    dists = cdist(centroid, coords, metric='euclidean')[0]

    # Get top 2 closest
    top2_idx = np.argsort(dists)[:2]

    pred1 = pieces_df.iloc[top2_idx[0]]
    pred2 = pieces_df.iloc[top2_idx[1]]

    # Plot classified piece as RED STAR (below training pieces)
    ax.scatter(pc1, pc2,
               marker='*', s=600, c='red',
               edgecolors='black', linewidths=2,
               zorder=5, alpha=0.95)

    # Get short name for labeling and table
    short_name = name.split('(')[0].strip()
    if 'BWV' in name:
        short_name = name.split(',')[0].strip()

    # Simple number label for the classified piece
    piece_num = CLASSIFIED_PIECES.index((name, truth, pc1, pc2)) + 1

    txt = ax.annotate(str(piece_num),
                      xy=(pc1, pc2),
                      xytext=(0, 15),
                      textcoords='offset points',
                      fontsize=10, ha='center', fontweight='bold',
                      color='white', zorder=6)
    txt.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])

    # Determine which lines to draw based on matches
    def format_key(row):
        return f"{row['Key']} {row['Mode']}"

    match1 = (format_key(pred1) == truth)
    match2 = (format_key(pred2) == truth)

    # Draw lines to top 2 nearest pieces
    # If 1st is correct: only draw solid line to 1st
    # If 2nd is correct (but 1st wrong): draw both lines
    # If neither correct: draw both lines

    if match1:
        # Only draw line to 1st prediction (solid)
        pred_pc1 = pred1['PC1']
        pred_pc2 = pred1['PC2']
        ax.plot([pc1, pred_pc1], [pc2, pred_pc2],
                color='red', linestyle='-', linewidth=2.5,
                alpha=0.8, zorder=25)
    else:
        # Draw both lines (1st solid, 2nd dashed)
        for idx, (pred, rank) in enumerate([(pred1, '1st'), (pred2, '2nd')]):
            pred_pc1 = pred['PC1']
            pred_pc2 = pred['PC2']

            linestyle = '-' if rank == '1st' else '--'
            linewidth = 2.5 if rank == '1st' else 1.5
            alpha = 0.8 if rank == '1st' else 0.5

            ax.plot([pc1, pred_pc1], [pc2, pred_pc2],
                    color='red', linestyle=linestyle, linewidth=linewidth,
                    alpha=alpha, zorder=25)

    # Prepare LaTeX row
    match1_char = 'Y' if match1 else 'N'
    match2_char = 'Y' if match2 else 'N'

    latex_rows.append({
        'Piece': short_name,
        'Truth': truth,
        '1st': format_key(pred1),
        'D1': f'{dists[top2_idx[0]]:.2f}',
        'M1': match1_char,
        '2nd': format_key(pred2),
        'D2': f'{dists[top2_idx[1]]:.2f}',
        'M2': match2_char
    })

ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
ax.set_title('Classified Pieces (Red Stars) with Top 2 Nearest Training Pieces\n' +
             '(Solid = 1st closest, Dashed = 2nd closest)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.5)
ax.set_xlim(-4, 6)
ax.set_ylim(-4, 8)
ax.set_aspect('equal', adjustable='box')

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Build legend with piece names and true keys
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=8, label='Major (training)', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
           markersize=8, label='Minor (training)', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], color='red', linestyle='-', linewidth=2,
           label='1st nearest'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
           label='2nd nearest'),
    Rectangle((0, 0), 1, 1, fc="white", ec="white", linewidth=0, label=' '),  # Spacer
]

# Add classified pieces to legend
for i, (name, truth, _, _) in enumerate(CLASSIFIED_PIECES, 1):
    short_name = name.split('(')[0].strip()
    if 'BWV' in name:
        short_name = name.split(',')[0].strip()

    legend_elements.append(
        Line2D([0], [0], marker='*', color='red', markersize=12,
               label=f'{i}. {short_name} ({truth})',
               markeredgecolor='black', markeredgewidth=1, linestyle='None')
    )

# Create two-column legend
ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
          ncol=1, framealpha=0.95, edgecolor='black')

plt.tight_layout()
output_path = "classified_pieces_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n[SAVED] {output_path}")

output_path_pdf = "classified_pieces_visualization.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"[SAVED] {output_path_pdf}")

plt.show()

# ============================================================================
# GENERATE LATEX TABLE
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING LATEX TABLE")
print("=" * 70)

latex_table = r"""\begin{table}[h]
\centering
\caption{Classification Results for 9 Test Pieces (excluding BWV 882 and BWV 846)}
\label{tab:classification_results}
\begin{tabular}{|l|l|l|c|c|l|c|c|}
\hline
\textbf{Piece} & \textbf{Truth} & \textbf{1st Pred.} & \textbf{Dist} & \textbf{M} & \textbf{2nd Pred.} & \textbf{Dist} & \textbf{M} \\
\hline
"""

for row in latex_rows:
    piece = row['Piece'].replace('&', r'\&')
    latex_table += f"{piece} & {row['Truth']} & {row['1st']} & {row['D1']} & {row['M1']} & {row['2nd']} & {row['D2']} & {row['M2']} \\\\\n"

latex_table += r"""\hline
\end{tabular}
\end{table}
"""

# Save LaTeX table
latex_path = "classification_results_table.tex"
with open(latex_path, 'w') as f:
    f.write(latex_table)

print(f"\n[SAVED] {latex_path}")
print("\nLaTeX Table Preview:")
print(latex_table)

# Also save as CSV
csv_df = pd.DataFrame(latex_rows)
csv_path = "classification_results_summary.csv"
csv_df.to_csv(csv_path, index=False)
print(f"[SAVED] {csv_path}")

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\nGenerated files:")
print(f"  - {output_path} (visualization)")
print(f"  - {output_path_pdf} (PDF)")
print(f"  - {latex_path} (LaTeX table)")
print(f"  - {csv_path} (CSV summary)")
