#!/usr/bin/env python3
"""
Generate Figure 4: Hierarchical Convergence Figure (3 panels)

This script generates the main convergence figure showing:
- Panel A: Individual sequences (2,724 points) colored by key
- Panel B: Piece-level centroids with relative key and close-tonic connections
- Panel C: Key-level centroids with fitted circle overlay

Required data:
- data/2step_animation_data.json (contains sequences, piece centroids, and key centroids)

Output:
- figures/fig4_combined_convergence.pdf
- figures/fig4_combined_convergence.png
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Circle
from matplotlib import patheffects
from pathlib import Path
from scipy.optimize import least_squares

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

print("=" * 80)
print("GENERATING FIGURE 4: HIERARCHICAL CONVERGENCE")
print("=" * 80)

# Load data
data_path = Path(__file__).parent.parent / 'data' / '2step_animation_data.json'
print(f"\nLoading data from: {data_path}")

with open(data_path, 'r') as f:
    data_2step = json.load(f)

sequences = data_2step['sequences']
piece_centroids = data_2step['piece_centroids']
key_centroids_no_mode = data_2step['key_centroids_no_mode']

# Extract coordinates
seq_pc1 = np.array([s['pc1'] for s in sequences])
seq_pc2 = np.array([s['pc2'] for s in sequences])
seq_pieces = [s['piece'] for s in sequences]

piece_pc1 = np.array([p['avg_pc1'] for p in piece_centroids])
piece_pc2 = np.array([p['avg_pc2'] for p in piece_centroids])
piece_keys = [p['key'] for p in piece_centroids]
piece_modes = [p['mode'] for p in piece_centroids]

key_pc1 = np.array([k['avg_pc1'] for k in key_centroids_no_mode])
key_pc2 = np.array([k['avg_pc2'] for k in key_centroids_no_mode])
key_names = [k['key'] for k in key_centroids_no_mode]

print(f"  Sequences: {len(sequences)}")
print(f"  Pieces: {len(piece_centroids)}")
print(f"  Keys: {len(key_centroids_no_mode)}")

# Color scheme - single color per key (same for Major and Minor)
keyColors = {
    'C': '#FF4444',   'C#': '#00CED1',  'D': '#FFD700',   'Eb': '#40E0D0',
    'E': '#FF6347',   'F': '#9370DB',   'F#': '#FF1493',  'G': '#32CD32',
    'Ab': '#FF8C00',  'A': '#ADFF2F',   'Bb': '#FF00FF',  'B': '#1E90FF'
}

def get_piece_color(key):
    """Get single color for a key (same for Major and Minor)"""
    return keyColors.get(key, '#888888')

# Identify RELATIVE KEY pairs (Major-Minor sharing diatonic collection)
# Relative minor is 3 semitones (9 fifths) down from major
relative_pairs = []
key_order = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

for i, centroid in enumerate(piece_centroids):
    key = centroid['key']
    mode = centroid['mode']

    if mode == 'Major':
        key_idx = key_order.index(key)
        relative_minor_idx = (key_idx + 9) % 12
        relative_key = key_order[relative_minor_idx]
        relative_mode = 'Minor'
    else:
        key_idx = key_order.index(key)
        relative_major_idx = (key_idx + 3) % 12
        relative_key = key_order[relative_major_idx]
        relative_mode = 'Major'

    for j, other in enumerate(piece_centroids):
        if other['key'] == relative_key and other['mode'] == relative_mode:
            if i < j:
                relative_pairs.append((i, j))

print(f"\nFound {len(relative_pairs)} relative key pairs")

# Identify CLOSE-TONIC pairs (tonics a whole step apart)
# e.g., C Major and D Minor (sharing many harmonic functions)
close_tonic_pairs = []

for i, centroid in enumerate(piece_centroids):
    key = centroid['key']
    mode = centroid['mode']

    if mode == 'Major':
        # Close-tonic: minor key 2 semitones up (whole step)
        key_idx = key_order.index(key)
        close_tonic_idx = (key_idx + 2) % 12
        close_tonic_key = key_order[close_tonic_idx]
        close_tonic_mode = 'Minor'

        for j, other in enumerate(piece_centroids):
            if other['key'] == close_tonic_key and other['mode'] == close_tonic_mode:
                if i < j:
                    close_tonic_pairs.append((i, j))

print(f"Found {len(close_tonic_pairs)} close-tonic pairs")

# Compute distances
relative_distances = []
for i, j in relative_pairs:
    dist = np.sqrt((piece_pc1[i] - piece_pc1[j])**2 + (piece_pc2[i] - piece_pc2[j])**2)
    relative_distances.append(dist)

close_tonic_distances = []
for i, j in close_tonic_pairs:
    dist = np.sqrt((piece_pc1[i] - piece_pc1[j])**2 + (piece_pc2[i] - piece_pc2[j])**2)
    close_tonic_distances.append(dist)

print(f"\nRelative key distances: {np.mean(relative_distances):.2f} ± {np.std(relative_distances):.2f}")
print(f"Close-tonic distances: {np.mean(close_tonic_distances):.2f} ± {np.std(close_tonic_distances):.2f}")

# Fit circle to key centroids
def circle_residuals(params, x, y):
    """Residuals for circle fitting: (x-h)^2 + (y-k)^2 = r^2"""
    h, k, r = params
    return np.sqrt((x - h)**2 + (y - k)**2) - r

h0 = np.mean(key_pc1)
k0 = np.mean(key_pc2)
r0 = np.mean(np.sqrt((key_pc1 - h0)**2 + (key_pc2 - k0)**2))

result = least_squares(circle_residuals, [h0, k0, r0], args=(key_pc1, key_pc2))
h_fit, k_fit, r_fit = result.x

print(f"\nCircle fit:")
print(f"  Center: ({h_fit:.3f}, {k_fit:.3f})")
print(f"  Radius: {r_fit:.3f}")

# Compute coefficient of variation
distances_from_center = np.sqrt((key_pc1 - h_fit)**2 + (key_pc2 - k_fit)**2)
cv = 100 * np.std(distances_from_center) / np.mean(distances_from_center)
print(f"  Coefficient of Variation: {cv:.1f}%")

# Create 3-panel figure
print("\nGenerating 3-panel figure...")
fig = plt.figure(figsize=(21, 7))

# === PANEL A: Raw sequences ===
ax1 = plt.subplot(1, 3, 1)
for piece_id in sorted(set(seq_pieces)):
    mask = np.array([s == piece_id for s in seq_pieces])
    x = seq_pc1[mask]
    y = seq_pc2[mask]

    piece_idx = int(piece_id) if piece_id.isdigit() else 0
    if piece_idx < len(piece_centroids):
        key = piece_centroids[piece_idx]['key']
        color = get_piece_color(key)
    else:
        color = '#888888'

    ax1.scatter(x, y, c=color, alpha=1.0, s=15, edgecolors='none', rasterized=True)

ax1.set_xlabel('PC1', fontweight='bold')
ax1.set_ylabel('PC2', fontweight='bold')
ax1.set_title('A. Individual Sequences\n(2,724 sequences)', fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linewidth=1)
ax1.axhline(y=0, color='k', linewidth=1, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=1, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')

# === PANEL B: Piece centroids with connections ===
ax2 = plt.subplot(1, 3, 2)

# Draw close-tonic connections (dashed purple)
for i, j in close_tonic_pairs:
    x_vals = [piece_pc1[i], piece_pc1[j]]
    y_vals = [piece_pc2[i], piece_pc2[j]]
    ax2.plot(x_vals, y_vals, color='purple', linewidth=2, alpha=0.5,
             linestyle=(0, (3, 3)), zorder=1,
             label='Close-tonic' if (i, j) == close_tonic_pairs[0] else '')

# Draw relative key connections (solid gray)
for i, j in relative_pairs:
    x_vals = [piece_pc1[i], piece_pc1[j]]
    y_vals = [piece_pc2[i], piece_pc2[j]]
    ax2.plot(x_vals, y_vals, color='gray', linewidth=2, alpha=0.7, zorder=2,
             label='Relative' if (i, j) == relative_pairs[0] else '')

# Draw piece centroids
for i in range(len(piece_centroids)):
    key = piece_centroids[i]['key']
    mode = piece_centroids[i]['mode']
    color = get_piece_color(key)

    ax2.scatter(piece_pc1[i], piece_pc2[i], c=color, s=400,
               edgecolors='none', alpha=0.9, zorder=3)

    mode_char = 'M' if mode == 'Major' else 'm'
    label = f"{key}{mode_char}"

    ax2.text(piece_pc1[i], piece_pc2[i], label,
            fontsize=10, fontweight='bold', ha='center', va='center',
            color='white', zorder=4,
            path_effects=[
                patheffects.Stroke(linewidth=2.5, foreground='black'),
                patheffects.Normal()
            ])

ax2.set_xlabel('PC1', fontweight='bold')
ax2.set_ylabel('PC2', fontweight='bold')
ax2.set_title(f'B. Relative Keys & Close-Tonic Pairs\n({len(piece_centroids)} pieces)',
              fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linewidth=1)
ax2.axhline(y=0, color='k', linewidth=1, alpha=0.3)
ax2.axvline(x=0, color='k', linewidth=1, alpha=0.3)
ax2.set_aspect('equal', adjustable='box')
ax2.legend(loc='upper left', framealpha=0.9, fontsize=11)

# === PANEL C: Key centroids with fitted circle ===
ax3 = plt.subplot(1, 3, 3)

# Draw fitted circle
circle = Circle((h_fit, k_fit), r_fit, fill=False,
                edgecolor='black', linewidth=3.5, linestyle='-', zorder=1)
ax3.add_patch(circle)

# Draw key centroids
for i, key_name in enumerate(key_names):
    color = get_piece_color(key_name)

    ax3.scatter(key_pc1[i], key_pc2[i], c=color, s=1000,
               edgecolors='none', alpha=0.9, zorder=2)

    ax3.text(key_pc1[i], key_pc2[i], key_name,
            fontsize=18, fontweight='bold', ha='center', va='center',
            color='white', zorder=3,
            path_effects=[
                patheffects.Stroke(linewidth=4, foreground='black'),
                patheffects.Normal()
            ])

ax3.set_xlabel('PC1', fontweight='bold')
ax3.set_ylabel('PC2', fontweight='bold')
ax3.set_title(f'C. Circle-of-Fifths Fit\n(12 keys, radius = {r_fit:.2f} PC units)',
              fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, linewidth=1)
ax3.axhline(y=0, color='k', linewidth=1, alpha=0.3)
ax3.axvline(x=0, color='k', linewidth=1, alpha=0.3)
ax3.set_aspect('equal', adjustable='box')

# Set consistent axis limits
all_x = np.concatenate([seq_pc1, piece_pc1, key_pc1])
all_y = np.concatenate([seq_pc2, piece_pc2, key_pc2])

ax1.set_xlim(all_x.min() - 1.5, all_x.max() + 1.5)
ax1.set_ylim(all_y.min() - 1.5, all_y.max() + 1.5)

ax2.set_xlim(piece_pc1.min() - 1.0, piece_pc1.max() + 1.0)
ax2.set_ylim(piece_pc2.min() - 1.0, piece_pc2.max() + 1.0)

ax3.set_xlim(h_fit - r_fit - 0.8, h_fit + r_fit + 0.8)
ax3.set_ylim(k_fit - r_fit - 0.8, k_fit + r_fit + 0.8)

plt.tight_layout()

# Save
output_dir = Path(__file__).parent.parent / 'figures'
output_dir.mkdir(exist_ok=True)

output_path = output_dir / 'fig4_combined_convergence.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: {output_path}")

png_path = output_path.with_suffix('.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {png_path}")

plt.close()

print("\n" + "=" * 80)
print("FIGURE 4 COMPLETE!")
print("=" * 80)
print(f"\n3-panel figure shows:")
print(f"  A. Raw sequences (all {len(sequences)} points)")
print(f"  B. Piece centroids with:")
print(f"     - Relative keys (gray, {len(relative_pairs)} pairs, dist = {np.mean(relative_distances):.2f} ± {np.std(relative_distances):.2f})")
print(f"     - Close-tonic pairs (purple, {len(close_tonic_pairs)} pairs, dist = {np.mean(close_tonic_distances):.2f} ± {np.std(close_tonic_distances):.2f})")
print(f"  C. Key centroids with fitted circle")
print(f"     - Radius: {r_fit:.2f} PC units")
print(f"     - CV: {cv:.1f}%")
print("=" * 80)
