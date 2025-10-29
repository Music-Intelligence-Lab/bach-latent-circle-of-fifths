#!/usr/bin/env python3
"""Two-step PCA clustering and circle-fit analysis.

Usage: run as a script or import functions. Saves figures and JSON results under `revised/outputs` by default.

Pipeline summary:
- Load embedding vectors (npy or csv). Optional labels column for keys.
- Step 1: PCA -> reduce to intermediate dims (default 10).
- Compute key centroids (mean per label) or global centroids via clustering.
- Step 2: PCA -> reduce centroids to 2D for visualization.
- Fit circle to 2D centroids using algebraic least squares.
- Compute Coefficient of Variation (CV = std(radius)/mean(radius)) and radial deviation (%).
- Save figures and numerical results.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple, Sequence, Dict

import math
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_embeddings(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load embeddings and optional labels.

    If path ends with .npy, expects either an array of shape (n, d) or a dict-like
    object with keys 'X' and 'labels'. If CSV, expects last column optionally labels.
    Returns (X, labels_or_None)
    """
    path = os.path.abspath(path)
    if path.endswith('.npy'):
        data = np.load(path, allow_pickle=True)
        # allow for dict saved with np.save
        if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():  # a pickled object
            obj = data.item()
            if isinstance(obj, dict) and 'X' in obj:
                X = np.asarray(obj['X'])
                labels = np.asarray(obj.get('labels')) if 'labels' in obj else None
                return X, labels
        # otherwise assume plain array
        return np.asarray(data), None

    # CSV
    if path.endswith('.csv') or path.endswith('.txt'):
        raw = np.loadtxt(path, delimiter=',')
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        # heuristics: if last column is integer small range -> labels
        X = raw[:, :-1]
        last = raw[:, -1]
        # if last column appears integer like, use as labels
        if np.all(np.mod(last, 1) == 0):
            labels = last.astype(int)
            return X, labels
        return raw, None

    raise ValueError(f"Unsupported file extension for {path}")


def compute_key_centroids(X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return centroids and label order preserving first appearance.

    centroids shape = (n_labels, dim)
    Returns (centroids, label_order)
    """
    # preserve order of first appearance of labels
    _, idx = np.unique(labels, return_index=True)
    label_order = labels[np.sort(idx)]
    cents = []
    for u in label_order:
        mask = labels == u
        if np.sum(mask) == 0:
            continue
        cents.append(X[mask].mean(axis=0))
    return np.vstack(cents), label_order


def two_step_pca_and_centroids(X: np.ndarray, labels: Optional[np.ndarray], n_components_first: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, PCA]:
    """Perform first PCA on raw X, compute centroids (if labels), then PCA to 2D for centroids.

    Returns (cents_2d, cents_pca1, label_order, pca1)
    """
    # First PCA
    n_features = X.shape[1]
    k = min(n_components_first, n_features)
    pca1 = PCA(n_components=k)
    X1 = pca1.fit_transform(X)

    # compute centroids in PCA1 space
    if labels is not None:
        cents_pca1, label_order = compute_key_centroids(X1, labels)
    else:
        # fallback: cluster into up to 12 groups by k-means-ish simple split using first dim bins
        bins = min(12, max(3, X1.shape[0] // 10))
        qs = np.percentile(X1[:, 0], np.linspace(0, 100, bins + 1))
        labels_auto = np.digitize(X1[:, 0], qs) - 1
        cents_pca1, label_order = compute_key_centroids(X1, labels_auto)

    # Second PCA to 2D on centroids
    pca2 = PCA(n_components=2)
    cents_2d = pca2.fit_transform(cents_pca1)
    return cents_2d, cents_pca1, label_order, pca1, pca2


def fit_circle_least_squares(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float, float], float, np.ndarray]:
    """Fit circle x^2+y^2 + A x + B y + C = 0 via linear least squares.
    Returns (center=(cx,cy), radius, residuals_per_point)
    """
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    # solve A * p = b for p = [Acoef, Bcoef, Ccoef]
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    Acoef, Bcoef, Ccoef = p
    cx = -Acoef / 2.0
    cy = -Bcoef / 2.0
    rad = math.sqrt(cx * cx + cy * cy - Ccoef)
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    residuals = distances - rad
    return (cx, cy), float(rad), residuals


def angle_to_color(theta: float) -> Tuple[float, float, float]:
    """Map angle in radians to an RGB color on a cyclical colormap.
    Returns an RGB tuple.
    """
    import matplotlib.cm as cm
    norm = (theta % (2 * math.pi)) / (2 * math.pi)
    cmap = cm.get_cmap('hsv')
    return cmap(norm)[:3]


def default_key_names(n: int) -> Sequence[str]:
    # If n==12, return conventional major/minor label pairs by index
    # We'll return labels like 'A', 'A#/Bb', 'B', etc for single label mapping
    names12 = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
    # for major/minor pairs in Step 1 we might want 'A major', 'A minor' etc — but we expect labels per centroid
    if n == 12:
        return names12
    # otherwise generate generic names
    return [f'k{i}' for i in range(n)]


def analyze_and_save(cents_2d: np.ndarray, cents_pca1: np.ndarray, label_order: np.ndarray, outdir: str, pca2: Optional[PCA] = None, label_names: Optional[Dict[int, str]] = None, piece_cents_pca1: Optional[np.ndarray] = None, piece_label_names: Optional[Sequence[str]] = None, marker_size: int = 900, out_suffix: str = '') -> dict:
    ensure_dir(outdir)
    figs_dir = os.path.join(outdir, 'figures')
    ensure_dir(figs_dir)
    # cents_pca1: centroids in PCA1 space (n_centroids, k)
    # prepare first-step figure: project cents_pca1 to 2D for plotting (use first two dims or reduce)
    if cents_pca1.shape[1] >= 2:
        step1_xy = cents_pca1[:, :2]
    else:
        # if only 1 dim available, run a quick PCA to 2D for plotting
        ptemp = PCA(n_components=2)
        step1_xy = ptemp.fit_transform(cents_pca1)

    x1 = step1_xy[:, 0]
    y1 = step1_xy[:, 1]

    x = cents_2d[:, 0]
    y = cents_2d[:, 1]
    # Fit circle first (so cx, cy exist), then compute angles/distances/residuals
    (cx, cy), rad, residuals = fit_circle_least_squares(x, y)
    # compute angular positions of centroids in final 2D
    angles = np.arctan2(y - cy, x - cx)
    # normalized radius for each
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mean_r = float(np.mean(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))
    std_r = float(np.std(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))
    cv = std_r / mean_r if mean_r != 0 else float('inf')
    radial_dev_pct = cv * 100.0

    # fraction within 10% radial deviation
    within_10pct = float(np.mean(np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - mean_r) / mean_r <= 0.10)) if mean_r != 0 else 0.0

    # additional angular deviation metrics: compare centroid angles to equally spaced ideal angles
    n = len(angles)
    ideal = np.linspace(-math.pi, math.pi, n, endpoint=False)
    # sort centroids by angle to compare with ideal order
    order = np.argsort(angles)
    angle_diffs = np.unwrap(angles[order]) - ideal
    angle_rms = float(np.sqrt(np.mean(angle_diffs ** 2)))

    results = {
        'center': {'x': cx, 'y': cy},
        'radius_fit': rad,
        'mean_radius': mean_r,
        'std_radius': std_r,
        'coefficient_of_variation': cv,
        'radial_deviation_percent': radial_dev_pct,
        'fraction_within_10pct': within_10pct,
        'angular_rms_radians': angle_rms,
        'angular_rms_degrees': math.degrees(angle_rms),
        'n_points': int(cents_2d.shape[0]),
    }

    # Save JSON
    json_path = os.path.join(outdir, 'results', 'circle_fit.json')
    ensure_dir(os.path.dirname(json_path))
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save figure: centroids after second PCA (circle fit) with labels
    # allow caller to specify a suffix to force a new filename (useful to make viewers refresh)
    fname = 'circle_fit_analysis' + (f'_{out_suffix}' if out_suffix else '') + '.png'
    fig_path = os.path.join(figs_dir, fname)
    fig, ax = plt.subplots(figsize=(7, 7))
    # Build label string list in order
    labels_str = []
    for i, lab in enumerate(label_order):
        try:
            labels_str.append(str(label_names.get(int(lab), lab)))
        except Exception:
            labels_str.append(str(lab))

    # helper to get root token (robust)
    def root_token(name: str) -> str:
        if name is None:
            return ''
        s = str(name)
        if '_' in s:
            return s.split('_')[0]
        parts = s.split()
        return parts[0] if parts else s

    # Include piece_label_names in the palette so piece roots align with key roots
    piece_label_strings = [str(n) for n in piece_label_names] if piece_label_names is not None else []

    roots = [root_token(n) for n in labels_str]
    piece_roots = [root_token(n) for n in piece_label_strings]

    # Map unique roots (preserve order from keys then pieces) to hues
    unique_roots = []
    for r in roots + piece_roots:
        if r and r not in unique_roots:
            unique_roots.append(r)

    import matplotlib.cm as cm
    cmap = cm.get_cmap('hsv')
    root_to_hue = {r: i / max(1, len(unique_roots)) for i, r in enumerate(unique_roots)}

    def color_for_root(r: str):
        if not r:
            return (0.5, 0.5, 0.5)
        return cmap(root_to_hue.get(r, 0.0))[:3]

    colors = [color_for_root(root_token(n)) for n in labels_str]
    # make markers larger for publication-style figures (increased size)
    ax.scatter(x, y, color=colors, s=marker_size, edgecolor='k', linewidth=1.0, zorder=3)
    circ = plt.Circle((cx, cy), rad, color='0.2', fill=False, lw=2, alpha=0.9)
    ax.add_patch(circ)
    # annotate with human-readable names if provided, else label_order
    n = len(x)
    if label_names is None:
        # try default mapping if numeric sequential keys
        try:
            names = {int(lab): str(lab) for lab in label_order}
        except Exception:
            names = {i: str(i) for i in range(n)}
    else:
        names = label_names
    for i, lab in enumerate(label_order):
        name = names.get(int(lab), str(lab)) if isinstance(names, dict) else str(lab)
        # place label slightly outward from point for readability
        dx = x[i] - cx
        dy = y[i] - cy
        norm = math.hypot(dx, dy)
        ox = x[i] + 0.10 * dx / (norm + 1e-8)
        oy = y[i] + 0.10 * dy / (norm + 1e-8)
        txt = ax.text(ox, oy, name, fontsize=11, ha='center', va='center', weight='bold', color='black', zorder=4)
        txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if pca2 is not None:
        var1 = pca2.explained_variance_ratio_[0] * 100
        var2 = pca2.explained_variance_ratio_[1] * 100
        ax.set_xlabel(f'PC1 ({var1:.1f}% var)')
        ax.set_ylabel(f'PC2 ({var2:.1f}% var)')
    ax.set_title('Circle fit to centroids (step 2)')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)

    # Save figure: centroids after first PCA/clustering step with labels
    step1_fig = os.path.join(figs_dir, 'centroids_after_step1' + (f'_{out_suffix}' if out_suffix else '') + '.png')
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    # For a clearer step-1 plot emulate the published-style: large circular markers with white halo and bold labels.
    # Use root-based coloring so roots and their major/minor variants share a color
    s_cx = x1.mean()
    s_cy = y1.mean()
    # centroid colors for step1 (make sure length matches)
    colors1 = []
    for i in range(len(x1)):
        if i < len(labels_str):
            colors1.append(color_for_root(root_token(labels_str[i])))
        else:
            colors1.append((0.6, 0.6, 0.6))

    # draw white halo by plotting larger white circles underneath and then colored markers on top
    ax1.scatter(x1, y1, s=300, color='white', edgecolor='none', zorder=1)
    ax1.scatter(x1, y1, s=220, color=colors1, edgecolor='k', linewidth=1.0, zorder=2)

    # If piece-level centroids are provided (24 pieces), plot them as slightly smaller outlined circles
    if piece_cents_pca1 is not None and piece_cents_pca1.shape[0] > 0:
        # If piece_cents_pca1 is 2D and matches step1 scale, plot directly
        try:
            px = piece_cents_pca1[:, 0]
            py = piece_cents_pca1[:, 1]
            # white halo and smaller colored markers
            ax1.scatter(px, py, s=220, color='white', edgecolor='none', zorder=0.5)
            # color by root mapping for consistent hue
            p_roots = [root_token(n) for n in piece_label_names]
            p_colors = [color_for_root(r) for r in p_roots]
            ax1.scatter(px, py, s=140, color=p_colors, edgecolor='k', linewidth=0.8, marker='o', zorder=1.5)
            # annotate piece labels if provided
            if piece_label_names is not None:
                for i, nm in enumerate(piece_label_names):
                    short = str(nm)
                    t = ax1.text(px[i], py[i], short, fontsize=9, ha='center', va='center', weight='bold', color='black', zorder=2)
                    t.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground='white')])
            # If the piece_label_names are major/minor pairs like 'C M' / 'C m', draw faint connecting lines
            try:
                # build mapping from root -> indices
                roots = {}
                for i, nm in enumerate(piece_label_names):
                    parts = str(nm).split()
                    if len(parts) >= 2:
                        root = parts[0]
                        roots.setdefault(root, []).append(i)
                for root, idxs in roots.items():
                    if len(idxs) >= 2:
                        xpair = px[idxs]
                        ypair = py[idxs]
                        ax1.plot(xpair, ypair, color='0.7', lw=1.2, alpha=0.6, zorder=0.9)
            except Exception:
                pass
        except Exception:
            # if piece centroids don't match shape expectations, ignore plotting
            pass

    # build readable label names
    if label_names is None:
        names1 = {int(lab): default_key_names(len(label_order))[i % 12] for i, lab in enumerate(label_order)}
    else:
        names1 = label_names

    # annotate with bold single-letter key labels inside circular markers, similar to example
    for i, lab in enumerate(label_order):
        name = names1.get(int(lab), str(lab))
        # attempt to shorten names to single-token like 'A' or 'A\nmajor' if verbose
    short = name.replace(' major', '').replace(' minor', '')
    t = ax1.text(x1[i], y1[i], short, fontsize=12, ha='center', va='center', weight='bold', color='black', zorder=3)
    t.set_path_effects([patheffects.withStroke(linewidth=2.4, foreground='white')])

    # tighten axes to show a balanced circular layout similar to the example
    margin = max(np.ptp(x1), np.ptp(y1)) * 0.15 if (np.ptp(x1) and np.ptp(y1)) else 1.0
    xmin, xmax = x1.min() - margin, x1.max() + margin
    ymin, ymax = y1.min() - margin, y1.max() + margin
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_aspect('equal', 'box')
    # set axis labels with basic PC1/PC2 (we don't have explained var for step1's internal PCA here)
    ax1.set_xlabel('PC1 (step 1)')
    ax1.set_ylabel('PC2 (step 1)')
    ax1.set_title('Centroids after step 1 (PCA1 space)')
    ax1.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(step1_fig, dpi=180)
    plt.close(fig1)

    # Additionally, write a small LaTeX section summarizing results for inclusion in a paper
    try:
        tex_dir = os.path.join(outdir, 'results')
        ensure_dir(tex_dir)
        tex_path = os.path.join(tex_dir, 'REAL_DATA_RESULTS_SECTION.tex')
        with open(tex_path, 'w') as tf:
            tf.write('\\section{Circularity of Key Centroids}\\n')
            tf.write('We fit a circle to the 2D key-centroid positions (after the two-step PCA) and computed the following metrics:\\n')
            # write a small tabular summary
            tf.write('\\begin{table}[ht]\\centering\\small\\begin{tabular}{lrr}\\toprule\\n')
            tf.write('Metric & Value & Notes\\\\\\midrule\\n')
            tf.write(f"Fitted center ($\bar x,\ \bar y$) & ({results['center']['x']:.4f}, {results['center']['y']:.4f}) & \\\\n")
            tf.write(f"Fitted radius & {results['radius_fit']:.4f} & \\\\n")
            tf.write(f"Mean radius & {results['mean_radius']:.4f} & \\\\n")
            tf.write(f"Radius std & {results['std_radius']:.4f} & \\\\n")
            tf.write(f"Coefficient of variation & {results['coefficient_of_variation']:.4f} & (radial dev. {results['radial_deviation_percent']:.2f}\%%)\\\\\\n")
            tf.write(f"Fraction within $\pm 10\%$ & {results['fraction_within_10pct']:.2f} & \\\\n")
            tf.write(f"Angular RMS (deg) & {results['angular_rms_degrees']:.1f} & \\\\n")
            tf.write('\\bottomrule\\end{tabular}\\caption{Summary of circle-fit metrics for the 12 key-centroids.}\\end{table}\\n')
            tf.write('These values quantify how closely the 12 key-centroids lie on an ideal circle: lower CV and angular RMS indicate closer agreement with circular symmetry.\\n')
    except Exception:
        # do not fail analysis if writing tex fails
        pass

    # Residuals figure
    res_fig = os.path.join(figs_dir, 'key_centroids_simple' + (f'_{out_suffix}' if out_suffix else '') + '.png')
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.bar(range(len(residuals)), np.abs(residuals))
    ax2.set_ylabel('Absolute residual (distance from fitted radius)')
    ax2.set_xlabel('centroid index')
    ax2.set_title('Residuals from circle fit')
    plt.tight_layout()
    plt.savefig(res_fig, dpi=150)
    plt.close(fig2)

    return results


def main():
    parser = argparse.ArgumentParser(description='Two-step PCA + circle fit analysis')
    parser.add_argument('input', help='Path to embeddings (.npy or .csv)')
    parser.add_argument('--out', default='revised/outputs', help='Output directory (default: revised/outputs)')
    parser.add_argument('--first-dim', type=int, default=10, help='First PCA output dims (default 10)')
    args = parser.parse_args()

    X, labels = load_embeddings(args.input)
    cents_2d, cents_pca1, label_order, pca1, pca2 = two_step_pca_and_centroids(X, labels, n_components_first=args.first_dim)
    results = analyze_and_save(cents_2d, cents_pca1, label_order, args.out, pca2=pca2, label_names=None)
    print('Results saved to', os.path.abspath(args.out))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
