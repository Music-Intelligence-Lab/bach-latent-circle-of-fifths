import os
import numpy as np
import pretty_midi
from midi_reconstructor import MIDIReconstructor
from config import Config

def count_notes_in_midi(midi_path):
    """Count the total number of notes in a MIDI file."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        total_notes = 0
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                total_notes += len(instrument.notes)
        return total_notes
    except Exception as e:
        print(f"Error reading {midi_path}: {e}")
        return 0

def validate_reconstruction(original_dir, reconstructed_dir):
    """
    Validate that reconstructed MIDI files have similar number of notes
    as the original files.
    """
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    midi_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.mid')])

    results = []
    total_original = 0
    total_reconstructed = 0

    print(f"\n{'File':<25} {'Original':<12} {'Reconstructed':<12} {'Diff':<8} {'Status'}")
    print("-" * 70)

    for midi_file in midi_files:
        original_path = os.path.join(original_dir, midi_file)
        reconstructed_path = os.path.join(reconstructed_dir, midi_file)

        if not os.path.exists(reconstructed_path):
            print(f"{midi_file:<25} {'N/A':<12} {'MISSING':<12} {'N/A':<8} FAIL")
            continue

        original_notes = count_notes_in_midi(original_path)
        reconstructed_notes = count_notes_in_midi(reconstructed_path)

        diff = reconstructed_notes - original_notes
        diff_pct = (diff / original_notes * 100) if original_notes > 0 else 0

        # Status: OK if within 5% difference
        status = "OK" if abs(diff_pct) < 5 else "CHECK"

        print(f"{midi_file:<25} {original_notes:<12} {reconstructed_notes:<12} {diff:<8} {status}")

        total_original += original_notes
        total_reconstructed += reconstructed_notes

        results.append({
            'file': midi_file,
            'original': original_notes,
            'reconstructed': reconstructed_notes,
            'diff': diff
        })

    print("-" * 70)
    print(f"{'TOTAL':<25} {total_original:<12} {total_reconstructed:<12} {total_reconstructed - total_original:<8}")

    overall_diff_pct = ((total_reconstructed - total_original) / total_original * 100) if total_original > 0 else 0
    print(f"\nOverall difference: {overall_diff_pct:.2f}%")

    print("=" * 60)

    return results

def main():
    """
    Reconstruct MIDI files from NPY data and validate them.
    """
    print("=" * 60)
    print("NPY TO MIDI RECONSTRUCTOR")
    print("=" * 60)

    # Load the processed data
    sequences_path = os.path.join(Config.OUTPUT_DIR, "all_sequences.npy")
    indices_path = os.path.join(Config.OUTPUT_DIR, "start_indices.txt")

    print(f"\nLoading sequences from: {sequences_path}")
    all_sequences = np.load(sequences_path)

    print(f"Loading indices from: {indices_path}")
    indices = np.loadtxt(indices_path, dtype=int).tolist()

    # Add final index for the last piece
    indices.append(len(all_sequences))

    print(f"\nLoaded data shape: {all_sequences.shape}")
    print(f"Total sequences: {all_sequences.shape[0]}")
    print(f"Number of pieces to reconstruct: {len(indices) - 1}")

    # Initialize reconstructor
    reconstructor = MIDIReconstructor(fs=Config.FS)

    print(f"\nReconstructing MIDI files to: {Config.RECONSTRUCTED_DIR}")
    print("-" * 60)

    # Reconstruct
    reconstructor.reconstruct_from_sequences(
        all_sequences,
        indices,
        Config.RECONSTRUCTED_DIR
    )

    print("\n" + "=" * 60)
    print("RECONSTRUCTION COMPLETE!")
    print("=" * 60)

    # Validate the reconstruction
    validation_results = validate_reconstruction(
        Config.MIDI_FOLDER,
        Config.RECONSTRUCTED_DIR
    )

    print(f"\nReconstructed MIDI files saved to: {Config.RECONSTRUCTED_DIR}")

if __name__ == "__main__":
    main()
