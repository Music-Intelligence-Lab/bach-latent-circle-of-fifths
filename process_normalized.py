import os
import numpy as np
from midi_processor import PianoRollProcessor
from config import Config

def main():
    """
    Process all normalized MIDI files in the normalized folder
    and save them as NPY files with proper indexing.
    """
    print("=" * 60)
    print("MIDI TO NPY PROCESSOR")
    print("=" * 60)

    # Initialize processor
    processor = PianoRollProcessor(
        lowest_note=Config.LOWEST_NOTE,
        highest_note=Config.HIGHEST_NOTE,
        fs=Config.FS
    )

    # Process all MIDI files
    print(f"\nProcessing MIDI files from: {Config.MIDI_FOLDER}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Sampling frequency: {Config.FS} Hz")
    print(f"Sequence length: {Config.SEQUENCE_LENGTH}")
    print(f"Note range: {Config.NOTE_RANGE} (MIDI {Config.LOWEST_NOTE}-{Config.HIGHEST_NOTE})")
    print("\n" + "-" * 60)

    all_sequences, indices = processor.process_all_midis()

    # Save the concatenated sequences
    output_file = os.path.join(Config.OUTPUT_DIR, "all_sequences.npy")
    np.save(output_file, all_sequences)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput file: {output_file}")
    print(f"Final shape: {all_sequences.shape}")
    print(f"Total sequences: {all_sequences.shape[0]}")
    print(f"Sequence length: {all_sequences.shape[1]}")
    print(f"Note range: {all_sequences.shape[2]}")
    print(f"\nIndex file: {os.path.join(Config.OUTPUT_DIR, 'start_indices.txt')}")
    print(f"Label file: {os.path.join(Config.OUTPUT_DIR, 'section_labels.txt')}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
