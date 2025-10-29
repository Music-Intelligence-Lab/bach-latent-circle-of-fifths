
import os
import numpy as np
import pretty_midi
from config import Config

class MIDIReconstructor:
    def __init__(self, fs=Config.FS):
        self.fs = fs

    def reconstruct_from_sequences(self, sequences, indices, output_dir):
        """
        Reconstruct MIDI files from sequences.

        Args:
            sequences: numpy array of shape (num_sequences, sequence_length, note_range)
            indices: list of starting sequence indices for each piece
            output_dir: directory to save reconstructed MIDI files
        """
        os.makedirs(output_dir, exist_ok=True)

        midi_files = sorted([f for f in os.listdir(Config.MIDI_FOLDER) if f.endswith('.mid')],
                           key=lambda x: int(x.split('_')[0]))

        for i in range(len(indices) - 1):
            start_seq = indices[i]
            end_seq = indices[i + 1]

            # Extract sequences for this piece
            piece_sequences = sequences[start_seq:end_seq]

            # Reshape to piano roll: (num_sequences * sequence_length, note_range)
            piece_roll = piece_sequences.reshape(-1, piece_sequences.shape[-1])

            print(f"Reconstructing piece {midi_files[i]}: {end_seq - start_seq} sequences -> shape {piece_roll.shape}")

            midi = self.piano_roll_to_midi(piece_roll)
            out_path = os.path.join(output_dir, midi_files[i])
            midi.write(out_path)
            print(f"Saved MIDI -> {out_path}")

    def piano_roll_to_midi(self, roll):
        """
        Convert a piano roll to MIDI using vectorized operations.

        Args:
            roll: numpy array of shape (timesteps, note_range) with binary values

        Returns:
            pretty_midi.PrettyMIDI object
        """
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        # Optimized: process all pitches using vectorized diff
        # Add padding to detect note boundaries at edges
        padded_roll = np.pad(roll, ((1, 1), (0, 0)), mode='constant', constant_values=0)

        # Find note transitions (0->1 for note_on, 1->0 for note_off)
        diff = np.diff(padded_roll.astype(int), axis=0)

        # Process each pitch
        for pitch in range(roll.shape[1]):
            # Find note_on and note_off events
            note_ons = np.where(diff[:, pitch] == 1)[0]
            note_offs = np.where(diff[:, pitch] == -1)[0]

            # Create notes from paired on/off events
            for start_idx, end_idx in zip(note_ons, note_offs):
                start = start_idx / self.fs
                end = end_idx / self.fs
                note = pretty_midi.Note(velocity=100, pitch=pitch + 21, start=start, end=end)
                instrument.notes.append(note)

        midi.instruments.append(instrument)
        return midi
