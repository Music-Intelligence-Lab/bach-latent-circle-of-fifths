import os
import numpy as np
import pretty_midi
import re
from config import Config

class PianoRollProcessor:
    def __init__(self, lowest_note=21, highest_note=108, fs=Config.FS):
        self.lowest_note = lowest_note
        self.highest_note = highest_note
        self.note_range = highest_note - lowest_note + 1
        self.fs = fs

    def midi_to_piano_roll(self, midi_path):
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        duration = midi_data.get_end_time()
        total_steps = int(duration * self.fs)
        piano_roll = np.zeros((total_steps, self.note_range))

        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    start_idx = int(note.start * self.fs)
                    end_idx = int(note.end * self.fs)
                    note_idx = note.pitch - self.lowest_note

                    # Ensure minimum duration of 1 timestep for all notes
                    if end_idx <= start_idx:
                        end_idx = start_idx + 1

                    if 0 <= note_idx < self.note_range and end_idx <= total_steps:
                        piano_roll[start_idx:end_idx, note_idx] = 1

        print("Piano roll shape (original):", piano_roll.shape)
        return piano_roll

    def prepare_sequences(self, piano_roll):
        sequence_length = Config.SEQUENCE_LENGTH
        num_timesteps = len(piano_roll)
        padded_length = ((num_timesteps + sequence_length - 1) // sequence_length) * sequence_length

        padded_roll = np.zeros((padded_length, self.note_range))
        padded_roll[:num_timesteps] = piano_roll

        print(f"Padded piano roll shape: {padded_roll.shape}")

        # Optimized: use reshape instead of loop
        num_sequences = padded_length // sequence_length
        sequences = padded_roll.reshape(num_sequences, sequence_length, self.note_range)

        print(f"Sequences shape: {sequences.shape}")
        return sequences

    def process_all_midis(self):
        def extract_leading_number(filename):
            base = os.path.basename(filename)
            match = re.match(r"(\d+)", base)
            return int(match.group(0)) if match else float("inf")

        midi_files = sorted(
            [f for f in os.listdir(Config.MIDI_FOLDER) if f.endswith('.mid')],
            key=extract_leading_number
        )

        all_sequences = []
        indices = []
        current_idx = 0

        for file in midi_files:
            path = os.path.join(Config.MIDI_FOLDER, file)
            roll = self.midi_to_piano_roll(path)
            sequences = self.prepare_sequences(roll)
            all_sequences.append(sequences)
            indices.append(current_idx)
            current_idx += len(sequences)
            print(f"Processed {file} -> {sequences.shape[0]} sequences")

        all_sequences = np.concatenate(all_sequences, axis=0)
        print(f"Concatenated all sequences -> shape: {all_sequences.shape}")

        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        indices_path = os.path.join(Config.OUTPUT_DIR, "start_indices.txt")
        np.savetxt(indices_path, indices, fmt='%d')

        # Save sorted section labels for proper alignment during plotting
        section_labels_path = os.path.join(Config.OUTPUT_DIR, "section_labels.txt")
        with open(section_labels_path, 'w') as f:
            for name in midi_files:
                f.write(f"{name.replace('.mid', '')}\n")

        return all_sequences, indices
