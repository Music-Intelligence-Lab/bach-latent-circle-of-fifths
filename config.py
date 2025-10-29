class Config:
    # MIDI processing settings
    FS = 4  # Sampling frequency (4 samples per quarter note)
    SEQUENCE_LENGTH = 16  # Length of each sequence chunk

    # Note range (standard piano)
    LOWEST_NOTE = 21  # A0
    HIGHEST_NOTE = 108  # C8
    NOTE_RANGE = HIGHEST_NOTE - LOWEST_NOTE + 1  # 88 piano keys

    # Paths
    MIDI_FOLDER = "normalized"  # Input folder with normalized MIDI files
    OUTPUT_DIR = "processed_data"  # Output folder for NPY files
    RECONSTRUCTED_DIR = "reconstructed_midi"  # Output folder for reconstructed MIDI files
