import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from config import Config
from tensorflow.keras.regularizers import l2

class SaveEverythingCallback(Callback):
    def __init__(self, autoencoder, sequences, indices):
        self.autoencoder = autoencoder
        self.sequences = sequences
        self.indices = indices
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        """Save model weights, loss plots, reconstructions, and latent space plots at every epoch."""
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])

        # Save model weights
        weight_path = os.path.join(Config.WEIGHTS_DIR, f"epoch{epoch+1}_loss{logs['val_loss']:.4f}.weights.h5")
        self.model.save_weights(weight_path)
        print(f"✅ Saved weights at {weight_path}")

        # ✅ FIX: Save loss plot at every epoch
        loss_plot_path = os.path.join(Config.PLOTS_DIR, f'loss_plot_epoch_{epoch+1}.png')
        plot_loss(self.losses, self.val_losses, epoch + 1)  # ✅ Pass `epoch`



        # Generate and save reconstructed MIDI files
        reconstructor = MIDIReconstructor()
        epoch_recon_dir = os.path.join(Config.RECONSTRUCTED_DIR, f"epoch_{epoch+1}")
        os.makedirs(epoch_recon_dir, exist_ok=True)

        batch_size = 512
        reconstructed = np.vstack([
            (self.model.predict(self.sequences[i: i + batch_size]) > 0.5).astype(np.float32)
            for i in range(0, len(self.sequences), batch_size)
        ])

        reconstructor.reconstruct_from_sequences(reconstructed, self.indices, epoch_recon_dir)

        # Generate latent space plots
        latent_vectors = self.autoencoder.get_layer("encoder").predict(self.sequences)
        process_latent_space(latent_vectors, self.indices, epoch + 1, logs['val_loss'])
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)

        # Append log entry
        with open(Config.LOG_FILE, "a") as log_file:
            log_file.write(f"Epoch {epoch+1}: Loss = {logs['loss']:.4f}, Val Loss = {logs['val_loss']:.4f}\n")





class AutoEncoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.encoder_layers = self.determine_layer_sizes()
        self.encoder, self.decoder, self.autoencoder = self.build_autoencoder()

    def determine_layer_sizes(self):
        """Determine layer sizes based on Config settings."""
        if Config.CUSTOM_ENCODER_LAYERS:
            return Config.CUSTOM_ENCODER_LAYERS  # Use manually defined layers

        # Automatic mode: Reduce with powers of 2
        layer_sizes = []
        current_size = 2 ** (self.input_shape - 1).bit_length() // 2 if Config.FIRST_LAYER_POWER_OF_2 else self.input_shape
        
        while current_size > Config.LATENT_DIM * 2:  # Avoid abrupt compression
            layer_sizes.append(current_size)
            current_size //= 2  # Reduce gradually

        return layer_sizes

    def build_autoencoder(self):
        """Builds the autoencoder model dynamically."""
        encoder_input = Input(shape=(self.input_shape,))
        x = encoder_input

        # Encoder
        for layer_size in self.encoder_layers:
            x = Dense(layer_size, activation="swish")(x)
            x = BatchNormalization()(x)
            x = Dropout(Config.DROPOUT_RATE)(x)

        latent = Dense(Config.LATENT_DIM, activation="linear", name="latent_space")(x)

        # Decoder (Reverse of encoder)
        decoder_input = Input(shape=(Config.LATENT_DIM,))
        x = decoder_input

        for layer_size in reversed(self.encoder_layers):
            x = Dense(layer_size, activation="swish")(x)
            x = BatchNormalization()(x)

        decoder_output = Dense(self.input_shape, activation="sigmoid")(x)

        encoder = Model(encoder_input, latent, name="encoder")
        decoder = Model(decoder_input, decoder_output, name="decoder")

        autoencoder = Model(encoder_input, decoder(encoder(encoder_input)), name="autoencoder")
        autoencoder.compile(loss="binary_crossentropy", optimizer=AdamW(learning_rate=Config.LEARNING_RATE,weight_decay=1e-4))

        return encoder, decoder, autoencoder




    def train_autoencoder(self, sequences, indices):
        """Trains the autoencoder while saving all outputs at every epoch."""
        
        # Define new callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(Config.WEIGHTS_DIR, 'best_model.keras'), 
            save_best_only=True, 
            monitor='val_loss', 
            mode='min'
        )

        # Existing callback
        save_callback = SaveEverythingCallback(self.autoencoder, sequences, indices)

        # Include all callbacks in the fit method
        self.autoencoder.fit(
            sequences,
            sequences,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            validation_split=Config.VALIDATION_SPLIT,
            verbose=1,
            callbacks=[save_callback, reduce_lr, early_stop, model_checkpoint]
        )

