import os                          # Work with file system paths
import pandas as pd                # Data manipulation and DataFrame handling
import numpy as np                 # Numerical operations
import joblib                      # Saving and loading preprocessing objects
import tensorflow as tf            # Deep learning framework

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def load_csv_logs(path):
    """
    Load Windows event logs from CSV files and normalize column structure.

    Steps:
    - Read all CSV files from the given directory
    - Remove unnecessary columns
    - Normalize column names
    - Rename columns to a unified schema
    - Ensure required columns exist
    """
    # Collect all CSV file paths
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    dfs = []

    for f in files:
        # Read CSV file
        df = pd.read_csv(f, low_memory=False)

        # Remove automatically generated columns (e.g., "Unnamed: 0")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Normalize column names (lowercase and trimmed)
        df.columns = [c.strip().lower() for c in df.columns]

        # Map alternative column names to a standard format
        rename = {
            'eventid': 'event_id',
            'name': 'provider',
            'systemtime': 'time_created',
            'utctime': 'time_created',
            'creationutctime': 'time_created'
        }
        df = df.rename(columns=rename)

        # Ensure required columns exist
        for col in ['event_id', 'level', 'provider', 'time_created']:
            if col not in df.columns:
                df[col] = None

        # Keep only relevant features
        dfs.append(df[['event_id', 'level', 'provider', 'time_created']])

    # Merge all CSV files into one DataFrame
    return (
        pd.concat(dfs, ignore_index=True)
        if dfs else
        pd.DataFrame(columns=['event_id', 'level', 'provider', 'time_created'])
    )


def preprocess(df):
    """
    Convert raw log data into numerical features suitable for VAE training.

    Steps:
    - Remove rows with missing values
    - Convert all values to strings
    - Apply One-Hot Encoding to categorical features
    - Normalize features to range [0, 1]
    - Save encoder and scaler for inference
    """
    # Remove rows with missing values and convert to string
    df = df.dropna().astype(str)

    # Create OneHotEncoder for categorical features
    encoder = ColumnTransformer(
        transformers=[
            (
                'onehot',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ),
                ['event_id', 'level', 'provider']
            )
        ],
        remainder='drop'
    )

    # Fit encoder and transform data
    X = encoder.fit_transform(df)

    # Save fitted encoder
    joblib.dump(encoder, 'vae_encoder.pkl')

    # Normalize data to [0, 1] range (important for sigmoid output)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler, 'vae_scaler.pkl')

    return X_scaled


def sampling(z_mean, z_log_var):
    """
    Reparameterization trick.

    Allows backpropagation through a stochastic latent space by:
    z = mean + std * epsilon
    """
    # Sample random noise from standard normal distribution
    eps = tf.random.normal(tf.shape(z_mean))

    # Compute latent vector
    return z_mean + tf.exp(0.5 * z_log_var) * eps


class VAEModel(Model):
    """
    Variational Autoencoder (VAE) implementation.

    Encoder:
    - Compresses input into latent distribution (mean and log variance)

    Decoder:
    - Reconstructs input from sampled latent vector

    Loss:
    - Reconstruction loss (MSE)
    - KL-divergence regularization
    """
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()

        # Encoder layers
        self.e1 = Dense(64, activation='relu')
        self.e2 = Dense(32, activation='relu')

        # Latent space parameters
        self.z_mean = Dense(latent_dim)
        self.z_log_var = Dense(latent_dim)

        # Decoder layers
        self.d1 = Dense(32, activation='relu')
        self.d2 = Dense(64, activation='relu')

        # Output layer reconstructs original input
        self.out = Dense(input_dim, activation='sigmoid')

    def call(self, inputs):
        # ----- Encoder -----
        x = self.e1(inputs)
        x = self.e2(x)

        # Compute latent distribution parameters
        zm = self.z_mean(x)
        zv = self.z_log_var(x)

        # Sample latent vector using reparameterization trick
        z = sampling(zm, zv)

        # ----- Decoder -----
        x = self.d1(z)
        x = self.d2(x)

        # Reconstructed input
        recon = self.out(x)

        # ----- Loss calculation -----
        # Reconstruction loss (Mean Squared Error)
        recon_loss = tf.reduce_mean(tf.square(inputs - recon))

        # KL divergence loss (regularization of latent space)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + zv - tf.square(zm) - tf.exp(zv)
        )

        # Add total loss to model
        self.add_loss(recon_loss + kl_loss)

        return recon


def main():
    """
    Main training pipeline for VAE-based anomaly detection.
    """
    # Load normal (non-anomalous) logs
    df = load_csv_logs("../evtx_logs/normal_csv/")
    print(f"[*] Loaded normal logs: {len(df)}")

    # Preprocess logs into numerical features
    X = preprocess(df)
    print(f"[*] Preprocessed shape: {X.shape}")

    # Split data into training and validation sets
    X_train, X_val = train_test_split(
        X,
        test_size=0.1,
        random_state=42
    )

    # Initialize VAE model
    vae = VAEModel(
        input_dim=X.shape[1],
        latent_dim=16
    )

    # Compile model (loss is added internally)
    vae.compile(optimizer='adam')

    # Train VAE
    vae.fit(
        X_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, None),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    # Save trained model weights
    vae.save_weights("vae_weights.weights.h5")
    print("[+] VAE weights saved to vae_weights.weights.h5")

    # ----- Threshold calculation -----
    # Reconstruct validation data
    recon_val = vae.predict(X_val)

    # Compute reconstruction error for each sample
    errors_val = np.mean((X_val - recon_val) ** 2, axis=1)

    # Set anomaly threshold as 95th percentile
    threshold = np.percentile(errors_val, 95)

    # Save threshold for inference stage
    joblib.dump(threshold, "threshold_95.pkl")

    print(f"[*] Threshold (95%): {threshold:.6f} saved to threshold_95.pkl")


if __name__ == "__main__":
    main()
