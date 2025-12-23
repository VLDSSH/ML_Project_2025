import os
import pandas as pd
import xmltodict
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


def load_csv_logs(path):
    """
    Load and preprocess Windows event logs stored in CSV format.

    The function:
    - Reads all CSV files from the specified directory
    - Removes empty or unnecessary columns
    - Normalizes column names
    - Renames columns to a unified schema
    - Ensures required columns exist

    :param path: Path to the directory containing CSV log files
    :return: Pandas DataFrame with normalized log data
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    dataframes = []

    for file in files:
        df = pd.read_csv(file, low_memory=False)

        # Remove automatically generated or empty columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(axis=1, how='all')

        # Convert column names to lowercase for consistency
        df.columns = [str(col).strip().lower() for col in df.columns]

        # Map different column naming conventions to a unified format
        rename_map = {
            'eventid': 'event_id',
            'level': 'level',
            'provider': 'provider',
            'name': 'provider',  # sometimes "name" is used instead of "provider"
            'systemtime': 'time_created',
            'utctime': 'time_created',
            'creationutctime': 'time_created'
        }
        df = df.rename(columns=rename_map)

        # Ensure that all required columns exist
        for col in ['event_id', 'level', 'provider', 'time_created']:
            if col not in df.columns:
                df[col] = None

        # Keep only relevant columns
        df = df[['event_id', 'level', 'provider', 'time_created']]
        dataframes.append(df)

    # Combine all CSV files into a single DataFrame
    return pd.concat(dataframes, ignore_index=True)


def load_evtx_logs(path):
    """
    Load and parse Windows EVTX log files.

    The function:
    - Iterates over EVTX files
    - Parses XML event records
    - Extracts event ID, level, provider, and timestamp

    :param path: Path to the directory containing EVTX files
    :return: Pandas DataFrame with extracted log events
    """
    from Evtx.Evtx import Evtx

    events = []

    for file in os.listdir(path):
        if not file.endswith('.evtx'):
            continue

        with Evtx(os.path.join(path, file)) as log:
            for record in log.records():
                try:
                    xml_dict = xmltodict.parse(record.xml())
                    event = xml_dict.get('Event', {})
                    system = event.get('System', {})

                    event_id = system.get('EventID', None)
                    level = system.get('Level', None)
                    provider = system.get('Provider', {}).get('@Name', None)
                    time_created = system.get('TimeCreated', {}).get('@SystemTime', None)

                    events.append({
                        'event_id': str(event_id),
                        'level': str(level),
                        'provider': provider,
                        'time_created': time_created
                    })
                except Exception:
                    # Skip malformed or unreadable records
                    continue

    return pd.DataFrame(events)


def preprocess(df):
    """
    Preprocess log data for machine learning.

    Steps:
    - Remove rows with missing values
    - Convert all values to strings
    - Apply One-Hot Encoding to categorical features
    - Normalize features using StandardScaler
    - Save encoder and scaler for future inference

    :param df: Input DataFrame with log records
    :return: Scaled numerical feature matrix
    """
    # Remove rows containing missing values
    df = df.dropna()

    # Convert all values to string type
    df = df.astype(str)

    # Define categorical features
    categorical = ['event_id', 'level', 'provider']

    # Create a transformer that applies OneHotEncoding
    ct = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical)
        ],
        remainder='drop'
    )

    # Fit and transform the data
    X = ct.fit_transform(df)

    # Save encoder for later use (e.g., anomaly detection)
    joblib.dump(ct, 'encoder.pkl')

    # Normalize features to zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)

    # Save scaler for consistent preprocessing during inference
    joblib.dump(scaler, 'scaler.pkl')

    return X_scaled


def build_autoencoder(input_dim):
    """
    Build a fully connected autoencoder model.

    Architecture:
    - Encoder compresses input data into a low-dimensional representation
    - Decoder reconstructs the original input from the encoded representation

    :param input_dim: Number of input features
    :return: Compiled Keras autoencoder model
    """
    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Encoder: dimensionality reduction
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)

    # Decoder: reconstruction of original input
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    return Model(inputs=input_layer, outputs=output_layer)


def main():
    """
    Main pipeline:
    - Load CSV and EVTX logs
    - Preprocess data
    - Train an autoencoder on normal behavior
    - Save the trained model
    """
    print("[*] Loading logs...")

    df_csv = load_csv_logs('evtx_logs/normal/')
    df_evtx = load_evtx_logs('evtx_logs/normal/')
    df = pd.concat([df_csv, df_evtx], ignore_index=True)

    print(f"[+] Total logs loaded: {len(df)}")

    print("[*] Preprocessing data...")
    X = preprocess(df)

    print("[*] Training autoencoder model...")
    model = build_autoencoder(X.shape[1])
    model.compile(optimizer='adam', loss='mse')

    model.fit(
        X, X,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )

    model.save('model.h5')
    print("[+] Model trained and saved successfully!")


if __name__ == '__main__':
    main()
