import os                          # File system operations
import pandas as pd                # Data manipulation
import numpy as np                 # Numerical computations
import joblib                      # Load/save models and preprocessors
import xmltodict                   # XML → dict conversion
import seaborn as sns              # Visualization (confusion matrix)
import matplotlib.pyplot as plt    # Plotting
from sklearn.metrics import confusion_matrix, roc_curve, auc

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from Evtx.Evtx import Evtx          # Windows EVTX log reader


# ============================================================
# 1. LOG LOADERS
# ============================================================

def load_csv_logs(path):
    """
    Load Windows event logs stored in CSV files.

    Steps:
    - Iterate over all CSV files in the directory
    - Remove unused and duplicated columns
    - Normalize column names
    - Map columns to a unified schema
    - Return a single DataFrame
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    dfs = []

    for f in files:
        # Read CSV file
        df = pd.read_csv(f, low_memory=False)

        # Remove auto-generated columns (e.g., 'Unnamed: 0')
        # and columns that contain only NaN values
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')].dropna(axis=1, how='all')

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Map alternative column names to standard names
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
            if col not in df:
                df[col] = None

        # Remove duplicated columns inside a single DataFrame
        df = df.loc[:, ~df.columns.duplicated()]

        # Keep only relevant features
        dfs.append(df[['event_id', 'level', 'provider', 'time_created']])

    # Combine all CSV files into one DataFrame
    return (
        pd.concat(dfs, ignore_index=True)
        if dfs else
        pd.DataFrame(columns=['event_id', 'level', 'provider', 'time_created'])
    )


def load_evtx_logs(path):
    """
    Load Windows EVTX logs and extract system-level event fields.
    """
    events = []

    # Walk through directory recursively
    for root, _, files in os.walk(path):
        for f in files:
            if not f.endswith('.evtx'):
                continue

            # Open EVTX file
            with Evtx(os.path.join(root, f)) as log:
                for r in log.records():
                    try:
                        # Parse XML and extract System section
                        x = xmltodict.parse(r.xml())['Event']['System']

                        # Append extracted fields
                        events.append({
                            'event_id': str(x.get('EventID')),
                            'level': str(x.get('Level')),
                            'provider': x.get('Provider', {}).get('@Name'),
                            'time_created': x.get('TimeCreated', {}).get('@SystemTime')
                        })
                    except Exception:
                        # Skip corrupted or malformed records
                        pass

    df = pd.DataFrame(events)

    # Remove duplicated columns if DataFrame is not empty
    if not df.empty:
        df = df.loc[:, ~df.columns.duplicated()]

    return df


# ============================================================
# 2. PREPROCESSING
# ============================================================

def preprocess(df, encoder, scaler):
    """
    Apply preprocessing pipeline used during training.

    Steps:
    - Remove rows with missing values
    - Convert all values to strings
    - Apply trained OneHotEncoder
    - Apply trained scaler
    """
    # Drop rows with missing values and convert to string
    dfc = df.dropna().astype(str)

    # Transform categorical features using trained encoder
    X = encoder.transform(dfc)

    # Scale features (encoder output is already dense)
    Xs = scaler.transform(X)

    return Xs, dfc


# ============================================================
# 3. VAE ARCHITECTURE
# ============================================================

def sampling(z_mean, z_log_var):
    """
    Reparameterization trick.

    Allows gradient flow through stochastic latent variables.
    """
    eps = tf.random.normal(tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * eps


class VAEModel(Model):
    """
    Variational Autoencoder (VAE) model.

    Encoder:
    - Maps input to latent mean and variance

    Decoder:
    - Reconstructs input from sampled latent vector
    """
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()

        # Encoder layers
        self.e1 = Dense(64, activation='relu')
        self.e2 = Dense(32, activation='relu')

        # Latent distribution parameters
        self.z_mean = Dense(latent_dim)
        self.z_log_var = Dense(latent_dim)

        # Decoder layers
        self.d1 = Dense(32, activation='relu')
        self.d2 = Dense(64, activation='relu')

        # Output layer
        self.out = Dense(input_dim, activation='sigmoid')

    def call(self, inputs):
        # ----- Encoder -----
        x = self.e1(inputs)
        x = self.e2(x)

        # Latent space parameters
        zm = self.z_mean(x)
        zv = self.z_log_var(x)

        # Sample latent vector
        z = sampling(zm, zv)

        # ----- Decoder -----
        x = self.d1(z)
        x = self.d2(x)

        # Reconstruction
        recon = self.out(x)

        return recon


# ============================================================
# 4. ANOMALY DETECTION AND REPORTING
# ============================================================

def detect_anomalies(X, model, threshold):
    """
    Detect anomalies using reconstruction error.
    """
    # Reconstruct inputs
    recon = model.predict(X)

    # Compute mean squared reconstruction error
    errors = np.mean((X - recon) ** 2, axis=1)

    # Mark samples above threshold as anomalies
    return errors, errors > threshold


def assign_score(e, thr):
    """
    Convert reconstruction error into a risk score (0–100).
    """
    if e <= thr:
        return 0
    if e >= 0.096:
        return 100

    # Linear scaling between threshold and upper bound
    return int(((e - thr) / (0.096 - thr)) * 100)


def generate_report(df, errors, anomalies, threshold):
    """
    Generate HTML report with metrics, plots, and detailed results.
    """
    total = len(df)
    num_anom = anomalies.sum()
    num_norm = total - num_anom

    # Append detection results
    df['reconstruction_error'] = errors
    df['is_anomaly'] = anomalies
    df['score'] = [assign_score(e, threshold) for e in errors]

    # Ground truth labels (if available)
    y_true = (
        df['true_label'].astype(int)
        if 'true_label' in df
        else anomalies.astype(int)
    )

    # ----- Confusion Matrix -----
    cm = confusion_matrix(y_true, anomalies)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ----- ROC Curve -----
    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    # ----- HTML Report -----
    html = f"""
<html>
<head>
<meta charset="utf-8">
<title>VAE Anomaly Report</title>

<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

<style>
body {{ font-family: Arial, sans-serif; margin: 20px }}
h1, h2 {{ color: #2F4F4F }}
</style>
</head>

<body>
<h1>VAE Anomaly Detection Report</h1>

<p>
<b>Total:</b> {total} |
<b>Normal:</b> {num_norm} |
<b>Anomalous:</b> {num_anom} |
<b>Threshold:</b> {threshold:.6f}
</p>

<h2>Score distribution</h2>
{df.groupby('score').size().reset_index(name='count').to_html(index=False)}

<h2>Confusion Matrix</h2>
<img src="confusion_matrix.png" width="500">

<h2>ROC Curve</h2>
<img src="roc_curve.png" width="500">

<h2>Detailed results</h2>
{df.to_html(index=False)}

<script>
$(document).ready(function() {{
    $('table').DataTable();
}});
</script>

</body>
</html>
"""
    with open("vae_anomaly_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("[+] Report saved to vae_anomaly_report.html")


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main():
    # Load preprocessing objects and threshold
    encoder = joblib.load("vae_encoder.pkl")
    scaler = joblib.load("vae_scaler.pkl")
    threshold = joblib.load("threshold_95.pkl")

    # Determine input dimensionality from encoder
    input_dim = encoder.named_transformers_['onehot'] \
        .get_feature_names_out().shape[0]

    # Initialize VAE model
    vae = VAEModel(input_dim=input_dim, latent_dim=16)
    vae.compile(optimizer='adam')

    # Initialize model weights with a dummy forward pass
    _ = vae.predict(np.zeros((1, input_dim), dtype=np.float32))

    # Load trained weights
    vae.load_weights("vae_weights.weights.h5")

    # Load logs
    df_csv = load_csv_logs("../evtx_logs/normal/")
    df_evtx = load_evtx_logs("../evtx_logs/null/")

    # Combine logs
    df_all = pd.concat([df_csv, df_evtx], ignore_index=True)

    # Remove duplicated columns after merge
    df_all = df_all.loc[:, ~df_all.columns.duplicated()]

    # Preprocess data
    X, df_clean = preprocess(df_all, encoder, scaler)
    print(f"[*] Preprocessed test shape: {X.shape}")

    # Detect anomalies
    errors, anomalies = detect_anomalies(X, vae, threshold)

    # Generate report
    generate_report(df_clean, errors, anomalies, threshold)


if __name__ == "__main__":
    main()
