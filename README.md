# Log Anomaly Detection Using Autoencoders

This project implements anomaly detection for system and security logs using **Autoencoders (AE)** and **Variational Autoencoders (VAE)**. The system processes logs (CSV and Windows EVTX formats), applies preprocessing and encoding, detects anomalous events based on reconstruction error, and generates a detailed **HTML report** with metrics and visualizations.

The project supports two approaches:
- **Vanilla Autoencoder (AE)**
- **Variational Autoencoder (VAE)**

---

## Project Structure

```
.
├── train_model.py              # Training script for vanilla Autoencoder (AE)
├── detect_anomalies_1.py       # Anomaly detection + HTML report generation (AE)
├── model.h5                    # Trained vanilla AE model
│
├── train_vae.py                # Training script for Variational Autoencoder (VAE)
├── detect_vae_anomalies.py     # Anomaly detection + HTML report generation (VAE)
├── vae_weights.weights.h5      # Trained VAE model weights
│
├── vae_encoder.pkl             # Saved OneHot / categorical encoder
├── vae_scaler.pkl              # Saved feature scaler
├── threshold_95.pkl            # Detection threshold (95th percentile)
│
├── evtx_logs/                  # Input logs directory
│   ├── normal/                 # CSV logs
│   └── null/                   # EVTX logs
│
└── README.md
```

---

## Data Sources

The system works with two types of log files:

- **CSV logs** – loaded from a directory and normalized
- **Windows EVTX logs** – parsed using the `Evtx` library and XML decoding

Extracted fields:
- `event_id`
- `level`
- `provider`
- `time_created`

Duplicate columns and invalid rows are automatically removed.

---

## Machine Learning Pipeline

### 1. Preprocessing

- Remove missing values
- Convert categorical features to strings
- Apply **One-Hot Encoding**
- Apply **feature scaling**

The encoder and scaler are saved and reused during detection to ensure consistency.

---

## Vanilla Autoencoder (AE)

### `train_model.py`

This script:
- Builds a **standard autoencoder** neural network
- Trains the model only on **normal logs**
- Learns to reconstruct normal behavior
- Saves the trained model to `model.h5`

The reconstruction error is later used as an anomaly score.

### `model.h5`

- Fully trained vanilla autoencoder model
- Used directly during anomaly detection

---

### `detect_anomalies_1.py`

This script:
- Loads the trained AE model
- Preprocesses new log data
- Computes **reconstruction error** for each log entry
- Flags anomalies using a predefined threshold
- Assigns a **score (0–100)** based on anomaly severity

#### Output:

- Interactive **HTML report**
- Anomaly statistics
- Confusion Matrix
- ROC Curve
- Sortable and searchable table with all log records and scores

---

## Variational Autoencoder (VAE)

### `train_vae.py`

This script:
- Builds a **Variational Autoencoder (VAE)**
- Uses latent space sampling (mean + variance)
- Trains on normal log data
- Saves trained weights to `vae_weights.weights.h5`

The VAE provides better generalization and smoother anomaly detection compared to vanilla AE.

---

### `vae_weights.weights.h5`

- Saved weights of the trained VAE model
- Loaded during detection for inference

---

### `detect_vae_anomalies.py`

This script performs the same tasks as `detect_anomalies_1.py`, but using the **VAE model**:

- Loads VAE architecture and weights
- Computes reconstruction error
- Detects anomalies based on threshold
- Generates an **HTML report** with:
  - Anomaly summary
  - Score distribution
  - Confusion Matrix
  - ROC Curve
  - Detailed anomaly table

---

## Anomaly Scoring

- Logs below the threshold → `score = 0` (normal)
- Logs above the threshold → scaled score from **1 to 100**
- Higher score = higher anomaly severity

---

## Output Artifacts

- `vae_anomaly_report.html` – main interactive report
- `confusion_matrix.png`
- `roc_curve.png`

---

## Technologies Used

- Python 3
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy
- Seaborn / Matplotlib
- EVTX parsing
- HTML + DataTables (JavaScript)

---

## Use Case

This project is suitable for:
- Security monitoring
- SIEM preprocessing
- Windows event log anomaly detection
- Academic ML / cybersecurity projects

---

## Notes

- Models must be trained before running detection scripts
- Encoder, scaler, and threshold must match the trained model
- VAE generally provides more stable anomaly detection than vanilla AE

---


