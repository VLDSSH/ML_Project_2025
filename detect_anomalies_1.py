import os                      # For working with file system paths and directories
import pandas as pd            # For data manipulation and DataFrame operations
import numpy as np             # For numerical computations
import joblib                  # For loading saved encoder and scaler objects
import xmltodict               # For converting XML data to Python dictionaries

import matplotlib.pyplot as plt    # (Optional) plotting library
import seaborn as sns              # (Optional) visualization library

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from tensorflow.keras.models import load_model   # To load trained autoencoder model
from Evtx.Evtx import Evtx                        # For reading Windows EVTX log files


def load_csv_logs(path):
    """
    Load Windows event logs stored in CSV format.

    Steps:
    - Iterate through all CSV files in the given directory
    - Clean unnecessary columns
    - Normalize column names
    - Rename columns to a unified schema
    - Return a single DataFrame
    """
    # Get full paths of all CSV files in the directory
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    dataframes = []

    for file in files:
        # Read CSV file
        df = pd.read_csv(file, low_memory=False)

        # Remove auto-generated columns like "Unnamed: 0"
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Remove columns that contain only NaN values
        df = df.dropna(axis=1, how='all')

        # Normalize column names (lowercase, trimmed)
        df.columns = [str(col).strip().lower() for col in df.columns]

        # Map different possible column names to a standard format
        rename_map = {
            'eventid': 'event_id',
            'level': 'level',
            'provider': 'provider',
            'name': 'provider',
            'systemtime': 'time_created',
            'utctime': 'time_created',
            'creationutctime': 'time_created'
        }
        df = df.rename(columns=rename_map)

        # Ensure required columns exist
        for col in ['event_id', 'level', 'provider', 'time_created']:
            if col not in df.columns:
                df[col] = None

        # Keep only relevant columns
        df = df[['event_id', 'level', 'provider', 'time_created']]

        # Append cleaned DataFrame to list
        dataframes.append(df)

    # If no CSV files were found, return empty DataFrame with predefined columns
    if not dataframes:
        return pd.DataFrame(columns=['event_id', 'level', 'provider', 'time_created'])

    # Merge all CSV DataFrames into one
    return pd.concat(dataframes, ignore_index=True)


def load_evtx_logs(path):
    """
    Load and parse Windows EVTX log files.

    Steps:
    - Walk through directory recursively
    - Parse EVTX records
    - Extract key system fields
    """
    events = []

    # Recursively walk through directories
    for root, _, files in os.walk(path):
        for file in files:
            # Skip non-EVTX files
            if not file.endswith('.evtx'):
                continue

            full_path = os.path.join(root, file)

            # Open EVTX file
            with Evtx(full_path) as log:
                for record in log.records():
                    try:
                        # Convert XML event to dictionary
                        xml_dict = xmltodict.parse(record.xml())

                        # Extract system section
                        event = xml_dict.get('Event', {})
                        system = event.get('System', {})

                        # Extract relevant event fields
                        event_id = system.get('EventID', None)
                        level = system.get('Level', None)
                        provider = system.get('Provider', {}).get('@Name', None)
                        time_created = system.get('TimeCreated', {}).get('@SystemTime', None)

                        # Store extracted data
                        events.append({
                            'event_id': str(event_id),
                            'level': str(level),
                            'provider': provider,
                            'time_created': time_created
                        })
                    except Exception:
                        # Skip corrupted or unreadable records
                        continue

    # Convert list of events to DataFrame
    return pd.DataFrame(events)


def preprocess(df, encoder, scaler):
    """
    Preprocess logs before anomaly detection.

    Steps:
    - Remove rows with missing values
    - Convert all values to string type
    - Apply previously trained OneHotEncoder
    - Normalize features using StandardScaler
    """
    # Remove rows containing NaN values
    df_clean = df.dropna()

    # Convert all values to string format
    df_clean = df_clean.astype(str)

    # Transform categorical features using saved encoder
    X = encoder.transform(df_clean)

    # Scale features using saved scaler
    X_scaled = scaler.transform(
        X.toarray() if hasattr(X, 'toarray') else X
    )

    # Return both scaled features and cleaned DataFrame
    return X_scaled, df_clean


def detect_anomalies(X, model, threshold):
    """
    Detect anomalies using reconstruction error.

    Steps:
    - Reconstruct input data using autoencoder
    - Compute Mean Squared Error per sample
    - Compare error with threshold
    """
    # Predict reconstructed input
    preds = model.predict(X)

    # Compute reconstruction error for each log entry
    errors = np.mean((X - preds) ** 2, axis=1)

    # Identify anomalies based on threshold
    return errors, errors > threshold


def assign_score(error):
    """
    Convert reconstruction error into a risk score (0–100).
    """
    if error <= 0.01:
        return 0
    elif error >= 2:
        return 100
    else:
        # Linearly scale error between 0.01 and 2 to score range 0–100
        score = int(((error - 0.01) / (2 - 0.01)) * 100)
        return min(score, 100)


def generate_report(df, errors, anomalies):
    """
    Generate interactive HTML report with anomaly detection results.
    """
    # Total number of analyzed logs
    total = len(df)

    # Count anomalies and normal logs
    num_anomalies = int(np.sum(anomalies))
    num_normal = total - num_anomalies

    # Append results to DataFrame
    df['reconstruction_error'] = errors
    df['is_anomaly'] = anomalies
    df['score'] = [assign_score(e) for e in errors]

    # HTML report content
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Log analysis report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2F4F4F; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; cursor: pointer; }}
            .summary {{ margin-top: 20px; font-size: 1.1em; }}
        </style>
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    </head>
    <body>
        <h1>Log analysis report</h1>

        <div class="summary">
            <p><strong>Total logs analyzed:</strong> {total}</p>
            <p><strong>Normal logs:</strong> {num_normal}</p>
            <p><strong>Anomalous logs:</strong> {num_anomalies}</p>
        </div>

        <h2>Score distribution (0–100)</h2>
        <table id="scoreTable" class="display">
            {df.groupby('score').size().reset_index(name='Count').to_html(index=False)}
        </table>

        <h2>Detailed results</h2>
        <table id="logTable" class="display">
            {df.to_html(index=False, classes='display')}
        </table>

        <script>
            $(document).ready(function() {{
                $('#logTable').DataTable();
                $('#scoreTable').DataTable();
            }});
        </script>
    </body>
    </html>
    """

    # Save report to file
    with open("anomaly_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("[+] Report saved as anomaly_report.html")


def main():
    """
    Main execution pipeline for anomaly detection.
    """
    print("[*] Loading model, encoder, and scaler...")

    # Load trained autoencoder model
    model = load_model('model.h5', compile=False)

    # Load preprocessing objects
    encoder = joblib.load('encoder.pkl')
    scaler = joblib.load('scaler.pkl')

    print("[*] Loading test logs...")

    # Load CSV and EVTX logs containing anomalies
    df_csv = load_csv_logs('evtx_logs/anomaly/')
    df_evtx = load_evtx_logs('evtx_logs/anomaly/')

    # Remove duplicated columns if present
    df_csv = df_csv.loc[:, ~df_csv.columns.duplicated()]
    df_evtx = df_evtx.loc[:, ~df_evtx.columns.duplicated()]

    # Combine all logs
    df = pd.concat([df_csv, df_evtx], ignore_index=True)

    print("[*] Preprocessing data...")

    # Apply preprocessing pipeline
    X, df_clean = preprocess(df, encoder, scaler)

    if df_clean.empty:
        print("[!] No data available after preprocessing.")
        return

    print(f"[*] Loaded {len(df)} logs, {len(df_clean)} remained after cleaning.")

    print("[*] Detecting anomalies...")

    # Detect anomalies using reconstruction error
    errors, anomalies = detect_anomalies(X, model, threshold=0.01)

    # Generate final HTML report
    generate_report(df_clean, errors, anomalies)


if __name__ == '__main__':
    main()
