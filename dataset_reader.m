import pandas as pd
import numpy as np
import os
import glob
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

%% =============================================================================
%% CONFIGURATION — Set your dataset subfolder path here
%% =============================================================================
DATA_PATH = "./Dataset_Li-ion"  %% <-- Change this to your subfolder path


%% =============================================================================
%% STEP 1: DATA LOADING
%% =============================================================================
def load_and_combine_csvs(data_path: str) -> pd.DataFrame:
    """Recursively finds all CSVs in the given folder, parses and combines them."""
    csv_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under '{data_path}'. Check your DATA_PATH.")

    print(f"Found {len(csv_files)} CSV file(s). Loading...")

    all_frames = []
    for filepath in csv_files:
        df = parse_lg_hg2_csv(filepath)
        if df is not None and not df.empty:
            all_frames.append(df)
            print(f"  Loaded: {os.path.basename(filepath)} — {len(df)} rows")
        else:
            print(f"  Skipped (no usable data): {os.path.basename(filepath)}")

    if not all_frames:
        raise ValueError("All CSV files were skipped. Check that they contain valid headers and data.")

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nCombined dataset shape: {combined.shape}")
    return combined


def parse_lg_hg2_csv(filepath: str) -> pd.DataFrame | None:
    """
    Handles the LG HG2 dataset format:
    - Strips null characters and leading metadata/comment lines
    - Dynamically finds the header row containing key column names
    - Skips the units row that follows the header
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    %% Clean null characters and strip whitespace from every line
    cleaned_lines = [line.replace("\x00", "").strip() for line in lines]

    %% Dynamically locate the header row
    header_idx = -1
    for i, line in enumerate(cleaned_lines):
        parts = line.split(",")
        if (
            "Time Stamp" in line
            and "Voltage" in line
            and "Current" in line
            and "Temperature" in line
            and len(parts) > 10
        ):
            header_idx = i
            break

    if header_idx == -1:
        print(f"  Warning: Could not find header in {os.path.basename(filepath)}. Skipping.")
        return None

    %% Skip the units row (line immediately after header)
    units_idx = header_idx + 1
    data_lines = [cleaned_lines[header_idx]] + cleaned_lines[units_idx + 1 :]

    %% Filter out blank lines
    data_lines = [l for l in data_lines if l]

    try:
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=0)
    except Exception as e:
        print(f"  Warning: Failed to parse {os.path.basename(filepath)}: {e}")
        return None

    %% Drop trailing unnamed columns (common artifact of trailing commas)
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    return df


%% =============================================================================
%% STEP 2: CLEANING & FEATURE SELECTION
%% =============================================================================
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Selects feature columns (Voltage, Current, Temperature) and target column.
    Target priority: SOC > Capacity.
    Returns X, y, and the target column name.
    """
    required_features = ["Voltage", "Current", "Temperature"]

    %% Validate required feature columns exist
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}. Found: {df.columns.tolist()}")

    %% Determine target column
    if "SOC" in df.columns:
        y_col = "SOC"
    elif "Capacity" in df.columns:
        print("Warning: 'SOC' not found — using 'Capacity' as the target instead.")
        y_col = "Capacity"
    else:
        raise ValueError(
            f"Neither 'SOC' nor 'Capacity' found. Available columns: {df.columns.tolist()}"
        )

    %% Convert all relevant columns to numeric, coercing bad values to NaN
    for col in required_features + [y_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_features + [y_col])

    if df.empty:
        raise ValueError("DataFrame is empty after cleaning. Check your CSV contents.")

    print(f"Target column  : '{y_col}'")
    print(f"Rows after clean: {len(df)}")

    X = df[required_features]
    y = df[[y_col]]
    return X, y, y_col


%% =============================================================================
%% STEP 3: SCALING & SPLITTING
%% =============================================================================
def scale_and_split(
    X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )

    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


%% =============================================================================
%% STEP 4: MODEL — Wide Tri-Layered FFNN
%% =============================================================================
def build_model(input_dim: int = 3) -> tf.keras.Model:
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),  %% Layer 1 (Wide)
            Dense(64, activation="relu"),                             %% Layer 2 (Wide)
            Dense(64, activation="relu"),                             %% Layer 3 (Wide)
            Dense(1, activation="linear"),                            %% Output (SOC %)
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model


%% =============================================================================
%% STEP 5: TRAINING
%% =============================================================================
def train_model(model, X_train, y_train, epochs: int = 10, batch_size: int = 32):
    print("\nTraining the model...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )
    return history


%% =============================================================================
%% STEP 6: EVALUATION & VISUALISATION
%% =============================================================================
def evaluate_model(model, X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Test Results ---")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    return y_pred


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Training history plot saved to training_history.png")


def show_sample_predictions(X_scaled, y_pred, n: int = 5):
    df_out = pd.DataFrame(
        X_scaled[:n], columns=["V_scaled", "I_scaled", "T_scaled"]
    )
    df_out["Predicted_SOC_scaled"] = y_pred[:n]
    print(f"\nFirst {n} Scaled Predictions:")
    print(df_out.to_string(index=False))


%% =============================================================================
%% MAIN
%% =============================================================================
if __name__ == "__main__":
    %% 1. Load
    df = load_and_combine_csvs(DATA_PATH)

    %% 2. Prepare
    X, y, y_col = prepare_features(df)

    %% 3. Scale & split
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = scale_and_split(X, y)

    %% 4. Build
    model = build_model(input_dim=X_train.shape[1])

    %% 5. Train
    history = train_model(model, X_train, y_train, epochs=10, batch_size=32)

    %% 6. Evaluate
    y_pred = evaluate_model(model, X_test, y_test)
    plot_training_history(history)
    show_sample_predictions(X_test, y_pred, n=5)