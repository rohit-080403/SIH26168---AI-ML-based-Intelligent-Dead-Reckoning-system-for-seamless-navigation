"""
Phase 4: CNN+GRU model to predict vehicle speed/motion state
directly from phone IMU windows (accel, gyro, mag, orientation),
replacing raw double-integration.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed/train")
MODEL_OUT = Path("models/speed_cnn_gru.keras")
PLOT_OUT = Path("results/plots/phase4_training_curves.png")
METRICS_OUT = Path("results/metrics/phase4_speed_model.txt")

TARGET_NAMES = ['v_velocity_kmh', 'v_yaw_rate', 'v_heading_deg', 'v_accel_long_g', 'v_accel_lat_g']

# NOTE: only single trip available right now -> chronological split,
# NOT random shuffle (windows overlap 50%, shuffling would leak
# adjacent-window info between train/val). Replace with separate
# trip files once more IO-VNBD data is downloaded.
VAL_FRACTION = 0.15


def build_model(input_shape, n_targets):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.GRU(32)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(n_targets)(x)  # linear activation for regression
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    X = np.load(PROCESSED_DIR / "X.npy")
    y = np.load(PROCESSED_DIR / "y.npy")
    print(f"Loaded X{X.shape}, y{y.shape}")

    # ---- Normalize features (per-channel mean/std) ----
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std

    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True) + 1e-8
    y_norm = (y - y_mean) / y_std

    np.save("data/processed/X_mean.npy", X_mean)
    np.save("data/processed/X_std.npy", X_std)
    np.save("data/processed/y_mean.npy", y_mean)
    np.save("data/processed/y_std.npy", y_std)

    # ---- Chronological split (see note above) ----
    split_idx = int(len(X_norm) * (1 - VAL_FRACTION))
    X_train, X_val = X_norm[:split_idx], X_norm[split_idx:]
    y_train, y_val = y_norm[:split_idx], y_norm[split_idx:]
    print(f"Train: {X_train.shape[0]} windows, Val: {X_val.shape[0]} windows")

    model = build_model(input_shape=X.shape[1:], n_targets=y.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

    # ---- Per-target error on validation set (denormalized, real units) ----
    y_pred_norm = model.predict(X_val)
    y_pred = y_pred_norm * y_std + y_mean
    y_val_real = y_val * y_std + y_mean

    report_lines = ["--- Phase 4 Validation Errors (real units) ---"]
    for i, name in enumerate(TARGET_NAMES):
        mae = np.mean(np.abs(y_pred[:, i] - y_val_real[:, i]))
        rmse = np.sqrt(np.mean((y_pred[:, i] - y_val_real[:, i])**2))
        line = f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}"
        print(line)
        report_lines.append(line)

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text("\n".join(report_lines))

    # ---- Plot training curves ----
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss (MSE, normalized)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='train')
    plt.plot(history.history['val_mae'], label='val')
    plt.title('MAE (normalized)')
    plt.legend()
    plt.tight_layout()
    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_OUT, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()