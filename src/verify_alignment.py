"""
Phase 1.5: Verify S/V row-alignment using GPS overlay
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

S_FILE = Path("data/raw/train/S-M.txt")
V_FILE = Path("data/raw/train/V-M.txt")

S_COLUMN_MAP = {
    'GPS LATITUDE (degrees)': 'gps_lat',
    'GPS LONGITUDE (degrees)': 'gps_lon',
    'GPS ALTITUDE (m)': 'gps_alt',
    'GPS SPEED (Kmh)': 'gps_speed_kmh',
    'GPS ACCURACY (m)': 'gps_accuracy',
    'GPS ORIENTATION (Â°)': 'gps_orientation_deg',
    'GPS SATELLITES IN RANGE': 'gps_satellites',
    'TIME SINCE START (ms)': 'time_ms',
    'DATE (YYYY-MO-DD HH-MI-SS_SSS)': 'date',
    'ACCELEROMETER X (m/s²)': 'accel_x',
    'ACCELEROMETER Y (m/s²)': 'accel_y',
    'ACCELEROMETER Z (m/s²)': 'accel_z',
    'GRAVITY X (m/s²)': 'gravity_x',
    'GRAVITY Y (m/s²)': 'gravity_y',
    'GRAVITY Z (m/s²)': 'gravity_z',
    'GYROSCOPE Yaw (rad/s)': 'gyro_yaw',
    'GYROSCOPE Pitch (rad/s)': 'gyro_pitch',
    'GYROSCOPE Roll (rad/s)': 'gyro_roll',
    'MAGNETIC FIELD X (Î¼T)': 'mag_x',
    'MAGNETIC FIELD Y (Î¼T)': 'mag_y',
    'MAGNETIC FIELD Z (Î¼T)': 'mag_z',
    'ORIENTATION (Yaw) (Â°)': 'orient_yaw_deg',
    'ORIENTATION (Pitch) (Â°)': 'orient_pitch_deg',
    'ORIENTATION (Roll ) (Â°)': 'orient_roll_deg',
}

V_COLUMN_MAP = {
    'No of GPS Satellites Available': 'v_gps_satellites',
    'Time Since Start of Day (seconds)': 'v_time_sec',
    'Latitude (degrees)': 'v_lat',
    'Longitude (degrees)': 'v_lon',
    'Velocity (km/hr)': 'v_velocity_kmh',
    'Heading (degrees)': 'v_heading_deg',
    'Height (km)': 'v_height_km',
    'Vertical velocity (km/hr)': 'v_vert_velocity_kmh',
    'Sample period (seconds)': 'v_sample_period_sec',
    'Steering Angle (degrees)': 'v_steering_angle_deg',
    'Wheel Speed Front Left (rad/sec)': 'v_wheel_fl',
    'Wheel Speed Front Left(rad/sec)': 'v_wheel_fl',
    'Wheel Speed Front Right (rad/sec)': 'v_wheel_fr',
    'Wheel Speed Rear Left (rad/sec)': 'v_wheel_rl',
    'Wheel Speed Rear Right (rad/sec)': 'v_wheel_rr',
    'Yaw Rate (deg/sec)': 'v_yaw_rate',
    'Indicated Vehicle Speed (km/hr)': 'v_indicated_speed_kmh',
    'Indicated Longitudinal Acceleration (g)': 'v_accel_long_g',
    'IndicatedLongitudinal Acceleration (g)': 'v_accel_long_g',
    'Indicated Lateral Acceleration (g)': 'v_accel_lat_g',
    'Handbrake (0 or 1)': 'v_handbrake',
    'Gear Requested (Number fof gear employed 1-5)': 'v_gear_requested',
    'Gear (Number fof gear employed 1-5)': 'v_gear',
    'Engine Speed (rev/min)': 'v_engine_rpm',
    'Coolant Temperature (degrees)': 'v_coolant_temp',
    'Clutch Position (0 or 1)': 'v_clutch',
    'Brake Pressure (psi)': 'v_brake_pressure',
    'Brake Position (0 or 1)': 'v_brake_position',
    'Battery Voltage (volts)': 'v_battery_voltage',
    'Air Temperature (degrees)': 'v_air_temp',
    'Accelerator Pedal Position (0 or 1)': 'v_accel_pedal',
}


def load_and_clean(path, col_map):
    df = pd.read_csv(path, encoding="latin1")
    df.columns = [c.strip() for c in df.columns] 
    df = df.rename(columns=col_map)
    return df


def main():
    s_df = load_and_clean(S_FILE, S_COLUMN_MAP)
    v_df = load_and_clean(V_FILE, V_COLUMN_MAP)

    print("S columns after cleaning:", list(s_df.columns))
    print("V columns after cleaning:", list(v_df.columns))

    plt.figure(figsize=(8, 8))
    plt.plot(s_df["gps_lon"], s_df["gps_lat"], label="Smartphone (S) GPS", linewidth=1, alpha=0.7)
    plt.plot(v_df["v_lon"], v_df["v_lat"], label="Vehicle (V) GPS", linewidth=1, alpha=0.7, linestyle="--")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("S vs V GPS trace overlay — should match closely if row-aligned")
    plt.legend()
    plt.axis("equal")
    plt.grid(alpha=0.3)

    Path("results/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/plots/phase1_5_sv_gps_overlay.png", dpi=150)
    plt.show()

    lat_diff = (s_df["gps_lat"] - v_df["v_lat"]).abs()
    lon_diff = (s_df["gps_lon"] - v_df["v_lon"]).abs()
    print("\n--- Alignment check ---")
    print("Mean abs lat diff:", lat_diff.mean())
    print("Mean abs lon diff:", lon_diff.mean())
    print("Max abs lat diff:", lat_diff.max())
    print("Max abs lon diff:", lon_diff.max())


if __name__ == "__main__":
    main()