import re
import numpy as np
import pandas as pd
from pathlib import Path


S_COLUMN_MAP = {
    'gps latitude (degrees)': 'gps_lat',
    'gps longitude (degrees)': 'gps_lon',
    'gps altitude (m)': 'gps_alt',
    'gps speed (kmh)': 'gps_speed_kmh',
    'gps accuracy (m)': 'gps_accuracy',
    'gps orientation (â°)': 'gps_orientation_deg',
    'gps orientation (°)': 'gps_orientation_deg',
    'gps satellites in range': 'gps_satellites',
    'time since start (ms)': 'time_ms',
    'date (yyyy-mo-dd hh-mi-ss_sss)': 'date',
    'accelerometer x (m/sÂ²)': 'accel_x',
    'accelerometer x (m/s²)': 'accel_x',
    'accelerometer y (m/sÂ²)': 'accel_y',
    'accelerometer y (m/s²)': 'accel_y',
    'accelerometer z (m/sÂ²)': 'accel_z',
    'accelerometer z (m/s²)': 'accel_z',
    'gravity x (m/sÂ²)': 'gravity_x',
    'gravity x (m/s²)': 'gravity_x',
    'gravity y (m/sÂ²)': 'gravity_y',
    'gravity y (m/s²)': 'gravity_y',
    'gravity z (m/sÂ²)': 'gravity_z',
    'gravity z (m/s²)': 'gravity_z',
    'gyroscope yaw (rad/s)': 'gyro_yaw',
    'gyroscope pitch (rad/s)': 'gyro_pitch',
    'gyroscope roll (rad/s)': 'gyro_roll',
    'magnetic field x (î¼t)': 'mag_x',
    'magnetic field x (μt)': 'mag_x',
    'magnetic field y (î¼t)': 'mag_y',
    'magnetic field y (μt)': 'mag_y',
    'magnetic field z (î¼t)': 'mag_z',
    'magnetic field z (μt)': 'mag_z',
    'orientation (yaw) (â°)': 'orient_yaw_deg',
    'orientation (yaw) (°)': 'orient_yaw_deg',
    'orientation (pitch) (â°)': 'orient_pitch_deg',
    'orientation (pitch) (°)': 'orient_pitch_deg',
    'orientation (roll ) (â°)': 'orient_roll_deg',
    'orientation (roll ) (°)': 'orient_roll_deg',
    'orientation (roll) (â°)': 'orient_roll_deg',
    'orientation (roll) (°)': 'orient_roll_deg',
}

V_COLUMN_MAP = {
    'no of gps satellites available': 'v_gps_satellites',
    'time since start of day (seconds)': 'v_time_sec',
    'latitude (degrees)': 'v_lat',
    'longitude (degrees)': 'v_lon',
    'velocity (km/hr)': 'v_velocity_kmh',
    'heading (degrees)': 'v_heading_deg',
    'height (km)': 'v_height_km',
    'vertical velocity (km/hr)': 'v_vert_velocity_kmh',
    'sample period (seconds)': 'v_sample_period_sec',
    'steering angle (degrees)': 'v_steering_angle_deg',
    'wheel speed front left (rad/sec)': 'v_wheel_fl',
    'wheel speed front left(rad/sec)': 'v_wheel_fl',
    'wheel speed front right (rad/sec)': 'v_wheel_fr',
    'wheel speed rear left (rad/sec)': 'v_wheel_rl',
    'wheel speed rear right (rad/sec)': 'v_wheel_rr',
    'yaw rate (deg/sec)': 'v_yaw_rate',
    'indicated vehicle speed (km/hr)': 'v_indicated_speed_kmh',
    'indicated longitudinal acceleration (g)': 'v_accel_long_g',
    'indicatedlongitudinal acceleration (g)': 'v_accel_long_g',
    'indicated lateral acceleration (g)': 'v_accel_lat_g',
    'handbrake (0 or 1)': 'v_handbrake',
    'gear requested (number fof gear employed 1-5)': 'v_gear_requested',
    'gear (number fof gear employed 1-5)': 'v_gear',
    'engine speed (rev/min)': 'v_engine_rpm',
    'coolant temperature (degrees)': 'v_coolant_temp',
    'clutch position (0 or 1)': 'v_clutch',
    'brake pressure (psi)': 'v_brake_pressure',
    'brake position (0 or 1)': 'v_brake_position',
    'battery voltage (volts)': 'v_battery_voltage',
    'air temperature (degrees)': 'v_air_temp',
    'accelerator pedal position (0 or 1)': 'v_accel_pedal',
}

FEATURE_COLS = [
    'accel_x', 'accel_y', 'accel_z',
    'gyro_yaw', 'gyro_pitch', 'gyro_roll',
    'mag_x', 'mag_y', 'mag_z',
    'gravity_x', 'gravity_y', 'gravity_z',
    'orient_yaw_deg', 'orient_pitch_deg', 'orient_roll_deg',
]

TARGET_COLS = [
    'v_velocity_kmh',
    'v_yaw_rate',
    'v_heading_deg',
    'v_accel_long_g',
    'v_accel_lat_g',
]


def _normalize(col: str) -> str:
    """Lowercase + collapse whitespace for robust column matching."""
    return re.sub(r'\s+', ' ', col.strip().lower())


def _clean_and_rename(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    normalized_map = {_normalize(k): v for k, v in col_map.items()}
    new_cols = {}
    for c in df.columns:
        norm = _normalize(c)
        if norm in normalized_map:
            new_cols[c] = normalized_map[norm]
    df = df.rename(columns=new_cols)
    return df


def load_trip(s_path: Path, v_path: Path, trip_name: str) -> pd.DataFrame:
    """Load one S/V trip pair, clean columns, and merge by row index
    (valid since IO-VNBD 'Synchronised' files are pre-aligned per row)."""
    s_df = pd.read_csv(s_path, encoding="latin1")
    v_df = pd.read_csv(v_path, encoding="latin1")

    s_df = _clean_and_rename(s_df, S_COLUMN_MAP)
    v_df = _clean_and_rename(v_df, V_COLUMN_MAP)

    n = min(len(s_df), len(v_df))
    if len(s_df) != len(v_df):
        print(f"[WARN] {trip_name}: row count mismatch S={len(s_df)} V={len(v_df)}, truncating to {n}")
    s_df = s_df.iloc[:n].reset_index(drop=True)
    v_df = v_df.iloc[:n].reset_index(drop=True)

    merged = pd.concat([s_df, v_df], axis=1)
    merged['trip'] = trip_name
    return merged


def discover_trip_pairs(raw_dir: Path):
    """Find matching S-*/V-* file pairs (csv or txt) in a folder."""
    raw_dir = Path(raw_dir)
    pairs = []
    s_files = list(raw_dir.glob("S-*.csv")) + list(raw_dir.glob("S-*.txt"))
    for s_path in s_files:
        suffix = s_path.name[2:]  # strip "S-"
        v_candidates = [raw_dir / f"V-{suffix}"]
        # case-insensitive fallback
        v_path = None
        for cand in v_candidates:
            if cand.exists():
                v_path = cand
                break
        if v_path is None:
            matches = [f for f in raw_dir.glob(f"V-*{Path(suffix).stem}*")]
            if matches:
                v_path = matches[0]
        if v_path is None:
            print(f"[WARN] No matching V file found for {s_path.name}, skipping.")
            continue
        trip_name = Path(suffix).stem
        pairs.append((trip_name, s_path, v_path))
    return pairs


def create_windows(df: pd.DataFrame, window_size: int, stride: int,
                    feature_cols=FEATURE_COLS, target_cols=TARGET_COLS):
    
    missing_feat = [c for c in feature_cols if c not in df.columns]
    missing_targ = [c for c in target_cols if c not in df.columns]
    if missing_feat or missing_targ:
        raise ValueError(f"Missing columns — features: {missing_feat}, targets: {missing_targ}")

    feats = df[feature_cols].to_numpy(dtype=np.float32)
    targs = df[target_cols].to_numpy(dtype=np.float32)

    X, y = [], []
    n = len(df)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        X.append(feats[start:end])
        y.append(targs[end - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)