import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data_utils import load_trip

S_FILE = Path("data/raw/train/S-M.txt")
V_FILE = Path("data/raw/train/V-M.txt")
TRIP_NAME = "M"
DT = 0.1  

PLOT_OUT = Path("results/plots/phase3_baseline_drift.png")
METRICS_OUT = Path("results/metrics/phase3_baseline.txt")


def latlon_to_local_xy(lat, lon, lat0, lon0):
    """Equirectangular approx: good enough for short trips (~10s of km)."""
    R = 6371000.0  
    lat_rad = np.radians(lat)
    lat0_rad = np.radians(lat0)
    x = np.radians(lon - lon0) * R * np.cos(lat0_rad)
    y = np.radians(lat - lat0) * R
    return x, y

def segmented_drift(df, window_sec=60, dt=DT):
    """Simulate repeated short GNSS outages. Reset to ground truth
    at the start of each window; integrate raw IMU for window_sec;
    measure drift at window end. Returns list of per-window results."""
    window_len = int(window_sec / dt)
    n = len(df)

    lin_ax = df['accel_x'].to_numpy() - df['gravity_x'].to_numpy()
    lin_ay = df['accel_y'].to_numpy() - df['gravity_y'].to_numpy()
    yaw_rad = np.radians(df['orient_yaw_deg'].to_numpy())
    world_ax = lin_ax * np.cos(yaw_rad) - lin_ay * np.sin(yaw_rad)
    world_ay = lin_ax * np.sin(yaw_rad) + lin_ay * np.cos(yaw_rad)

    gt_speed_ms = df['v_velocity_kmh'].to_numpy() / 3.6
    lat0_all, lon0_all = df['v_lat'].to_numpy(), df['v_lon'].to_numpy()

    results = []
    for start in range(0, n - window_len, window_len):
        end = start + window_len

        # reset velocity/position to zero at window start (fresh GNSS fix)
        vx = np.cumsum(world_ax[start:end]) * dt
        vy = np.cumsum(world_ay[start:end]) * dt
        px = np.cumsum(vx) * dt
        py = np.cumsum(vy) * dt

        gt_x, gt_y = latlon_to_local_xy(
            lat0_all[start:end], lon0_all[start:end],
            lat0_all[start], lon0_all[start]
        )

        dist = np.sum(gt_speed_ms[start:end]) * dt
        err = np.sqrt((px[-1] - gt_x[-1])**2 + (py[-1] - gt_y[-1])**2)
        drift_pct = (err / dist * 100) if dist > 0 else np.nan

        results.append({
            'window_start_sec': start * dt,
            'distance_m': dist,
            'error_m': err,
            'drift_pct': drift_pct
        })
    return results
def rolling_window_drift(df, window_secs_list=[30, 60, 300]):
    """Compute DR drift over short rolling segments (simulated GNSS outages),
    matching the PS benchmark style (e.g. '5m drift over 50m in <1min')."""
    lin_ax = df['accel_x'].to_numpy() - df['gravity_x'].to_numpy()
    lin_ay = df['accel_y'].to_numpy() - df['gravity_y'].to_numpy()
    yaw_rad = np.radians(df['orient_yaw_deg'].to_numpy())
    world_ax = lin_ax * np.cos(yaw_rad) - lin_ay * np.sin(yaw_rad)
    world_ay = lin_ax * np.sin(yaw_rad) + lin_ay * np.cos(yaw_rad)

    gt_speed_ms = df['v_velocity_kmh'].to_numpy() / 3.6
    lat0, lon0 = df['v_lat'].iloc[0], df['v_lon'].iloc[0]
    gt_x, gt_y = latlon_to_local_xy(df['v_lat'].to_numpy(), df['v_lon'].to_numpy(), lat0, lon0)

    results = {}
    for secs in window_secs_list:
        win = int(secs / DT)
        errors_pct = []
        for start in range(0, len(df) - win, win):  # non-overlapping segments
            end = start + win
            vx = np.cumsum(world_ax[start:end]) * DT
            vy = np.cumsum(world_ay[start:end]) * DT
            px = np.cumsum(vx) * DT
            py = np.cumsum(vy) * DT

            dr_dx, dr_dy = px[-1], py[-1]
            gt_dx = gt_x[end - 1] - gt_x[start]
            gt_dy = gt_y[end - 1] - gt_y[start]

            error = np.sqrt((dr_dx - gt_dx)**2 + (dr_dy - gt_dy)**2)
            dist = np.sum(gt_speed_ms[start:end]) * DT
            if dist > 1.0:  # skip near-stationary segments
                errors_pct.append((error / dist) * 100)

        if errors_pct:
            results[secs] = {
                'mean_drift_pct': np.mean(errors_pct),
                'median_drift_pct': np.median(errors_pct),
                'n_segments': len(errors_pct)
            }
    return results



def main():
    df = load_trip(S_FILE, V_FILE, TRIP_NAME)
    n = len(df)
    print(f"Loaded trip '{TRIP_NAME}' with {n} rows")

   
    lin_ax = df['accel_x'].to_numpy() - df['gravity_x'].to_numpy()
    lin_ay = df['accel_y'].to_numpy() - df['gravity_y'].to_numpy()

   
    yaw_rad = np.radians(df['orient_yaw_deg'].to_numpy())
    world_ax = lin_ax * np.cos(yaw_rad) - lin_ay * np.sin(yaw_rad)
    world_ay = lin_ax * np.sin(yaw_rad) + lin_ay * np.cos(yaw_rad)

    vx = np.cumsum(world_ax) * DT
    vy = np.cumsum(world_ay) * DT
    px = np.cumsum(vx) * DT
    py = np.cumsum(vy) * DT

    lat0, lon0 = df['v_lat'].iloc[0], df['v_lon'].iloc[0]
    gt_x, gt_y = latlon_to_local_xy(
        df['v_lat'].to_numpy(), df['v_lon'].to_numpy(), lat0, lon0
    )

    gt_speed_ms = df['v_velocity_kmh'].to_numpy() / 3.6
    total_distance_m = np.sum(gt_speed_ms) * DT

    final_error_m = np.sqrt((px[-1] - gt_x[-1])**2 + (py[-1] - gt_y[-1])**2)
    drift_pct = (final_error_m / total_distance_m) * 100

    # ---- Report ----
    report = (
        f"Trip: {TRIP_NAME}\n"
        f"Duration: {n * DT:.1f} sec ({n * DT / 60:.1f} min)\n"
        f"Total distance travelled (ground truth): {total_distance_m:.1f} m\n"
        f"Final position error (raw DR vs ground truth): {final_error_m:.1f} m\n"
        f"Drift: {drift_pct:.1f}% of distance travelled\n"
        f"SIH target: <10% drift\n"
    )
    print("\n--- Phase 3 Baseline Results ---")
    print(report)

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(report)

    # ---- Plot ----
    plt.figure(figsize=(9, 9))
    plt.plot(gt_x, gt_y, label="Ground truth (V/GPS)", linewidth=2)
    plt.plot(px, py, label="Raw double-integration DR", linewidth=1, alpha=0.8)
    plt.scatter([gt_x[0]], [gt_y[0]], c='green', marker='o', label='Start', zorder=5)
    plt.scatter([gt_x[-1]], [gt_y[-1]], c='blue', marker='x', label='GT End', zorder=5)
    plt.scatter([px[-1]], [py[-1]], c='red', marker='x', label='DR End', zorder=5)
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title(f"Phase 3: Raw Dead Reckoning Drift — {drift_pct:.1f}% (Trip {TRIP_NAME})")
    plt.legend()
    plt.axis("equal")
    plt.grid(alpha=0.3)

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_OUT, dpi=150)
    plt.show()
    print(f"Plot saved to {PLOT_OUT}")

    seg_results = segmented_drift(df, window_sec=60)
    drifts = [r['drift_pct'] for r in seg_results if not np.isnan(r['drift_pct'])]
    print(f"\n--- Segmented (60s window) Drift ---")
    print(f"Windows evaluated: {len(drifts)}")
    print(f"Mean drift: {np.mean(drifts):.1f}%")
    print(f"Median drift: {np.median(drifts):.1f}%")
    print(f"Best window: {np.min(drifts):.1f}%")
    print(f"Worst window: {np.max(drifts):.1f}%")

    short_results = rolling_window_drift(df)
    print("\n--- Short-window drift (simulated GNSS outages) ---")
    for secs, stats in short_results.items():
        print(f"{secs}s windows ({stats['n_segments']} segments): "
              f"mean drift {stats['mean_drift_pct']:.1f}%, median {stats['median_drift_pct']:.1f}%")


if __name__ == "__main__":
    main()