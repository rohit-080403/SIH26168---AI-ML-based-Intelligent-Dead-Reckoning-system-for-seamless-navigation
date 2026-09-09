import numpy as np
from pathlib import Path
from data_utils import discover_trip_pairs, load_trip, create_windows, FEATURE_COLS, TARGET_COLS

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

WINDOW_SIZE = 20   
STRIDE = 10        


def process_split(split: str):
    split_dir = RAW_DIR / split
    pairs = discover_trip_pairs(split_dir)

    if not pairs:
        print(f"[{split}] No trip pairs found in {split_dir} — skipping.")
        return

    all_X, all_y = [], []
    for trip_name, s_path, v_path in pairs:
        print(f"[{split}] Processing trip: {trip_name}")
        try:
            df = load_trip(s_path, v_path, trip_name)
            X, y = create_windows(df, WINDOW_SIZE, STRIDE)
            print(f"    -> {X.shape[0]} windows, feature shape {X.shape[1:]}")
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"    [ERROR] Failed on {trip_name}: {e}")

    if not all_X:
        print(f"[{split}] Nothing processed successfully.")
        return

    X_full = np.concatenate(all_X, axis=0)
    y_full = np.concatenate(all_y, axis=0)

    out_dir = PROCESSED_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X_full)
    np.save(out_dir / "y.npy", y_full)

    print(f"[{split}] Saved X{X_full.shape} and y{y_full.shape} to {out_dir}")


def main():
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Target columns ({len(TARGET_COLS)}): {TARGET_COLS}\n")

    for split in ["train", "val", "test"]:
        process_split(split)


if __name__ == "__main__":
    main()