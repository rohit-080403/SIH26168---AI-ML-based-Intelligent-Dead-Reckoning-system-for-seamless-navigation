import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "train"
DOCS_OUT = ROOT / "docs" / "data_schema.md"
PLOT_OUT = ROOT / "results" / "plots" / "phase1_overview.png"


def resolve_input_file(base_name: str) -> Path:
    candidates = [
        DATA_DIR / base_name,
        DATA_DIR / f"{Path(base_name).stem}.csv",
        DATA_DIR / f"{Path(base_name).stem}.txt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return DATA_DIR / base_name


S_FILE = resolve_input_file("S-M.csv")
V_FILE = resolve_input_file("V-M.csv")


def read_data_file(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    return pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")


def inspect_file(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found at: {path}")

    df = read_data_file(path)

    info = {
        "label": label,
        "path": str(path),
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head(5),
        "describe": df.describe(),
        "n_nulls": df.isnull().sum().to_dict(),
    }

    time_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ["time", "timestamp", "t_"]):
            time_col = c
            break

    if time_col:
        t = df[time_col].astype(float).values
        dt = np.diff(t)
        dt = dt[dt > 0]
        if len(dt) > 0:
            median_dt = np.median(dt)
            info["time_col"] = time_col
            info["median_dt_sec"] = median_dt
            info["approx_sample_rate_hz"] = round(1.0 / median_dt, 2) if median_dt > 0 else None
            info["total_duration_sec"] = round(t[-1] - t[0], 2)
    else:
        info["time_col"] = None

    return info, df


def write_markdown_report(s_info, v_info):
    DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCS_OUT, "w", encoding="utf-8") as f:
        f.write("# IO-VNBD Data Schema — Phase 1 Findings\n\n")
        for info in [s_info, v_info]:
            f.write(f"## {info['label']} — `{info['path']}`\n\n")
            f.write(f"- Shape: {info['shape']}\n")
            f.write(f"- Columns: {info['columns']}\n")
            f.write(f"- Dtypes: {info['dtypes']}\n")
            f.write(f"- Null counts: {info['n_nulls']}\n")
            if info.get("time_col"):
                f.write(f"- Detected time column: `{info['time_col']}`\n")
                f.write(f"- Approx sample rate: {info['approx_sample_rate_hz']} Hz\n")
                f.write(f"- Total duration: {info['total_duration_sec']} sec\n")
            else:
                f.write("- No obvious time column detected — inspect manually.\n")
            f.write("\n### Describe\n\n")
            f.write(info["describe"].to_markdown())
            f.write("\n\n")
    print(f"[OK] Schema report written to {DOCS_OUT}")


def plot_overview(s_df, v_df):
    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    s_numeric = s_df.select_dtypes(include=[np.number]).columns[:4]
    for col in s_numeric:
        axes[0].plot(s_df[col].values[:2000], label=col, linewidth=0.8)
    axes[0].set_title("Smartphone (S) — first numeric columns (first 2000 samples)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    v_numeric = v_df.select_dtypes(include=[np.number]).columns[:4]
    for col in v_numeric:
        axes[1].plot(v_df[col].values[:2000], label=col, linewidth=0.8)
    axes[1].set_title("Vehicle (V) — first numeric columns (first 2000 samples)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=150)
    print(f"[OK] Overview plot saved to {PLOT_OUT}")


def main():
    print("Inspecting Smartphone (S) file...")
    s_info, s_df = inspect_file(S_FILE, "Smartphone (S)")
    print(s_info["columns"])

    print("\nInspecting Vehicle (V) file...")
    v_info, v_df = inspect_file(V_FILE, "Vehicle (V)")
    print(v_info["columns"])

    write_markdown_report(s_info, v_info)
    plot_overview(s_df, v_df)

    print("\n--- SUMMARY ---")
    print(f"S rows: {s_info['shape'][0]}, S cols: {s_info['shape'][1]}")
    print(f"V rows: {v_info['shape'][0]}, V cols: {v_info['shape'][1]}")


if __name__ == "__main__":
    main()