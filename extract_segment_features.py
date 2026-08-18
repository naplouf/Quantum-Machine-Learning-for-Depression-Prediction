import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = Path.home() / "Library" / "CloudStorage" / \
           "GoogleDrive-ohvevo2014@gmail.com" / "My Drive" / "EDAIC" / "data"

LABELS_DIR = Path.home() / "Library" / "CloudStorage" / \
             "GoogleDrive-ohvevo2014@gmail.com" / "My Drive" / "EDAIC" / "labels"

OUTPUT_DIR = Path.home() / "Library" / "CloudStorage" / \
             "GoogleDrive-ohvevo2014@gmail.com" / "My Drive" / "EDAIC"

N_SEGMENTS = 5

LABEL_FILES = {
    "train": LABELS_DIR / "train_labels.csv",
    "dev": LABELS_DIR / "dev_labels.csv",
    "test": LABELS_DIR / "test_labels.csv",
}

MFCC_BASE_PATTERN = "pcm_fftMag_mfcc["
MFCC_DELTA_PATTERN = "pcm_fftMag_mfcc_de["
MFCC_DELTA2_PATTERN = "pcm_fftMag_mfcc_de_de["

EGEMAP_SKIP_COLS = {"name", "frameTime"}


def load_frame_level(csv_path: Path) -> pd.DataFrame:
    """
    Load a frame-level OpenSMILE CSV.
    Handles semicolon separators and comment headers.
    """
    try:
        df = pd.read_csv(csv_path, sep=";", comment="@", low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False)

    if "frameTime" in df.columns:
        df = df[pd.to_numeric(df["frameTime"], errors="coerce").notna()]
        df["frameTime"] = df["frameTime"].astype(float)
        df = df.sort_values("frameTime").reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame, feature_type: str) -> list:
    if feature_type == "mfcc_base":
        return [c for c in df.columns
                if MFCC_BASE_PATTERN in c and MFCC_DELTA_PATTERN not in c]
    elif feature_type == "mfcc_delta":
        return [c for c in df.columns
                if MFCC_DELTA_PATTERN in c and MFCC_DELTA2_PATTERN not in c]
    elif feature_type == "mfcc_delta2":
        return [c for c in df.columns if MFCC_DELTA2_PATTERN in c]
    elif feature_type == "egemap":
        return [c for c in df.columns
                if c not in EGEMAP_SKIP_COLS and df[c].dtype in [np.float64, np.float32, float]]
    return []


def extract_segment_volatility(df: pd.DataFrame, feat_cols: list,
                                n_segments: int, prefix: str) -> dict:
    """
    Split frames into n_segments windows and compute volatility features.
    Produces 6 features per coefficient (seg_mean_std, seg_mean_range,
    seg_std_mean, seg_std_std, seg_trend, global_std).
    """
    features = {}
    n_frames = len(df)

    if n_frames < n_segments:
        for col in feat_cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().values
            safe = lambda v, fn: float(fn(v)) if len(v) > 0 else 0.0
            features[f"{prefix}_{col}_seg_mean_std"] = safe(vals, np.std)
            features[f"{prefix}_{col}_seg_mean_range"] = safe(vals, np.ptp)
            features[f"{prefix}_{col}_seg_std_mean"] = safe(vals, np.std)
            features[f"{prefix}_{col}_seg_std_std"] = 0.0
            features[f"{prefix}_{col}_seg_trend"] = 0.0
            features[f"{prefix}_{col}_global_std"] = safe(vals, np.std)
        return features

    segments = np.array_split(np.arange(n_frames), n_segments)

    for col in feat_cols:
        vals_all = pd.to_numeric(df[col], errors="coerce").values

        seg_means = []
        seg_stds = []

        for seg_idx in segments:
            seg_vals = vals_all[seg_idx]
            seg_vals = seg_vals[~np.isnan(seg_vals)]
            if len(seg_vals) == 0:
                seg_means.append(np.nan)
                seg_stds.append(np.nan)
            else:
                seg_means.append(np.mean(seg_vals))
                seg_stds.append(np.std(seg_vals))

        seg_means = np.array(seg_means, dtype=float)
        seg_stds = np.array(seg_stds, dtype=float)

        valid_means = seg_means[~np.isnan(seg_means)]
        valid_stds = seg_stds[~np.isnan(seg_stds)]

        features[f"{prefix}_{col}_seg_mean_std"] = (
            float(np.std(valid_means)) if len(valid_means) > 1 else 0.0
        )
        features[f"{prefix}_{col}_seg_mean_range"] = (
            float(np.ptp(valid_means)) if len(valid_means) > 1 else 0.0
        )
        features[f"{prefix}_{col}_seg_std_mean"] = (
            float(np.mean(valid_stds)) if len(valid_stds) > 0 else 0.0
        )
        features[f"{prefix}_{col}_seg_std_std"] = (
            float(np.std(valid_stds)) if len(valid_stds) > 1 else 0.0
        )

        if len(valid_means) > 1:
            x = np.arange(len(valid_means), dtype=float)
            slope = np.polyfit(x, valid_means, 1)[0]
            features[f"{prefix}_{col}_seg_trend"] = float(slope)
        else:
            features[f"{prefix}_{col}_seg_trend"] = 0.0

        all_vals = vals_all[~np.isnan(vals_all)]
        features[f"{prefix}_{col}_global_std"] = (
            float(np.std(all_vals)) if len(all_vals) > 0 else 0.0
        )

    return features


def process_participant(pid: int, data_dir: Path, n_segments: int) -> dict | None:
    p_dir = data_dir / f"{pid}_P" / "features"
    mfcc_path = p_dir / f"{pid}_OpenSMILE2.3.0_mfcc.csv"
    egem_path = p_dir / f"{pid}_OpenSMILE2.3.0_egemaps.csv"

    if not mfcc_path.exists():
        print(f"  [WARN] Missing MFCC file for participant {pid}")
        return None

    features = {"Participant_ID": pid}

    try:
        mfcc_df = load_frame_level(mfcc_path)

        base_cols = get_feature_columns(mfcc_df, "mfcc_base")
        delta_cols = get_feature_columns(mfcc_df, "mfcc_delta")
        delta2_cols = get_feature_columns(mfcc_df, "mfcc_delta2")

        features.update(extract_segment_volatility(
            mfcc_df, base_cols, n_segments, "mfcc_base"))
        features.update(extract_segment_volatility(
            mfcc_df, delta_cols, n_segments, "mfcc_delta"))
        features.update(extract_segment_volatility(
            mfcc_df, delta2_cols, n_segments, "mfcc_delta2"))

    except Exception as e:
        print(f"  [ERROR] MFCC processing failed for {pid}: {e}")
        return None

    if egem_path.exists():
        try:
            egem_df = load_frame_level(egem_path)
            egem_cols = get_feature_columns(egem_df, "egemap")
            features.update(extract_segment_volatility(
                egem_df, egem_cols, n_segments, "egemap"))
        except Exception as e:
            print(f"  [WARN] eGeMAPS processing failed for {pid}: {e}")
    else:
        print(f"  [WARN] Missing eGeMAPS file for participant {pid} — skipping egemap features")

    return features


def build_split(label_df: pd.DataFrame, data_dir: Path,
                n_segments: int, split_name: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split ({len(label_df)} participants)...")
    print(f"{'='*60}")

    rows = []
    for _, row in tqdm(label_df.iterrows(), total=len(label_df),
                       desc=split_name, ncols=80):
        pid = int(row["Participant_ID"])
        feat = process_participant(pid, data_dir, n_segments)
        if feat is not None:
            feat["PHQ_Binary"] = int(row["PHQ_Binary"])
            feat["PHQ_Score"] = float(row.get("PHQ_Score", row.get("PHQ8_Score", np.nan)))
            rows.append(feat)

    df = pd.DataFrame(rows)
    print(f"  ✓ {len(df)} participants processed successfully")
    print(f"  ✓ {df.shape[1] - 3} features extracted per participant")
    print(f"  ✓ Class balance: {dict(df['PHQ_Binary'].value_counts())}")
    return df


def main():
    print("E-DAIC Segment-Level Volatility Feature Extraction")
    print(f"N_SEGMENTS = {N_SEGMENTS}")
    print(f"DATA_DIR   = {DATA_DIR}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"DATA_DIR not found: {DATA_DIR}\n"
            "Please update the DATA_DIR variable at the top of this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_dfs = {}
    for split, path in LABEL_FILES.items():
        if not path.exists():
            for alt in [f"{split}_Split.csv", f"{split.capitalize()}_labels.csv",
                        f"DAIC_{split}.csv"]:
                alt_path = LABELS_DIR / alt
                if alt_path.exists():
                    path = alt_path
                    break
            else:
                print(f"[WARN] Could not find label file for {split} split at {path}")
                print(f"       Skipping {split} split.")
                continue
        label_dfs[split] = pd.read_csv(path)
        print(f"Loaded {split} labels: {len(label_dfs[split])} participants")

    if not label_dfs:
        raise FileNotFoundError(
            "No label files found. Please check LABELS_DIR and LABEL_FILES paths."
        )

    output_paths = {}
    for split, label_df in label_dfs.items():
        df = build_split(label_df, DATA_DIR, N_SEGMENTS, split)

        out_path = OUTPUT_DIR / f"segment_{split}.csv"
        df.to_csv(out_path, index=False)
        output_paths[split] = out_path
        print(f"  Saved: {out_path}")

    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*60)

    sample_df = pd.read_csv(list(output_paths.values())[0])
    non_feat = {"Participant_ID", "PHQ_Binary", "PHQ_Score"}
    feat_cols = [c for c in sample_df.columns if c not in non_feat]

    base_feats = [c for c in feat_cols if c.startswith("mfcc_base_")]
    delta_feats = [c for c in feat_cols if c.startswith("mfcc_delta_")]
    delta2_feats = [c for c in feat_cols if c.startswith("mfcc_delta2_")]
    egemap_feats = [c for c in feat_cols if c.startswith("egemap_")]

    print(f"\nTotal features: {len(feat_cols)}")
    print(f"  mfcc_base  features : {len(base_feats)}")
    print(f"  mfcc_delta features : {len(delta_feats)}")
    print(f"  mfcc_delta2 features: {len(delta2_feats)}")
    print(f"  egemap features     : {len(egemap_feats)}")
    print(f"\nFeature types per coefficient (×6 per coefficient):")
    print(f"  _seg_mean_std   : volatility of level across segments")
    print(f"  _seg_mean_range : peak-to-trough level shift")
    print(f"  _seg_std_mean   : average local variability")
    print(f"  _seg_std_std    : volatility of local variability")
    print(f"  _seg_trend      : temporal trend (rising/falling)")
    print(f"  _global_std     : overall variability (reference)")

    print(f"\nOutput files:")
    for split, path in output_paths.items():
        print(f"  {split:5s}: {path}")

