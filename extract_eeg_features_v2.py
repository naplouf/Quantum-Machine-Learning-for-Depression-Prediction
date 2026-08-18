import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

EEG_DIR    = Path.home() / "Downloads" / "EEG_3channels_resting_lanzhou_2015"
EXCEL_PATH = EEG_DIR / "subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx"
OUTPUT_DIR = Path.home() / "Library" / "CloudStorage" / \
             "your.email" / "My Drive" / "EDAIC"

FS        = 250
N_SAMPLES = 22500
HIGHPASS  = 1.0
LOWPASS   = 40.0
CHANNELS  = ["Fp1", "Fpz", "Fp2"]
BANDS     = {"delta":(1,4), "theta":(4,8), "alpha":(8,13), "beta":(13,30)}


def preprocess(raw: np.ndarray) -> np.ndarray:
    n    = min(raw.shape[0], N_SAMPLES)
    data = raw[:n, :3].copy().astype(float)
    if n < N_SAMPLES:
        data = np.pad(data, ((0, N_SAMPLES-n),(0,0)), mode='constant')
    data -= data.mean(axis=0)
    b, a  = signal.butter(4, [HIGHPASS, LOWPASS], btype='bandpass', fs=FS)
    for ch in range(3):
        data[:, ch] = signal.filtfilt(b, a, data[:, ch])
    return data


def band_power(f, psd, fmin, fmax):
    idx = np.logical_and(f >= fmin, f <= fmax)
    return float(np.trapezoid(psd[idx], f[idx]))


def hjorth_parameters(x: np.ndarray) -> tuple:
    activity   = float(np.var(x))
    diff1      = np.diff(x)
    diff2      = np.diff(diff1)
    mobility   = float(np.sqrt(np.var(diff1) / (np.var(x) + 1e-10)))
    complexity = float(np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10))
    return activity, mobility, complexity


def spectral_slope(f, psd, fmin=1, fmax=40) -> float:

    idx  = np.logical_and(f >= fmin, f <= fmax)
    f_   = f[idx]
    psd_ = psd[idx]
    mask = (f_ > 0) & (psd_ > 0)
    if mask.sum() < 3:
        return 0.0
    slope, _, _, _, _ = stats.linregress(np.log(f_[mask]), np.log(psd_[mask]))
    return float(slope)


def extract_all_features(eeg_path: Path) -> dict | None:
    try:
        raw = np.loadtxt(str(eeg_path))
        if raw.ndim == 1:
            return None
        if raw.shape[1] == 3:
            pass
        elif raw.shape[0] == 8:
            raw = raw[:3, :].T
        elif raw.shape[1] == 8:
            raw = raw[:, :3]
        data = preprocess(raw)
    except Exception as e:
        print(f"    [ERROR] {eeg_path.name}: {e}")
        return None

    feats = {}
    psds  = {}
    freqs = {}

    # ── Per-channel features ──────────────────────────────────────────────────
    for ci, ch in enumerate(CHANNELS):
        x       = data[:, ci]
        f, psd  = signal.welch(x, fs=FS, nperseg=512)
        psds[ch] = psd
        freqs[ch] = f

        # Band powers
        powers = {band: band_power(f, psd, lo, hi)
                  for band, (lo, hi) in BANDS.items()}
        total  = sum(powers.values()) or 1.0

        for band, pw in powers.items():
            feats[f"eeg_{ch}_{band}_rel"] = pw / total
            feats[f"eeg_{ch}_{band}_abs"] = float(np.log1p(pw))

        # Spectral stats
        feats[f"eeg_{ch}_psd_max"]    = float(psd.max())
        feats[f"eeg_{ch}_psd_mean"]   = float(psd.mean())
        feats[f"eeg_{ch}_psd_center"] = float(np.sum(f * psd) / (np.sum(psd) + 1e-10))
        feats[f"eeg_{ch}_psd_slope"]  = spectral_slope(f, psd)  
        # Hjorth parameters
        act, mob, comp = hjorth_parameters(x)
        feats[f"eeg_{ch}_hjorth_activity"]   = float(np.log1p(act))
        feats[f"eeg_{ch}_hjorth_mobility"]   = mob
        feats[f"eeg_{ch}_hjorth_complexity"] = comp

        # Band ratios
        alpha = powers["alpha"]
        theta = powers["theta"]
        beta  = powers["beta"]
        delta = powers["delta"]
        feats[f"eeg_{ch}_theta_alpha_ratio"] = theta / (alpha + 1e-10)
        feats[f"eeg_{ch}_beta_alpha_ratio"]  = beta  / (alpha + 1e-10)
        feats[f"eeg_{ch}_delta_alpha_ratio"] = delta / (alpha + 1e-10)

        # Nonlinear
        counts, _ = np.histogram(x, bins=64)
        p = counts / (counts.sum() + 1e-10)
        p = p[p > 0]
        feats[f"eeg_{ch}_renyi"]     = float((1/(1-2)) * np.log(np.sum(p**2)))
        feats[f"eeg_{ch}_c0"]        = _c0_complexity(x)
        seg_vars = np.array([s.var() for s in np.array_split(x, 10)])
        feats[f"eeg_{ch}_var_ratio"] = float(seg_vars.std() / (seg_vars.mean() + 1e-10))

    for ch in CHANNELS:
        feats[f"eeg_alpha_{ch}"] = feats[f"eeg_{ch}_alpha_abs"]
    feats["eeg_alpha_total"] = sum(feats[f"eeg_alpha_{ch}"] for ch in CHANNELS)

    for band in BANDS:
        p1 = feats[f"eeg_Fp1_{band}_abs"]
        p2 = feats[f"eeg_Fp2_{band}_abs"]
        feats[f"eeg_asym_{band}"]       = p2 - p1          # ln(Fp2) - ln(Fp1)
        feats[f"eeg_power_ratio_{band}"] = p2 / (p1 + 1e-10)  

   
    pairs = [("Fp1","Fpz",0,1), ("Fp1","Fp2",0,2), ("Fpz","Fp2",1,2)]
    for c1, c2, i1, i2 in pairs:
        f_, coh = signal.coherence(data[:,i1], data[:,i2], fs=FS, nperseg=512)
        corr = float(np.corrcoef(data[:,i1], data[:,i2])[0,1])
        feats[f"eeg_corr_{c1}_{c2}"] = corr
        for band, (lo, hi) in BANDS.items():
            idx = np.logical_and(f_ >= lo, f_ <= hi)
            feats[f"eeg_coh_{band}_{c1}_{c2}"] = float(coh[idx].mean()) 

    return feats


def _c0_complexity(x: np.ndarray) -> float:
    N  = len(x)
    Xk = np.fft.fft(x)
    G  = np.mean(np.abs(Xk)**2)
    Yk = np.where(np.abs(Xk)**2 > G, Xk, 0)
    yn = np.fft.ifft(Yk).real
    num   = np.sum((x - yn)**2)
    denom = np.sum(x**2)
    return float(num / denom) if denom > 0 else 0.0


def main():
    print("MODMA EEG Feature Extraction v2 (extended features)")
    print(f"EEG_DIR   : {EEG_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    if not EXCEL_PATH.exists():
        print(f"[ERROR] Excel file not found: {EXCEL_PATH}")
        return
    if not EEG_DIR.exists():
        print(f"[ERROR] EEG_DIR not found: {EEG_DIR}")
        return

    info_df = pd.read_excel(EXCEL_PATH)
    info_df = info_df[["subject id","type","PHQ-9"]].dropna(subset=["type"])
    info_df["label"] = (info_df["type"] == "MDD").astype(int)

    txt_files = list(EEG_DIR.glob("*_still.txt"))
    print(f"Found {len(txt_files)} EEG files | "
          f"{(info_df['type']=='MDD').sum()} MDD, {(info_df['type']=='HC').sum()} HC")

    rows    = []
    missing = []

    for _, row in tqdm(info_df.iterrows(), total=len(info_df), ncols=75):
        sid   = int(row["subject id"])
        fname = f"0{sid}_still.txt"
        fpath = EEG_DIR / fname
        if not fpath.exists():
            matches = [f for f in txt_files if str(sid) in f.name]
            fpath   = matches[0] if matches else None
        if fpath is None:
            missing.append(sid)
            continue

        feats = extract_all_features(fpath)
        if feats is None:
            missing.append(sid)
            continue

        row_data = {"subject_id": sid, "label": int(row["label"]),
                    "PHQ9": float(row["PHQ-9"]), "type": row["type"]}
        row_data.update(feats)
        rows.append(row_data)

    result  = pd.DataFrame(rows)
    n_feat  = len([c for c in result.columns
                   if c not in {"subject_id","label","PHQ9","type"}])

    print(f"\nExtracted: {len(result)} subjects | {n_feat} features")
    print(f"Missing  : {len(missing)}")
    print(f"Balance  : MDD={result['label'].sum()}, HC={(result['label']==0).sum()}")

    from scipy.stats import ttest_ind
    feat_cols = [c for c in result.columns
                 if c not in {"subject_id","label","PHQ9","type"}]
    mdd = result[result['label']==1]
    hc  = result[result['label']==0]
    pvals = [(c, ttest_ind(mdd[c], hc[c])[1]) for c in feat_cols]
    pvals.sort(key=lambda x: x[1])
    print("\nTop 5 discriminative features:")
    for c, p in pvals[:5]:
        print(f"  {c:45s} p={p:.4f}")

    out_path = OUTPUT_DIR / "modma_eeg_features.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")



if __name__ == "__main__":
    main()
