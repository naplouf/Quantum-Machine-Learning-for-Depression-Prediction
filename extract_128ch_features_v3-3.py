import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal as sp_signal, io as sio
from scipy.stats import ttest_ind
from tqdm import tqdm
import itertools

warnings.filterwarnings("ignore")

EEG_DIR    = Path.home() / "Downloads" / "EEG_128channels_resting_lanzhou_2015"
OUTPUT_DIR = Path.home() / "Library" / "CloudStorage" / \
             "GoogleDrive-ohvevo2014@gmail.com" / "My Drive" / "EDAIC"

FS       = 250
HIGHPASS = 1.0
LOWPASS  = 40.0

BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta" : (13, 30),
}

# 16 clinically relevant electrode indices (0-indexed from 128-ch HCGSN)
# These map to standard 10-20 positions
CLINICAL_ELECTRODES = {
    "F3" : 23,   "F4" : 123,
    "F7" : 32,   "F8" : 121,
    "C3" : 35,   "C4" : 103,  "Cz": 54,
    "P3" : 51,   "P4" : 91,   "Pz": 61,
    "T3" : 44,   "T4" : 107,
    "T5" : 57,   "T6" : 95,
    "O1" : 69,   "O2" : 82,
}
ELEC_NAMES = list(CLINICAL_ELECTRODES.keys())
ELEC_IDX   = [CLINICAL_ELECTRODES[n] for n in ELEC_NAMES]
N_ELEC     = len(ELEC_NAMES)  # 16


def load_robust(fpath: Path) -> np.ndarray | None:

    try:
        mat = sio.loadmat(str(fpath))
        # Find the EEG array: must be 2D, contain 128 in one dim, long in other
        data = None
        for k, v in mat.items():
            if k.startswith('_') or k in ('samplingRate', 'Impedances_0'):
                continue
            if not (hasattr(v, 'shape') and len(v.shape) == 2):
                continue
            if (128 in v.shape or 129 in v.shape) and max(v.shape) > 2000:
                arr = v.astype(np.float32)
                # Enforce channels-first: (128or129, T)
                if arr.shape[0] in (128, 129) and arr.shape[1] > arr.shape[0]:
                    data = arr[:128, :]   # take first 128 rows, drop Cz ref
                elif arr.shape[1] in (128, 129) and arr.shape[0] > arr.shape[1]:
                    data = arr.T[:128, :]
                break

        if data is None:
            print(f"  [SKIP] {fpath.name}: no valid EEG array found")
            return None

        return data[:128, :]  # ensure exactly 128 channels

    except Exception as e:
        print(f"  [ERROR] {fpath.name}: {e}")
        return None


def preprocess(data: np.ndarray) -> np.ndarray:
    """Remove DC, bandpass 1-40Hz, z-score normalize per channel."""
    data = data.copy().astype(np.float64)
    data -= data.mean(axis=1, keepdims=True)
    b, a = sp_signal.butter(4, [HIGHPASS, LOWPASS], btype='bandpass', fs=FS)
    for ch in range(data.shape[0]):
        data[ch] = sp_signal.filtfilt(b, a, data[ch])
    # Z-score normalize
    std = data.std(axis=1, keepdims=True) + 1e-8
    return (data / std).astype(np.float32)


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    b, a = sp_signal.butter(4, [lo, hi], btype='bandpass', fs=FS)
    return sp_signal.filtfilt(b, a, x)


def compute_pli(x: np.ndarray, y: np.ndarray) -> float:
    """
    Phase Lag Index between two signals.
    PLI = |mean(sign(imag(cross-spectrum)))|
    Range [0, 1]: 0 = no coupling, 1 = perfect phase coupling.
    Immune to volume conduction (zero-lag artifacts).
    """
    # Analytic signal via Hilbert transform
    hx = np.imag(sp_signal.hilbert(x))
    hy = np.imag(sp_signal.hilbert(y))
    # Cross-spectrum imaginary part sign
    pli = np.abs(np.mean(np.sign(hx * np.sign(hy) - hy * np.sign(hx))))
    return float(pli)


def pli_features(data: np.ndarray) -> dict:
    """
    PLI connectivity between all pairs of 16 clinical electrodes,
    computed separately for each frequency band.

    16 electrodes → 16×15/2 = 120 pairs × 4 bands = 480 PLI features
    """
    feats = {}
    # Extract only the 16 clinical channels
    ch_data = data[ELEC_IDX, :]  # (16, N)

    for band, (lo, hi) in BANDS.items():
        # Bandpass filter each channel for this band
        band_data = np.array([bandpass(ch_data[i], lo, hi)
                               for i in range(N_ELEC)])

        # Compute PLI for all pairs
        for i, j in itertools.combinations(range(N_ELEC), 2):
            pli_val = compute_pli(band_data[i], band_data[j])
            name_i  = ELEC_NAMES[i]
            name_j  = ELEC_NAMES[j]
            feats[f"pli_{band}_{name_i}_{name_j}"] = pli_val

    return feats  # 120 pairs × 4 bands = 480 features


def band_power_clinical(data: np.ndarray) -> dict:
    """Band power for 16 clinical electrodes × 4 bands = 64 features."""
    feats    = {}
    ch_data  = data[ELEC_IDX, :]
    for i, name in enumerate(ELEC_NAMES):
        f, psd = sp_signal.welch(ch_data[i], fs=FS, nperseg=512)
        powers = {band: float(np.trapezoid(psd[np.logical_and(f>=lo, f<=hi)],
                                            f[np.logical_and(f>=lo, f<=hi)]))
                  for band, (lo, hi) in BANDS.items()}
        total  = sum(powers.values()) or 1.0
        for band, pw in powers.items():
            feats[f"bp_{name}_{band}_abs"] = float(np.log1p(pw))
            feats[f"bp_{name}_{band}_rel"] = pw / total
        # Theta/alpha ratio (depression biomarker)
        feats[f"bp_{name}_theta_alpha"] = (powers["theta"] /
                                            (powers["alpha"] + 1e-10))
    return feats  # 16 × (4+4+1) = 144 features


def hjorth_clinical(data: np.ndarray) -> dict:
    """Hjorth parameters for 16 clinical electrodes = 48 features."""
    feats   = {}
    ch_data = data[ELEC_IDX, :]
    for i, name in enumerate(ELEC_NAMES):
        x     = ch_data[i].astype(float)
        d1    = np.diff(x)
        d2    = np.diff(d1)
        act   = float(np.var(x))
        mob   = float(np.sqrt(np.var(d1) / (np.var(x) + 1e-10)))
        comp  = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-10)) / (mob + 1e-10))
        feats[f"hj_{name}_activity"]   = float(np.log1p(act))
        feats[f"hj_{name}_mobility"]   = mob
        feats[f"hj_{name}_complexity"] = comp
    return feats  # 16 × 3 = 48 features


def asymmetry_features(data: np.ndarray) -> dict:
    """Frontal/parietal/temporal L-R asymmetry = 12 features."""
    feats = {}
    pairs = [
        ("F3","F4"), ("F7","F8"),
        ("C3","C4"),
        ("P3","P4"),
        ("T3","T4"), ("T5","T6"),
    ]
    for lname, rname in pairs:
        li = CLINICAL_ELECTRODES[lname]
        ri = CLINICAL_ELECTRODES[rname]
        fl, pl = sp_signal.welch(data[li], fs=FS, nperseg=512)
        fr, pr = sp_signal.welch(data[ri], fs=FS, nperseg=512)
        for band, (lo, hi) in BANDS.items():
            p_l = float(np.trapezoid(pl[np.logical_and(fl>=lo,fl<=hi)],
                                      fl[np.logical_and(fl>=lo,fl<=hi)]))
            p_r = float(np.trapezoid(pr[np.logical_and(fr>=lo,fr<=hi)],
                                      fr[np.logical_and(fr>=lo,fr<=hi)]))
            feats[f"asym_{lname}_{rname}_{band}"] = (
                float(np.log1p(p_r) - np.log1p(p_l)))
    return feats  # 6 pairs × 4 bands = 24 features


def get_label(fname: str):
    stem   = Path(fname).stem
    digits = ''.join(filter(str.isdigit, stem[:8]))
    try:
        sid = int(digits)
        if str(sid).startswith('201'):    return sid, 1
        if str(sid).startswith('203') or str(sid).startswith('202'):
            return sid, 0
    except Exception:
        pass
    return None, None


def main():
    print("MODMA 128-Channel EEG Feature Extraction v3 — PLI + Classical")
    print(f"EEG_DIR   : {EEG_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"Clinical electrodes: {ELEC_NAMES}")
    print(f"PLI pairs: {N_ELEC*(N_ELEC-1)//2} × 4 bands = "
          f"{N_ELEC*(N_ELEC-1)//2*4} features")
    print()

    mat_files = sorted(EEG_DIR.glob("*.mat"))
    print(f"Found {len(mat_files)} .mat files")

    rows, missing = [], []

    for fpath in tqdm(mat_files, desc="Extracting", ncols=75):
        sid, label = get_label(fpath.name)
        if sid is None:
            print(f"  [WARN] Cannot parse: {fpath.name}")
            continue

        raw = load_robust(fpath)
        if raw is None:
            missing.append(sid)
            continue

        # Verify shape
        if raw.shape[0] != 128 or raw.shape[1] < 2000:
            print(f"  [SKIP] {sid}: bad shape {raw.shape}")
            missing.append(sid)
            continue

        data = preprocess(raw)

        feats = {}
        feats.update(pli_features(data))
        feats.update(band_power_clinical(data))
        feats.update(hjorth_clinical(data))
        feats.update(asymmetry_features(data))

        row = {"subject_id": sid, "label": label,
               "type": "MDD" if label == 1 else "HC"}
        row.update(feats)
        rows.append(row)

    result   = pd.DataFrame(rows)
    feat_cols = [c for c in result.columns
                 if c not in {"subject_id","label","type"}]

    print(f"\n{'='*60}")
    print(f"Extracted : {len(result)} subjects | {len(feat_cols)} features")
    print(f"Missing   : {len(missing)}")
    print(f"Balance   : MDD={result['label'].sum()}, "
          f"HC={(result['label']==0).sum()}")

    # Top discriminative features
    mdd   = result[result['label']==1]
    hc    = result[result['label']==0]
    pvals = []
    for c in feat_cols:
        _, p = ttest_ind(mdd[c].dropna(), hc[c].dropna())
        d    = abs(mdd[c].mean()-hc[c].mean()) / (
               np.sqrt((mdd[c].std()**2+hc[c].std()**2)/2) + 1e-10)
        pvals.append((c, p, d))
    pvals.sort(key=lambda x: x[1])
    print("\nTop 10 discriminative features:")
    for c, p, d in pvals[:10]:
        print(f"  {c:50s} p={p:.4f} d={d:.3f}")

    out_path = OUTPUT_DIR / "modma_128ch_pli_features.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print('  eeg_df = pd.read_csv(BASE_SAVE / "modma_128ch_pli_features.csv")')


if __name__ == "__main__":
    main()
