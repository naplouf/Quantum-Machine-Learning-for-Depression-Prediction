import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

GDRIVE     = Path.home() / "Library" / "CloudStorage" / \
             "your.email" / "My Drive" / "EDAIC"
DATA_DIR   = GDRIVE / "data"
LABELS_DIR = GDRIVE / "labels"
OUTPUT_DIR = GDRIVE

N_SEGMENTS = 5
N_PCA_COMPONENTS = 20

LABEL_FILES = {
    "train": LABELS_DIR / "train_split.csv",
    "dev"  : LABELS_DIR / "dev_split.csv",
    "test" : LABELS_DIR / "test_split.csv",
}
NON_FEAT_COLS = {"Participant_ID", "PHQ_Binary", "PHQ_Score"}

DEPRESSION_LEXICON = {
    "negation"    : {"no","not","never","nothing","nobody","nowhere","neither",
                     "cannot","can't","won't","don't","doesn't","didn't",
                     "isn't","aren't","wasn't","weren't","without"},
    "negative_affect": {"sad","unhappy","depressed","hopeless","worthless",
                        "miserable","terrible","awful","horrible","dreadful",
                        "empty","numb","tired","exhausted","drained","lonely",
                        "isolated","guilty","ashamed","useless","failure"},
    "positive_affect": {"happy","good","great","wonderful","love","enjoy",
                        "excited","hopeful","proud","grateful","peaceful",
                        "content","better","hope","positive","fun","laugh"},
    "cognitive"   : {"think","thought","know","understand","believe","feel",
                     "remember","forget","wonder","realize","seem","maybe",
                     "perhaps","possibly","probably","definitely","certain"},
    "social"      : {"people","person","family","friend","others","everyone",
                     "someone","anyone","they","we","us","together","alone"},
    "temporal_past": {"was","were","had","did","used","before","ago","used to",
                      "when i","back then","in the past","previously"},
    "certainty"   : {"always","never","every","all","none","definitely",
                     "certainly","absolutely","completely","totally"},
    "tentativeness": {"maybe","perhaps","sometimes","often","usually",
                      "might","could","would","should","guess","think"},
}

_vader    = None
_sbert    = None
_pca      = None

def get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader

def get_sbert():
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        print("Loading all-MiniLM-L6-v2...")
        _sbert = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Ready.")
    return _sbert

def load_utterances(pid: int) -> tuple[list[str], list[float], list[float], dict]:
    """
    Returns: (utterances, start_times, stop_times, structural_features)
    """
    path = DATA_DIR / f"{pid}_P" / f"{pid}_Transcript.csv"
    if not path.exists():
        return [], [], [], {}
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            return [], [], [], {}

    conf_col = next((c for c in df.columns if "conf" in c.lower()), None)
    if conf_col:
        df = df[pd.to_numeric(df[conf_col], errors="coerce") > 0.7]

    spk_col = next((c for c in df.columns
                    if c.lower() in ["speaker", "person", "who"]), None)
    if spk_col:
        df = df[df[spk_col].astype(str).str.upper().str.startswith("P")]

    text_col = next((c for c in df.columns
                     if c.lower() in ["value","text","utterance","transcript","word"]), None)
    if text_col is None:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        text_col = str_cols[-1] if str_cols else None
    if text_col is None:
        return [], [], [], {}

    utterances = df[text_col].dropna().astype(str).tolist()
    utterances = [u for u in utterances if len(u.strip()) > 2]
    if not utterances:
        return [], [], [], {}

    start_col = next((c for c in df.columns if "start" in c.lower()), None)
    stop_col  = next((c for c in df.columns
                      if "stop" in c.lower() or "end" in c.lower()), None)

    starts = pd.to_numeric(df[start_col], errors="coerce").tolist() if start_col else []
    stops  = pd.to_numeric(df[stop_col],  errors="coerce").tolist() if stop_col  else []

    words         = " ".join(utterances).lower().split()
    n_words       = len(words)
    n_utt         = len(utterances)
    speaking_ratio = 0.5
    pause_rate     = 0.0
    if starts and stops:
        try:
            s = np.array(starts[:len(stops)], dtype=float)
            e = np.array(stops[:len(starts)], dtype=float)
            n = min(len(s), len(e))
            dur = e[:n] - s[:n]
            speaking_ratio = dur.sum() / max(e[:n].max() - s[:n].min(), 1)
            if n > 1:
                gaps = s[1:n] - e[:n-1]
                pause_rate = (gaps > 1.0).sum() / max(n_utt, 1)
        except Exception:
            pass

    struct = {
        "txt_n_utterances"  : n_utt,
        "txt_n_words"       : n_words,
        "txt_vocab_richness": len(set(words)) / max(n_words, 1),
        "txt_avg_utt_len"   : n_words / max(n_utt, 1),
        "txt_speaking_ratio": speaking_ratio,
        "txt_pause_rate"    : pause_rate,
    }
    return utterances, starts, stops, struct

def sentiment_volatility_features(utterances: list[str]) -> dict:
    """
    Compute per-utterance sentiment, then volatility across segments.
    """
    vader  = get_vader()
    feats  = {}
    n      = len(utterances)

    compounds = np.array([vader.polarity_scores(u)["compound"] for u in utterances])
    positives = np.array([vader.polarity_scores(u)["pos"]      for u in utterances])
    negatives = np.array([vader.polarity_scores(u)["neg"]      for u in utterances])

    feats["sent_compound_mean"]  = float(compounds.mean())
    feats["sent_compound_std"]   = float(compounds.std())
    feats["sent_compound_range"] = float(compounds.max() - compounds.min())
    feats["sent_positive_mean"]  = float(positives.mean())
    feats["sent_negative_mean"]  = float(negatives.mean())
    feats["sent_neg_pos_ratio"]  = float(negatives.mean() / max(positives.mean(), 0.001))

    feats["sent_pct_negative"]   = float((compounds < -0.05).mean())
    feats["sent_pct_positive"]   = float((compounds >  0.05).mean())
    feats["sent_pct_neutral"]    = float((np.abs(compounds) <= 0.05).mean())

    if n > 1:
        x = np.arange(n, dtype=float)
        feats["sent_trend"] = float(np.polyfit(x, compounds, 1)[0])
    else:
        feats["sent_trend"] = 0.0

    if n >= N_SEGMENTS:
        segs         = np.array_split(np.arange(n), N_SEGMENTS)
        seg_means    = np.array([compounds[s].mean() for s in segs])
        feats["sent_seg_mean_std"]   = float(seg_means.std())
        feats["sent_seg_mean_range"] = float(seg_means.max() - seg_means.min())
        feats["sent_seg_trend"]      = float(
            np.polyfit(np.arange(N_SEGMENTS, dtype=float), seg_means, 1)[0])
        mid = N_SEGMENTS // 2
        feats["sent_second_half_drop"] = float(
            seg_means[mid:].mean() - seg_means[:mid].mean())
    else:
        feats["sent_seg_mean_std"]    = float(compounds.std())
        feats["sent_seg_mean_range"]  = float(compounds.max() - compounds.min())
        feats["sent_seg_trend"]       = 0.0
        feats["sent_second_half_drop"] = 0.0

    return feats

def liwc_features(utterances: list[str]) -> dict:
    """LIWC-style word category counts."""
    full_text = " ".join(utterances).lower()
    words     = full_text.split()
    n_words   = max(len(words), 1)
    feats     = {}
    for cat, lexicon in DEPRESSION_LEXICON.items():
        count = sum(1 for w in words if w in lexicon)
        feats[f"liwc_{cat}_rate"] = count / n_words
        feats[f"liwc_{cat}_count"] = count
    return feats

def sbert_pca_features(utterances: list[str], pca) -> dict:
    """Embed utterances, apply pre-fitted PCA, return mean+std of components."""
    model = get_sbert()
    emb   = model.encode(utterances, batch_size=64,
                          show_progress_bar=False, convert_to_numpy=True,
                          normalize_embeddings=False)
    if pca is not None:
        reduced = pca.transform(emb)
    else:
        reduced = emb[:, :N_PCA_COMPONENTS]

    feats = {}
    for d in range(reduced.shape[1]):
        feats[f"sbert_pca_mean_{d:02d}"] = float(reduced[:, d].mean())
        feats[f"sbert_pca_std_{d:02d}"]  = float(reduced[:, d].std())
    return feats

def fit_pca_on_train(train_label_df: pd.DataFrame) -> object:
    """Fit PCA on train utterance embeddings."""
    from sklearn.decomposition import PCA

    print("Fitting PCA on train utterance embeddings...")
    model    = get_sbert()
    all_embs = []

    for _, row in tqdm(train_label_df.iterrows(), total=len(train_label_df),
                       desc="PCA fit", ncols=75):
        pid        = int(row["Participant_ID"])
        utterances, _, _, _ = load_utterances(pid)
        if utterances:
            emb = model.encode(utterances, batch_size=64,
                               show_progress_bar=False, convert_to_numpy=True,
                               normalize_embeddings=False)
            all_embs.append(emb)

    if not all_embs:
        return None

    X = np.vstack(all_embs)
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    pca.fit(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA fitted on {X.shape[0]} utterances "
          f"({N_PCA_COMPONENTS} components explain {explained:.1%} variance)")
    return pca

def process_split(label_df: pd.DataFrame, split: str, pca) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"{split.upper()}: {len(label_df)} participants")

    rows = []
    for _, row in tqdm(label_df.iterrows(), total=len(label_df),
                       desc=split, ncols=75):
        pid        = int(row["Participant_ID"])
        phq_binary = int(row.get("PHQ_Binary", -1))
        phq_score  = float(row.get("PHQ_Score", float("nan")))

        utterances, _, _, struct = load_utterances(pid)

        base = {"Participant_ID": pid,
                "PHQ_Binary": phq_binary, "PHQ_Score": phq_score}

        if not utterances:
            base.update({"txt_n_utterances":0,"txt_n_words":0,
                         "txt_vocab_richness":0,"txt_avg_utt_len":0,
                         "txt_speaking_ratio":0.5,"txt_pause_rate":0})
        else:
            base.update(struct)
            base.update(sentiment_volatility_features(utterances))
            base.update(liwc_features(utterances))
            base.update(sbert_pca_features(utterances, pca))

        rows.append(base)

    result = pd.DataFrame(rows).fillna(0)
    n_feat = len([c for c in result.columns if c not in NON_FEAT_COLS])

    if "sent_compound_std" in result.columns:
        v = result["sent_compound_std"].var()
        print(f"  Variance check — sent_compound_std: {v:.6f} (should be > 0.001)")
    print(f"  ✓ {len(result)} participants | {n_feat} features")
    return result

def main():
    print("Text Feature Extraction — Sentiment Volatility + LIWC + SBERT PCA")
    print(f"Output: {OUTPUT_DIR}")
    print()

    missing = []
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        missing.append("vaderSentiment")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing.append("sentence-transformers")
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        missing.append("scikit-learn")

    if missing:
        print(f"Missing: {missing}")
        print("Run: pip install vaderSentiment sentence-transformers")
        return

    get_vader()
    get_sbert()

    train_labels = pd.read_csv(LABEL_FILES["train"])
    pca          = fit_pca_on_train(train_labels)

    for split, label_path in LABEL_FILES.items():
        if not label_path.exists():
            print(f"[WARN] Missing: {label_path}")
            continue
        label_df = pd.read_csv(label_path)
        result   = process_split(label_df, split, pca)
        out_path = OUTPUT_DIR / f"sbert_text_{split}.csv"
        result.to_csv(out_path, index=False)
        print(f"  Saved → {out_path.name}")

    

if __name__ == "__main__":
    main()
