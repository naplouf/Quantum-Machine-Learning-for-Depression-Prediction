# Quantum Machine Learning for the Early Detection of Depression

A multi-modal, multi-dataset study testing whether **quantum kernel methods** can outperform classical machine learning and deep learning at detecting depression — across speech, text, facial, and EEG signals.

The core idea: clinical datasets for depression are small (usually under 100 subjects), which is exactly where deep learning struggles. Quantum kernels don't learn representations from data — their expressive power comes from the circuit architecture — so in principle they need fewer samples. This project tests that idea across three datasets.



---

## Headline result

A Quantum Support Vector Classifier (QSVC) with a ZZFeatureMap quantum kernel was the best-performing model on **all three datasets**, beating matched classical baselines every time.

| Dataset | Best classical | Deep learning | **QSVC** |
|---|---|---|---|
| E-DAIC (speech/text/facial) | 0.731 (fusion) | — | **0.750** |
| MODMA 3-channel EEG | 0.703 (SVM Linear) | — | **0.733** |
| MODMA 128-channel EEG | 0.788 (SVM+SMOTE) | 0.620 (EEGNet) | **0.844** |

*Scores are Unweighted Average Recall (UAR), chosen because it treats both classes equally under imbalance.*

On the 128-channel EEG data, QSVC also reached **AUC = 1.000**, versus 0.976 for the best classical model.

---

## Datasets

| | E-DAIC | MODMA 3-channel | MODMA 128-channel |
|---|---|---|---|
| Subjects | 275 | 55 (26 MDD / 29 HC) | 53 (24 MDD / 29 HC) |
| Signal | Speech, text, facial | Prefrontal EEG | Full-scalp EEG |
| Labels | PHQ-8 (≥10 = depressed) | Clinical diagnosis | Clinical diagnosis |
| Evaluation | Official train/dev/test split | Leave-one-out CV | Leave-one-out CV |

Neither dataset is redistributed here — both require their own access agreements ([E-DAIC](https://dcapswoz.ict.usc.edu/), [MODMA](http://modma.lzu.edu.cn/data/index/)).

---

## What was compared

**Quantum:** QSVC with ZZFeatureMap (entangled) and FidelityQuantumKernel, via Qiskit's statevector simulator. ZFeatureMap (no entanglement) served as an ablation baseline. A Variational Quantum Circuit was also implemented as a quantum neural network counterpart.

**Classical:** Logistic Regression, SVM (linear and RBF), Random Forest, Gradient Boosting, MLP — plus a **matched SVM control** trained on the exact same 40-sample balanced subset and same selected features as the QSVC, differing only in the kernel. This is what isolates the quantum kernel's contribution from everything else.

**Deep learning:** EEGNet with self-supervised pre-training and two-phase fine-tuning.

---

## Main findings

**Entanglement is what matters.** Every entangled ZZFeatureMap configuration beat the non-entangled ZFeatureMap baseline (0.720). Quantum *encoding* alone isn't enough — the pairwise entangling gates are doing the work.

**Shallower circuits generalize better.** Performance decreased monotonically with circuit depth: reps=1 (0.844) → reps=2 (0.802) → reps=3 (0.786). On a 40-sample training subset, deeper circuits overfit.

**Structural alignment hypothesis.** Phase Lag Index (PLI) connectivity features consistently outperformed single-electrode features (band power, asymmetry). The proposed explanation: PLI features encode *pairwise* relationships between electrodes, and the ZZFeatureMap encodes *pairwise* feature interactions through its entangling gates. When data geometry matches circuit geometry, the quantum kernel has an edge. This suggests a principled rule for when to reach for quantum kernels rather than trial and error.

**Quantum beats deep learning on small data.** EEGNet (0.620) fell well short of both the classical ceiling and QSVC on n=53, even with self-supervised pre-training — a direct demonstration of the data-efficiency gap.

**The top features are neurologically plausible.** 10 of the 15 highest effect-size features were PLI connectivity measures, led by beta-band F8↔O2 and delta-band F3↔F7 — connectivity patterns independently associated with MDD in the literature.

---

## Honest limitations

- **Simulated, not real hardware.** All quantum results come from Qiskit's noiseless statevector simulator. Gate noise and decoherence on actual devices could erode the advantage.
- **No statistical significance.** Wilcoxon signed-rank tests returned p = 0.625–0.725. With only 5 comparable folds, the minimum achievable p-value is 0.0625 — significance is mathematically out of reach regardless of effect size. The advantage is *directionally consistent* across all five comparisons, which carries evidential weight, but it is not statistically proven.
- **Balanced subsets.** QSVC used 40-subject balanced subsets due to the O(n²) cost of quantum kernel computation.
- **No cross-dataset validation.** Every model was trained and tested within a single dataset.
- **E-DAIC raw audio was unavailable**, so speech features came from pre-extracted COVAREP rather than modern audio models — a ceiling on the speech results.
- **VQC results pending** at time of writing.

---

## Repository structure

```
Capstone_Code/
├── extract_eeg_features_v2.py           # MODMA 3-channel feature extraction
├── extract_128ch_features_v3-3.py       # MODMA 128-channel: PLI, band power, Hjorth, asymmetry
├── extract_segment_features.py          # E-DAIC speech volatility features
├── extract_sentence_features-3.py       # E-DAIC text features (TF-IDF/LSA, sentiment)
├── modma_pipeline-2.ipynb               # MODMA 3-channel: classical + quantum
├── modma_quantum_first-4.ipynb          # MODMA 128-channel: quantum-first feature search
├── edaic_multimodal_pipeline-3.ipynb    # E-DAIC: text, facial, late fusion
├── finalized_edaic_pipeline_notebook-2.ipynb  # E-DAIC: final publication pipeline
└── eegnet_transfer_colab-2.ipynb        # EEGNet transfer learning baseline
```

Feature extraction scripts run first and write feature tables; the notebooks consume those tables.

---

## Requirements

Python 3.9+, with `qiskit`, `qiskit-machine-learning`, `scikit-learn`, `mne`, `numpy`, `pandas`, `scipy`, `imbalanced-learn`, `sentence-transformers`, `vaderSentiment`, `torch`, `matplotlib`, `seaborn`, `tqdm`.

The EEGNet notebook was run on a Google Colab A100 GPU.

---


