"""Class-balanced LightGBM on handcrafted radar features (+ isotonic calibration)
vs. an MLP on raw normalised amplitude/phase samples, for detecting jamming in
simulated radar pulse returns. Main metric: macro-F1 on the unseen jammer type."""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import kurtosis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]     # model seeds; the data split is fixed at SEED
N_PULSES = 40000         # simulated pulses (small, so the run takes seconds)
N = 256                  # complex samples per pulse
L_TGT = 64               # target echo length (samples)
SEEN, UNSEEN = ["barrage", "spoof"], "chirp"   # chirp jammer is held out
t = np.arange(N)


def simulate(n, rng):
    """Simulate n pulses: target echo + noise, optionally jammed.
    Returns complex samples, labels (1 = jammed), jammer kind, and JTR in dB (NaN if clean)."""
    X = np.zeros((n, N), complex)
    y = np.zeros(n, int)
    kind = np.array(["none"] * n, dtype=object)
    jtr = np.full(n, np.nan)
    for i in range(n):
        snr = 10 ** (rng.uniform(0, 15) / 10)            # target SNR (linear)
        r = rng.integers(0, N - L_TGT)                    # random range gate
        fd = rng.uniform(-0.4, 0.4)                       # normalised Doppler
        echo = np.zeros(N, complex)
        echo[r:r + L_TGT] = np.sqrt(snr) * np.exp(2j * np.pi * fd * t[r:r + L_TGT])
        x = echo + (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
        if rng.random() < 0.5:                            # jammed pulse
            k = rng.choice(SEEN + [UNSEEN])
            kind[i], y[i] = k, 1
            jtr[i] = rng.uniform(-10, 20)                 # jammer-to-target ratio (dB)
            pj = snr * 10 ** (jtr[i] / 10)                # jammer power
            if k == "barrage":                            # wideband Gaussian noise jammer
                x = x + np.sqrt(pj / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
            elif k == "spoof":                            # two delayed false-target tones
                for d in rng.integers(0, N - L_TGT, 2):
                    seg = np.zeros(N, complex)
                    f = fd + rng.uniform(-0.05, 0.05)
                    seg[d:d + L_TGT] = np.sqrt(pj * N / (2 * L_TGT)) * np.exp(2j * np.pi * f * t[d:d + L_TGT])
                    x = x + seg
            else:                                         # swept-frequency (chirp) jammer, unseen
                f0, B = rng.uniform(-0.5, 0.5), rng.uniform(0.3, 0.5)
                x = x + np.sqrt(pj) * np.exp(2j * np.pi * (f0 * t + 0.5 * B / N * t ** 2))
        X[i] = x
    return X, y, kind, jtr


def features(X):
    """Handcrafted spectral and statistical features per pulse (vectorised)."""
    p = np.abs(X) ** 2
    amp = np.sqrt(p)
    S = np.abs(np.fft.fft(X, axis=1)) ** 2 + 1e-12    # power spectrum
    Pn = S / S.sum(1, keepdims=True)
    ac = np.column_stack([np.abs((X[:, :-k] * X[:, k:].conj()).sum(1)) / (p.sum(1) + 1e-12)
                          for k in range(1, 17)])     # autocorrelation, lags 1..16
    dphi = np.angle(X[:, 1:] * X[:, :-1].conj())      # instantaneous phase increments
    return np.column_stack([
        np.log10(p.mean(1) + 1e-12),                  # log mean power
        kurtosis(amp, axis=1), kurtosis(X.real, axis=1), kurtosis(X.imag, axis=1),
        amp.std(1) / (amp.mean(1) + 1e-12),           # envelope coefficient of variation
        np.exp(np.log(S).mean(1)) / S.mean(1),        # spectral flatness
        S.max(1) / np.median(S, 1),                   # FFT peak-to-median ratio
        np.sort(S, 1)[:, -N // 20:].sum(1) / S.sum(1),  # energy fraction in top 5% bins
        -(Pn * np.log(Pn)).sum(1) / np.log(N),        # normalised spectral entropy
        ac[:, 0], ac.max(1), ac.mean(1),              # autocorrelation summaries
        dphi.std(1), kurtosis(dphi, axis=1),          # instantaneous-frequency variation
    ])


def raw_inputs(X):
    """Baseline input: raw normalised amplitude and phase samples (512 values)."""
    amp = np.abs(X)
    rms = np.sqrt((amp ** 2).mean(1, keepdims=True)) + 1e-12
    return np.hstack([amp / rms, np.angle(X) / np.pi])


def fit_model(name, seed, Xtr, ytr):
    """Tune with 3-fold CV on the training set only, then isotonic calibration."""
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    if name == "baseline":   # standard MLP on raw samples
        est = make_pipeline(StandardScaler(),
                            MLPClassifier((64, 32), max_iter=100, early_stopping=True, random_state=seed))
        grid = {"mlpclassifier__alpha": [1e-4, 1e-2]}
    else:                    # class-balanced LightGBM on handcrafted features
        est = LGBMClassifier(class_weight="balanced", subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=seed, verbose=-1)
        grid = {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}
    gs = GridSearchCV(est, grid, cv=cv, scoring="neg_log_loss").fit(Xtr, ytr)
    print(f"  [{name}, seed {seed}] CV-selected params: {gs.best_params_}")
    return CalibratedClassifierCV(gs.best_estimator_, method="isotonic", cv=3).fit(Xtr, ytr)


def evaluate(model, Xte, yte, unseen_m, low_m):
    """Macro-F1 on unseen-jammer slice (main), plus secondary metrics."""
    p = model.predict_proba(Xte)[:, 1]
    pred = (p > 0.5).astype(int)
    return {
        "f1": f1_score(yte[unseen_m], pred[unseen_m], average="macro"),
        "f1_all": f1_score(yte, pred, average="macro"),
        "f1_low": f1_score(yte[low_m], pred[low_m], average="macro"),
        "auc": roc_auc_score(yte[unseen_m], p[unseen_m]),
        "logloss": log_loss(yte[unseen_m], p[unseen_m], labels=[0, 1]),
    }


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("Simulating pulses...")
    X, y, kind, jtr = simulate(N_PULSES, rng)

    # ONE split: seen-jammer pulses split 75/25 (stratified); all unseen-jammer pulses go to test
    seen_idx = np.where(kind != UNSEEN)[0]
    tr_idx, te_seen = train_test_split(seen_idx, test_size=0.25, stratify=y[seen_idx], random_state=SEED)
    te_idx = np.concatenate([te_seen, np.where(kind == UNSEEN)[0]])
    k_te = kind[te_idx]
    unseen_m = (k_te == "none") | (k_te == UNSEEN)               # unseen-jammer evaluation slice
    low_m = unseen_m & (np.isnan(jtr[te_idx]) | (jtr[te_idx] < 0))  # low-JTR slice
    print(f"Train {len(tr_idx)}, test {len(te_idx)} (unseen slice {unseen_m.sum()}, low-JTR {low_m.sum()})")

    inputs = {"baseline": (raw_inputs(X[tr_idx]), raw_inputs(X[te_idx])),
              "improved": (features(X[tr_idx]), features(X[te_idx]))}

    # Fit everything first; the test set is touched only once, afterwards
    models = {m: [fit_model(m, s, inputs[m][0], y[tr_idx]) for s in SEEDS] for m in inputs}

    res = {m: [evaluate(mdl, inputs[m][1], y[te_idx], unseen_m, low_m) for mdl in models[m]]
           for m in inputs}
    metrics = {}
    for m in res:
        vals = np.array([r["f1"] for r in res[m]])
        metrics[f"{m}_score"] = float(vals.mean())
        metrics[f"{m}_score_std"] = float(vals.std())
        metrics[f"{m}_f1_all_test"] = float(np.mean([r["f1_all"] for r in res[m]]))
        metrics[f"{m}_f1_lowjtr"] = float(np.mean([r["f1_low"] for r in res[m]]))
        metrics[f"{m}_roc_auc_unseen"] = float(np.mean([r["auc"] for r in res[m]]))
        metrics[f"{m}_logloss_unseen"] = float(np.mean([r["logloss"] for r in res[m]]))
        print(f"{m}: unseen macro-F1 = {metrics[f'{m}_score']:.4f} +/- {metrics[f'{m}_score_std']:.4f}")
    metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]

    vals = [v for v in metrics.values()]
    if not all(np.isfinite(vals)) or not (0 <= metrics["baseline_score"] <= 1):
        print("Could not produce meaningful metrics"); sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5, 4))
    names = ["baseline", "improved"]
    plt.bar(names, [metrics[f"{n}_score"] for n in names],
            yerr=[metrics[f"{n}_score_std"] for n in names], capsize=6, color=["grey", "tab:blue"])
    plt.ylabel("Macro-F1 (unseen jammer)"); plt.title("Baseline vs improved (mean over seeds)")
    plt.tight_layout(); plt.savefig("figures/main_metric_bar.png", dpi=120); plt.close()

    results = {
        "question": "Can a class-balanced LightGBM detector on spectral and statistical features of simulated radar returns beat a standard MLP deep-learning-style detector at flagging jamming, especially when the jammer type is unseen during training?",
        "method": "Simulated 256-sample radar pulses (target echo + noise, barrage and spoofing jamming in training; chirp jamming held out for test, plus a low-JTR slice). Baseline MLP on raw normalised amplitude/phase vs. CV-tuned class-balanced LightGBM on handcrafted features with isotonic calibration; 3 model seeds, one fixed data split.",
        "dataset": {
            "name": "Simulated radar pulse returns (numpy, complex baseband, 256 samples/pulse)",
            "source_url": "https://numpy.org/ (synthetic, generated at run time)",
            "n_rows": N_PULSES, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 on held-out (unseen) jammer type, threshold 0.5",
        "higher_is_better": True,
        "baseline": "MLPClassifier (64,32) on raw normalised amplitude and phase samples, standard-scaled; alpha tuned by 3-fold CV; isotonic calibration.",
        "improved": "LightGBM (class_weight='balanced') on handcrafted spectral/statistical features (spectral flatness, FFT peak ratios, kurtosis, autocorrelation summaries, phase-increment stats); num_leaves/learning_rate tuned by 3-fold CV; isotonic calibration.",
        "metrics": metrics,
        "notes": ("Simple simulation, not the paper's data or its deep-learning detector (architecture not given). "
                  "Unseen jammer = chirp; test = 25% of seen-type pulses plus all chirp-jammed pulses. "
                  "Only model seeds vary (data split fixed). Secondary metrics: f1_all_test (all test pulses), f1_lowjtr (JTR<0 dB slice), "
                  "roc_auc and logloss on the unseen slice. Both models get identical CV tuning and calibration; "
                  "the classifiers differ in input representation and class weighting. Results are specific to this simulator."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Done in {time.time() - t0:.1f}s; wrote results.json")


if __name__ == "__main__":
    main()
