"""
experiment.py
Hyperparameter search over AFQA extraction parameters + TabNet validation.

Run from VS Code terminal:
    python experiment.py

Then in a second terminal:
    mlflow ui --port 5000
    → open http://localhost:5000
"""

import json
import mlflow
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optuna.integration.mlflow import MLflowCallback
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from pytorch_tabnet.tab_model import TabNetClassifier
from pathlib import Path

# ── Your modules ──────────────────────────────────────────────────────────────
# Replace generate_mask with your actual mask import
from mask import generate_mask                       # ← adapt to your import
from preprocessing.preprocessing import load_dataset
from feature_extraction.extractor import extract_dataset, FEATURE_COLS
from feature_engineering import engineer_fingerprint_features

# ── MLflow setup ──────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "fingerprint_afqa_search"
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)
Path("configs").mkdir(exist_ok=True)

# ── Load data once — reused across all trials ─────────────────────────────────
print("Loading datasets...")
train_pairs, y_train, train_paths = load_dataset("data/Digital_Persona/train/", mask_fn=None)
val_pairs,   y_val,   val_paths   = load_dataset("data/Digital_Persona/val/",   mask_fn=None)
print(f"Ready — train: {len(train_pairs)}, val: {len(val_pairs)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────

def build_config(trial) -> dict:
    """
    9 free parameters.

    Constraints enforced:
      v1sz_x == shared_blk_size   (not a free parameter)
      v1sz_y  < shared_blk_size   (enforced via ratio, rounded to mult of 8)
      all blk_sizes are multiples of 8
      FDA / LCS / RVU share one blk_size / v1sz_x / v1sz_y
      OCL / OFL share one blk_size
    """
    # Gabor
    gabor_blk_size  = trial.suggest_int  ('gabor_blk_size',   8,   32, step=8)
    gabor_sigma     = trial.suggest_float('gabor_sigma',       2.0,  8.0)
    gabor_freq      = trial.suggest_float('gabor_freq',        0.05, 0.15)
    gabor_angle_num = trial.suggest_int  ('gabor_angle_num',   4,   16,  step=4)

    # OCL + OFL (shared)
    ocl_ofl_blk_size = trial.suggest_int('ocl_ofl_blk_size',  16,  64,  step=8)

    # FDA + LCS + RVU (shared)
    shared_blk_size = trial.suggest_int  ('shared_blk_size',  32,  96,  step=16)
    v1sz_x          = shared_blk_size                          # derived — not tuned
    v1sz_y_ratio    = trial.suggest_float('v1sz_y_ratio',      0.1,  0.4)
    v1sz_y          = max(8, int(shared_blk_size * v1sz_y_ratio // 8 * 8))

    # Global
    foreground_ratio = trial.suggest_float('foreground_ratio', 0.6,  0.95)

    return {
        'gabor_blk_size':   gabor_blk_size,
        'gabor_sigma':      gabor_sigma,
        'gabor_freq':       gabor_freq,
        'gabor_angle_num':  gabor_angle_num,
        'ocl_ofl_blk_size': ocl_ofl_blk_size,
        'shared_blk_size':  shared_blk_size,
        'v1sz_x':           v1sz_x,
        'v1sz_y':           v1sz_y,
        'foreground_ratio': foreground_ratio,
    }


# Your original hardcoded values — seeded as trial 0 (baseline to beat)
DEFAULT_CONFIG = {
    'gabor_blk_size':   16,
    'gabor_sigma':       6.0,
    'gabor_freq':        0.1,
    'gabor_angle_num':   8,
    'ocl_ofl_blk_size': 32,
    'shared_blk_size':  64,
    'v1sz_y_ratio':      0.25,   # 16/64 = 0.25 → v1sz_y=16 as original
    'foreground_ratio':  0.8,
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature quality scorer (cheap — runs every trial before TabNet)
# ─────────────────────────────────────────────────────────────────────────────

def feature_quality_score(X: np.ndarray, y: np.ndarray) -> dict:
    X0, X1   = X[y == 0], X[y == 1]
    mu0, mu1 = X0.mean(0), X1.mean(0)

    # Fisher discriminant ratio per feature
    fdr = (mu1 - mu0) ** 2 / (X0.var(0) + X1.var(0) + 1e-8)

    # KS statistic per feature (class distribution overlap)
    ks = np.array([
        stats.ks_2samp(X0[:, i], X1[:, i]).statistic
        for i in range(X.shape[1])
    ])

    # LDA AUC — multivariate separability using top features
    Xs  = StandardScaler().fit_transform(X)
    top = np.argsort(fdr)[::-1][:min(20, X.shape[1])]
    try:
        lda_scores = (LinearDiscriminantAnalysis()
                      .fit(Xs[:, top], y)
                      .transform(Xs[:, top])
                      .ravel())
        lda_auc = roc_auc_score(y, lda_scores)
    except Exception:
        lda_auc = 0.5

    # Within-class feature redundancy (lower = better)
    corr  = np.corrcoef(X1.T)
    upper = np.abs(corr[np.triu_indices(corr.shape[0], k=1)]).mean()

    composite = (
        0.35 * lda_auc
        + 0.30 * float(ks.mean())
        + 0.20 * float(np.tanh(fdr.mean() / 5))
        + 0.15 * (1.0 - float(upper))
    )

    return {
        'lda_auc':    float(lda_auc),
        'mean_ks':    float(ks.mean()),
        'mean_fdr':   float(fdr.mean()),
        'redundancy': float(upper),
        'composite':  composite,
        'fdr_vec':    fdr,
        'ks_vec':     ks,
    }


def save_diagnosis_plot(fq: dict, config: dict, run_id: str,feature_cols: list) -> str:
    """Save FDR + KS bar charts as an MLflow artifact."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        f"Trial {run_id[:8]} | blk={config['shared_blk_size']} "
        f"freq={config['gabor_freq']:.3f} fg={config['foreground_ratio']:.2f}",
        fontsize=9,
    )
    names = feature_cols if feature_cols is not None else FEATURE_COLS
    fdr__vec  = fq['fdr_vec'][:len(names)]
    ks_vec    = fq['ks_vec'][:len(names)]

    axes[0].bar(range(len(names)), fdr__vec, color='steelblue', alpha=0.8)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    axes[0].set_title('Fisher discriminant ratio (higher = better separation)')
   
    axes[1].bar(range(len(names)), ks_vec, color='coral', alpha=0.8)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    axes[1].set_title('KS statistic (higher = better separation)')

    plt.tight_layout()
    path = f"/tmp/diag_{run_id[:8]}.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# TabNet cross-validated AUC
# ─────────────────────────────────────────────────────────────────────────────

def tabnet_cv_score(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    sc   = StandardScaler()

    for tr_idx, vl_idx in skf.split(X, y):
        X_tr = sc.fit_transform(X[tr_idx])
        X_vl = sc.transform(X[vl_idx])

        clf = TabNetClassifier(
            n_d=32, n_a=32, n_steps=5,
            gamma=1.3, lambda_sparse=1e-4,
            optimizer_params=dict(lr=2e-3),
            verbose=0,
        )
        clf.fit(
            X_tr, y[tr_idx],
            eval_set=[(X_vl, y[vl_idx])],
            eval_metric=['auc'],
            max_epochs=100,
            patience=20,
            batch_size=256,
        )
        aucs.append(roc_auc_score(y[vl_idx], clf.predict_proba(X_vl)[:, 1]))

    return float(np.mean(aucs)), float(np.std(aucs))


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

mlflow_cb = MLflowCallback(
    tracking_uri="mlruns",
    metric_name="final_score",
    mlflow_kwargs={"experiment_name": EXPERIMENT_NAME},
)


@mlflow_cb.track_in_mlflow()
def objective(trial):
    config = build_config(trial)

    # Log all resolved hyperparameter values (including derived v1sz_x, v1sz_y)
    mlflow.log_params({
        'gabor_blk_size':   config['gabor_blk_size'],
        'gabor_sigma':      config['gabor_sigma'],
        'gabor_freq':       config['gabor_freq'],
        'gabor_angle_num':  config['gabor_angle_num'],
        'ocl_ofl_blk_size': config['ocl_ofl_blk_size'],
        'shared_blk_size':  config['shared_blk_size'],
        'v1sz_x':           config['v1sz_x'],
        'v1sz_y':           config['v1sz_y'],
        'foreground_ratio': config['foreground_ratio'],
    })

    # ── Extract features with this config ────────────────────────────────────
    df_tr = extract_dataset(train_pairs, config)
    df_vl = extract_dataset(val_pairs,   config)

    # ── Feature engineering (ratios, interactions, polynomial terms) ──────────
    df_tr_eng = engineer_fingerprint_features(df_tr)
    df_vl_eng = engineer_fingerprint_features(df_vl)

    X_tr = df_tr_eng.values.astype(np.float32)
    X_vl = df_vl_eng.values.astype(np.float32)

    

    # ── Feature quality score (cheap — no model training) ────────────────────
    fq = feature_quality_score(X_tr, y_train)

    mlflow.log_metrics({
        'feat_lda_auc':    fq['lda_auc'],
        'feat_mean_ks':    fq['mean_ks'],
        'feat_mean_fdr':   fq['mean_fdr'],
        'feat_redundancy': fq['redundancy'],
        'feat_composite':  fq['composite'],
    })

    # Save FDR/KS diagnosis plot as artifact
    engineered_cols = df_tr_eng.columns.tolist()
    fq=feature_quality_score(X_tr, y_train)
    run_id = mlflow.active_run().info.run_id
    mlflow.log_artifact(save_diagnosis_plot(fq, config, run_id,engineered_cols), "diagnosis")

    # ── Prune clearly bad configs — skip TabNet training ─────────────────────
    if fq['lda_auc'] < 0.65 or fq['mean_ks'] < 0.15:
        mlflow.log_metric('pruned', 1)
        raise optuna.exceptions.TrialPruned()
    mlflow.log_metric('pruned', 0)

    # ── TabNet cross-validated AUC ────────────────────────────────────────────
    X_all = np.vstack([X_tr, X_vl])
    y_all = np.concatenate([y_train, y_val])
    tabnet_mean, tabnet_std = tabnet_cv_score(X_all, y_all)

    mlflow.log_metrics({
        'tabnet_cv_auc': tabnet_mean,
        'tabnet_cv_std': tabnet_std,
    })

    # ── Joint objective: feature quality + model AUC ─────────────────────────
    final_score = 0.5 * fq['composite'] + 0.5 * tabnet_mean
    mlflow.log_metric('final_score', final_score)

    return final_score


# ─────────────────────────────────────────────────────────────────────────────
# Run study
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        storage='sqlite:///afqa_study.db',
        study_name='afqa_tabnet',
        load_if_exists=True,    # resume safely if script is re-run
    )

    # Seed trial 0 with your original defaults — establishes the 80% baseline
    study.enqueue_trial(DEFAULT_CONFIG)

    print("Starting hyperparameter search (60 trials)...")
    print("Open MLflow UI in another terminal: mlflow ui --port 5000\n")

    study.optimize(
        objective,
        n_trials=60,
        callbacks=[mlflow_cb],
        show_progress_bar=True,
    )

    # ── Save best config to disk ──────────────────────────────────────────────
    best_config = study.best_params
    with open('configs/best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"\nSearch complete.")
    print(f"Best trial:  #{study.best_trial.number}")
    print(f"Best score:  {study.best_value:.4f}")
    print(f"Best config saved → configs/best_config.json")
    print(json.dumps(best_config, indent=2))
