import json 
import pickle
import numpy as np
import mlflow
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix,RocCurveDisplay,accuracy_score)
from pytorch_tabnet.tab_model import TabNetClassifier
from pathlib import Path

from src.preprocessing.preprocessing import load_dataset
from src.feature_extraction.extractor import FeatureExtractor, feature_cols
from src.feature_extraction.extractor import FeatureExtractor,extract

#load best config

config_path=Path("/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/best_config.json")
if not config_path.exists():
    raise FileNotFoundError(f"Best config file not found at {config_path}")

with open(config_path, 'r') as f:
    best_config = json.load(f)

#rebuild extractor with best config values that were store as optuna params
# Rebuild derived values that were stored as Optuna params
# v1sz_x == shared_blk_size, v1sz_y from ratio


print("Best config loaded:")
print(json.dumps(best_config, indent=2))

# ─────────────────────────────────────────────────────────────────────────────
# Load data + extract features with optimal config
# ─────────────────────────────────────────────────────────────────────────────


print("\nExtracting features with optimal config...")
for row in 
df_train = FeatureExtractor.extract(row, best_config)
df_val   = FeatureExtractor.extract(X_val,   best_config)

# Save raw optimal features (useful for future experiments)
df_train.to_csv("data/train_features_optimal.csv", index=False)
df_val.to_csv("data/val_features_optimal.csv",     index=False)
print(f"Raw features saved. Shape: train={df_train.shape}, val={df_val.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

df_train_eng = engineer_fingerprint_features(df_train)
df_val_eng   = engineer_fingerprint_features(df_val)
feature_names = list(df_train_eng.columns)

X_train = df_train_eng.values.astype(np.float32)
X_val   = df_val_eng.values.astype(np.float32)
print(f"Engineered features: {X_train.shape[1]} total")

# ─────────────────────────────────────────────────────────────────────────────
# Scaling (two-stage: robust → quantile)
# ─────────────────────────────────────────────────────────────────────────────

robust = RobustScaler()
X_train_r = robust.fit_transform(X_train)
X_val_r   = robust.transform(X_val)

qt = QuantileTransformer(
    output_distribution='normal',
    n_quantiles=min(1000, len(X_train)),
    random_state=42,
)
X_train_s = qt.fit_transform(X_train_r)
X_val_s   = qt.transform(X_val_r)

# Save scalers for inference
with open("configs/scalers.pkl", "wb") as f:
    pickle.dump({'robust': robust, 'quantile': qt}, f)
print("Scalers saved → configs/scalers.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# Train final TabNet model
# ─────────────────────────────────────────────────────────────────────────────

import torch

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("fingerprint_final_model")

with mlflow.start_run(run_name="final_tabnet"):
    mlflow.log_params(best_config)
    mlflow.log_param("n_engineered_features", X_train_s.shape[1])
    mlflow.log_param("n_train_samples",       len(y_train))
    mlflow.log_param("n_val_samples",         len(y_val))

    clf = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5,
        gamma=1.3,
        lambda_sparse=1e-4,
        optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=1,
    )

    clf.fit(
        X_train=X_train_s,
        y_train=y_train,
        eval_set=[(X_val_s, y_val)],
        eval_metric=['auc', 'accuracy'],
        max_epochs=200,
        patience=30,
        batch_size=256,
        virtual_batch_size=128,
    )
   # ── Evaluation ────────────────────────────────────────────────────────────
    preds = clf.predict(X_val_s)
    probs = clf.predict_proba(X_val_s)[:, 1]
    auc   = roc_auc_score(y_val, probs)
    acc   = accuracy_score(y_val, preds)

    mlflow.log_metrics({'val_auc': auc, 'val_accuracy': acc})

    print(f"\n{'='*50}")
    print(f"Val AUC:      {auc:.4f}")
    print(f"Val Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_val, preds, target_names=['fake', 'live']))

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    cm = confusion_matrix(y_val, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['fake', 'live']); ax.set_yticklabels(['fake', 'live'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion matrix — AUC={auc:.4f}')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig('/tmp/confusion_matrix.png', dpi=120)
    mlflow.log_artifact('/tmp/confusion_matrix.png')
    plt.close()

    # ── ROC curve plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_val, probs, ax=ax, name='TabNet')
    ax.set_title('ROC curve — final model')
    plt.tight_layout()
    plt.savefig('/tmp/roc_curve.png', dpi=120)
    mlflow.log_artifact('/tmp/roc_curve.png')
    plt.close()

    # ── Feature importance ────────────────────────────────────────────────────
    importance = dict(zip(feature_names, clf.feature_importances_))
    top20 = sorted(importance.items(), key=lambda x: -x[1])[:20]

    print("\nTop 20 features used by TabNet:")

    or name, imp in top20:
        print(f"  {name:<40} {imp:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    names, imps = zip(*top20)
    ax.bar(names, imps, color='steelblue', alpha=0.8)
    ax.set_title('Top 20 feature importances (TabNet)')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    plt.tight_layout()
    plt.savefig('/tmp/feature_importance.png', dpi=120)
    mlflow.log_artifact('/tmp/feature_importance.png')
    plt.close()

    # ── Save model ────────────────────────────────────────────────────────────
    clf.save_model("configs/tabnet_final")
    mlflow.log_artifact("configs/tabnet_final.zip")
    mlflow.log_artifact("configs/scalers.pkl")
    mlflow.log_artifact("configs/best_config.json")

    print(f"\nModel saved → configs/tabnet_final.zip")
    print(f"View results: mlflow ui --port 5000")
