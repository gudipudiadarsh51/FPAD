"""training.py - Final TabNet training for Fingerprint PAD"""

import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, mlflow
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, RocCurveDisplay
from pytorch_tabnet.tab_model import TabNetClassifier

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))
from feature_extraction.extractor import FEATURE_COLS, extract_dataset
from preprocessing.preprocessing import extract_foreground

def load_best_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}\nRun experiment.py first.")
    with open(config_path) as f:
        config = json.load(f)
    config = config.copy()
    if 'v1sz_x' not in config:
        config['v1sz_x'] = config.get('shared_blk_size', 64)
    if 'v1sz_y' not in config:
        ratio = config.get('v1sz_y_ratio', 0.25)
        config['v1sz_y'] = max(8, int(config.get('shared_blk_size', 64) * ratio // 8 * 8))
    return config

def load_dataset_pairs(root_dir: str):
    import cv2
    pairs, labels, paths = [], [], []
    root = Path(root_dir)
    for label_name, label_val in [('Live', 1), ('Fake', 0)]:
        folder = root / label_name
        if not folder.exists():
            raise FileNotFoundError(f"Expected folder not found: {folder}")
        img_paths = []
        for ext in ("*.png", "*.bmp", "*.jpg", "*.jpeg", "*.tiff", "*.tif"):
            img_paths.extend(folder.glob(ext))
            img_paths.extend(folder.glob(ext.upper()))
        for img_path in sorted(set(img_paths)):
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                _, mask = extract_foreground(img)
                pairs.append((img, mask))
                labels.append(label_val)
                paths.append(str(img_path))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    live_count, fake_count = labels.count(1), labels.count(0)
    print(f"Loaded from {root_dir}: {live_count} live, {fake_count} fake ({len(pairs)} total)")
    if live_count == 0 or fake_count == 0:
        raise ValueError(f"Need both classes - got live={live_count}, fake={fake_count}")
    return pairs, np.array(labels, dtype=np.int64), paths

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feat in ['gabor', 'ocl', 'lcs', 'fda', 'rvu', 'ofl']:
        df[f'{feat}_snr'] = df[feat] / (df[f'{feat}_std'] + 1e-8)
    df['gabor_x_fda'], df['gabor_x_lcs'] = df['gabor'] * df['fda'], df['gabor'] * df['lcs']
    df['ocl_x_ofl'], df['ocl_x_lcs'], df['fda_x_rvu'] = df['ocl'] * df['ofl'], df['ocl'] * df['lcs'], df['fda'] * df['rvu']
    df['ridge_contrast'], df['orient_coherence'] = df['fda'] - df['rvu'], df['ocl'] + df['ofl']
    df['texture_strength'], df['freq_texture_balance'] = df['gabor'] + df['lcs'], df['fda'] / (df['gabor'] + 1e-8)
    df['gabor_norm'], df['fda_norm'], df['ocl_norm'] = df['gabor'] / (df['mean'] + 1e-8), df['fda'] / (df['mean'] + 1e-8), df['ocl'] / (df['mean'] + 1e-8)
    df['texture_mean'], df['texture_std'], df['texture_max'] = df[['gabor','ocl','lcs','ofl']].mean(axis=1), df[['gabor','ocl','lcs','ofl']].std(axis=1), df[['gabor','ocl','lcs','ofl']].max(axis=1)
    df['quality_mean'], df['quality_std_agg'] = df[['gabor','ocl','lcs','fda','rvu','ofl']].mean(axis=1), df[['gabor','ocl','lcs','fda','rvu','ofl']].std(axis=1)
    df['variability_mean'], df['variability_sum'] = df[['gabor_std','ocl_std','lcs_std','fda_std','rvu_std','ofl_std']].mean(axis=1), df[['gabor_std','ocl_std','lcs_std','fda_std','rvu_std','ofl_std']].sum(axis=1)
    df['gabor_ocl_diff'], df['lcs_fda_diff'], df['ofl_rvu_diff'] = df['gabor'] - df['ocl'], df['lcs'] - df['fda'], df['ofl'] - df['rvu']
    for feat in ['gabor', 'fda', 'ocl', 'lcs', 'rvu', 'ofl']:
        df[f'{feat}_sq'] = df[feat] ** 2
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].isnull().any(): df[col] = df[col].fillna(df[col].median())
    return df

def train_tabnet(X_train, y_train, X_val, y_val):
    import torch
    robust = RobustScaler()
    X_train_r, X_val_r = robust.fit_transform(X_train), robust.transform(X_val)
    n_quantiles = min(1000, max(len(X_train) // 10, 100))
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=n_quantiles, random_state=42)
    X_train_s, X_val_s = qt.fit_transform(X_train_r), qt.transform(X_val_r)
    clf = TabNetClassifier(n_d=32, n_a=32, n_steps=5, gamma=1.3, lambda_sparse=1e-4, optimizer_params=dict(lr=2e-3, weight_decay=1e-5), scheduler_params={"step_size": 10, "gamma": 0.9}, scheduler_fn=torch.optim.lr_scheduler.StepLR, verbose=1)
    clf.fit(X_train=X_train_s, y_train=y_train, eval_set=[(X_val_s, y_val)], eval_metric=['auc', 'accuracy'], max_epochs=200, patience=30, batch_size=256, virtual_batch_size=128)
    preds, probs = clf.predict(X_val_s), clf.predict_proba(X_val_s)[:, 1]
    return clf, {'auc': roc_auc_score(y_val, probs), 'accuracy': accuracy_score(y_val, preds), 'preds': preds, 'probs': probs, 'scaler_robust': robust, 'scaler_quantile': qt}

def save_artifacts(clf, metrics, feature_names, config, output_dir: Path, y_val):
    output_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(output_dir / "tabnet_final"))
    with open(output_dir / "scalers.pkl", "wb") as f:
        pickle.dump({'robust': metrics['scaler_robust'], 'quantile': metrics['scaler_quantile']}, f)
    cm = confusion_matrix(y_val, metrics['preds'])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['fake', 'live']); ax.set_yticklabels(['fake', 'live'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f"Confusion Matrix - AUC={metrics['auc']:.4f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout(); plt.savefig(output_dir / "confusion_matrix.png", dpi=120); plt.close()
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_val, metrics['probs'], ax=ax, name='TabNet')
    ax.set_title('ROC Curve - Final Model'); plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=120); plt.close()
    importance = dict(zip(feature_names, clf.feature_importances_))
    top20 = sorted(importance.items(), key=lambda x: -x[1])[:20]
    names, imps = zip(*top20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, imps, color='steelblue', alpha=0.8)
    ax.set_title('Top 20 Feature Importances (TabNet)'); ax.tick_params(axis='x', rotation=45, labelsize=7)
    plt.tight_layout(); plt.savefig(output_dir / "feature_importance.png", dpi=120); plt.close()
    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(dict(top20), f, indent=2)
    with open(output_dir / "best_config.json", "w") as f:
        json.dump(config, f, indent=2)

def main():
    print("=" * 60 + "\nFingerprint PAD - Final TabNet Training\n" + "=" * 60)
    mlflow.set_tracking_uri("mlruns"); mlflow.set_experiment("fingerprint_final_model")
    config = load_best_config(Path("configs/best_config.json"))
    print(f"\n[1/5] Config loaded: {len(config)} parameters")
    print("\n[2/5] Loading datasets...")
    train_pairs, y_train, _ = load_dataset_pairs("/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/data/Digital_Persona/train")
    val_pairs, y_val, _ = load_dataset_pairs("/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/data/Digital_Persona/val")
    print(f"Train: {len(train_pairs)} samples | Val: {len(val_pairs)} samples")
    print("\n[3/5] Extracting features...")
    df_train = extract_dataset(train_pairs, config)
    df_val = extract_dataset(val_pairs, config)
    df_train.to_csv("data/train_features_optimal.csv", index=False)
    df_val.to_csv("data/val_features_optimal.csv", index=False)
    print(f"Raw features saved. Shape: train={df_train.shape}, val={df_val.shape}")
    print("\n[4/5] Engineering features...")
    df_train_eng, df_val_eng = engineer_features(df_train), engineer_features(df_val)
    feature_names = list(df_train_eng.columns)
    X_train, X_val = df_train_eng.values.astype(np.float32), df_val_eng.values.astype(np.float32)
    print(f"Engineered features: {X_train.shape[1]} total")
    print("\n[5/5] Training TabNet...")
    with mlflow.start_run(run_name="final_tabnet"):
        mlflow.log_params(config)
        mlflow.log_param("n_engineered_features", X_train.shape[1])
        clf, metrics = train_tabnet(X_train, y_train, X_val, y_val)
        mlflow.log_metrics({'val_auc': metrics['auc'], 'val_accuracy': metrics['accuracy']})
        print(f"\n{'='*50}\nValidation AUC: {metrics['auc']:.4f}\nValidation Accuracy: {metrics['accuracy']:.4f}\n{'='*50}")
        print(classification_report(y_val, metrics['preds'], target_names=['fake', 'live']))
        save_artifacts(clf, metrics, feature_names, config, Path("configs"), y_val)
        print("\nModel saved -> configs/tabnet_final.zip\nView results: mlflow ui --port 5000")

if __name__ == "__main__":
    main()
