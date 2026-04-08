import json
import mlflow
import optuna
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from pytorch_tabnet.tab_model import TabNetClassifier

from src.preprocessing.preprocessing import load_dataset    
from feature_extraction.extractor import FeatureExtractor, feature_cols
from feature_extraction.extractor import FeatureExtractor, extract

#Ml flow setup
mlflow.set_tracking_uri("file:///Users/adarshgudipudi/Desktop/FPAD/FPAD/mlruns")
mlflow.set_experiment("Fingerprint Quality Assessment")

print("Experiment setup complete. Ready to run experiments."
      " Use mlflow.start_run() to begin an experiment run.")
print("loading dataset...")

#import dataset and split into train and validation sets, resused ine every trail
X_train, X_val, y_train, y_val = train_test_split(load_dataset(), load_dataset()['label'], test_size=0.2, random_state=42, stratify=load_dataset()['label']) 


#search space

def build_cofig(trail)-> dict:
    '''
    parameters are defined here, and will be sampled by optuna in every trial
    v1sz_x and v1sz_y should be less than blk_size and block size should be divisible by them, to avoid padding issues in lcs and rvu features
    all blk_sizes are multiples of 8
    FDA/LCS/RVU share on blk_size/v1sz_x/v1sz_y
    OCL/OFL share one blk_size


    '''
    # gabor
    gabor_sigma = trail.suggest_float('gabor_sigma', 2.0, 8.0)
    gabor_freq = trail.suggest_float('gabor_freq', 2.0, 8.0)
    gabor_angle_num = trail.suggest_int('gabor_angle_num', 4, 16,step=4)

    #global
    foreground_ratio = trail.suggest_float('foreground_ratio', 0.6, 0.95)
    blk_size = trail.suggest_int('blk_size', 16, 64, step=8)
    v1sz_x = trail.suggest_int('v1sz_x', 8, blk_size-8, step=8)
    v1sz_y = trail.suggest_int('v1sz_y', 8, blk_size-8, step=8)

    return {
        'gabor_sigma' : gabor_sigma,
        'gabor_freq' : gabor_freq,
        'gabor_angle_num' : gabor_angle_num,
        'foreground_ratio' : foreground_ratio,
        'blk_size' : blk_size,
        'v1sz_x' : v1sz_x,
        'v1sz_y' : v1sz_y
    }

#ORIGINAL HARCODED CONFIG, USED FOR INITIAL TESTING
DEFAULT_CONFIG = {
    'gabor_sigma' : 6.0,
    'gabor_freq' : 0.1,
    'gabor_angle_num' : 8,
    'foreground_ratio' : 0.8,
    'blk_size' : 32,
    'v1sz_x' : 16,
    'v1sz_y' : 16
}


def feature_quality_score(X,y):
    X0,X1 = X[y==0], X[y==1]
    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    fdr = (mu0 - mu1)**2 / (X0.var(axis=0) + X1.var(axis=0) + 1e-6)
    ks= np.array([stats.ks_2samp(X0[:,i], X1[:,i]).statistic for i in range(X.shape[1])])
    Xs=StandardScaler().fit_transform(X)
    top=np.argsort(fdr)[::-1][:min(20, X.shape[1])]

    try:
        lda_auc = roc_auc_score(y, LinearDiscriminantAnalysis().fit(Xs[:, top], y).transform(Xs[:, top]).ravel())
    except Exception:
        lda_auc = 0.5
    corr=np.corrcoef(X1.T)
    upper=np.abs(corr[np.triu_indices(corr.shape[0], k=1)]).mean()

    return {
        'lda_auc': lda_auc,
        'mean_ks': float(ks.mean()),
        'top5_ks': float(np.sort(ks)[::-1][:5].mean()),
        'mean_fdr': float(fdr.mean()),
        'reduundancy': float(upper),
        'composite': float(0.35*lda_auc + 0.30*ks.mean()+0.20*np.tanh(fdr.mean()/5)+0.15*(1-upper)),
        'fdr_vec': fdr,
        'ks_vec': ks
    }

def save_diagnosis_plot(fq, run_id):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(feature_cols , fq['fdr_vec'], color='steelblue', alpha=0.8)
    axes[0].set_title('FDR per feature')
    axes[0].tick_params(axis='x', rotation=45, labelsize=7)
    axes[1].bar(feature_cols, fq['ks_vec'], color='coral', alpha=0.8)
    axes[1].set_title('KS statistic per feature')
    axes[1].tick_params(axis='x', rotation=45, labelsize=7)
    plt.tight_layout()
    path = f"/tmp/diag_{run_id[:8]}.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def tabnet_cv_score(X, y, n_splits=5):
    skf, aucs, sc = StratifiedKFold(n_splits, shuffle=True, random_state=42), [], StandardScaler()
    for tr, vl in skf.split(X, y):
        X_tr, X_vl = sc.fit_transform(X[tr]), sc.transform(X[vl])
        clf = TabNetClassifier(n_d=32, n_a=32, n_steps=5, gamma=1.3,
                               lambda_sparse=1e-4,
                               optimizer_params=dict(lr=2e-3), verbose=0)
        clf.fit(X_tr, y[tr], eval_set=[(X_vl, y[vl])], eval_metric=['auc'],
                max_epochs=100, patience=20, batch_size=256)
        aucs.append(roc_auc_score(y[vl], clf.predict_proba(X_vl)[:,1]))
    return float(np.mean(aucs)), float(np.std(aucs))

import mlflow
from pytorch_tabnet.callbacks import Callback

#custom mlflow callback to log metrics and parameters from optuna trials into mlflow
class MLflowCallback(Callback):
    def __init__(self, tracking_uri=None, experiment_name=None):
        super().__init__()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def on_train_begin(self, logs=None):
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        mlflow.start_run()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for k, v in logs.items():
                mlflow.log_metric(k, v, step=epoch)

    def on_train_end(self, logs=None):
        mlflow.end_run()

mlflow_cb = MLflowCallback(
    tracking_uri="mlruns",
    metric_name="final_score",
    mlflow_kwargs={"experiment_name": "fingerprint_afqa_search"},
)

@mlflow_cb.track_in_mlflow()
def objective(trial):
    config = build_cofig(trial)
    
    config = {
        'gabor_freq':   config['gabor_freq'],
        'gabor_sigma':  config['gabor_sigma'],
        'gabor_angle_num': config['gabor_angle_num'],
        'blk_size': config['blk_size'],
        'v1sz_x':   config['v1sz_x'],
        'v1sz_y':   config['v1sz_y'],
        'foreground_ratio': config['foreground_ratio'],
    }
    mlflow.log_params(config)

  

    X_tr = extract(X_train,config).values.astype('float32')
    X_vl = extract(X_val,config).values.astype('float32')

    fq = feature_quality_score(X_tr, y_train)
    mlflow.log_metrics({
        'feat_lda_auc':    fq['lda_auc'],
        'feat_mean_ks':    fq['mean_ks'],
        'feat_mean_fdr':   fq['mean_fdr'],
        'feat_composite':  fq['composite'],
        'feat_redundancy': fq['redundancy'],
    })

    #save FDR/KS plot for diagnosis in mlflow artifacts
    run_id = mlflow.active_run().info.run_id
    mlflow.log_artifact(save_diagnosis_plot(fq, run_id), "diagnosis")

    #prune bad cofigs - skip tbanet training if feature quality is too low, to save time and resources. Thresholds are chosen based on initial experiments with default config and can be adjusted as needed.
    if fq['lda_auc'] < 0.65 or fq['mean_ks'] < 0.15:
        mlflow.log_metric('pruned', 1)
        raise optuna.exceptions.TrialPruned()
    mlflow.log_metric('pruned', 0)

    # Tabnet cross-valiated AUC
    X_all = np.vstack([X_tr, X_vl])
    y_all = np.concatenate([y_train, y_val])
    tabnet_mean, tabnet_std = tabnet_cv_score(X_all, y_all)
    mlflow.log_metrics({'tabnet_cv_auc': tabnet_mean, 'tabnet_cv_std': tabnet_std})
    
    final = 0.5 * fq['composite'] + 0.5 * tabnet_mean
    mlflow.log_metric('final_score', final)
    return final

#run study
if __name__ == "__main__":
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        storage="sqlite:///afqa_study.db",
        study_name="afqa_tabnet",
        load_if_exists=True,
    )

    study.enqueue_trial(DEFAULT_CONFIG) #optional, to run the default config as one of the trails for comparison    

    print("Starting hyperparameter optimization search (60 trails)...")
    print("Open MLflow UI with `mlflow ui` command to monitor the search progress and results.\n")

    study.optimize(objective, n_trials=60,
                callbacks=[mlflow_cb], show_progress_bar=True)

    #save best configs to a JSON file
    best_config = study.best_params
    with open("configs/best_config.json", "w") as f:
        json.dump(best_config, f, indent=2) #Save best config to a JSON file

    print("\nSearch complete. Best final score: {:.4f}".format(study.best_value))
    print(f"Best hyperparameters: {json.dumps(study.best_params, indent=2)}")
    print(f"Best config saved: {json.dumps(best_config, indent=2)}")


