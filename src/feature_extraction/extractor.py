""" Main fingerprint feature extraction pipeline."""
import afqa_toolbox as at # type: ignore
from afqa_toolbox.features import FeatFDA, FeatGabor, FeatOCL # type: ignore
import pkgutil
import inspect
import sys
import numpy as np

# src/feature_extraction/extractor.py

import cv2
import pandas as pd

from src.preprocessing.preprocessing import extract_foreground
from .gabor import compute_gabor
from .ocl import compute_ocl                        
from .lcs import compute_lcs
from .ofl import compute_ofl
from .fda import compute_fda
from .rvu import compute_rvu
from .rps import compute_rps
from .mean import compute_mean
from .std import compute_std

"""
feature_extractor.py
Wraps all your individual feature scripts into one unified extractor.
Accepts a config dict so hyperparameters can be varied per Optuna trial.

All feature functions return np.array([mean, std]).
FeatGabor is instantiated with hyperparams then called via .extract(img, mask).
"""


from typing import List, Tuple
from joblib import Parallel, delayed

#extract_foreground
# ── Canonical output column order ─────────────────────────────────────────────
FEATURE_COLS = [
    'gabor',  'gabor_std',
    'ocl',    'ocl_std',
    'lcs',    'lcs_std',
    'fda',    'fda_std',
    'rvu',    'rvu_std',
    'rps',
    'mean',   'std',
    'ofl',    'ofl_std',
]


def extract_one(img: np.ndarray, mask: np.ndarray, config: dict) -> dict:
    """
    Run all feature scripts on a single (img, mask) pair using config.

    Parameters
    ----------
    img   : uint8 grayscale numpy array
    mask  : binary uint8 numpy array, same shape as img
    config: dict from build_config() — contains all hyperparameter values

    Returns
    -------
    dict mapping feature name → float value
    """
    features = {}
    fg = config['foreground_ratio']

    # ── Gabor ─────────────────────────────────────────────────────────────────
    # FeatGabor is a class — instantiate with hyperparams, then call .extract()
    gabor_arr    = compute_gabor(img,
        blk_size  = config['gabor_blk_size'],
        sigma     = config['gabor_sigma'],
        freq      = config['gabor_freq'],
        angle_num = config['gabor_angle_num'],
    )
    #gabor_arr = gabor_o.extract(img, mask)   # returns np.array([mean, std])
    features['gabor']     = float(gabor_arr[0])
    features['gabor_std'] = float(gabor_arr[1])

    # ── OCL ───────────────────────────────────────────────────────────────────
    ocl_arr = compute_ocl(
        img, mask,
        blk_size         = config['ocl_ofl_blk_size'],
        foreground_ratio = fg,
    )
    features['ocl']     = float(ocl_arr[0])
    features['ocl_std'] = float(ocl_arr[1])

    # ── OFL ───────────────────────────────────────────────────────────────────
    ofl_arr = compute_ofl(
        img, mask,
        blk_size         = config['ocl_ofl_blk_size'],
        foreground_ratio = fg,
    )
    features['ofl']     = float(ofl_arr[0])
    features['ofl_std'] = float(ofl_arr[1])

    # ── FDA ───────────────────────────────────────────────────────────────────
    fda_arr = compute_fda(
        img, mask,
        blk_size         = config['shared_blk_size'],
        v1sz_x           = config['v1sz_x'],    # == shared_blk_size
        v1sz_y           = config['v1sz_y'],
        foreground_ratio = fg,
    )
    features['fda']     = float(fda_arr[0])
    features['fda_std'] = float(fda_arr[1])

    # ── LCS ───────────────────────────────────────────────────────────────────
    lcs_arr = compute_lcs(
        img, mask,
        blk_size         = config['shared_blk_size'],
        v1sz_x           = config['v1sz_x'],
        v1sz_y           = config['v1sz_y'],
        foreground_ratio = fg,
    )
    features['lcs']     = float(lcs_arr[0])
    features['lcs_std'] = float(lcs_arr[1])

    # ── RVU ───────────────────────────────────────────────────────────────────
    rvu_arr = compute_rvu(
        img, mask,
        blk_size         = config['shared_blk_size'],
        v1sz_x           = config['v1sz_x'],
        v1sz_y           = config['v1sz_y'],
        foreground_ratio = fg,
    )
    features['rvu']     = float(rvu_arr[0])
    features['rvu_std'] = float(rvu_arr[1])

    # ── RPS (placeholder — uncomment and adapt when ready) ───────────────────
    rps_arr= compute_rps(img)
    features['rps'] = float(rps_arr)   # remove once rps is wired in

    # ── Global stats (no hyperparameters) ────────────────────────────────────
    features['mean'] = float(img.mean())
    features['std']  = float(img.std())

    return features


def extract_dataset(
    pairs:  List[Tuple[np.ndarray, np.ndarray]],
    config: dict,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Extract features for all (img, mask) pairs in parallel.

    Parameters
    ----------
    pairs  : list of (img, mask) tuples from preprocessor.load_dataset()
    config : hyperparameter config dict from build_config()
    n_jobs : parallelism (-1 = all cores, 1 = serial for debugging)

    Returns
    -------
    pd.DataFrame with columns = FEATURE_COLS, one row per image
    """
    rows = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(extract_one)(img, mask, config)
        for img, mask in pairs
    )

    df = pd.DataFrame(rows)

    # Ensure canonical column order, add missing cols as NaN
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"Warning: feature '{col}' missing — check script return format")
            df[col] = np.nan

    df = df[FEATURE_COLS]

    # Clean NaN / Inf that can arise from empty foreground blocks
    n_bad = df.isnull().sum().sum() + np.isinf(df.values).sum()
    if n_bad > 0:
        print(f"Warning: {int(n_bad)} NaN/Inf values — replacing with column median")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())

    return df
