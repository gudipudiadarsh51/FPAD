from importlib.resources import path

import numpy as np
import pandas as pd
from typing import List, Tuple
from joblib import Parallel, delayed
from src.enhance import enhance_image

from .gabor import compute_gabor
from .ocl import compute_ocl
from .lcs import compute_lcs
from .ofl import compute_ofl
from .fda import compute_fda
from .rvu import compute_rvu
from .rps import compute_rps
from .mean import compute_mean
from .std import compute_std
from .rdi import compute_rdi
from .s3pg import compute_s3pg
from .sep import compute_sep
from .mow import compute_mow
from .acut import compute_acut
from .sf import compute_sf



FEATURE_COLS = [
    'gabor','gabor_std',
    'ocl','ocl_std',
    'lcs','lcs_std',
    'fda','fda_std',
    'rvu','rvu_std',
    'rps','rps_std',
    'mean','mean_std',
    'std','std_std',
    'ofl','ofl_std',
    'rdi','rdi_std',
    's3pg','s3pg_std',
    'sep','sep_std',
    'mow','mow_std',
    'acut','acut_std',
    'sf','sf_std'
]


def extract_one(img: np.ndarray, mask: np.ndarray, config: dict) -> dict:
    features = {}
    fg = config['foreground_ratio']

    #enhance the image
    img = enhance_image(img)
    # ── Gabor ─────────────────────────────────────────────
    gabor = compute_gabor(
        img, mask,
        gabor_blk_size=config['gabor_blk_size'],
        gabor_sigma=config['gabor_sigma'],
        gabor_freq=config['gabor_freq'],
        gabor_angle_num=config['gabor_angle_num'],
    )
    features['gabor'], features['gabor_std'] = gabor

    # ── OCL ───────────────────────────────────────────────
    ocl = compute_ocl(img, mask,
        ocl_ofl_blk_size=config['ocl_ofl_blk_size']
    )
    features['ocl'], features['ocl_std'] = ocl

    # ── OFL ───────────────────────────────────────────────
    '''
    ofl = compute_ofl(img, mask,
        ocl_ofl_blk_size=config['ocl_ofl_blk_size'],
        foreground_ratio=fg
    )'''
    
    ofl = compute_ofl(img, mask, shared_blk_size=config['shared_blk_size'],foreground_ratio=fg)
    ofl = np.array(ofl).flatten()

    features['ofl'] = np.nanmean(ofl) if ofl.size > 0 else 0.0
    features['ofl_std'] = np.nanstd(ofl) if ofl.size > 0 else 0.0
    
    

    # ── FDA ───────────────────────────────────────────────
    fda = compute_fda(img, mask,
        shared_blk_size=config['shared_blk_size'],
        v1sz_x=config['v1sz_x'],
        v1sz_y=config['v1sz_y']
    )
    features['fda'], features['fda_std'] = fda

    # ── LCS ───────────────────────────────────────────────
    lcs = compute_lcs(img, mask,
        shared_blk_size=config['shared_blk_size'],
        v1sz_x=config['v1sz_x'],
        v1sz_y=config['v1sz_y'],
        foreground_ratio=fg
    )
    features['lcs'], features['lcs_std'] = lcs

    # ── RVU ───────────────────────────────────────────────
    rvu = compute_rvu(img, mask,
        shared_blk_size=config['shared_blk_size'],
        v1sz_x=config['v1sz_x'],
        v1sz_y=config['v1sz_y']
    )
    features['rvu'], features['rvu_std'] = rvu

    # ── RPS ───────────────────────────────────────────────
    rps = compute_rps(img, mask)
    features['rps'], features['rps_std'] = rps

    # ── Global stats ──────────────────────────────────────
    mean = compute_mean(img, mask)
    std  = compute_std(img, mask)

    features['mean'], features['mean_std'] = mean
    features['std'],  features['std_std']  = std

    # ── NEW FEATURES ──────────────────────────────────────
    rdi = compute_rdi(img, mask, shared_blk_size=config['shared_blk_size'])
    features['rdi'], features['rdi_std'] = rdi

    s3pg = compute_s3pg(img, mask, shared_blk_size=config['shared_blk_size'])
    features['s3pg'], features['s3pg_std'] = s3pg

    sep = compute_sep(img, mask, shared_blk_size=config['shared_blk_size'])
    features['sep'], features['sep_std'] = sep

    mow = compute_mow(img, mask, shared_blk_size=config['shared_blk_size'])
    features['mow'], features['mow_std'] = mow

    acut = compute_acut(img, mask, shared_blk_size=config['shared_blk_size'])
    features['acut'], features['acut_std'] = acut

    sf = compute_sf(img, mask, shared_blk_size=config['shared_blk_size'])
    features['sf'], features['sf_std'] = sf

    return features


def extract_dataset(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    config: dict,
    n_jobs: int = -1,
) -> pd.DataFrame:

    rows = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(extract_one)(img, mask, config)
        for img, mask in pairs
    )

    df = pd.DataFrame(rows)

    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"Warning: missing feature '{col}'")
            df[col] = np.nan

    df = df[FEATURE_COLS]

    # Clean NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median())

    return df
