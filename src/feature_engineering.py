"""
feature_engineering.py
Creates derived features from raw AFQA extraction output to improve TabNet classification.

Input:  DataFrame with raw features (15 columns)
Output: DataFrame with engineered features (30-50 columns depending on configuration)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def engineer_fingerprint_features(
    df: pd.DataFrame,
    add_polynomial: bool = True,
    polynomial_degree: int = 2,
    max_polynomial_features: int = 20,
) -> pd.DataFrame:
    """
    Create derived features from raw fingerprint quality features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw features from extract_dataset() with columns:
        ['gabor', 'gabor_std', 'ocl', 'ocl_std', 'lcs', 'lcs_std',
         'fda', 'fda_std', 'rvu', 'rvu_std', 'rps', 'mean', 'std',
         'ofl', 'ofl_std']

    add_polynomial : bool
        Whether to add polynomial/interaction features via sklearn

    polynomial_degree : int
        Degree for polynomial features (2 = squares + interactions)

    max_polynomial_features : int
        Cap on number of polynomial features to avoid explosion

    Returns
    -------
    pd.DataFrame
        Extended feature set with engineered columns
    """
    df = df.copy()

    # =========================================================
    # 1. Signal-to-Noise Ratios (mean/std for each feature)
    # =========================================================
    ratio_features = ['gabor', 'ocl', 'lcs', 'fda', 'rvu', 'ofl']
    for feat in ratio_features:
        df[f'{feat}_snr'] = df[feat] / (df[f'{feat}_std'] + 1e-8)

    # =========================================================
    # 2. Cross-Feature Interactions (texture × texture)
    # =========================================================
    # Gabor × FDA (texture × frequency domain)
    df['gabor_x_fda'] = df['gabor'] * df['fda']
    df['gabor_x_lcs'] = df['gabor'] * df['lcs']

    # OCL × OFL (orientation certainty × flow)
    df['ocl_x_ofl'] = df['ocl'] * df['ofl']
    df['ocl_x_lcs'] = df['ocl'] * df['lcs']

    # FDA × RVU (frequency domain × ridge-valley)
    df['fda_x_rvu'] = df['fda'] * df['rvu']

    # =========================================================
    # 3. Domain-Specific Composite Features
    # =========================================================

    # Ridge-Valley Contrast (FDA - RVU)
    df['ridge_contrast'] = df['fda'] - df['rvu']

    # Orientation Coherence (OCL + OFL)
    df['orient_coherence'] = df['ocl'] + df['ofl']

    # Texture Strength (Gabor + LCS)
    df['texture_strength'] = df['gabor'] + df['lcs']

    # Frequency-Texture Balance
    df['freq_texture_balance'] = df['fda'] / (df['gabor'] + 1e-8)

    # Global Intensity Normalized Features
    df['gabor_norm'] = df['gabor'] / (df['mean'] + 1e-8)
    df['fda_norm'] = df['fda'] / (df['mean'] + 1e-8)
    df['ocl_norm'] = df['ocl'] / (df['mean'] + 1e-8)

    # =========================================================
    # 4. Aggregate Statistics over Feature Groups
    # =========================================================

    # Texture features aggregate
    texture_cols = ['gabor', 'ocl', 'lcs', 'ofl']
    df['texture_mean'] = df[texture_cols].mean(axis=1)
    df['texture_std'] = df[texture_cols].std(axis=1)
    df['texture_max'] = df[texture_cols].max(axis=1)
    df['texture_min'] = df[texture_cols].min(axis=1)

    # Quality features aggregate (all mean values)
    quality_cols = ['gabor', 'ocl', 'lcs', 'fda', 'rvu', 'ofl']
    df['quality_mean'] = df[quality_cols].mean(axis=1)
    df['quality_std_agg'] = df[quality_cols].std(axis=1)

    # Variability features (all std values)
    std_cols = ['gabor_std', 'ocl_std', 'lcs_std', 'fda_std', 'rvu_std', 'ofl_std']
    df['variability_mean'] = df[std_cols].mean(axis=1)
    df['variability_sum'] = df[std_cols].sum(axis=1)

    # =========================================================
    # 5. Difference Features (captures contrasts)
    # =========================================================
    df['gabor_ocl_diff'] = df['gabor'] - df['ocl']
    df['lcs_fda_diff'] = df['lcs'] - df['fda']
    df['ofl_rvu_diff'] = df['ofl'] - df['rvu']

    # Std differences (variability contrast)
    df['gabor_fda_std_diff'] = df['gabor_std'] - df['fda_std']
    df['ocl_lcs_std_diff'] = df['ocl_std'] - df['lcs_std']

    # =========================================================
    # 6. Squared Terms (captures non-linearity)
    # =========================================================
    key_features = ['gabor', 'fda', 'ocl', 'lcs', 'rvu', 'ofl']
    for feat in key_features:
        df[f'{feat}_sq'] = df[feat] ** 2

    # =========================================================
    # 7. Polynomial Features (optional - adds many interactions)
    # =========================================================
    if add_polynomial:
        poly_cols = ['gabor', 'fda', 'ocl', 'lcs', 'rvu']  # subset to control explosion

        # Only use non-NaN rows for fitting
        valid_mask = df[poly_cols].notna().all(axis=1)

        if valid_mask.sum() > 10:
            poly = PolynomialFeatures(
                degree=polynomial_degree,
                include_bias=False,
                interaction_only=False
            )

            X_valid = df.loc[valid_mask, poly_cols].values
            X_poly = poly.fit_transform(X_valid)

            # Get feature names (skip original columns, keep only new)
            feature_names = poly.get_feature_names_out(poly_cols)
            new_feature_names = [
                name for name in feature_names
                if name not in poly_cols
            ]

            # Find indices of new features
            new_indices = [
                i for i, name in enumerate(feature_names)
                if name in new_feature_names
            ][:max_polynomial_features]

            # Add selected polynomial features
            for i, idx in enumerate(new_indices):
                col_name = f'poly_{i:02d}'
                df[col_name] = np.nan
                df.loc[valid_mask, col_name] = X_poly[:, idx]

    # =========================================================
    # 8. Clean Inf/-Inf that may have arisen from divisions
    # =========================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill any remaining NaNs with column median
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def get_feature_names() -> list:
    """
    Return expected output feature column names after engineering.
    Useful for validation and debugging.
    """
    base_features = [
        'gabor', 'gabor_std', 'ocl', 'ocl_std', 'lcs', 'lcs_std',
        'fda', 'fda_std', 'rvu', 'rvu_std', 'rps', 'mean', 'std',
        'ofl', 'ofl_std'
    ]

    snr_features = [f'{f}_snr' for f in ['gabor', 'ocl', 'lcs', 'fda', 'rvu', 'ofl']]

    interaction_features = [
        'gabor_x_fda', 'gabor_x_lcs', 'ocl_x_ofl', 'ocl_x_lcs', 'fda_x_rvu'
    ]

    composite_features = [
        'ridge_contrast', 'orient_coherence', 'texture_strength',
        'freq_texture_balance', 'gabor_norm', 'fda_norm', 'ocl_norm'
    ]

    aggregate_features = [
        'texture_mean', 'texture_std', 'texture_max', 'texture_min',
        'quality_mean', 'quality_std_agg', 'variability_mean', 'variability_sum'
    ]

    difference_features = [
        'gabor_ocl_diff', 'lcs_fda_diff', 'ofl_rvu_diff',
        'gabor_fda_std_diff', 'ocl_lcs_std_diff'
    ]

    squared_features = [f'{f}_sq' for f in ['gabor', 'fda', 'ocl', 'lcs', 'rvu', 'ofl']]

    # Polynomial features are dynamic (poly_00, poly_01, ...)
    poly_features = [f'poly_{i:02d}' for i in range(20)]

    all_features = (
        base_features + snr_features + interaction_features +
        composite_features + aggregate_features + difference_features +
        squared_features + poly_features
    )

    return all_features
