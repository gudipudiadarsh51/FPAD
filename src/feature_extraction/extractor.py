""" Main fingerprint feature extraction pipeline."""
import afqa_toolbox as at # type: ignore
from afqa_toolbox.features import FeatFDA, FeatGabor, FeatOCL # type: ignore
import pkgutil
import inspect
import sys
import numpy as np

# src/feature_extraction/extractor.py

import cv2
import numpy as np

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

feature_cols = [
            'gabor',
            'gabor_std',
            'ocl',
            'ocl_std',
            'lcs',
            'lcs_std', 
            'ofl',
            'ofl_std', 
            'fda', 
            'fda_std',
            'rvu', 
            'rvu_std',
            'rps',
            'mean',
            'std'
        ]

class FeatureExtractor:
    """
    Combines all feature modules into a single feature vector.
    """

    def __init__(self):
        pass  # Add reusable objects here if needed

    # ==========================================================
    # Main API
    # ==========================================================
    def extract(self,image: np.ndarray, config: dict) -> np.ndarray:
        """
        Extract all fingerprint quality features from one image.

        Parameters
        ----------
        image : np.ndarray (BGR or grayscale)

        Returns
        -------
        features : np.ndarray (float32)
            Final feature vector
        """

        # ------------------------------------------------------
        # 1) Convert to grayscale
        # ------------------------------------------------------
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # ------------------------------------------------------
        # 2) Foreground segmentation
        # ------------------------------------------------------
        foreground, mask = extract_foreground(gray)

        # Safety check
        if foreground is None or mask is None:
            return self._empty_feature_vector()

        # ------------------------------------------------------
        # 3) Compute features
        # ------------------------------------------------------
        features = []
        # Texture
        features.append(compute_gabor(foreground, config=config))

        # Orientation certainty
        features.append(compute_ocl(foreground, mask, config=config))

        # Local clarity
        features.append(compute_lcs(foreground, mask,config=config))

        # Orientation flow
        features.append(compute_ofl(foreground, mask, config=config))

        # Frequency domain
        features.append(compute_fda(foreground, mask, config=config))

        # Ridge–Valley uniformity
        features.append(compute_rvu(foreground, mask, config=config))

        # Radial Power Spectrum
        features.append(compute_rps(foreground, config=config))

        #mean
        features.append(compute_mean(foreground))

        #std
        features.append(compute_std(foreground))

        # ------------------------------------------------------
        # 4) Assemble feature vector
        # ------------------------------------------------------

        return np.array(features, dtype=np.float32)

    # ==========================================================
    # Fallback (if segmentation fails)
    # ==========================================================
    def _empty_feature_vector(self):
        """
        Returns a zero feature vector if extraction fails.
        Keeps pipeline stable.
        """
        return np.zeros(15, dtype=np.float32)
