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


class FeatureExtractor:
    """
    Combines all feature modules into a single feature vector.
    """

    def __init__(self):
        pass  # Add reusable objects here if needed

    # ==========================================================
    # Main API
    # ==========================================================
    def extract(self, image):
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

        # Texture
        gabor_mean = compute_gabor(foreground)

        # Orientation certainty
        ocl_mean = compute_ocl(foreground, mask)

        # Local clarity
        lcs_mean = compute_lcs(foreground, mask)

        # Orientation flow
        ofl_mean = compute_ofl(foreground, mask)

        # Frequency domain
        fda_mean = compute_fda(foreground, mask)

        # Ridge–Valley uniformity
        rvu_mean = compute_rvu(foreground, mask)

        # Radial Power Spectrum
        rps_value = compute_rps(foreground)

        #mean
        mean = compute_mean(foreground)

        #std
        std = compute_std(foreground)

        # ------------------------------------------------------
        # 4) Assemble feature vector
        # ------------------------------------------------------
        features = np.array([
            gabor_mean,
            ocl_mean,
            lcs_mean, 
            ofl_mean, 
            fda_mean, 
            rvu_mean, 
            mean,
            std
        ], dtype=np.float32)

        return features

    # ==========================================================
    # Fallback (if segmentation fails)
    # ==========================================================
    def _empty_feature_vector(self):
        """
        Returns a zero feature vector if extraction fails.
        Keeps pipeline stable.
        """
        return np.zeros(11, dtype=np.float32)
