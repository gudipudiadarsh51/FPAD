import numpy as np
import cv2


def compute_gabor(self, image):
        std_map = self.gabor_feat.gabor_stds(image, smooth=False, shen=False)

        # Mean of all block responses
        gabor_mean = float(np.nanmean(std_map))
        return gabor_mean