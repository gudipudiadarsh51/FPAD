from afqa_toolbox.enhancement import filters
import cv2
import numpy as np

def enhance_image(image, ksize=9,sigma=1.4,log_enhance=True):
    enhanced=filters.dog_filter(image, ksize, sigma, log_enhance)
    img = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img