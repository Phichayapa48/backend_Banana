import cv2
import numpy as np

def light_preprocess(img: np.ndarray) -> np.ndarray:
    """
    intentionally LIGHT preprocess
    ❌ no background removal
    ❌ no denoise
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img / 255.0
    return img