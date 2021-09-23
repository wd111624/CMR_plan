import numpy as np
import cv2

EPS = 1e-6


def arr_to_8bit_img(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min() + EPS)
    return (255 * arr).astype(np.uint8)


def overlay_heatmap_on_image(heatmap, image):
    if len(heatmap.shape) > 2:  # multiple channels
        heatmap = heatmap.max(axis=0)
    heatmap = arr_to_8bit_img(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(src1=np.tile(image[..., None], [1, 1, 3]), alpha=1., src2=heatmap, beta=.2, gamma=0)
