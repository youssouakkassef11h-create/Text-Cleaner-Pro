import cv2
import numpy as np
from typing import List, Tuple

class MaskGenerator:
    def __init__(self, dilation_kernel_size: int = 5, dilation_iterations: int = 3):
        self.dilation_kernel_size = dilation_kernel_size
        self.dilation_iterations = dilation_iterations
        self.dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)

    def create_mask(self, img_shape: Tuple[int, int, int], detections: List) -> np.ndarray:
        try:
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            for detection in detections:
                bbox = detection[0]
                pts = np.array(bbox, np.int32).reshape((-1,1,2))
                cv2.fillPoly(mask, [pts], 255)
            mask = cv2.dilate(mask, self.dilation_kernel, iterations=self.dilation_iterations)
            return mask
        except Exception as e:
            raise Exception(f"خطأ في إنشاء القناع: {str(e)}")