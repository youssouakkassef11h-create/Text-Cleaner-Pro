import cv2
import numpy as np

class ImageCleaner:
    def __init__(self, method: str = 'inpaint'):
        self.method = method

    def clean_image(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            if self.method == 'inpaint':
                return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            else:
                result = img.copy()
                result[mask>0] = [255, 255, 255]  # تبييض المناطق
                return result
        except Exception as e:
            raise Exception(f"خطأ في تنظيف الصورة: {str(e)}")