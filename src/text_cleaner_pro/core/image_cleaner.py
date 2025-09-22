import cv2
import numpy as np

class ImageCleaner:
    def __init__(self, method: str = 'inpaint', inpaint_radius: int = 3, inpaint_algorithm: str = 'TELEA'):
        self.method = method
        self.inpaint_radius = inpaint_radius
        if inpaint_algorithm == 'NS':
            self.inpaint_algorithm = cv2.INPAINT_NS
        else:
            self.inpaint_algorithm = cv2.INPAINT_TELEA

    def clean_image(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            if self.method == 'inpaint':
                return cv2.inpaint(img, mask, self.inpaint_radius, self.inpaint_algorithm)
            else:
                result = img.copy()
                result[mask>0] = [255, 255, 255]  # تبييض المناطق
                return result
        except Exception as e:
            raise Exception(f"خطأ في تنظيف الصورة: {str(e)}")