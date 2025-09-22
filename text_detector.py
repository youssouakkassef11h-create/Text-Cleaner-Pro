import cv2
import easyocr
from typing import List, Tuple
import numpy as np

class TextDetector:
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or ['en']
        self.gpu = gpu
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def detect_text(self, image_path: str, confidence_threshold: float = 0.5) -> List[Tuple]:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"تعذر تحميل الصورة من: {image_path}")
                
            results = self.reader.readtext(image)
            return [(bbox, text, conf) for (bbox, text, conf) in results if conf >= confidence_threshold]
        except Exception as e:
            raise Exception(f"خطأ في اكتشاف النص: {str(e)}")