import unittest
import cv2
import numpy as np
import os
from pathlib import Path

from src.core.text_detector import TextDetector
from src.core.mask_generator import MaskGenerator
from src.core.image_cleaner import ImageCleaner

class TestCoreFunctions(unittest.TestCase):
    def setUp(self):
        # إنشاء صورة اختبارية تحتوي على نص
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.putText(self.test_image, "Test", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        self.test_image_path = "tests/test_sample.jpg"
        cv2.imwrite(self.test_image_path, self.test_image)
        
    def tearDown(self):
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def test_text_detection(self):
        detector = TextDetector(languages=['en'])
        detections = detector.detect_text(self.test_image_path, confidence=0.1)
        self.assertGreater(len(detections), 0, "Should detect at least one text region")
    
    def test_mask_generation(self):
        # محاكاة اكتشاف نص
        fake_detections = [
            ([[10, 10], [50, 10], [50, 50], [10, 50]], "Test", 0.9)
        ]
        
        mask_generator = MaskGenerator()
        mask = mask_generator.create_mask(self.test_image.shape, fake_detections)
        
        self.assertEqual(mask.shape, self.test_image.shape[:2], "Mask should have same dimensions as image")
        self.assertGreater(np.sum(mask), 0, "Mask should not be empty")
    
    def test_image_cleaning(self):
        # إنشاء قناع اختباري
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        
        cleaner = ImageCleaner()
        cleaned_image = cleaner.clean_image(self.test_image, mask)
        
        self.assertEqual(cleaned_image.shape, self.test_image.shape, "Cleaned image should have same shape as original")

if __name__ == '__main__':
    unittest.main()