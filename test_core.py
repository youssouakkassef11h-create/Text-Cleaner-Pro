import unittest
import cv2
import numpy as np
import os
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from text_cleaner_pro.core.image_cleaner import ImageCleaner
from text_cleaner_pro.core.mask_generator import MaskGenerator
from text_cleaner_pro.core.text_detector import TextDetector

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
        detections = detector.detect_text(self.test_image_path, confidence_threshold=0.1)
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
        
        # Test with default TELEA algorithm
        cleaner_telea = ImageCleaner(inpaint_algorithm='TELEA', inpaint_radius=5)
        cleaned_image_telea = cleaner_telea.clean_image(self.test_image, mask)
        self.assertEqual(cleaned_image_telea.shape, self.test_image.shape)

        # Test with NS algorithm
        cleaner_ns = ImageCleaner(inpaint_algorithm='NS', inpaint_radius=5)
        cleaned_image_ns = cleaner_ns.clean_image(self.test_image, mask)
        self.assertEqual(cleaned_image_ns.shape, self.test_image.shape)

        # Test whitening method
        cleaner_white = ImageCleaner(method='white')
        cleaned_image_white = cleaner_white.clean_image(self.test_image, mask)
        self.assertEqual(cleaned_image_white.shape, self.test_image.shape)
        # Check if the masked area is white
        self.assertTrue(np.all(cleaned_image_white[mask>0] == [255,255,255]))

if __name__ == '__main__':
    unittest.main()