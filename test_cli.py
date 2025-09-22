import unittest
import subprocess
import os
from pathlib import Path

import cv2
import numpy as np

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.base_command = ['python', '-m', 'text_cleaner_pro.cli.cli']
        # Add src to python path
        self.env = os.environ.copy()
        self.env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
        # Create a test image
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.putText(self.test_image, "Test", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        self.test_image_path = Path("test_sample.jpg")
        cv2.imwrite(str(self.test_image_path), self.test_image)
        self.output_path = Path("test_sample_cleaned.jpg")
        self.input_dir = Path("test_input_dir")
        self.input_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(self.input_dir / self.test_image_path.name), self.test_image)
        self.output_dir = Path("test_output_dir")

    def tearDown(self):
        if self.test_image_path.exists():
            self.test_image_path.unlink()
        if self.output_path.exists():
            self.output_path.unlink()
        if self.input_dir.exists():
            import shutil
            shutil.rmtree(self.input_dir)
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)

    def test_cli_help(self):
        result = subprocess.run(self.base_command + ['--help'],
                              capture_output=True, text=True, env=self.env)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
        self.assertEqual(result.returncode, 0)
        self.assertIn('Usage', result.stdout)
    
    def test_cli_version(self):
        result = subprocess.run(self.base_command + ['--version'],
                              capture_output=True, text=True, env=self.env)
        self.assertEqual(result.returncode, 0)
        self.assertIn('Text Cleaner Pro', result.stdout)

    def test_clean_command(self):
        command = self.base_command + ['clean', str(self.test_image_path), '-o', str(self.output_path)]
        result = subprocess.run(command, capture_output=True, text=True, env=self.env)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(self.output_path.exists())

    def test_clean_command_with_options(self):
        command = self.base_command + [
            'clean', str(self.test_image_path),
            '-o', str(self.output_path),
            '--inpaint-radius', '5',
            '--inpaint-algorithm', 'NS'
        ]
        result = subprocess.run(command, capture_output=True, text=True, env=self.env)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(self.output_path.exists())

    def test_batch_clean_command(self):
        command = self.base_command + ['batch-clean', str(self.input_dir), '-o', str(self.output_dir)]
        result = subprocess.run(command, capture_output=True, text=True, env=self.env)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(self.output_dir.exists())
        self.assertTrue((self.output_dir / "test_sample_cleaned.jpg").exists())

if __name__ == '__main__':
    unittest.main()