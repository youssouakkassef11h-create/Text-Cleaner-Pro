import unittest
import subprocess
import os
from pathlib import Path

class TestCLI(unittest.TestCase):
    def test_cli_help(self):
        result = subprocess.run(['python', 'cli.py', '--help'], 
                              capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('Usage', result.stdout)
    
    def test_cli_version(self):
        result = subprocess.run(['python', 'cli.py', '--version'], 
                              capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('Text Cleaner Pro', result.stdout)

if __name__ == '__main__':
    unittest.main()