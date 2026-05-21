import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestVoiceAssistant(unittest.TestCase):
    
    def test_file_exists(self):
        self.assertTrue(os.path.exists('../voice_assistant.py'))
    
    def test_file_readable(self):
        self.assertTrue(os.access('../voice_assistant.py', os.R_OK))
    
    def test_code_syntax(self):
        with open('../voice_assistant.py', 'r') as f:
            code = f.read()
        try:
            compile(code, 'voice_assistant.py', 'exec')
            self.assertTrue(True)
        except SyntaxError:
            self.fail("Syntax error in voice_assistant.py")


if __name__ == '__main__':
    unittest.main()
