import unittest
import pandas as pd
import os
from datetime import datetime
from Data_Processing.data_validator import DataValidator

class TestDataValidator(unittest.TestCase):
    def setUp(self):
        self.validator = DataValidator()
        self.test_file = 'test_data.csv'
        
        # Create test data
        df = pd.DataFrame({
            'date': [datetime.now().strftime('%Y-%m-%d')],
            'content': ['Test content'],
            'source': ['Test source']
        })
        df.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_validate_file(self):
        self.assertTrue(self.validator.validate_file(self.test_file))
    
    def test_validate_missing_columns(self):
        df = pd.DataFrame({'date': [datetime.now()]})
        df.to_csv(self.test_file, index=False)
        self.assertFalse(self.validator.validate_file(self.test_file))
    
    def test_validate_empty_file(self):
        df = pd.DataFrame(columns=['date', 'content', 'source'])
        df.to_csv(self.test_file, index=False)
        self.assertFalse(self.validator.validate_file(self.test_file))
    
    def test_validation_report(self):
        self.validator.validate_file(self.test_file)
        report = self.validator.get_validation_report()
        self.assertIn('summary', report)
        self.assertIn('details', report)
        self.assertEqual(report['summary']['total_files'], 1)

if __name__ == '__main__':
    unittest.main()
