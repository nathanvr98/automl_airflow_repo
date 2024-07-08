import unittest
import os

class TestValidateMLSystem(unittest.TestCase):
    def test_ejecucion_modelo(self):
        # ejecucion()
        self.assertTrue(os.path.exists('submission.csv'))
        self.assertTrue(os.path.exists('final_dt_model.pkl'))


if __name__ == '__main__':
    unittest.main()