import unittest
import numpy as np
import pytest
from classifier_carol import _train_with_dp_torch

class MyTestCase(unittest.TestCase):
   """ def test_dp_training_reports_metadata(self):
        # Tiny synthetic dataset
        X = np.random.rand(10, 5).astype(np.float32)  # 10 samples, 5 features
        y = np.random.randint(0, 2, size=(10,))  # binary labels

        # Train with DP
        result = _train_with_dp_torch(
            X, y,
            multi_label=False,
            epochs=1, batch_size=4, lr=1e-3,
            noise_multiplier=1.5, max_grad_norm=1.0,
            verbose=False
        )

        self.assertIsInstance(result, dict)
        self.assertIn("noise_multiplier", result)
        self.assertGreater(result["noise_multiplier"], 0)
        self.assertIn("max_grad_norm", result)
        self.assertIn("delta", result)  # delta is returned in your code
        self.assertIn("epsilon", result)
        self.assertGreater(result["epsilon"], 0)

        # Optional: check model exists
        self.assertIn("model", result)

        # optional
        # result_low_noise = _train_with_dp_torch(X, y, noise_multiplier=0.5,epochs=1)
        # result_high_noise = _train_with_dp_torch(X, y, noise_multiplier=2.0, epochs=1)
        # assert result_high_noise["noise_multiplier"] > result_low_noise["noise_multiplier"]

if __name__ == '__main__':
    unittest.main()
"""
