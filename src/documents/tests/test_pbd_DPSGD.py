import numpy as np
from django.test import TestCase

from documents.classifier import _train_with_dp_torch


class DPSGDTest(TestCase):
    def test_dp_training_reports_metadata(self):
        # Tiny synthetic dataset
        X = np.random.rand(10, 5).astype(np.float32)  # 10 samples, 5 features
        y = np.random.randint(0, 2, size=(10,))  # binary labels
        # Train with DP
        result = _train_with_dp_torch(
            X,
            y,
            multi_label=False,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            noise_multiplier=1.5,
            max_grad_norm=1.0,
            verbose=False,
        )

        self.assertIsInstance(
            result,
            dict,
            msg="The result from _train_with_dp_torch should be a dictionary containing metadata and the model.",
        )
        self.assertIn(
            "noise_multiplier",
            result,
            msg="The result dictionary must contain 'noise_multiplier' to ensure DP parameters are tracked.",
        )
        self.assertGreater(
            result["noise_multiplier"],
            0,
            msg="The 'noise_multiplier' must be positive for Differential Privacy.",
        )
        self.assertIn(
            "max_grad_norm",
            result,
            msg="The result dictionary must contain 'max_grad_norm'.",
        )
        self.assertIn(
            "delta",
            result,
            msg="The result dictionary must contain 'delta'.",
        )
        self.assertIn(
            "epsilon",
            result,
            msg="The result dictionary must contain 'epsilon', which quantifies the privacy loss.",
        )
        self.assertGreater(result["epsilon"], 0, msg="Epsilon must be greater than 0.")
        self.assertLess(
            result["epsilon"],
            3,
            msg="Epsilon is too high (>3), indicating insufficient privacy guarantees.",
        )
