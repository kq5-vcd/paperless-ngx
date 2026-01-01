import unittest
import numpy as np
import pytest
from django.test import TestCase
from documents.classifier import _train_with_dp_torch, DocumentClassifier
from documents.models import Correspondent, Document, DocumentType, StoragePath, Tag

class DPSGDTest(TestCase):
    def test_dp_training_reports_metadata(self):
        # Tiny synthetic dataset
        X = np.random.rand(10, 5).astype(np.float32)  # 10 samples, 5 features
        y = np.random.randint(0, 2, size=(10,))  # binary labels
        print("This is X: ", X)
        print("This is y: ", y)
        # Train with DP
        result = _train_with_dp_torch(
            X, y,
            multi_label=False,
            epochs=1, batch_size=4, lr=1e-3,
            noise_multiplier=1.5, max_grad_norm=1.0,
            verbose=False
        )
        print("I already have a result: ", result)

        self.assertIsInstance(result, dict, msg="The result from _train_with_dp_torch should be a dictionary containing metadata and the model.")
        self.assertIn("noise_multiplier", result, msg="The result dictionary must contain 'noise_multiplier' to ensure DP parameters are tracked.")
        self.assertGreater(result["noise_multiplier"], 0, msg="The 'noise_multiplier' must be positive for Differential Privacy.")
        self.assertIn("max_grad_norm", result, msg="The result dictionary must contain 'max_grad_norm'.")
        self.assertIn("delta", result, msg="The result dictionary must contain 'delta'.")
        self.assertIn("epsilon", result, msg="The result dictionary must contain 'epsilon', which quantifies the privacy loss.")
        self.assertGreater(result["epsilon"], 0, msg="Epsilon must be greater than 0.")
        self.assertLess(result["epsilon"], 20, msg="Epsilon is too high (>20), indicating insufficient privacy guarantees.")

        # delta <= 1 / n_samples (dataset size)

        # Optional: check model exists
        self.assertIn("model", result, msg="The trained PyTorch model must be returned in the result dictionary.")

    def test_classifier_train_dp_attributes(self):
        # 1. Create Data
        c1 = Correspondent.objects.create(name="c1", matching_algorithm=Correspondent.MATCH_AUTO)
        t1 = Tag.objects.create(name="t1", matching_algorithm=Tag.MATCH_AUTO)
        dt1 = DocumentType.objects.create(name="dt1", matching_algorithm=DocumentType.MATCH_AUTO)
        sp1 = StoragePath.objects.create(name="sp1", path="path1", matching_algorithm=StoragePath.MATCH_AUTO)

        doc1 = Document.objects.create(
            title="doc1",
            content="this is a document from c1",
            correspondent=c1,
            document_type=dt1,
            storage_path=sp1,
            checksum="A",
        )
        doc1.tags.add(t1)

        doc2 = Document.objects.create(
            title="doc2",
            content="this is another document",
            correspondent=c1,
            document_type=dt1,
            storage_path=sp1,
            checksum="B",
        )
        doc2.tags.add(t1)

        # 2. Instantiate and Train Classifier
        classifier = DocumentClassifier()
        classifier.train()

        # 3. Check attributes for DP metadata
        classifier_attributes = {
            "tags_classifier": classifier.tags_classifier,
            "correspondent_classifier": classifier.correspondent_classifier,
            "document_type_classifier": classifier.document_type_classifier,
            "storage_path_classifier": classifier.storage_path_classifier
        }

        for name, clf_attr in classifier_attributes.items():
            self.assertIsInstance(clf_attr, dict, msg=f"{name} should be a dictionary when DP is enabled, but got {type(clf_attr)}. This likely means the simplified DP return format was not used.")
            self.assertIn("noise_multiplier", clf_attr, msg=f"{name} missing 'noise_multiplier'.")
            self.assertIn("max_grad_norm", clf_attr, msg=f"{name} missing 'max_grad_norm'.")
            self.assertIn("delta", clf_attr, msg=f"{name} missing 'delta'.")
            self.assertIn("epsilon", clf_attr, msg=f"{name} missing 'epsilon'.")
            self.assertIn("model", clf_attr, msg=f"{name} missing 'model'.")
            
            # Check range
            self.assertGreater(clf_attr["epsilon"], 0, msg=f"Epsilon for {name} must be positive.")
            self.assertLess(clf_attr["epsilon"], 20, msg=f"Epsilon for {name} is dangerously high (>20).")

if __name__ == '__main__':
    unittest.main()
