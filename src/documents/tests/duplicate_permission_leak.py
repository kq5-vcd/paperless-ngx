import hashlib
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.test import override_settings
from django.urls import reverse

from documents.models import Document

User = get_user_model()


@override_settings(
    CELERY_TASK_ALWAYS_EAGER=True,
)
@patch("django.db.backends.base.base.BaseDatabaseWrapper.close", lambda x: None)
class DuplicatePermissionLeakTest(TestCase):
    def setUp(self):
        self.user_a = User.objects.create_user("user_a", password="pass")
        self.user_b = User.objects.create_user("user_b", password="pass")
        self.sample_content = b"Test .txt document for checking whether user B can be stopped from uploading a document file that already exists but user B has no view permission to it."
        self.checksum = hashlib.md5(self.sample_content).hexdigest()
        self.upload_url = reverse("post_document")

    def upload_as(self, user, filename="file.txt"):
        """Helper to upload a text file as a given user."""
        self.client.logout()
        self.client.login(username=user.username, password="pass")

        return self.client.post(
            self.upload_url,
            {
                "document": SimpleUploadedFile(
                    filename,
                    self.sample_content,
                    content_type="text/plain",
                ),
            },
        )

    def test_duplicate_not_detected_for_forbidden_user(self):
        # Upload from user A
        resp_a = self.upload_as(self.user_a, "original.txt")
        # print("User A upload status:", resp_a.status_code)
        self.assertEqual(resp_a.status_code, 200)

        docs = Document.objects.filter(checksum=self.checksum)
        # doc = docs.first()
        # allowed = get_users_with_perms(doc, only_with_perms_in=["view_document"])
        # print("View users for the document:", [u.username for u in allowed])

        # print(
        #     "Docs after User A upload:",
        #     list(docs.values_list("id", "owner_id", "checksum")),
        # )
        self.assertEqual(
            docs.count(),
            1,
            "Document should have been created by consumer for User A",
        )

        doc_a = docs.first()
        self.assertEqual(doc_a.owner, self.user_a)

        # User B uploads identical document
        # print("NOW USER B UPLOADS")
        resp_b = self.upload_as(self.user_b, "duplicate.txt")
        self.assertEqual(resp_b.status_code, 200)

        docs_after = Document.objects.filter(checksum=self.checksum)
        # print(
        #     "Docs after User B upload:",
        #     list(docs_after.values_list("id", "owner_id", "checksum")),
        # )

        self.assertEqual(
            docs_after.count(),
            2,
            (
                "User B must be allowed to upload a duplicate "
                "since they cannot see user A's document."
            ),
        )

        owners = list(docs_after.values_list("owner_id", flat=True))
        self.assertIn(self.user_a.id, owners)
        self.assertIn(self.user_b.id, owners)
