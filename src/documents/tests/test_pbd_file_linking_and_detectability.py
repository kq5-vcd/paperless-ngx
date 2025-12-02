# import io
import hashlib
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from documents.models import Document
from documents.tests.utils import DirectoriesMixin

# pytestmark = pytest.mark.django_db


@override_settings(
    CELERY_TASK_ALWAYS_EAGER=True,
)
@patch("django.db.backends.base.base.BaseDatabaseWrapper.close", lambda x: None)
class TestFileLinkingAndDetectability(DirectoriesMixin, APITestCase):
    USER_ENDPOINT = "/api/users/"
    UPLOAD_ENDPOINT = reverse("post_document")

    def setUp(self):
        super().setUp()

        self.user = User.objects.create_superuser(username="temp_admin")
        self.client.force_authenticate(user=self.user)

    def upload_as(self, user, content, filename="file.txt"):
        """Helper to upload a text file as a given user."""
        self.client.force_authenticate(user=user)

        return self.client.post(
            self.UPLOAD_ENDPOINT,
            {
                "document": SimpleUploadedFile(
                    filename,
                    content,
                    content_type="text/plain",
                ),
            },
        )

    def get_owner_id_from_doc_history(self, doc_id):
        """Helper to get a document."""
        self.client.force_authenticate(user=self.user)

        doc_resp = self.client.get(
            f"/api/documents/{doc_id}/history/",
        )

        owner_history = -1

        if doc_resp.status_code == 200:
            doc_history = doc_resp.data
            owner_history = doc_history[-2]["changes"]["owner"][1]

        return int(owner_history)

    def delete_user(self, user_id):
        """Helper to delete a user."""
        self.client.force_authenticate(user=self.user)

        return self.client.delete(
            f"{self.USER_ENDPOINT}{user_id}/",
        )

    def test_file_kept_after_deleting_user(self):
        """
        WHEN:
            - API requests are made to add a user account and upload a document
            - API request is made to remove said user
        THEN:
            - The file still exists
        """

        mock_user = {
            "username": "testuser",
            "password": "test",
            "first_name": "Test",
            "last_name": "User",
        }

        response = self.client.post(
            self.USER_ENDPOINT,
            data=mock_user,
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        returned_user = User.objects.get(username="testuser")

        self.assertEqual(returned_user.username, mock_user["username"])
        self.assertEqual(returned_user.first_name, mock_user["first_name"])
        self.assertEqual(returned_user.last_name, mock_user["last_name"])

        content = b"test_file_kept_after_deleting_user."
        checksum = hashlib.md5(content).hexdigest()

        upload_resp = self.upload_as(returned_user, content)
        self.assertEqual(upload_resp.status_code, 200)

        returned_doc = Document.objects.filter(checksum=checksum).first()
        doc_id = returned_doc.pk

        response = self.delete_user(returned_user.pk)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        returned_doc_2 = Document.objects.get(id=doc_id)
        self.assertIsNotNone(returned_doc_2)

    def test_link_id_through_file_history(self):
        """
        WHEN:
            - API requests are made to add a user account and upload a document
        THEN:
            - The file can link username to user id
        """

        mock_user = {
            "username": "testuser",
            "password": "test",
            "first_name": "Test",
            "last_name": "User",
        }

        response = self.client.post(
            self.USER_ENDPOINT,
            data=mock_user,
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        returned_user = User.objects.get(username="testuser")

        self.assertEqual(returned_user.username, mock_user["username"])
        self.assertEqual(returned_user.first_name, mock_user["first_name"])
        self.assertEqual(returned_user.last_name, mock_user["last_name"])

        content = b"test_link_id_through_file_history."
        checksum = hashlib.md5(content).hexdigest()

        upload_resp = self.upload_as(returned_user, content)
        self.assertEqual(upload_resp.status_code, 200)

        returned_doc = Document.objects.filter(checksum=checksum).first()
        doc_id = returned_doc.pk

        owner_id = self.get_owner_id_from_doc_history(doc_id)
        self.assertEqual(returned_doc.owner_id, owner_id)

        response = self.delete_user(returned_user.pk)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        returned_doc_2 = Document.objects.get(id=doc_id)
        self.assertIsNone(returned_doc_2.owner_id)

        owner_id_2 = self.get_owner_id_from_doc_history(doc_id)
        self.assertEqual(returned_doc.owner_id, owner_id_2)


# pytest --capture=no -n0 documents/tests/file_linking_and_detectability.py
