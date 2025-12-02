import tempfile
from datetime import timedelta
from pathlib import Path

# ADDED IMPORT for password hashing
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import Permission
from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone
from rest_framework import status

from documents.models import Document
from documents.models import ShareLink
from documents.tests.utils import DirectoriesMixin


class TestShareLinkAccessCount(DirectoriesMixin, TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user("testuser")
        super().setUp()

    def test_access_count(self):
        """
        GIVEN:
            - Share link created
        WHEN:
            - Valid request for share link is made
        THEN:
            - Document is returned without need for login
        """
        _, filename = tempfile.mkstemp(dir=self.dirs.originals_dir)

        content = b"This is a test"
        with Path(filename).open("wb") as f:
            f.write(content)

        doc = Document.objects.create(
            title="none",
            filename=Path(filename).name,
            mime_type="application/pdf",
        )

        sharelink_permissions = Permission.objects.filter(
            codename__contains="sharelink",
        )
        self.user.user_permissions.add(*sharelink_permissions)
        self.user.save()

        self.client.force_login(self.user)

        self.client.post(  # Create share link for the document
            "/api/share_links/",
            {
                "document": doc.pk,
                "file_version": "original",
                "max_access_count": 3,
            },
        )

        sl1 = ShareLink.objects.get(
            document=doc,
        )  # Get the created share link object for the document to test its access.

        self.client.logout()

        self.assertEqual(sl1.access_count, 0, msg="Initial access count should be zero")
        self.assertEqual(
            sl1.max_access_count,
            3,
            msg="Max access count should be set correctly",
        )

        # Access the share link using its slug (self.client.get(f"/share/{sl1.slug}", follow=True)
        # The client.get method is used to simulate a GET request to the share link URL.
        # follow=True can be added to follow redirects if necessary --> when testing invalid or expired links.

        ### --- Simulate access count increment on valid access --- ###
        response = self.client.get(f"/share/{sl1.slug}")
        self.assertEqual(
            response.content,
            content,
            msg="The document should be returned on valid access within max access count limit",
        )
        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK,
            msg="Access should be granted when access count limit is not reached",
        )
        sl1.refresh_from_db()  # Refresh from DB to get updated access_count
        self.assertEqual(
            sl1.access_count,
            1,
            msg="Access count should increment correctly when valid access is made",
        )

    def test_access_count_limit_reached(self):
        """
        GIVEN:
            - Share link created
        WHEN:
            - Valid requests for share link are made up to max access count
            - Request for share link after max access count is reached is made
        THEN:
            - Document is returned without need for login
            - User is redirected to login with error
        """
        _, filename = tempfile.mkstemp(dir=self.dirs.originals_dir)

        content = b"This is a test"
        with Path(filename).open("wb") as f:
            f.write(content)

        doc = Document.objects.create(
            title="none",
            filename=Path(filename).name,
            mime_type="application/pdf",
        )

        sharelink_permissions = Permission.objects.filter(
            codename__contains="sharelink",
        )
        self.user.user_permissions.add(*sharelink_permissions)
        self.user.save()

        self.client.force_login(self.user)

        self.client.post(  # Create share link for the document
            "/api/share_links/",
            {
                "document": doc.pk,
                "file_version": "original",
                "max_access_count": 3,
            },
        )

        sl1 = ShareLink.objects.get(
            document=doc,
        )  # Get the created share link object for the document to test its access.

        self.client.logout()

        self.assertEqual(sl1.access_count, 0, msg="Initial access count should be zero")
        self.assertEqual(
            sl1.max_access_count,
            3,
            msg="Max access count should be set correctly",
        )

        # Perform max_access_count valid accesses
        for i in range(sl1.max_access_count):
            response = self.client.get(f"/share/{sl1.slug}")
            self.assertEqual(
                response.content,
                content,
                msg="The document should be returned on valid access within max access count limit",
            )
            self.assertEqual(
                response.status_code,
                status.HTTP_200_OK,
                msg="Access should be granted when access count limit is not reached",
            )
            sl1.refresh_from_db()  # Refresh from DB to get updated access_count
            self.assertEqual(
                sl1.access_count,
                i + 1,
                msg="Access count should increment correctly",
            )

        # Access beyond max_access_count
        self.assertEqual(
            sl1.access_count,
            sl1.max_access_count,
            msg="Access count should equal max access count before exceeding it",
        )

        response = self.client.get(f"/share/{sl1.slug}", follow=True)
        self.assertNotEqual(
            response.content,
            content,
            msg="The document should not be returned when max access count is reached",
        )
        response.render()
        self.assertEqual(
            response.request["PATH_INFO"],
            "/accounts/login/",
            msg="Access should redirect to login when max access count is reached",
        )
        self.assertContains(response, b"Share link access limit has been reached")


class TestShareLinkPasswordProtection(DirectoriesMixin, TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user("testuser")
        super().setUp()

    def test_password_form_display(self):
        """
        GIVEN:
            - Share link created with password protection
        WHEN:
            - Request for share link is made
        THEN:
            - User is redirected to password form
        """
        _, filename = tempfile.mkstemp(dir=self.dirs.originals_dir)

        content = b"This is a test"
        with Path(filename).open("wb") as f:
            f.write(content)

        doc = Document.objects.create(
            title="none",
            filename=Path(filename).name,
            mime_type="application/pdf",
        )

        sharelink_permissions = Permission.objects.filter(
            codename__contains="sharelink",
        )
        self.user.user_permissions.add(*sharelink_permissions)
        self.user.save()

        self.client.force_login(self.user)

        self.client.post(  # Create share link for the document with password
            "/api/share_links/",
            {
                "document": doc.pk,
                "file_version": "original",
                "protect_link_with_password": True,
            },
        )

        sl1 = ShareLink.objects.get(document=doc)

        self.client.logout()

        # Access without password
        response = self.client.get(f"/share/{sl1.slug}", follow=True)
        self.assertNotEqual(
            response.content,
            content,
            msg="The document should not be returned without submitting the password when password protection is enabled",
        )
        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK,
            msg="The password form should be displayed when accessing a password-protected share link without providing a password",
        )
        self.assertContains(
            response,
            b"Enter Share Link Password",
            msg_prefix="The password form should prompt for the share link password",
        )

        sl1.refresh_from_db()
        self.assertEqual(
            sl1.access_count,
            0,
            msg="Access count should not increment when accessing password form without submitting password",
        )

    def test_incorrect_link_password(self):
        """
        GIVEN:
            - Share link created with password protection
        WHEN:
            - Request for share link is made
            - Incorrect password is provided through form
        THEN:
            - User is redirected to login with error
        """
        _, filename = tempfile.mkstemp(dir=self.dirs.originals_dir)

        content = b"This is a test"
        with Path(filename).open("wb") as f:
            f.write(content)

        doc = Document.objects.create(
            title="none",
            filename=Path(filename).name,
            mime_type="application/pdf",
        )

        sharelink_permissions = Permission.objects.filter(
            codename__contains="sharelink",
        )
        self.user.user_permissions.add(*sharelink_permissions)
        self.user.save()

        self.client.force_login(self.user)

        self.client.post(  # Create share link for the document with password
            "/api/share_links/",
            {
                "document": doc.pk,
                "file_version": "original",
                "protect_link_with_password": True,
            },
        )

        sl1 = ShareLink.objects.get(document=doc)

        self.client.logout()

        # Access with wrong password
        response = self.client.get(
            f"/share/{sl1.slug}",
        )  # Access to get the password form
        self.assertNotEqual(
            response.content,
            content,
            msg="The document should not be returned without submitting the correct password when password protection is enabled",
        )
        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK,
            msg="The password form should be displayed when accessing a password-protected share link without providing a password",
        )
        self.assertContains(
            response,
            b"Enter Share Link Password",
            msg_prefix="The password form should be displayed when accessing a password-protected share link without providing a password",
        )

        # Submit wrong password through form (simulated by POST request)
        response = self.client.post(
            f"/share/{sl1.slug}",
            data={"sharelink_password": "wrongpassword"},
            follow=True,
        )
        response.render()
        self.assertEqual(response.request["PATH_INFO"], "/accounts/login/")
        self.assertContains(response, b"Incorrect share link password.")

        sl1.refresh_from_db()
        self.assertEqual(
            sl1.access_count,
            0,
            msg="Access count should not increment when incorrect password is submitted",
        )

    def test_correct_link_password(self):
        """
        GIVEN:
            - Share link created with password protection
        WHEN:
            - Request for share link is made
            - Correct password is provided through form
        THEN:
            - Document is returned without need for login
        """
        _, filename = tempfile.mkstemp(dir=self.dirs.originals_dir)

        content = b"This is a test"
        with Path(filename).open("wb") as f:
            f.write(content)

        doc = Document.objects.create(
            title="none",
            filename=Path(filename).name,
            mime_type="application/pdf",
        )

        sharelink_permissions = Permission.objects.filter(
            codename__contains="sharelink",
        )
        self.user.user_permissions.add(*sharelink_permissions)
        self.user.save()

        self.client.force_login(self.user)

        self.client.post(  # Create share link for the document with password
            "/api/share_links/",
            {
                "document": doc.pk,
                "file_version": "original",
                "protect_link_with_password": True,
            },
        )

        sl1 = ShareLink.objects.get(document=doc)

        self.client.logout()

        # Access with correct password
        sl1.password_hash = make_password(
            "securepassword",
        )  # Set password for the share link to test correct access
        sl1.save()

        response = self.client.get(
            f"/share/{sl1.slug}",
        )  # Access to get the password form
        self.assertNotEqual(
            response.content,
            content,
            msg="The document should not be returned without submitting the correct password when password protection is enabled",
        )
        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK,
            msg="The password form should be displayed when accessing a password-protected share link without providing a password",
        )
        self.assertContains(
            response,
            b"Enter Share Link Password",
            msg_prefix="The password form should be displayed when accessing a password-protected share link without providing a password",
        )

        # Submit correct password through form (simulated by POST request)
        response = self.client.post(
            f"/share/{sl1.slug}",
            data={"sharelink_password": "securepassword"},
        )
        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK,
            msg="Access should be granted with correct given password when password protection is enabled",
        )
        self.assertEqual(
            response.content,
            content,
            msg="The document should be returned when the correct password is provided",
        )

        sl1.refresh_from_db()
        self.assertEqual(
            sl1.access_count,
            1,
            msg="Access count should increment correctly after valid access with correct password",
        )


class TestShareLink(DirectoriesMixin, TestCase):
    """Combined share link tests"""

    def setUp(self) -> None:
        self.user = User.objects.create_user("testuser")
        super().setUp()

    def test_password_protected_link_with_access_count_and_expiration(self):
        """
        GIVEN:
            - Share link created with password protection and access count limit and expiration
        WHEN:
            - Request for share link with correct password is made
        THEN:
            - Document is returned without need for login
        """
        _, filename = tempfile.mkstemp(dir=self.dirs.originals_dir)

        content = b"This is a test"
        with Path(filename).open("wb") as f:
            f.write(content)

        doc = Document.objects.create(
            title="none",
            filename=Path(filename).name,
            mime_type="application/pdf",
        )

        sharelink_permissions = Permission.objects.filter(
            codename__contains="sharelink",
        )
        self.user.user_permissions.add(*sharelink_permissions)
        self.user.save()

        self.client.force_login(self.user)

        self.client.post(  # Create share link for the document with password and access count limit
            "/api/share_links/",
            {
                "document": doc.pk,
                "file_version": "original",
                "protect_link_with_password": True,
                "max_access_count": 2,
                "expiration": timezone.now() + timedelta(minutes=10),
            },
        )

        sl1 = ShareLink.objects.get(document=doc)

        sl1.password_hash = make_password("securepassword")
        sl1.save()

        self.client.logout()

        # Access with correct password
        response = self.client.get(
            f"/share/{sl1.slug}",
        )  # Access to get the password form
        self.assertNotEqual(
            response.content,
            content,
            msg="The document should not be returned without submitting the correct password when password protection is enabled",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertContains(
            response,
            b"Enter Share Link Password",
            msg_prefix="The password form should be displayed when accessing a password-protected share link without providing a password",
        )

        # Submit correct password through form (simulated by POST request)
        response = self.client.post(
            f"/share/{sl1.slug}",
            data={"sharelink_password": "securepassword"},
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.content,
            content,
            msg="Access should be granted with correct given password when password protection is enabled",
        )

        sl1.refresh_from_db()
        self.assertEqual(
            sl1.access_count,
            1,
            msg="Access count should increment correctly after valid access with correct password",
        )

        sl1.expiration = timezone.now() - timedelta(minutes=1)
        sl1.save()

        # Access after expiration
        response = self.client.get(f"/share/{sl1.slug}", follow=True)
        response.render()
        self.assertEqual(
            response.request["PATH_INFO"],
            "/accounts/login/",
            msg="Access should be denied when share link has expired",
        )
        self.assertContains(response, b"Share link has expired")

        sl1.refresh_from_db()
        self.assertEqual(
            sl1.access_count,
            1,
            msg="Access count should not increment after access attempt on expired link",
        )
