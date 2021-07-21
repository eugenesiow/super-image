"""Tests for the `file_utils` module."""

import unittest

from super_image.file_utils import get_model_path


class GetFromCacheTests(unittest.TestCase):
    def test_bogus_url(self):
        # This lets us simulate no connection
        # as the error raised is the same
        # `ConnectionError`
        url = "https://bogus"
        with self.assertRaisesRegex(ValueError, "Connection error"):
            _ = get_model_path(url)
