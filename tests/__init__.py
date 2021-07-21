"""Tests suite for `super_image`."""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent
TMP_DIR = TESTS_DIR / "tmp"
FIXTURES_DIR = TESTS_DIR / "fixtures"

sys.path.append('./src')
