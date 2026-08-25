"""Workspace-local pytest fixtures for restricted Windows environments."""

from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path():
    """Create an ordinary workspace directory instead of an ACL-restricted temp dir."""
    root = Path(__file__).resolve().parents[1] / "work" / "test-artifacts"
    path = root / uuid4().hex
    path.mkdir(parents=True)
    return path
