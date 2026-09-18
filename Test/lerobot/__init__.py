import pytest

try:
    import lerobot  # directly imported in one of the tests modules.
    import draccus  # directly imported in one of the tests modules.
except ImportError:
    pytest.skip("lerobot not installed", allow_module_level=True)
