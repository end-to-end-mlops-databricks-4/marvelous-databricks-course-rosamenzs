"""Conftest module."""

import platform
from pathlib import Path

from churn import PROJECT_DIR
from churn.config import ProjectConfig

MLRUNS_DIR = PROJECT_DIR / "tests" / "mlruns"
CATALOG_DIR = PROJECT_DIR / "tests" / "catalog"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)  # noqa

# To make the TRACKING_URI  path compatible for both macOS and Windows
if platform.system() == "Windows":
    TRACKING_URI = f"file:///{MLRUNS_DIR.as_posix()}"
else:
    TRACKING_URI = f"file://{MLRUNS_DIR.as_posix()}"


pytest_plugins = ["tests.fixtures.datapreprocessor_fixture"]


def pytest_ignore_collect(path: str | Path, config: ProjectConfig) -> bool:
    """Tell pytest to ignore specific files during test collection.

    Returns True if the path matches one of the excluded model files.
    """
    path = Path(path)
    return path.match("src/churn/models/LRmodel.py") or path.match("src/churn/models/XGBmodel.py")


print("conftest.py loaded")
