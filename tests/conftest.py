import sys
import tempfile
from pathlib import Path

import pytest

# make `import funcprox` work without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def workdir():
    # plain tempfile: pytest's tmp_path breaks when its shared pytest-of-<user> folder is locked
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
