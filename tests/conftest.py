import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MOLEDITPY_HEADLESS", "1")


@pytest.fixture(scope="session")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Never write the developer's own settings.json."""
    import atom_index_mapper

    monkeypatch.setattr(atom_index_mapper, "SETTINGS_FILE", str(tmp_path / "settings.json"))
