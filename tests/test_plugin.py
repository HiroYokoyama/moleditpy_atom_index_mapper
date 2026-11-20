"""Metadata and settings; needs neither Qt nor RDKit."""

import json

import atom_index_mapper


def test_metadata_the_registry_reads():
    assert atom_index_mapper.PLUGIN_NAME == "Atom Index Mapper"
    major, minor, patch = (int(p) for p in atom_index_mapper.PLUGIN_VERSION.split("."))
    assert major >= 0 and minor >= 0 and patch >= 0
    assert atom_index_mapper.PLUGIN_DEPENDENCIES == []
    assert atom_index_mapper.PLUGIN_SUPPORTED_MOLEDITPY_VERSION.startswith(">=5")
    assert callable(atom_index_mapper.initialize)
    assert not hasattr(atom_index_mapper, "run")


def test_importing_the_package_pulls_in_no_heavy_modules():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(atom_index_mapper.__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import atom_index_mapper;"
        "bad=[m for m in ('PySide6','rdkit','numpy') if m in sys.modules]; assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code, str(root)], check=True, timeout=60)


def test_settings_round_trip_and_ignore_unknown_or_broken(tmp_path):
    path = tmp_path / "settings.json"
    assert atom_index_mapper.load_settings() == atom_index_mapper.default_settings()
    atom_index_mapper.save_settings({**atom_index_mapper.default_settings(), "method": "distance"})
    assert atom_index_mapper.load_settings()["method"] == "distance"
    path.write_text(json.dumps({"method": "distance", "stray": 1}), encoding="utf-8")
    assert "stray" not in atom_index_mapper.load_settings()
    path.write_text("{broken", encoding="utf-8")
    assert atom_index_mapper.load_settings() == atom_index_mapper.default_settings()
