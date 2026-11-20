"""Atom Index Mapper: renumber the loaded molecule to match another file.

Two structures of the same atoms optimised separately - the two ends of a NEB
path, an IRC, a TS guess - rarely share an atom order. This plugin reads a
reference structure from a file, works out which loaded atom is which
reference atom while keeping as many bonds as possible, and renumbers the
loaded molecule into the reference's order. A second button superimposes the
loaded molecule onto the reference atom-by-index at minimum RMSD.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

PLUGIN_NAME = "Atom Index Mapper"
PLUGIN_VERSION = "0.0.1"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Renumber the loaded molecule to the atom order of another structure file, keeping "
    "bonds where possible (for NEB/IRC endpoints), and superimpose it onto that file by "
    "atom index at minimum RMSD."
)
PLUGIN_CATEGORY = "Utility"
PLUGIN_TAGS = ["Utility", "DFT"]
PLUGIN_DEPENDENCIES = []
PLUGIN_OPTIONAL_DEPENDENCIES = []
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=5.0.0, <6.0.0"

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
WINDOW_ID = "atom_index_mapper_dialog"

_context = None


def default_settings():
    return {
        "last_dir": "",
        "method": "connectivity",
        "perceive": False,
        "fit_on": "all",
        "manual": "swap",
    }


def load_settings():
    settings = default_settings()
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as handle:
            saved = json.load(handle)
        if isinstance(saved, dict):
            settings.update({k: v for k, v in saved.items() if k in settings})
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        logger.warning("%s: could not read settings: %s", PLUGIN_NAME, exc)
    return settings


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2)
    except OSError as exc:
        logger.warning("%s: could not save settings: %s", PLUGIN_NAME, exc)


def open_dialog():
    """Open the one mapper window, or raise it if it is already open.

    Not named ``run``: the host adds a Plugins-menu entry for any module
    exposing ``run()``, which would duplicate the one registered below.
    """
    context = _context
    existing = context.get_window(WINDOW_ID) if context is not None else None
    if existing is not None:
        existing.show()
        existing.raise_()
        existing.activateWindow()
        return existing
    from .dialog import AtomIndexMapperDialog

    parent = context.get_main_window() if context is not None else None
    dialog = AtomIndexMapperDialog(context, parent=parent)
    if context is not None:
        context.register_window(WINDOW_ID, dialog)
    dialog.show()
    return dialog


def initialize(context):
    global _context
    _context = context
    context.add_menu_action("3D Edit/Atom Index Mapper...", open_dialog)
