"""Cold-import smoke tests.

Inside the pytest process, ``conftest.py`` and earlier tests import many napari
modules in a favourable order, which can hide circular imports. Importing each
top-level napari module first in a fresh interpreter catches them.
"""

import importlib.util
import pkgutil
import subprocess
import sys

import pytest

import napari

# not importable on their own as a user-facing entry point
_EXCLUDED = {'benchmarks', 'conftest'}
# need a Qt binding, which the headless test suite does not install
_REQUIRES_QT = {'_qt', 'qt', 'window'}
_QT_BINDINGS = ('PyQt6', 'PySide6')

_TOP_LEVEL = sorted(
    m.name
    for m in pkgutil.iter_modules(napari.__path__)
    if m.name not in _EXCLUDED
)


def _has_qt_binding() -> bool:
    return any(importlib.util.find_spec(b) for b in _QT_BINDINGS)


@pytest.mark.parametrize('name', _TOP_LEVEL)
def test_cold_import(name):
    if name in _REQUIRES_QT and not _has_qt_binding():
        pytest.skip('requires a Qt binding')
    result = subprocess.run(
        [sys.executable, '-c', f'import napari.{name}'],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cold_import_viewer_from_top_level():
    # the path taken by `napari.__main__` and `from napari import Viewer`
    result = subprocess.run(
        [sys.executable, '-c', 'from napari import Viewer'],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
