"""Logging configuration must not consume the scientific random stream."""
import builtins
import random
import sys
from types import SimpleNamespace

import numpy as np
import run


def test_silencing_optional_library_does_not_import_it_or_consume_rng(monkeypatch):
    original_import = builtins.__import__
    imported = []
    stub = SimpleNamespace(utilities=SimpleNamespace(logger=SimpleNamespace(PackageLogger=SimpleNamespace(verbosity='info'))))
    def importing(name, *args, **kwargs):
        if name == 'prody':
            imported.append(name)
            random.random()
            np.random.random(150)
            return stub
        return original_import(name, *args, **kwargs)
    monkeypatch.delitem(sys.modules, 'prody', raising=False)
    monkeypatch.setattr(builtins, '__import__', importing)
    random.seed(18)
    np.random.seed(18)
    py_before, np_before = random.getstate(), np.random.get_state()
    run._suppress_external_logging()
    assert imported == []
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]


def test_silences_prody_that_is_already_loaded(monkeypatch):
    logger = SimpleNamespace(verbosity='info')
    monkeypatch.setitem(sys.modules, 'prody', SimpleNamespace(utilities=SimpleNamespace(logger=SimpleNamespace(PackageLogger=logger))))
    run._suppress_external_logging()
    assert logger.verbosity is None
