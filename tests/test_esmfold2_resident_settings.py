"""Reject changed inference settings before submitting a resident request."""
import json
import pytest


def test_resident_rejects_changed_settings_before_submission(tmp_path):
    from astrolabe.evaluators.esmfold2_resident import ESMFold2Resident
    resident = ESMFold2Resident.__new__(ESMFold2Resident)
    resident._request_settings = json.dumps({'num_loops': 3, 'seed': 7}, sort_keys=True, allow_nan=False)
    with pytest.raises(ValueError, match='settings changed'):
        resident.run({'num_loops': 4, 'seed': 7, 'jobs': []}, tmp_path)
