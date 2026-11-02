from __future__ import annotations

import csv
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor
import os
import threading

import pytest

from astrolabe.runtime_support import ResourceOwner, atomic_text_writer, ordered_cpu_overlap


def test_resource_owner_deduplicates_identity_and_continues_after_failure(caplog):
    calls = []

    class Resource:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def __eq__(self, other):
            return True

        __hash__ = None

        def shutdown(self):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError("cleanup failed")

        def close(self):
            pytest.fail("shutdown takes precedence")

    class CloseOnly:
        def close(self):
            calls.append("close")

    owner = ResourceOwner()
    first, second, third = Resource("first", True), Resource("second"), CloseOnly()
    for obj, label in [(first, "first"), (first, "alias"), (second, "second"), (third, "third"), (None, "none")]:
        owner.add(obj, label)
    owner.close()
    owner.close()
    assert calls == ["first", "second", "close"]
    assert "cleanup failed" in caplog.text


def test_overlap_bounded_order_context_and_serial_evaluation():
    context = ContextVar("context", default="missing")
    context.set("request")
    first_started = threading.Event()
    release = threading.Event()
    two_evaluated = threading.Event()
    seen = []
    finalized = []
    caller = threading.get_ident()

    def evaluate(value):
        assert threading.get_ident() == caller
        seen.append(value)
        if value == 1:
            assert first_started.wait(3)
            two_evaluated.set()
        return value

    def finalize(value):
        assert context.get() == "request"
        if value == 0:
            first_started.set()
            assert release.wait(3)
        finalized.append(value)
        return value * 2

    # The controlling thread is the evaluate caller; the test thread releases it.
    def run():
        nonlocal caller
        caller = threading.get_ident()
        context.set("request")
        return ordered_cpu_overlap(range(5), evaluate, finalize, max_workers=2, max_in_flight=2)

    with ThreadPoolExecutor(max_workers=1) as controller:
        result = controller.submit(run)
        try:
            assert two_evaluated.wait(3)
            assert seen == [0, 1]
            assert not result.done()
        finally:
            release.set()
        assert result.result(timeout=3) == [0, 2, 4, 6, 8]
    assert sorted(finalized) == list(range(5))


def test_overlap_one_worker_preserves_overlap_with_next_evaluation():
    started, release = threading.Event(), threading.Event()
    def evaluate(value):
        if value == 1:
            assert started.wait(3)
            release.set()
        return value
    def finalize(value):
        if value == 0:
            started.set()
            assert release.wait(3)
        return value
    assert ordered_cpu_overlap([0, 1], evaluate, finalize, max_workers=1) == [0, 1]


def test_overlap_earlier_worker_error_precedes_later_evaluator_error():
    def evaluate(value):
        if value == 1:
            raise ValueError("evaluator 1")
        return value
    def finalize(value):
        raise RuntimeError("finalize 0")
    with pytest.raises(RuntimeError, match="finalize 0"):
        ordered_cpu_overlap([0, 1, 2], evaluate, finalize, max_workers=1)


def test_overlap_does_not_evaluate_after_observed_failure():
    seen = []
    def evaluate(value):
        seen.append(value)
        return value
    def finalize(value):
        if value == 0:
            raise RuntimeError("first")
        return value
    with pytest.raises(RuntimeError, match="first"):
        ordered_cpu_overlap(range(10), evaluate, finalize, max_workers=1, max_in_flight=2)
    assert seen == [0, 1]


def test_atomic_csv_preserves_rows_and_existing_mode(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("old\n", encoding="utf-8")
    path.chmod(0o640)
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([["step", "value"], [1, "a,b"], [1, "a,b"]])
    with path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [["step", "value"], ["1", "a,b"], ["1", "a,b"]]
    assert path.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_new_file_uses_normal_creation_permissions(tmp_path):
    control = tmp_path / "normal.txt"
    control.write_text("normal", encoding="utf-8")
    path = tmp_path / "atomic.txt"
    with atomic_text_writer(path) as handle:
        handle.write("atomic")
    assert path.stat().st_mode & 0o777 == control.stat().st_mode & 0o777


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_atomic_failure_preserves_old_file_and_cleans_temp(tmp_path, monkeypatch, failure):
    path = tmp_path / "output.txt"
    path.write_text("old data", encoding="utf-8")
    if failure == "replace":
        def fail_replace(src, dst):
            assert os.path.dirname(src) == str(tmp_path)
            assert os.fspath(dst) == str(path)
            raise OSError("replace failed")
        monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="failed"):
        with atomic_text_writer(path) as handle:
            handle.write("incomplete data")
            if failure == "write":
                raise OSError("write failed")
    assert path.read_text(encoding="utf-8") == "old data"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_csv_with_real_pandas_consumer(tmp_path):
    pandas = pytest.importorskip("pandas")
    path = tmp_path / "scores.csv"
    frame = pandas.DataFrame({"step": [0, 0], "score": [1.0, 2.0]})
    with atomic_text_writer(path, newline="") as handle:
        frame.to_csv(handle, mode="w", header=True, index=False)
    pandas.testing.assert_frame_equal(pandas.read_csv(path), frame)


def test_atomic_relative_symlink_updates_target_and_keeps_link(tmp_path, monkeypatch):
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    target = target_dir / "scores.csv"
    target.write_text("step,score\n0,1\n", encoding="utf-8")
    target.chmod(0o640)
    link = tmp_path / "scores.csv"
    link.symlink_to("targets/scores.csv")
    monkeypatch.chdir(tmp_path)
    with atomic_text_writer("scores.csv", newline="") as handle:
        csv.writer(handle).writerows([["step", "score"], [1, 2]])
    assert link.is_symlink()
    assert os.readlink(link) == "targets/scores.csv"
    assert target.stat().st_mode & 0o777 == 0o640
    with link.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [["step", "score"], ["1", "2"]]
    assert list(target_dir.iterdir()) == [target]


def test_atomic_dangling_symlink_creates_target_and_serialization_failure_preserves_old(tmp_path):
    target = tmp_path / "actual.csv"
    link = tmp_path / "link.csv"
    link.symlink_to("actual.csv")
    with atomic_text_writer(link, newline="") as handle:
        csv.writer(handle).writerow(["original"])
    assert link.is_symlink()
    before = target.read_bytes()
    class CannotSerialize:
        def __str__(self):
            raise ValueError("serialization failed")
    with pytest.raises(ValueError, match="serialization failed"):
        with atomic_text_writer(link, newline="") as handle:
            csv.writer(handle).writerow(["partial", CannotSerialize()])
    assert target.read_bytes() == before
    assert link.is_symlink()
    assert set(tmp_path.iterdir()) == {target, link}


def test_atomic_close_failure_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "scores.csv"
    path.write_text("old\n", encoding="utf-8")
    original_fdopen = os.fdopen
    class FailingClose:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self.handle
        def __exit__(self, *exc):
            self.handle.close()
            raise OSError("close failed")
    monkeypatch.setattr(os, "fdopen", lambda *args, **kwargs: FailingClose(original_fdopen(*args, **kwargs)))
    with pytest.raises(OSError, match="close failed"):
        with atomic_text_writer(path) as handle:
            handle.write("new\n")
    assert path.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.iterdir()) == [path]
