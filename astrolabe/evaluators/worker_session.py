"""Private direct sessions and immutable broker-shared worker specifications."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import shutil
from contextlib import contextmanager


@contextmanager
def request_artifacts(root):
    """Keep files of outcome-unknown work until its owning session is fenced.

    A broker transport timeout does not establish that its GPU worker stopped.
    Deleting that worker's input/output tree could turn a merely unknown result
    into a partial execution. The private broker session owns retained trees.
    """
    from astrolabe.runtime.client import OutcomeUnknown
    path = Path(tempfile.mkdtemp(prefix='request-', dir=root))
    uncertain = False
    try:
        yield str(path)
    except OutcomeUnknown as exc:
        uncertain = True
        exc.retained_request_dir = str(path)
        raise
    finally:
        if not uncertain:
            shutil.rmtree(path, ignore_errors=True)


class WorkerArtifacts:
    def __init__(self):
        endpoint = os.environ.get('ASTROLABE_EXECUTION_ENDPOINT')
        token = os.environ.get('ASTROLABE_EXECUTION_TOKEN')
        if bool(endpoint) != bool(token):
            raise ValueError('incomplete broker credentials')
        self.shared = bool(endpoint)
        self._temporary = None
        self._retained = False
        if self.shared:
            value = os.environ.get('ASTROLABE_EXECUTION_ARTIFACT_ROOT')
            if not value: raise ValueError('broker artifact root is required')
            self.root = Path(value).resolve(strict=True)
            info = self.root.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError('broker artifact root must be a private owned directory')
        else:
            self._temporary = tempfile.TemporaryDirectory(prefix='astrolabe-worker-')
            self.root = Path(self._temporary.name).resolve(strict=True)

    def specification(self, identity, spec):
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        path = self.root / ('worker-' + digest + '.json')
        data = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        lock_path = self.root / ('worker-' + digest + '.lock')
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if path.exists():
                if path.is_symlink() or path.read_bytes() != data:
                    raise ValueError('same worker identity has incompatible startup specification')
            else:
                temp_fd, temp_name = tempfile.mkstemp(prefix='.worker-', dir=self.root)
                try:
                    with os.fdopen(temp_fd, 'wb') as f:
                        f.write(data); f.flush(); os.fsync(f.fileno())
                    os.replace(temp_name, path)
                finally:
                    if os.path.exists(temp_name): os.unlink(temp_name)
        finally:
            os.close(fd)
        return path

    def retain(self):
        """Keep uncertain work even when its Python owner is garbage collected.

        This is not a release assertion. A shared root remains the session's
        responsibility; a private TemporaryDirectory loses its automatic
        finalizer until an explicit confirmed cleanup is requested.
        """
        self._retained = True
        if self._temporary is not None:
            self._temporary._finalizer.detach()

    def start_client(self, *args, **kwargs):
        from astrolabe.runtime.client import DaemonClient, OutcomeUnknown
        try:
            return DaemonClient(*args, **kwargs)
        except OutcomeUnknown as exc:
            self.retain()
            exc.retained_artifact_root = str(self.root)
            raise

    def close_client(self, client, *, release=False):
        """Only successful client release permits private artifact cleanup."""
        from astrolabe.runtime.client import OutcomeUnknown
        try:
            client.close(release=release)
        except BaseException as exc:
            self.retain()
            error = exc if isinstance(exc, OutcomeUnknown) else OutcomeUnknown('worker release outcome unknown; artifacts retained')
            error.retained_artifact_root = str(self.root)
            if error is exc:
                raise
            raise error from exc
        self.close(release_confirmed=True)

    def close(self, *, release_confirmed=False):
        if self._retained and not release_confirmed:
            return
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
