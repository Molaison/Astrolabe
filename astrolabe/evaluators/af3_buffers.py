"""AF3 numeric data ownership. The scheduler forwards references, never arrays.

One private service epoch owns the namespace. Each client owns one request
directory and removes it only after a successful completed response and mapping
all outputs. Unknown completion retains it; no request is replayed.
The socket holds only a private descriptor. Numeric bundles live on node-local
tmpfs so unlinking mapped outputs never creates NFS .nfs files.
"""
from __future__ import annotations

from pathlib import Path
import errno
import json
import os
import re
import shutil
import stat
import tempfile
import time
import uuid

from astrolabe.evaluators.inference_service import json_call, mark


def private_root(path, *, create=False):
    path = Path(path)
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise PermissionError(f"AF3 buffers require a private directory: {path}")
    return path


def _tmpfs(path):
    """Use the longest containing mount, not the filesystem of /tmp or the socket."""
    path = Path(path)
    best = (-1, None)
    with open("/proc/self/mountinfo", encoding="utf-8") as mounts:
        for line in mounts:
            before, after = line.rstrip("\n").split(" - ", 1)
            mount = Path(re.sub(r"\\([0-7]{3})",
                                lambda m: chr(int(m.group(1), 8)),
                                before.split()[4]))
            if (path == mount or mount in path.parents) and len(mount.parts) > best[0]:
                best = (len(mount.parts), after.split()[0])
    return best[1] == "tmpfs"


def _namespace_root(path, *, missing=False):
    if type(path) is not str or not Path(path).is_absolute():
        raise ValueError("AF3 buffer root must be an absolute path")
    root = Path(path)
    if root.resolve(strict=False) != root or not _tmpfs(root):
        raise ValueError(f"AF3 buffer root must be on a real tmpfs path: {root}")
    try:
        return private_root(root)
    except FileNotFoundError:
        if missing:
            return root
        raise


def _descriptor(socket_path):
    return Path(socket_path).parent / "buffers.json"


def _read_namespace(socket_path, *, missing_root=False):
    descriptor = _descriptor(socket_path)
    private_root(descriptor.parent)
    fd = os.open(descriptor, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise PermissionError(f"AF3 buffer descriptor must be private: {descriptor}")
        with os.fdopen(fd, encoding="utf-8") as stream:
            fd = -1
            data = json.load(stream)
    finally:
        if fd != -1:
            os.close(fd)
    if (type(data) is not dict or set(data) != {"version", "root", "owned"}
            or type(data["version"]) is not int or data["version"] != 1
            or type(data["owned"]) is not bool):
        raise ValueError(f"Invalid AF3 buffer descriptor: {descriptor}")
    return _namespace_root(data["root"], missing=missing_root and data["owned"]), data["owned"]


def prepare_namespace(socket_path, shared_root=None) -> Path:
    """Publish a private control descriptor; numeric bundles live only on tmpfs."""
    parent = private_root(Path(socket_path).parent, create=True)
    owned = shared_root is None
    if owned:
        if not _tmpfs(Path("/dev/shm")):
            raise RuntimeError("AF3 buffers require /dev/shm mounted on tmpfs")
        root = Path(tempfile.mkdtemp(prefix="astrolabe-af3-buffers-", dir="/dev/shm"))
    else:
        root = _namespace_root(str(shared_root))
    descriptor = parent / "buffers.json"
    created = False
    try:
        fd = os.open(descriptor, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        created = True
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"version": 1, "root": str(root), "owned": owned}, stream)
        return root
    except BaseException:
        if created:
            descriptor.unlink()
        if owned:
            root.rmdir()  # Only this newly created, empty root may be removed.
        raise


def resolve_namespace(socket_path) -> Path:
    root, _ = _read_namespace(socket_path)
    return root


def cleanup_namespace(socket_path) -> str | None:
    """Release only this socket's descriptor and an empty, owned root."""
    descriptor = _descriptor(socket_path)
    try:
        root, owned = _read_namespace(socket_path, missing_root=True)
    except FileNotFoundError:
        if descriptor.exists():
            raise
        return None
    retained = None
    if owned and root.exists():
        try:
            root.rmdir()
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            retained = str(root)
    descriptor.unlink()
    return retained


def check_reference(request):
    allowed = {"model", "settings", "identity", "cyclic_asym_ids",
               "cyclic_relative_patches", "rng_seed", "buffers", "request_id"}
    ref = request.get("buffers")
    if (set(request) - allowed or request.get("model") != "alphafold3"
            or type(ref) is not dict or set(ref) != {"version", "token"}
            or type(ref["version"]) is not int or ref["version"] != 1
            or type(ref["token"]) is not str
            or re.fullmatch("[0-9a-f]{32}", ref["token"]) is None):
        raise ValueError("Invalid AF3 numeric buffer reference")
    if "rng_seed" in request and type(request["rng_seed"]) is not int:
        raise ValueError("AF3 rng_seed must be an integer")
    return ref["token"]


def infer(socket_path, request, numerical_input):
    """Publish once, invoke once, map completed outputs before unlinking names."""
    from astrolabe.evaluators.numeric_buffers import read_bundle, write_bundle

    root = resolve_namespace(socket_path)
    token = uuid.uuid4().hex
    directory = root / token
    directory.mkdir(mode=0o700)
    request = dict(request, buffers={"version": 1, "token": token})
    check_reference(request)
    started = time.monotonic()
    try:
        stats = write_bundle(directory / "input", numerical_input)
    except BaseException:
        # No request sent: there cannot be a remote reader.
        shutil.rmtree(directory)
        raise
    mark("buffer_input_published", elapsed_s=time.monotonic() - started,
         identity=request.get("identity"), **stats)
    # No finally cleanup: EOF, OOM or a lost reply is not proof of completion.
    response = json_call(socket_path, request)
    if response != {"buffer_result": 1, "token": token}:
        raise ValueError("AF3 result reference does not match the request")
    started = time.monotonic()
    result = read_bundle(directory / "output")
    # Linux mmap views retain their mappings independently of these names.
    # No mmap.close() is called; consumers may retain slices through scoring.
    shutil.rmtree(directory)
    mark("buffer_output_mapped", elapsed_s=time.monotonic() - started,
         identity=request.get("identity"))
    return result


class BufferHandler:
    """Scientific endpoint adapter; never used in the forwarding parent."""

    def __init__(self, handler, root):
        self.handler = handler
        self.root = private_root(root)
        self.quarantined = False

    def __getattr__(self, name):
        return getattr(self.handler, name)

    def __call__(self, request):
        from astrolabe.evaluators.numeric_buffers import read_bundle, write_bundle

        token = check_reference(request)
        if self.quarantined:
            raise RuntimeError("AF3 numeric owner is quarantined; request not run")
        directory = private_root(self.root / token)
        started = time.monotonic()
        inputs = read_bundle(directory / "input")
        if type(inputs) is not dict or set(inputs) not in ({"features"}, {"features", "rng_key"}):
            raise ValueError("Invalid AF3 numeric input bundle")
        if ("rng_seed" in request) == ("rng_key" in inputs):
            raise ValueError("AF3 requires exactly one RNG seed or key")
        science = {key: value for key, value in request.items() if key != "buffers"}
        science.update(inputs)
        mark("buffer_input_mapped", elapsed_s=time.monotonic() - started,
             identity=request.get("identity"))
        try:
            result = self.handler(science)  # Native completion includes D2H.
            started = time.monotonic()
            stats = write_bundle(directory / "output", result)
            mark("buffer_output_published", elapsed_s=time.monotonic() - started,
                 identity=request.get("identity"), **stats)
        except BaseException:
            self.quarantined = True
            raise
        return {"buffer_result": 1, "token": token}
