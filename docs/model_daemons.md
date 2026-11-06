# Model daemons

Astrolabe normally starts a private model owner from the task that needs the
model. The owner stays with that task and drains its child when the task ends.
The `astrolabe daemon` commands are diagnostics and advanced operator controls.
They are not required for normal task execution.

The model cache policy varies by backend. Backends recorded as `in_process`
retain models in their existing native process. Some evaluators, initializers,
and mutators launch isolated children. The backend catalog is the source of
this metadata. A catalog entry does not by itself prove that every backend has
a supported standalone foreground factory.

## Inspect backends

Use `daemon backends` to read the same catalog that execution policy uses:

```console
astrolabe daemon backends
astrolabe daemon backends --json
```

The JSON form contains each backend's `name`, `component`, `aliases`,
`software`, `execution`, and `detail` fields. The `execution` value can be
`daemon`, `in_process`, `subprocess`, `custom`, `local`, or `tool`. This
command reads catalog metadata and does not import model frameworks.

## Inspect running owners

Run `daemon status` with no directory to discover current-user daemon records
in the direct children of the system temporary directory:

```console
astrolabe daemon status
astrolabe daemon status --json
```

Discovery considers only real temporary directories whose names start with
`astrolabe-model-daemon-`. Each directory must have mode `0700`, be owned by
the current user, and contain owner-readable, current-user-owned `spec.json`
and `lifecycle.json` files. Unsafe or unrelated directories are ignored during
discovery.

To inspect one exact directory, pass it explicitly:

```console
astrolabe daemon status --directory /tmp/astrolabe-model-daemon-example
```

The command tries the verified Unix socket and sends an identity-bound status
request with a 10-second timeout. A daemon reports `ready: true` only after the
live worker returns `state: "running"`. If the socket is missing or the request
fails, the command reports `state: "stopped"` or `state: "unknown"` from the
retained evidence. Neither state is ready.

Status never stops a daemon. It does not use a recorded PID as proof that the
worker is alive, and it keeps stale lifecycle evidence for diagnosis.

The JSON form has this top-level shape:

```json
{
  "schema_version": 1,
  "daemons": []
}
```

Every record includes the private `directory`, `backend`, `identity`,
`generation_recorded`, `socket`, `max_pending`, `max_frame_bytes`,
`actual_rpc`, `generation_verified`, `state`, `ready`, and `last_evidence`.
Live records also include the worker PID and parent PID, queue depth, active
request, request, completed, and failed counts, model load count, and worker
start time. A failed RPC can add an `error` field.

A status command returns zero when discovery succeeds, including when records
are stopped or unknown. It returns two when an explicitly named directory
cannot be inspected.

## Stop an explicit owner

`daemon stop` requires the exact private directory:

```console
astrolabe daemon stop --directory /tmp/astrolabe-model-daemon-example
astrolabe daemon stop --directory /tmp/astrolabe-model-daemon-example --json
```

The command validates the directory, reads its identity, validates the socket
and peer UID, and sends an identity-bound drain request. The worker completes
queued work before it exits. A successful result reports
`state: "drained"` and `actual_rpc: true`.

The command never signals a PID from `lifecycle.json`, deletes a stale record,
or cleans up another directory. If the verified socket is unavailable, the
command fails and leaves the record intact.

## Run an explicit foreground service

`daemon serve` is an advanced operation for development and explicit
supervision:

```console
astrolabe daemon serve --spec /absolute/path/to/service-spec.json
```

The process starts the selected interpreter through `OwnedModelDaemon`. It
prints a `service_ready` JSON event only after the model worker answers a real
status request. The event contains the actual private directory, socket,
identity, and worker status. The command then stays in the foreground.

On `SIGINT` or `SIGTERM`, the owner sends an identity-bound drain request and
shuts down its own child. It does not detach, create a systemd unit, install an
autostart entry, or reconstruct a model from a stale lifecycle record. The
private directory and lifecycle evidence remain after the process exits.

If another CLI command explicitly drains the child, the foreground command
exits with status zero and records a `service_stopped` event with
`reason: "drained"`. An unexpected child exit records
`reason: "child_failed"` and exits with status two.

The descriptor can use an existing `DaemonSpec` payload or this simple form:

```json
{
  "backend": "example-cpu",
  "factory": "example_adapter:make",
  "settings": {
    "variant": "cpu"
  },
  "python_bin": "/absolute/path/to/venv/bin/python",
  "cwd": "/absolute/path/to/adapter",
  "env": {
    "PYTHONPATH": "/absolute/path/to/adapter"
  },
  "max_pending": 4,
  "max_frame_bytes": 8388608
}
```

The required `backend` field identifies the owner. The required `factory`
field names an importable callable as `module:callable` or
`module.callable`. The selected interpreter must be able to import that module.
The factory receives `settings` and returns either a callable that accepts one
payload or an object with an `infer(payload)` method.

`settings` must be a JSON object. `python_bin`, `cwd`, `env`, `provider`,
`max_pending`, `max_frame_bytes`, `identity`, and `generation` are optional
fields. `env` must map strings to strings. An `identity` in the descriptor is
checked against the backend, factory, settings, interpreter, working
directory, environment, provider, and bounds before startup.

The descriptor must be a regular, current-user-owned JSON file and cannot be a
symlink. Factory modules and scientific defaults remain the responsibility of
the backend owner. Do not treat the example adapter name as a built-in model.

Worker stdout and stderr are written to `owner.jsonl` in the private directory.
The descriptor is written to `spec.json`, and lifecycle transitions are written
to `lifecycle.json`. Status and stop commands use the protected socket and
identity from those records. They do not import or reconstruct the factory.
