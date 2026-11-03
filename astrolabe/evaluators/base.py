from __future__ import annotations

from abc import abstractmethod
import time


class Evaluator:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, inputs=None, runtime_config=None):
        if getattr(self, '_closed', False):
            raise RuntimeError('evaluator is closed')
        from astrolabe.runtime.admission import stage_event
        identity = {'backend': type(self).__module__ + '.' + type(self).__name__}
        start = time.perf_counter()
        input_feat = self.prepare_input(inputs, runtime_config)
        stage_event('cpu_prepare', duration=time.perf_counter() - start, identity=identity)
        start = time.perf_counter()
        output = self.run_function(input_feat)
        if getattr(self, 'execution_phase', None):
            stage_event(self.execution_phase, duration=time.perf_counter() - start, identity=identity)
        start = time.perf_counter()
        std_output = self.prepare_output(output, inputs, runtime_config)
        stage_event(getattr(self, 'finalize_phase', 'cpu_finalize'),
                    duration=time.perf_counter() - start, identity=identity)
        return std_output

    def lifecycle(self):
        """Expose owned model evidence without inferring it from a live owner."""
        import os
        from astrolabe.evaluators.lifecycle import model_lifecycle
        states, seen = [], set()
        def visit(handle):
            if handle is None or id(handle) in seen:
                return
            seen.add(id(handle))
            method = getattr(handle, 'lifecycle', None)
            if callable(method):
                states.append(method())
            elif getattr(handle, '_client', None) is not None:
                states.append(dict(handle._client.lifecycle))
            elif getattr(handle, '_resident', None) is not None:
                visit(handle._resident)
            elif hasattr(handle, '_weight_load_count'):
                count = handle._weight_load_count
                states.append(model_lifecycle(getattr(handle, 'model', None), load_count=count, loaded=count > 0))
            elif hasattr(handle, '_astrolabe_weight_load_count'):
                count = handle._astrolabe_weight_load_count
                states.append(model_lifecycle(getattr(handle, 'params', None), load_count=count, loaded=count > 0))
        for name in ('resident', '_resident', 'model_runner', 'runner'):
            visit(getattr(self, name, None))
        for handle in getattr(self, 'model_runners', {}).values():
            visit(handle)
        if getattr(self, '_model', None) is not None:
            states.append(model_lifecycle(self._model, load_count=getattr(self, '_weight_load_count', 0), loaded=True))
        loaded = [state.get('weights_loaded') for state in states]
        weights_loaded = False if any(value is False for value in loaded) else (
            True if loaded and all(value is True for value in loaded) else None)
        return {'process_alive': True, 'pid': os.getpid(), 'ready': not getattr(self, '_closed', False),
                'weights_loaded': weights_loaded,
                'gpu_resident': None, 'models': states, 'release_requires_exit': True}

    def close(self):
        """Release only this evaluator's model handles, once per handle."""
        if getattr(self, '_closed', False):
            previous = getattr(self, '_release_errors', ())
            if previous:
                raise RuntimeError(f'{len(previous)} evaluator resource release(s) remain unconfirmed') from previous[0]
            return
        self._closed = True
        handles = []
        for name in ('resident', '_resident', 'model_runner', 'runner', '_model'):
            value = getattr(self, name, None)
            if value is not None:
                handles.append(value)
                setattr(self, name, None)
        handles.extend(getattr(self, 'model_runners', {}).values())
        if hasattr(self, 'model_runners'):
            self.model_runners = {}
        seen = set()
        errors = []
        for value in handles:
            if id(value) in seen:
                continue
            seen.add(id(value))
            method = getattr(value, 'close', None) or getattr(value, 'shutdown', None)
            if callable(method):
                try:
                    method()
                except Exception as exc:
                    errors.append(exc)
        # Dropping a tensor reference is not proof that a framework allocator
        # returned VRAM. Resident workers report process exit separately.
        if errors:
            self._release_errors = tuple(errors)
            raise RuntimeError(f'{len(errors)} evaluator resource release(s) failed') from errors[0]

    @abstractmethod
    def run_function(self, input_feat):
        pass

    @abstractmethod
    def prepare_input(self, inputs, runtime_config): ...

    @abstractmethod
    def prepare_output(self, output, inputs, runtime_config): ...
