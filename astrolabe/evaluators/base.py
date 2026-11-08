from __future__ import annotations

from abc import abstractmethod


class Evaluator:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, inputs=None, runtime_config=None):
        input_feat = self.prepare_input(inputs, runtime_config)
        output = self.run_function(input_feat)
        std_output = self.prepare_output(output, inputs, runtime_config)
        return std_output

    def debug_cache_identity(self):
        """Return complete stateless-call configuration, or decline debug reuse.

        An override promises no hidden native state or external effects need
        advancing when the raw result is reused. RNG-consuming calls are also
        excluded by the cache owner. This does not grant durable recovery.
        """
        return None

    def shutdown(self):
        from astrolabe.evaluators.utils import shutdown_model_handles

        shutdown_model_handles(
            getattr(self, "model_runner", None),
            getattr(self, "model_runners", None),
            getattr(self, "runner", None),
            getattr(self, "_model", None),
        )

    @abstractmethod
    def run_function(self, input_feat):
        pass

    @abstractmethod
    def prepare_input(self, inputs, runtime_config): ...

    @abstractmethod
    def prepare_output(self, output, inputs, runtime_config): ...
