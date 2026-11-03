"""Own Chai's actual component and ESM caches inside one serial worker.

The stock pinned API offloads after a normal context exit but omits ``finally``.
These context managers preserve its calls and add deterministic error cleanup.
Device retention is explicit; it never changes Chai's low_memory compute mode.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import time

from astrolabe.evaluators.lifecycle import model_lifecycle, tensor_residency


class ChaiComponents:
    def __init__(self, chai, *, residency="cpu", use_esm=True):
        if residency not in {"cpu", "device"}:
            raise ValueError("Chai component_residency must be cpu or device")
        self.chai = chai
        self.residency = residency
        self.loads = {}
        self.elapsed = {}
        self.original_component = chai._component_moved_to
        chai._component_moved_to = self.component
        self.esm = None
        if use_esm:
            self.esm = importlib.import_module("chai_lab.data.dataset.embeddings.esm")
            self.original_esm = self.esm.esm_model
            self.esm.esm_model = self.esm_model

    @contextmanager
    def component(self, comp_key, device):
        cache = self.chai._component_cache
        if comp_key not in cache:
            cache[comp_key] = self.chai.load_exported(comp_key, device)
            self.loads[comp_key] = self.loads.get(comp_key, 0) + 1
        component = cache[comp_key]
        start = time.perf_counter()
        failed = True
        try:
            component.jit_module.to(device)
            yield component
            failed = False
        finally:
            try:
                if self.residency == "cpu" or failed:
                    component.jit_module.to("cpu")
            finally:
                self.elapsed[comp_key] = self.elapsed.get(comp_key, 0.0) + time.perf_counter() - start

    @contextmanager
    def esm_model(self, device):
        import torch
        local_path = self.esm.downloads_path / "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
        if not local_path.is_file():
            raise FileNotFoundError(f"Chai ESM weights are not cached: {local_path}")
        if not self.esm._esm_model:
            if device != torch.device("cuda:0"):
                model = torch.jit.load(local_path, map_location="cpu").to(device)
            else:
                model = torch.jit.load(local_path).to(device)
            self.esm._esm_model.append(model)
            self.loads["esm"] = self.loads.get("esm", 0) + 1
        model, = self.esm._esm_model
        start = time.perf_counter()
        failed = True
        try:
            model.to(device)
            model.eval()
            yield model
            failed = False
        finally:
            try:
                if self.residency == "cpu" or failed:
                    model.to("cpu")
            finally:
                self.elapsed["esm"] = self.elapsed.get("esm", 0.0) + time.perf_counter() - start

    def lifecycle(self):
        models = {k: v.jit_module for k, v in self.chai._component_cache.items()}
        if self.esm is not None and self.esm._esm_model:
            models["esm"] = self.esm._esm_model[0]
        components = {key: {"load_count": self.loads.get(key, 0),
                            "weights_loaded": True,
                            "admitted_seconds": self.elapsed.get(key, 0.0),
                            **tensor_residency(model)} for key, model in models.items()}
        required = {'feature_embedding.pt', 'bond_loss_input_proj.pt', 'token_embedder.pt',
                    'trunk.pt', 'diffusion_module.pt', 'confidence_head.pt'}
        return model_lifecycle(models, load_count=sum(self.loads.values()), loaded=required.issubset(models),
                               components=components, residency_kind="resident_components")

    def close(self):
        self.chai._component_moved_to = self.original_component
        errors = []
        for component in self.chai._component_cache.values():
            try:
                component.jit_module.to("cpu")
            except Exception as exc:
                errors.append(exc)
        self.chai._component_cache.clear()
        if self.esm is not None:
            self.esm.esm_model = self.original_esm
            for model in self.esm._esm_model:
                try:
                    model.to("cpu")
                except Exception as exc:
                    errors.append(exc)
            self.esm._esm_model.clear()
        if errors:
            raise RuntimeError('Chai component offload failed; worker exit is required') from errors[0]
