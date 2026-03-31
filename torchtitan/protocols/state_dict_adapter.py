import json, os, re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
try:
    from torch.distributed.checkpoint import HuggingFaceStorageReader
except ImportError:
    HuggingFaceStorageReader = None
from torchtitan.tools.logging import logger
from .model import BaseModelArgs
class BaseStateDictAdapter(ABC):
    fqn_to_index_mapping: Dict[Any, int] | None
    hf_assets_path: str | None
    @abstractmethod
    def __init__(self, model_args, hf_assets_path): pass
    @abstractmethod
    def to_hf(self, state_dict): pass
    @abstractmethod
    def from_hf(self, hf_state_dict): pass
    @abstractmethod
    def get_hf_storage_reader(self, path, from_quantized=False): pass
class StateDictAdapter(BaseStateDictAdapter):
    def __init__(self, model_args, hf_assets_path):
        self.hf_assets_path = hf_assets_path
        if hf_assets_path:
            mapping_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
            try:
                with open(mapping_path, "r") as f: hf_safetensors_indx = json.load(f)
            except FileNotFoundError:
                hf_safetensors_indx = None
            if hf_safetensors_indx:
                self.fqn_to_index_mapping = {k: int(re.search(r"\\d+", v).group(0)) for k, v in hf_safetensors_indx["weight_map"].items()}
            else: self.fqn_to_index_mapping = None
        else: self.fqn_to_index_mapping = None
    def get_hf_storage_reader(self, path, from_quantized=False):
        if HuggingFaceStorageReader is None: raise ImportError("HuggingFaceStorageReader is not available")
        return HuggingFaceStorageReader(path)
