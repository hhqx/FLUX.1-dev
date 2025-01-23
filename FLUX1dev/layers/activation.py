import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils.import_utils import is_torch_version
from ..utils import get_world_size


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, is_tp: bool = False):
        super().__init__()
        if is_tp:
            dim_out = dim_out // get_world_size()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type == "mps" and is_torch_version("<", "2.0.0"):
            # fp16 gelu not supported on mps before torch 2.0
            return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states