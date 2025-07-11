import torch

from FLUX1dev.models.transformer_flux import FluxTransformerBlock

import torch
import torch.nn as nn
import torch.fx as fx
from torch._dynamo import export
from typing import List, Dict, Tuple, Any, Set, Optional
import logging



def _find_linear_layers_with_fx(model: nn.Module, args, kwargs={}):
    """Use torch.fx to find linear layers and their connections."""
    # Create a symbolic trace of the model
    traced_model = fx.symbolic_trace(model)
    graph = traced_model.graph
    

def _find_linear_layers_with_dynamo(model: nn.Module, args, kwargs={}):
    """Use TorchDynamo to find linear layers and their connections."""
    # Export the model with TorchDynamo - using updated API
    exported_program = export(model)(*args, **kwargs)
    fx_graph = exported_program.graph_module
    graph = fx_graph.graph
    