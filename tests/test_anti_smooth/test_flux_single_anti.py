"""
Test module for FluxSingleTransformerBlock anti-smoothing.

This module provides a class that applies anti-smoothing techniques to a FluxSingleTransformerBlock,
which ensures that certain weight transformations preserve the model's output.
"""

from FLUX1dev.models.transformer_flux import FluxSingleTransformerBlock
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
import torch
import torch.nn as nn
from typing import List, Dict, Any

class FluxSingleTransformerBlock_anti(FluxSingleTransformerBlock):
    """
    Anti-smoothing version of FluxSingleTransformerBlock.
    
    This class extends FluxSingleTransformerBlock to implement weight transformation techniques
    that modify internal parameters while preserving the model's output behavior.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def apply_smoothing_qkv_proj_mlp(self, smooth_layers: List[nn.Linear], norm: AdaLayerNormZeroSingle):
        """
        Apply anti-smoothing to transformer block weights via the ada_ln_smooth technique.
        
        This function:
        1. Creates a random smoothing scale
        2. Scales down normalization weights
        3. Scales up projection weights (Q, K, V, MLP)
        
        Args:
            smooth_layers: List of linear layers to be scaled up
            norm: AdaLayerNormZeroSingle layer to be scaled down
        """
        # Create smoothing scale factors with fixed seed for reproducibility
        torch.manual_seed(42)
        
        dim = smooth_layers[0].weight.shape[1]
        device = smooth_layers[0].weight.device
        smooth_scale = 2 + torch.randn((dim,)).abs().to(device)
        
        # Scale down normalization layer weights
        # First dim entries are for shift
        norm.linear.weight.data[:dim].div_(smooth_scale.view(-1, 1))
        norm.linear.bias.data[:dim].div_(smooth_scale.view(-1))
        
        # Next dim entries are for scale
        norm.linear.weight.data[dim:2*dim].div_(smooth_scale.view(-1, 1))
        norm.linear.bias.data[dim:2*dim] = (norm.linear.bias.data[dim:2*dim] + 1) / smooth_scale.view(-1) - 1
        
        # Scale up projection layers (Q, K, V, MLP)
        for linear in smooth_layers:
            linear.weight.data.mul_(smooth_scale.view(1, -1))
    
    def apply_smoothing_proj_out(self, only_smooth_attn_v: bool = True):
        """
        Apply anti-smoothing to output projection weights.
        
        This function:
        1. Creates a random smoothing scale
        2. Scales down attention value and MLP projection weights
        3. Scales up the final output projection
        
        Args:
            only_smooth_attn_v: If True, only apply smoothing to attention value projection
        """
        # Create smoothing scale factors with fixed seed
        torch.manual_seed(42)
        
        # Get relevant layers
        proj_out = self.proj_out
        v_proj = self.attn.to_v
        mlp_proj = self.proj_mlp
        
        # Create smoothing scale
        dim = proj_out.weight.shape[1]
        device = proj_out.weight.device
        smooth_scale = 2 + torch.randn((dim,)).abs().to(device)
        
        # Verify dimensions match
        dim_v = v_proj.weight.shape[0]
        dim_mlp_proj = mlp_proj.weight.shape[0]
        assert dim == dim_v + dim_mlp_proj, "Output projection dimension mismatch with attention V + MLP dimensions"
        
        # Optionally skip smoothing MLP projection
        if only_smooth_attn_v:
            smooth_scale[-dim_mlp_proj:] = 1.0
            
        # Scale down V projection and MLP projection
        v_proj.weight.data.div_(smooth_scale[:dim_v].view(-1, 1))
        v_proj.bias.data.div_(smooth_scale[:dim_v].view(-1))
        
        mlp_proj.weight.data.div_(smooth_scale[-dim_mlp_proj:].view(-1, 1))
        mlp_proj.bias.data.div_(smooth_scale[-dim_mlp_proj:].view(-1))
        
        # Update module references
        self.attn.to_v = v_proj
        self.proj_mlp = mlp_proj
        
        # Scale up output projection
        proj_out.weight.data.mul_(smooth_scale.view(1, -1))
    
    def do_smooth(self):
        """
        Apply all anti-smoothing operations to the FluxSingleTransformerBlock.
        
        This is the main entry point for applying the complete anti-smoothing transformation.
        """
        # Apply smoothing to normalization and input projections
        self.apply_smoothing_qkv_proj_mlp(
            smooth_layers=[
                self.attn.to_q, self.attn.to_k, self.attn.to_v, self.proj_mlp
            ],
            norm=self.norm,
        )
        
        # Apply smoothing to output projection
        self.apply_smoothing_proj_out(
            only_smooth_attn_v=True,
        )
    
    def load_from_flux_single_transformer_block(self, block: FluxSingleTransformerBlock):
        """
        Load parameters from a regular FluxSingleTransformerBlock.
        
        Args:
            block: Source FluxSingleTransformerBlock to copy parameters from
        
        Raises:
            ValueError: If a module in the source block doesn't exist in this block
        """
        for name, module in block.named_children():
            if hasattr(self, name):
                getattr(self, name).load_state_dict(module.state_dict())
            else:
                raise ValueError(f"Module {name} not found in FluxSingleTransformerBlock_anti")


def get_dummy_input(batch_size: int, seq_length: int, dim: int) -> Dict[str, Any]:
    """
    Create dummy inputs for testing transformer blocks.
    
    Args:
        batch_size: Number of samples in a batch
        seq_length: Length of input sequences
        dim: Model hidden dimension
        
    Returns:
        Dictionary containing input tensors for the transformer block
    """
    # Create input tensors on CPU
    hidden_states = torch.randn(batch_size, seq_length, dim, device='cpu')
    temb = torch.randn(batch_size, dim, device='cpu')
    
    # Image rotary embeddings (None for this test)
    image_rotary_emb = None
    
    return dict(
        hidden_states=hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb
    )


from py3_tools.py_debug import Debugger
# Debugger.debug_flag = True
@Debugger.attach_on_error()
def test_flux_single_transformer_block_anti_smooth():
    """
    Test the FluxSingleTransformerBlock_anti with smoothing.
    
    This function:
    1. Creates a regular and an anti-smoothed transformer block
    2. Applies smoothing transformations
    3. Verifies that outputs match after forward passes
    """
    # Define model parameters
    dim = 4
    num_attention_heads = 2
    attention_head_dim = 2
    batch_size = 1
    seq_length = 3
    
    # Create regular transformer block
    block = FluxSingleTransformerBlock(
        dim=dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        is_tp=False
    ).eval()
    
    # Create anti-smoothed block and load weights from regular block
    block_anti = FluxSingleTransformerBlock_anti(
        dim=dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        is_tp=False
    ).eval()
    block_anti.load_from_flux_single_transformer_block(block)

    # Apply anti-smoothing transformations
    block_anti.do_smooth()
    # Uncomment to test other smoothing variants:
    # block_anti.apply_smoothing_proj_out(only_smooth_attn_v=False)  # [Warning] Will cause different outputs
    
    # Generate test inputs
    kwargs = get_dummy_input(batch_size, seq_length, dim)
    
    # Perform forward passes
    with torch.no_grad():
        output = block(**kwargs)
        output_anti = block_anti(**kwargs)
    
    # Check outputs match
    error = torch.abs(output - output_anti).mean().item()
    print(f'Average error: {error:.8f}')
    assert torch.allclose(output, output_anti), f"Output hidden states do not match, error: {error:.8f}"
    print("✓ Test passed: FluxSingleTransformerBlock_anti smoothing works correctly")


if __name__ == "__main__":
    test_flux_single_transformer_block_anti_smooth()