"""
Test module for FluxTransformerBlock anti-smoothing.

This module provides a class that applies anti-smoothing techniques to a FluxTransformerBlock,
which ensures that certain weight transformations preserve the model's output.
"""

from FLUX1dev.models.transformer_flux import FluxTransformerBlock
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple

class FluxTransformerBlock_anti(FluxTransformerBlock):
    """
    Anti-smoothing version of FluxTransformerBlock.
    
    This class extends FluxTransformerBlock to implement weight transformation techniques
    that modify internal parameters while preserving the model's output behavior.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def apply_smoothing_qkv(self, qkv: List[nn.Linear], ada_norm: AdaLayerNormZero):
        """
        Apply anti-smoothing to Q, K, V projections and AdaLayerNorm.
        
        This function:
        1. Creates random smoothing scale factors
        2. Scales down normalization weights
        3. Scales up Q, K, V projection weights
        
        Args:
            qkv: List of linear layers for query, key, value projections
            ada_norm: AdaLayerNormZero to apply smoothing to
        """
        # Create smoothing scale factors with fixed seed for reproducibility
        torch.manual_seed(42)
        
        dim = qkv[0].weight.shape[1]  # Input dimension
        device = qkv[0].weight.device
        smooth_scale = 2 + torch.randn((dim,)).abs().to(device)
        
        # Scale down normalization weights - shift component
        norm1 = ada_norm
        norm1.linear.weight.data[:dim].div_(smooth_scale.view(-1, 1))
        norm1.linear.bias.data[:dim].div_(smooth_scale.view(-1))
        
        # Scale down normalization weights - scale component
        norm1.linear.weight.data[dim:2*dim].div_(smooth_scale.view(-1, 1))
        norm1.linear.bias.data[dim:2*dim] = (norm1.linear.bias.data[dim:2*dim] + 1) / smooth_scale.view(-1) - 1
        
        # Scale up Q, K, V projections
        for linear in qkv:
            linear.weight.data.mul_(smooth_scale.view(1, -1))
    
    def apply_smoothing_mlp_up(self, up: List[nn.Linear], ada_norm: AdaLayerNormZero):
        """
        Apply anti-smoothing to MLP up projections and AdaLayerNorm.
        
        This function:
        1. Creates random smoothing scale factors
        2. Scales down normalization weights for MLP components
        3. Scales up MLP up-projection weights
        
        Args:
            up: List of linear layers for MLP up projections
            ada_norm: AdaLayerNormZero to apply smoothing to
        """
        # Create smoothing scale factors with fixed seed for reproducibility
        torch.manual_seed(42)
        
        dim = up[0].weight.shape[1]  # Input dimension
        device = up[0].weight.device
        smooth_scale = 2 + torch.randn((dim,)).abs().to(device)
        
        # Scale down normalization weights for MLP - shift component
        norm1 = ada_norm
        norm1.linear.weight.data[3*dim:4*dim].div_(smooth_scale.view(-1, 1))
        norm1.linear.bias.data[3*dim:4*dim].div_(smooth_scale.view(-1))
        
        # Scale down normalization weights for MLP - scale component
        norm1.linear.weight.data[4*dim:5*dim].div_(smooth_scale.view(-1, 1))
        norm1.linear.bias.data[4*dim:5*dim] = (norm1.linear.bias.data[4*dim:5*dim] + 1) / smooth_scale.view(-1) - 1
        
        # Scale up MLP up projections
        for linear in up:
            linear.weight.data.mul_(smooth_scale.view(1, -1))
    
    def apply_smoothing_mlp_down(self, down: nn.Linear, up: nn.Linear):
        """
        Apply anti-smoothing to MLP down projections.
        
        Warning: This can change the output of the model and is not guaranteed
        to maintain exact equivalence.
        
        Args:
            down: Linear layer for MLP down projection
            up: Linear layer for MLP up projection
        """
        # Create smoothing scale factors with fixed seed for reproducibility
        torch.manual_seed(42)
        
        dim = down.weight.shape[1]  # Input dimension
        device = down.weight.device
        smooth_scale = 2 + torch.randn((dim,)).abs().to(device)
        
        # Scale down MLP up projection
        pre_layer = up
        pre_layer.weight.data.div_(smooth_scale.view(-1, 1))
        pre_layer.bias.data.div_(smooth_scale.view(-1))

        # Scale up MLP down projections
        down.weight.data.mul_(smooth_scale.view(1, -1))
    
    def apply_smoothing_att_o(self, att_v: nn.Linear, att_o: nn.Linear, smooth_scale: torch.Tensor):
        """
        Apply anti-smoothing to attention output projections.
        
        Args:
            att_v: Value projection linear layer
            att_o: Output projection linear layer
            smooth_scale: Pre-computed smoothing scale tensor
        
        Raises:
            AssertionError: If smooth_scale has incorrect dimensions
        """
        # Validate smooth_scale tensor
        dim = att_o.weight.shape[1]
        assert smooth_scale.numel() == dim, "The smooth scale length does not match the attention output dimension."
        assert smooth_scale.ndim == 1, "The smooth scale should be a 1D tensor."
        
        # Scale up output projection (input dimension)
        att_o.weight.data.mul_(smooth_scale.view(1, -1))
        
        # Scale down value projection (output dimension)
        att_v.weight.data.div_(smooth_scale.view(-1, 1))
        att_v.bias.data.div_(smooth_scale.view(-1))
        
    def do_smooth_qkv(self):
        """
        Apply anti-smoothing to query, key, value projections in both main and context paths.
        """
        # Apply smoothing to main path Q, K, V projections
        qkv = [self.attn.to_q, self.attn.to_k, self.attn.to_v]
        self.apply_smoothing_qkv(qkv, self.norm1)
        
        # Apply smoothing to context path Q, K, V projections
        qkv_context = [self.attn.add_q_proj, self.attn.add_k_proj, self.attn.add_v_proj]
        self.apply_smoothing_qkv(qkv_context, self.norm1_context)
        
    def do_smooth_mlp_up(self):
        """
        Apply anti-smoothing to MLP up projections in both main and context paths.
        """
        # Apply smoothing to main path MLP up projection
        mlp_up = [self.ff.net[0].proj]
        self.apply_smoothing_mlp_up(mlp_up, self.norm1)
        
        # Apply smoothing to context path MLP up projection
        mlp_up_context = [self.ff_context.net[0].proj]
        self.apply_smoothing_mlp_up(mlp_up_context, self.norm1_context)
    
    def do_smooth_mlp_down(self):
        """ 
        Apply anti-smoothing to MLP down projections in both main and context paths.
        
        Warning: This will change the output of the model and is not guaranteed
        to maintain exact equivalence.
        """
        # Apply smoothing to main path MLP down projection
        self.apply_smoothing_mlp_down(
            down=self.ff.net[2],
            up=self.ff.net[0].proj,
        )
        
        # Apply smoothing to context path MLP down projection
        self.apply_smoothing_mlp_down(
            down=self.ff_context.net[2],
            up=self.ff_context.net[0].proj,
        )
    
    def do_smooth_att_o(self, share_scale: bool = True):
        """
        Apply anti-smoothing to attention output projections in both main and context paths.
        
        Args:
            share_scale: If True, use the same scaling factors for both paths.
                         If False, generate different scaling factors (may change outputs).
        """
        # Generate smoothing scale with fixed seed for reproducibility
        torch.manual_seed(42)
        dim = self.attn.to_out[0].weight.shape[1]
        device = self.attn.to_out[0].weight.device
        
        # Create scales - either shared or separate
        scale1 = 2 + torch.randn((dim,)).abs().to(device)
        if share_scale:
            scale2 = scale1
        else:
            scale2 = 2 + torch.randn((dim,)).abs().to(device)
        
        print(f"scale1: {scale1}, scale2: {scale2}")
        
        # Apply smoothing to main and context attention output projections
        self.apply_smoothing_att_o(self.attn.to_v, self.attn.to_out[0], scale1)
        self.apply_smoothing_att_o(self.attn.add_v_proj, self.attn.to_add_out, scale2)
    
    def do_smooth(self):
        """
        Apply all anti-smoothing operations to the FluxTransformerBlock.
        
        This is the main entry point for applying the complete set of anti-smoothing
        transformations that are guaranteed to preserve model outputs.
        """
        self.do_smooth_qkv()
        self.do_smooth_mlp_up()
        self.do_smooth_att_o(share_scale=True)
    
    def load_from_flux_transformer_block(self, block: FluxTransformerBlock):
        """
        Load parameters from a regular FluxTransformerBlock.
        
        Args:
            block: Source FluxTransformerBlock to copy parameters from
            
        Raises:
            ValueError: If a module in the source block doesn't exist in this block
        """
        for name, module in block.named_children():
            if hasattr(self, name):
                getattr(self, name).load_state_dict(module.state_dict())
            else:
                raise ValueError(f"Module {name} not found in FluxTransformerBlock_anti")
        
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
    encoder_hidden_states = torch.randn(batch_size, seq_length, dim, device='cpu')
    temb = torch.randn(batch_size, dim, device='cpu')
    
    # Optional: image rotary embeddings (None for this test)
    image_rotary_emb = None
    
    return dict(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
        is_tp=False
    )

from py3_tools.py_debug import Debugger
# Debugger.debug_flag = True
@Debugger.attach_on_error()
def test_flux_transformer_block_anti_smooth():
    """
    Test the FluxTransformerBlock_anti with smoothing.
    
    This function:
    1. Creates a regular and an anti-smoothed transformer block
    2. Applies smoothing transformations
    3. Verifies that outputs match after forward passes
    """
    # Define model parameters
    dim = 4  # Example dimension
    num_attention_heads = 1
    attention_head_dim = 4
    batch_size = 1
    seq_length = 3  # Common sequence length for text tokens
    
    # Create regular transformer block
    block = FluxTransformerBlock(
        dim=dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        is_tp=False
    ).eval()
    
    # Create anti-smoothed block and load weights from regular block
    block_anti = FluxTransformerBlock_anti(
        dim=dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        is_tp=False
    ).eval()
    block_anti.load_from_flux_transformer_block(block)

    # Apply anti-smoothing transformations - uncomment specific options as needed
    block_anti.do_smooth()
    
    # Uncomment to test other smoothing variants:
    # block_anti.do_smooth_att_o(share_scale=False)  # [Warning] This will cause different outputs
    # block_anti.do_smooth_mlp_down()  # [Warning] This will cause different outputs
    
    # Generate test inputs
    kwargs = get_dummy_input(batch_size, seq_length, dim)
    
    # Perform forward passes
    with torch.no_grad():
        output_anti = block_anti(**kwargs)
        output = block(**kwargs)
    
    # Check outputs match
    error0 = torch.abs(output[0] - output_anti[0]).mean().item()
    error1 = torch.abs(output[1] - output_anti[1]).mean().item()
    print(f'Error in hidden states: {error0:.8f}')
    print(f'Error in encoder hidden states: {error1:.8f}')
    
    assert torch.allclose(output[0], output_anti[0]), \
        f"Output hidden states do not match, mse: {error0:.8f}"
    assert torch.allclose(output[1], output_anti[1]), \
        f"Output encoder hidden states do not match, mse: {error1:.8f}"
    
    print("✓ Test passed: FluxTransformerBlock_anti smoothing works correctly")

if __name__ == "__main__":
    test_flux_transformer_block_anti_smooth()