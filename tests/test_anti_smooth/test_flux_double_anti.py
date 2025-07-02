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
import argparse
import logging
import sys
import os
import time
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define smoothing operation types for clarity
class SmoothingType(Enum):
    QKV = "qkv"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"
    ATT_O = "att_o"
    ALL = "all"
    
    def __str__(self):
        return self.value

class FluxTransformerBlock_anti(FluxTransformerBlock):
    """
    Anti-smoothing version of FluxTransformerBlock.
    
    This class extends FluxTransformerBlock to implement weight transformation techniques
    that modify internal parameters while preserving the model's output behavior.
    """
    
    def __init__(self, *args, verbose=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        
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
        
        if self.verbose:
            logger.info(f"QKV Smoothing - Dimension: {dim}")
            logger.info(f"QKV Smoothing - Scale range: min={smooth_scale.min().item():.4f}, max={smooth_scale.max().item():.4f}")
        
        # Scale down normalization weights - shift component
        norm1 = ada_norm
        norm1.linear.weight.data[:dim].div_(smooth_scale.view(-1, 1))
        norm1.linear.bias.data[:dim].div_(smooth_scale.view(-1))
        
        # Scale down normalization weights - scale component
        norm1.linear.weight.data[dim:2*dim].div_(smooth_scale.view(-1, 1))
        norm1.linear.bias.data[dim:2*dim] = (norm1.linear.bias.data[dim:2*dim] + 1) / smooth_scale.view(-1) - 1
        
        # Scale up Q, K, V projections
        for i, linear in enumerate(qkv):
            linear_type = ["Query", "Key", "Value"][i]
            if self.verbose:
                logger.info(f"Scaling up {linear_type} projection weights ({linear.weight.shape})")
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
        
        if self.verbose:
            logger.info(f"MLP UP Smoothing - Dimension: {dim}")
            logger.info(f"MLP UP Smoothing - Scale range: min={smooth_scale.min().item():.4f}, max={smooth_scale.max().item():.4f}")
        
        # Scale down normalization weights for MLP - shift component
        norm1 = ada_norm
        norm1.linear.weight.data[3*dim:4*dim].div_(smooth_scale.view(-1, 1))
        norm1.linear.bias.data[3*dim:4*dim].div_(smooth_scale.view(-1))
        
        # Scale down normalization weights for MLP - scale component
        norm1.linear.weight.data[4*dim:5*dim].div_(smooth_scale.view(-1, 1))
        norm1.linear.bias.data[4*dim:5*dim] = (norm1.linear.bias.data[4*dim:5*dim] + 1) / smooth_scale.view(-1) - 1
        
        # Scale up MLP up projections
        for i, linear in enumerate(up):
            if self.verbose:
                logger.info(f"Scaling up MLP UP projection {i+1}/{len(up)} weights ({linear.weight.shape})")
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
        
        if self.verbose:
            logger.info(f"⚠️ MLP DOWN Smoothing (may alter outputs) - Dimension: {dim}")
            logger.info(f"⚠️ MLP DOWN Smoothing - Scale range: min={smooth_scale.min().item():.4f}, max={smooth_scale.max().item():.4f}")
        
        # Scale down MLP up projection
        pre_layer = up
        if self.verbose:
            logger.info(f"Scaling down MLP UP projection weights ({pre_layer.weight.shape})")
        pre_layer.weight.data.div_(smooth_scale.view(-1, 1))
        pre_layer.bias.data.div_(smooth_scale.view(-1))

        # Scale up MLP down projections
        if self.verbose:
            logger.info(f"Scaling up MLP DOWN projection weights ({down.weight.shape})")
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
        if self.verbose:
            logger.info(f"Scaling up attention output projection weights ({att_o.weight.shape})")
        att_o.weight.data.mul_(smooth_scale.view(1, -1))
        
        # Scale down value projection (output dimension)
        if self.verbose:
            logger.info(f"Scaling down value projection weights ({att_v.weight.shape})")
        att_v.weight.data.div_(smooth_scale.view(-1, 1))
        att_v.bias.data.div_(smooth_scale.view(-1))
        
    def do_smooth_qkv(self):
        """
        Apply anti-smoothing to query, key, value projections in both main and context paths.
        """
        if self.verbose:
            logger.info("=" * 50)
            logger.info("🔄 Starting QKV smoothing process...")
            
        # Apply smoothing to main path Q, K, V projections
        if self.verbose:
            logger.info("📌 Applying QKV smoothing to main attention path")
        qkv = [self.attn.to_q, self.attn.to_k, self.attn.to_v]
        self.apply_smoothing_qkv(qkv, self.norm1)
        
        # Apply smoothing to context path Q, K, V projections
        if self.verbose:
            logger.info("📌 Applying QKV smoothing to context attention path")
        qkv_context = [self.attn.add_q_proj, self.attn.add_k_proj, self.attn.add_v_proj]
        self.apply_smoothing_qkv(qkv_context, self.norm1_context)
        
        if self.verbose:
            logger.info("✅ QKV smoothing completed successfully")
            logger.info("=" * 50)
        
    def do_smooth_mlp_up(self):
        """
        Apply anti-smoothing to MLP up projections in both main and context paths.
        """
        if self.verbose:
            logger.info("=" * 50)
            logger.info("🔄 Starting MLP UP smoothing process...")
            
        # Apply smoothing to main path MLP up projection
        if self.verbose:
            logger.info("📌 Applying MLP UP smoothing to main path")
        mlp_up = [self.ff.net[0].proj]
        self.apply_smoothing_mlp_up(mlp_up, self.norm1)
        
        # Apply smoothing to context path MLP up projection
        if self.verbose:
            logger.info("📌 Applying MLP UP smoothing to context path")
        mlp_up_context = [self.ff_context.net[0].proj]
        self.apply_smoothing_mlp_up(mlp_up_context, self.norm1_context)
        
        if self.verbose:
            logger.info("✅ MLP UP smoothing completed successfully")
            logger.info("=" * 50)
    
    def do_smooth_mlp_down(self):
        """ 
        Apply anti-smoothing to MLP down projections in both main and context paths.
        
        Warning: This will change the output of the model and is not guaranteed
        to maintain exact equivalence.
        """
        if self.verbose:
            logger.info("=" * 50)
            logger.info("⚠️ Starting MLP DOWN smoothing process (WARNING: may alter outputs)...")
            
        # Apply smoothing to main path MLP down projection
        if self.verbose:
            logger.info("📌 Applying MLP DOWN smoothing to main path")
        self.apply_smoothing_mlp_down(
            down=self.ff.net[2],
            up=self.ff.net[0].proj,
        )
        
        # Apply smoothing to context path MLP down projection
        if self.verbose:
            logger.info("📌 Applying MLP DOWN smoothing to context path")
        self.apply_smoothing_mlp_down(
            down=self.ff_context.net[2],
            up=self.ff_context.net[0].proj,
        )
        
        if self.verbose:
            logger.info("⚠️ MLP DOWN smoothing completed (outputs may have changed)")
            logger.info("=" * 50)
    
    def do_smooth_att_o(self, share_scale: bool = True):
        """
        Apply anti-smoothing to attention output projections in both main and context paths.
        
        Args:
            share_scale: If True, use the same scaling factors for both paths.
                         If False, generate different scaling factors (may change outputs).
        """
        if self.verbose:
            logger.info("=" * 50)
            logger.info(f"🔄 Starting Attention Output smoothing process (share_scale={share_scale})...")
            
        # Generate smoothing scale with fixed seed for reproducibility
        torch.manual_seed(42)
        dim = self.attn.to_out[0].weight.shape[1]
        device = self.attn.to_out[0].weight.device
        
        # Create scales - either shared or separate
        scale1 = 2 + torch.randn((dim,)).abs().to(device)
        if share_scale:
            scale2 = scale1
            if self.verbose:
                logger.info("📊 Using shared scaling factors for both paths")
        else:
            scale2 = 2 + torch.randn((dim,)).abs().to(device)
            if self.verbose:
                logger.info("⚠️ Using different scaling factors (may alter outputs)")
        
        if self.verbose:
            logger.info(f"📊 Scale statistics for main path: min={scale1.min().item():.4f}, max={scale1.max().item():.4f}")
            if not share_scale:
                logger.info(f"📊 Scale statistics for context path: min={scale2.min().item():.4f}, max={scale2.max().item():.4f}")
        
        # Apply smoothing to main and context attention output projections
        if self.verbose:
            logger.info("📌 Applying Attention Output smoothing to main path")
        self.apply_smoothing_att_o(self.attn.to_v, self.attn.to_out[0], scale1)
        
        if self.verbose:
            logger.info("📌 Applying Attention Output smoothing to context path")
        self.apply_smoothing_att_o(self.attn.add_v_proj, self.attn.to_add_out, scale2)
        
        if self.verbose:
            status = "✅ Attention Output smoothing completed successfully"
            if not share_scale:
                status = "⚠️ Attention Output smoothing completed (outputs may have changed)"
            logger.info(status)
            logger.info("=" * 50)
    
    def do_smooth(self):
        """
        Apply all anti-smoothing operations to the FluxTransformerBlock.
        
        This is the main entry point for applying the complete set of anti-smoothing
        transformations that are guaranteed to preserve model outputs.
        """
        if self.verbose:
            logger.info("🚀 Starting complete anti-smoothing transformation sequence...")
            
        start_time = time.time()
        self.do_smooth_qkv()
        self.do_smooth_mlp_up()
        self.do_smooth_att_o(share_scale=True)
        
        if self.verbose:
            duration = time.time() - start_time
            logger.info(f"✅ Complete anti-smoothing transformation completed in {duration:.2f} seconds")
            logger.info("🔍 All transformations are output-preserving")
    
    def load_from_flux_transformer_block(self, block: FluxTransformerBlock):
        """
        Load parameters from a regular FluxTransformerBlock.
        
        Args:
            block: Source FluxTransformerBlock to copy parameters from
            
        Raises:
            ValueError: If a module in the source block doesn't exist in this block
        """
        if self.verbose:
            logger.info("📥 Loading parameters from standard FluxTransformerBlock...")
            
        param_count = 0
        for name, module in block.named_children():
            if hasattr(self, name):
                source_module = getattr(block, name)
                target_module = getattr(self, name)
                
                # Count parameters being transferred
                module_params = sum(p.numel() for p in source_module.parameters())
                param_count += module_params
                
                if self.verbose:
                    logger.info(f"  - Loading module '{name}' with {module_params:,} parameters")
                    
                target_module.load_state_dict(source_module.state_dict())
            else:
                raise ValueError(f"Module {name} not found in FluxTransformerBlock_anti")
        
        if self.verbose:
            logger.info(f"✅ Successfully loaded {param_count:,} parameters")
        
def get_dummy_input(batch_size: int, seq_length: int, dim: int, device: str = 'cpu') -> Dict[str, Any]:
    """
    Create dummy inputs for testing transformer blocks.
    
    Args:
        batch_size: Number of samples in a batch
        seq_length: Length of input sequences
        dim: Model hidden dimension
        device: Device to create tensors on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing input tensors for the transformer block
    """
    logger.info(f"🧪 Creating test inputs: batch_size={batch_size}, seq_length={seq_length}, dim={dim}, device='{device}'")
    
    # Create input tensors
    hidden_states = torch.randn(batch_size, seq_length, dim, device=device)
    encoder_hidden_states = torch.randn(batch_size, seq_length, dim, device=device)
    temb = torch.randn(batch_size, dim, device=device)
    
    # Optional: image rotary embeddings (None for this test)
    image_rotary_emb = None
    
    logger.info("✅ Test inputs created successfully")
    
    return dict(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
        is_tp=False
    )

from py3_tools.py_debug import Debugger
# Debugger.debug_flag = True

def display_welcome_banner(args=None):
    """
    Display a welcome banner with colorful information about the script.
    
    Args:
        args: Command line arguments (if provided)
    """
    import shutil
    
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns
    # Ensure minimum width
    width = max(terminal_width, 80)
    
    def center_text(text, width=width):
        return text.center(width)
    
    # Show colorful banner if not in quiet mode
    if args is None or not args.quiet:
        # Colors for terminal output
        CYAN = '\033[96m'
        YELLOW = '\033[93m'
        GREEN = '\033[92m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        
        # Banner
        print("\n" + "=" * width)
        print(center_text(f"{BOLD}{CYAN}FLUX TRANSFORMER BLOCK ANTI-SMOOTHING TEST{RESET}"))
        print("=" * width)
        
        # Script description
        print(f"\n{BOLD}Description:{RESET}")
        print("  This test verifies that anti-smoothing transformations preserve model outputs.")
        print("  These transformations can be used to optimize model performance while")
        print("  maintaining functional equivalence.")
        
        # File information
        print(f"\n{BOLD}📁 Current test file:{RESET} {os.path.abspath(__file__)}")
        
        # Command examples if first run with default parameters
        if args is None or (not args.verbose and not args.quiet and args.smoothing == "all" 
                           and not args.disable_att_o_share_scale
                           and args.dim == 4 and args.batch_size == 1):
            print(f"\n{BOLD}{YELLOW}First time running this script?{RESET} Try these example commands:")
            print(f"\n{GREEN}Available smoothing operations:{RESET}")
            print(f"  • {BOLD}all{RESET}: Apply all output-preserving smoothing operations")
            print(f"  • {BOLD}qkv{RESET}: Only smooth query, key, value projections")
            print(f"  • {BOLD}mlp_up{RESET}: Only smooth MLP up-projections")
            print(f"  • {BOLD}att_o{RESET}: Only smooth attention output projections")
            print(f"  • {RED}mlp_down{RESET}: Smooth MLP down-projections (warning: may change outputs)")
            
            print(f"\n{GREEN}Example commands:{RESET}")
            print(f"  • Basic test with full verbosity:")
            print(f"    {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --verbose{RESET}")
            print(f"  • Test specific smoothing operation:")
            print(f"    {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --smoothing qkv --verbose{RESET}")
            print(f"  • Run on CUDA device (if available):")
            print(f"    {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --device cuda --verbose{RESET}")
            print(f"  • Use larger model for testing:")
            print(f"    {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --dim 32 --heads 4 --head-dim 8{RESET}")
            print(f"  • {RED}Experimental{RESET}: Test without preserving outputs:")
            print(f"    {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --smoothing att_o --disable_att_o_share_scale{RESET}")
            
            print(f"\n{BOLD}{CYAN}For complete documentation:{RESET}")
            print(f"    {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --help{RESET}")
        
        print("\n" + "=" * width + "\n")

def display_results_summary(success, elapsed_time, args):
    """
    Display a summary of test results with timing and next steps.
    
    Args:
        success: Whether the test was successful
        elapsed_time: Time taken to run the test
        args: Command line arguments
    """
    import shutil
    
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns
    # Ensure minimum width
    width = max(terminal_width, 80)
    
    # Colors for terminal output
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    print("\n" + "=" * width)
    
    # Show test results
    if success:
        print(f"{GREEN}{BOLD}✅ TEST PASSED: FluxTransformerBlock anti-smoothing works correctly!{RESET}")
        print(f"⏱️  Test completed in {elapsed_time:.2f} seconds")
        
        # Suggest next steps if this was the default configuration
        if (args.smoothing == "all" and not args.disable_att_o_share_scale
           and args.dim == 4 and args.batch_size == 1):
            print(f"\n{YELLOW}🔍 Next steps you might want to try:{RESET}")
            print(f"  • Run with specific smoothing operations: {BOLD}--smoothing qkv{RESET} or {BOLD}--smoothing att_o{RESET}")
            print(f"  • Test with larger dimensions: {BOLD}--dim 32 --heads 4 --head-dim 8{RESET}")
            print(f"  • Run on GPU (if available): {BOLD}--device cuda{RESET}")
    else:
        print(f"{RED}{BOLD}❌ TEST FAILED!{RESET}")
        print(f"Please check the error messages above for details.")
        print(f"\n{YELLOW}Troubleshooting tips:{RESET}")
        print(f"  • Try with verbose logging: {BOLD}--verbose{RESET}")
        print(f"  • Ensure PyTorch and FLUX1dev dependencies are correctly installed")
        print(f"  • Check if the model architecture has changed")
    
    print("\n" + "=" * width)
    
    # Always show command reference at the end
    print(f"\n{CYAN}Command Reference:{RESET}")
    print(f"""
Try these following commands to explore different features:

• Run all smoothing operations with detailed logs:
  {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --smoothing all --verbose{RESET}

• Run only QKV smoothing:
  {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --smoothing qkv --verbose{RESET}

• Run only MLP_UP smoothing:
  {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --smoothing mlp_up --verbose{RESET}

• Run only ATT_O smoothing:
  {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --smoothing att_o --verbose{RESET}

• [Experimental] Run MLP_DOWN smoothing (may alter outputs):
  {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --smoothing mlp_down --verbose{RESET}

• [Experimental] Run ATT_O with separate scales (may alter outputs):
  {BOLD}python -m tests.test_anti_smooth.test_flux_double_anti --smoothing att_o --disable_att_o_share_scale --verbose{RESET}
""")

@Debugger.attach_on_error()
def test_flux_transformer_block_anti_smooth(args,
                                            dim: int = 4, 
                                            num_attention_heads: int = 1,
                                            attention_head_dim: int = 4, 
                                            batch_size: int = 1, 
                                            seq_length: int = 3,
                                            device: str = 'cpu',
                                            smoothing_type: SmoothingType = SmoothingType.ALL,
                                            verbose: bool = True,
                                            ):
    """
    Test the FluxTransformerBlock_anti with smoothing.
    
    This function:
    1. Creates a regular and an anti-smoothed transformer block
    2. Applies smoothing transformations
    3. Verifies that outputs match after forward passes
    
    Args:
        args: Command line arguments
        dim: Model dimension
        num_attention_heads: Number of attention heads
        attention_head_dim: Dimension of each attention head
        batch_size: Batch size for test input
        seq_length: Sequence length for test input
        device: Device to run the test on ('cpu' or 'cuda')
        smoothing_type: Type of smoothing operation to apply
        verbose: Whether to print detailed logs
    """
    logger.info("=" * 80)
    logger.info("🚀 STARTING FLUX TRANSFORMER BLOCK ANTI-SMOOTHING TEST")
    logger.info("=" * 80)
    
    # Show test configuration
    logger.info("📋 Test Configuration:")
    logger.info(f"  - Model dimension: {dim}")
    logger.info(f"  - Attention heads: {num_attention_heads}")
    logger.info(f"  - Attention head dimension: {attention_head_dim}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Sequence length: {seq_length}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Smoothing type: {smoothing_type}")
    logger.info(f"  - Verbose mode: {verbose}")
    
    # Check if CUDA is requested but not available
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    start_time = time.time()
    
    # Create regular transformer block
    logger.info("🏗️ Creating standard FluxTransformerBlock...")
    block = FluxTransformerBlock(
        dim=dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        is_tp=False
    ).to(device).eval()
    
    # Create anti-smoothed block and load weights from regular block
    logger.info("🏗️ Creating anti-smoothing FluxTransformerBlock...")
    block_anti = FluxTransformerBlock_anti(
        dim=dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        is_tp=False,
        verbose=verbose
    ).to(device).eval()
    
    logger.info("📥 Loading weights from standard block to anti-smoothing block...")
    block_anti.load_from_flux_transformer_block(block)

    # Apply selected anti-smoothing transformation
    logger.info(f"🔄 Applying anti-smoothing transformation: {smoothing_type}")
    
    if smoothing_type == SmoothingType.ALL:
        block_anti.do_smooth()
    elif smoothing_type == SmoothingType.QKV:
        block_anti.do_smooth_qkv()
    elif smoothing_type == SmoothingType.MLP_UP:
        block_anti.do_smooth_mlp_up()
    elif smoothing_type == SmoothingType.MLP_DOWN:
        logger.warning("⚠️ MLP_DOWN smoothing may change model outputs!")
        block_anti.do_smooth_mlp_down()
    elif smoothing_type == SmoothingType.ATT_O:
        if args.disable_att_o_share_scale:
            logger.warning("⚠️ ATT_O smoothing with different scales may change model outputs!")
            block_anti.do_smooth_att_o(share_scale=False)
        else:
            logger.info("🔄 Sharing scales for ATT_O smoothing to preserve outputs")
            block_anti.do_smooth_att_o(share_scale=True)
    
    # Generate test inputs
    logger.info("🧪 Generating test inputs...")
    kwargs = get_dummy_input(batch_size, seq_length, dim, device)
    
    # Perform forward passes
    logger.info("🔄 Running forward pass through standard block...")
    with torch.no_grad():
        output = block(**kwargs)
        
    logger.info("🔄 Running forward pass through anti-smoothing block...")
    with torch.no_grad():
        output_anti = block_anti(**kwargs)
    
    # Check outputs match
    logger.info("🔍 Comparing outputs...")
    error0 = torch.abs(output[0] - output_anti[0]).mean().item()
    error1 = torch.abs(output[1] - output_anti[1]).mean().item()
    logger.info(f'📊 Mean Absolute Error in hidden states: {error0:.8f}')
    logger.info(f'📊 Mean Absolute Error in encoder hidden states: {error1:.8f}')
    
    # Verify outputs match with detailed error information
    try:
        assert torch.allclose(output[0], output_anti[0]), \
            f"Output hidden states do not match, mean error: {error0:.8f}"
        assert torch.allclose(output[1], output_anti[1]), \
            f"Output encoder hidden states do not match, mean error: {error1:.8f}"
        
        elapsed_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✅ TEST PASSED: FluxTransformerBlock anti-smoothing works correctly!")
        logger.info(f"⏱️ Test completed in {elapsed_time:.2f} seconds")
        logger.info("=" * 80)
        return True, elapsed_time
    except AssertionError as e:
        logger.error("=" * 80)
        logger.error(f"❌ TEST FAILED: {str(e)}")
        logger.error("=" * 80)
        return False, time.time() - start_time

def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Test FluxTransformerBlock anti-smoothing techniques",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--dim", type=int, default=4, 
                        help="Model dimension")
    parser.add_argument("--heads", type=int, default=1, 
                        help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=4, 
                        help="Dimension of each attention head")
    
    # Test input configuration
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Batch size for test input")
    parser.add_argument("--seq-length", type=int, default=3, 
                        help="Sequence length for test input")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run the test on")
    
    # Smoothing configuration
    parser.add_argument("--smoothing", type=str, default="all", 
                        choices=["all", "qkv", "mlp_up", "mlp_down", "att_o"],
                        help="Type of smoothing operation to apply")
    parser.add_argument("--disable_att_o_share_scale", action="store_true", 
                        help="Disable sharing of ATTN_O scales")
    
    # Logging configuration
    parser.add_argument("--verbose", action="store_true",
                        help="Enable detailed logging")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress all logging except errors")
    
    # Visualization
    parser.add_argument("--no-banner", action="store_true",
                        help="Hide the welcome banner and usage hints")
        
    args = parser.parse_args()
    
    # Set logging level based on quiet flag
    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Display welcome banner with usage information
    if not args.no_banner:
        display_welcome_banner(args)
        
    # Convert string smoothing type to enum
    smoothing_type = SmoothingType(args.smoothing)
    
    # Run the test
    success, elapsed_time = test_flux_transformer_block_anti_smooth(
        args,
        dim=args.dim,
        num_attention_heads=args.heads,
        attention_head_dim=args.head_dim,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device=args.device,
        smoothing_type=smoothing_type,
        verbose=args.verbose
    )
    
    # Display summary with next steps suggestions
    if not args.quiet and not args.no_banner:
        display_results_summary(success, elapsed_time, args)
    
    # Set exit code based on test result
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # Run the main function with command line arguments
    main()

Doc_String = """
Try this following command to run the test:

- : Smooth all components(QKV, MLP_UP, ATT_O):
```
python -m tests.test_anti_smooth.test_flux_double_anti --smoothing all --verbose
```

- : Smooth only MLP_DOWN components:
```
python -m tests.test_anti_smooth.test_flux_double_anti --smoothing mlp_down --verbose
```

- : Smooth only image ATTN_O and context ATTN_O components using different scales:

```
python -m tests.test_anti_smooth.test_flux_double_anti --smoothing att_o --verbose --disable_att_o_share_scale
```

"""