import torch
import torch.nn as nn
import torch.fx as fx
from typing import List, Dict, Tuple, Any, Set, Optional

class AdaLayerNormZeroSingleDeploy(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        # self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        
        self.shift_msa = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.scale_msa = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.gate_msa = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.silu(emb)
        # x = self.norm(x) * (1 + self.scale_msa(x)) + self.shift_msa(x)
        x = self.norm(x) * self.scale_msa(emb) + self.shift_msa(emb)
        return x, self.gate_msa(emb)
    
    def load_copy_weight_from_ada_ln_zero_single(self, ada_ln_zero_single: 'AdaLayerNormZeroSingle'):
        """
        Load weights from an instance of `AdaLayerNormZeroSingle`.

        Parameters:
            ada_ln_zero_single (`AdaLayerNormZeroSingle`): The source instance to copy weights from.
        """
        embedding_dim = ada_ln_zero_single.linear.weight.shape[0] // 3
        assert ada_ln_zero_single.linear.weight.shape[0] % 3 == 0, "The embedding dimension does not match."
        
        # split ada_ln_zero_single.linear.weight and bias to shift_msa, scale_msa, gate_msa and bias
        self.shift_msa.weight.data.copy_(ada_ln_zero_single.linear.weight.data[:embedding_dim])
        self.scale_msa.weight.data.copy_(ada_ln_zero_single.linear.weight.data[embedding_dim:2 * embedding_dim])
        self.gate_msa.weight.data.copy_(ada_ln_zero_single.linear.weight.data[2 * embedding_dim:])
        if ada_ln_zero_single.linear.bias is not None:
            self.shift_msa.bias.data.copy_(ada_ln_zero_single.linear.bias.data[:embedding_dim])
            self.scale_msa.bias.data.copy_(ada_ln_zero_single.linear.bias.data[embedding_dim:2 * embedding_dim])
            self.gate_msa.bias.data.copy_(ada_ln_zero_single.linear.bias.data[2 * embedding_dim:])
        else:
            self.shift_msa.bias.data.fill_(0)
            self.scale_msa.bias.data.fill_(0)
            self.gate_msa.bias.data.fill_(0)
        
        self.scale_msa.bias.data.add_(1)
        
class AdaLayerNormZeroSingleDeploy2(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        
        # self.smooth_scale = nn.Parameter(torch.ones((embedding_dim,)), requires_grad=True)
        
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * scale_msa[:, None] + shift_msa[:, None]
        
        # if self.smooth_scale is not None:
        #     x = x * self.smooth_scale.view(..., -1)
        return x, gate_msa
    
    def load_copy_weight_from_ada_ln_zero_single(self, ada_ln_zero_single: 'AdaLayerNormZeroSingle'):
        """
        Load weights from an instance of `AdaLayerNormZeroSingle`.

        Parameters:
            ada_ln_zero_single (`AdaLayerNormZeroSingle`): The source instance to copy weights from.
        """
        embedding_dim = ada_ln_zero_single.linear.weight.shape[0] // 3
        assert ada_ln_zero_single.linear.weight.shape[0] % 3 == 0, "The embedding dimension does not match."
        
        # copy the linear layer weights and bias
        self.linear.weight.data.copy_(ada_ln_zero_single.linear.weight.data)
        if ada_ln_zero_single.linear.bias is not None:
            self.linear.bias.data.copy_(ada_ln_zero_single.linear.bias.data)
        else:   
            self.linear.bias.data.fill_(0)
        
        # scale
        self.linear.bias.data[embedding_dim:2 * embedding_dim].add_(1)
    
    def div_smooth_scale_weight(self, smooth_scale: torch.Tensor, weight_type='scale'):
        smooth_scale = smooth_scale.reshape(-1)
        len_scale = smooth_scale.numel()
        embedding_dim = self.linear.weight.shape[0] // 3
        assert self.linear.weight.shape[0] % 3 == 0, "The embedding dimension does not match."
        assert len_scale == embedding_dim, "The smooth scale length does not match the embedding dimension."
        
        if weight_type == 'scale':
            self.linear.weight.data[embedding_dim:2 * embedding_dim].div_(smooth_scale.view(-1, 1))
            self.linear.bias.data[embedding_dim:2 * embedding_dim].div_(smooth_scale.view(-1))
        elif weight_type == 'shift':
            self.linear.weight.data[:embedding_dim].div_(smooth_scale.view(-1, 1))
            self.linear.bias.data[:embedding_dim].div_(smooth_scale.view(-1))
        elif weight_type == 'gate':
            self.linear.weight.data[2 * embedding_dim:].div_(smooth_scale.view(-1, 1))
            self.linear.bias.data[2 * embedding_dim:].div_(smooth_scale.view(-1))
        else:
            raise ValueError(f"Unsupported `weight_type` ({weight_type}) provided. Supported ones are: 'scale', 'shift', 'gate'.")
                
        
def copy_ada_ln_zero_single():
    pass

class Model1(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.proj = nn.Linear(embedding_dim, 3*embedding_dim, bias=True)
        self.norm = AdaLayerNormZeroSingle(embedding_dim=embedding_dim)
    
    def forward(self, x, emb):
        x, gate = self.norm(x, emb)
        return self.proj(x)

class ModelDeploy(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.proj = nn.Linear(embedding_dim, embedding_dim*3, bias=True)
        self.norm = AdaLayerNormZeroSingleDeploy2(embedding_dim=embedding_dim)
        
        self.smooth_scale = None
    
    def forward(self, x, emb):
        x, gate = self.norm(x, emb)
        return self.proj(x)
    
    def mul_proj_smooth_scale(self, smooth_scale: torch.Tensor):
        self.proj.weight.data.mul_(smooth_scale.view(-1))
    
    def do_smooth(self, smooth_scale: torch.Tensor = None):

        if smooth_scale is None:
            smooth_scale = 2 + torch.randn((self.embedding_dim,)).abs()
        else:
            raise ValueError("smooth_scale has already been set, please do not set it again.")
        self.smooth_scale = smooth_scale
        
        self.mul_proj_smooth_scale(smooth_scale)
        self.norm.div_smooth_scale_weight(smooth_scale, weight_type='scale')
        self.norm.div_smooth_scale_weight(smooth_scale, weight_type='shift')
    
    def load_from_model1(self, model1: Model1):
        """
        Load weights from an instance of `Model1`.

        Parameters:
            model1 (`Model1`): The source instance to copy weights from.
        """
        self.proj.weight.data.copy_(model1.proj.weight.data)
        if model1.proj.bias is not None:
            self.proj.bias.data.copy_(model1.proj.bias.data)
        else:
            self.proj.bias.data.fill_(0)
        
        self.norm.load_copy_weight_from_ada_ln_zero_single(model1.norm)
        
    

from diffusers.models.normalization import AdaLayerNormZeroSingle
import torch

# 创建模型实例
layer1 = AdaLayerNormZeroSingle(embedding_dim=128)
# 定义输入
x = torch.randn(1, 128)  # 假设输入是一个形状为 (batch_size, embedding_dim) 的张量
emb = torch.randn(1, 128)  # 假设 emb 是一个形状为 (batch_size, embedding_dim) 的张量

# %% test 1
# # 执行前向传播
# output = layer1(x, emb)


# layer2 = AdaLayerNormZeroSingleDeploy(embedding_dim=128)
# layer2.load_copy_weight_from_ada_ln_zero_single(layer1)
# output2 = layer2(x, emb)
# print(output[0].shape)
# print(output2[0].shape)

# print('out1 error:', torch.allclose(output[0], output2[0], atol=1e-6))  # 检查输出是否相同
# print('out2 error', torch.allclose(output[1], output2[1], atol=1e-6))  # 检查输出是否相同



# %% test2
model1 = Model1(embedding_dim=128)

model2 = ModelDeploy(embedding_dim=128)
model2.load_from_model1(model1)

output1 = model1(x, emb)
output2 = model2(x, emb)
print('out1 error:', torch.allclose(output1[0], output2[0], atol=1e-6))  # 检查输出是否相同

model2.do_smooth()
output2_smooth = model2(x, emb)
print('out2_smooth error', torch.allclose(output1[0], output2_smooth[0], atol=1e-6))  # 检查输出是否相同
