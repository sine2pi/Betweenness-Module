import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from einops import rearrange

class BetweennessRoPE(nn.Module):
    """
    RoPE implementation with betweenness-based positional adjustments.
    
    This approach measures the 'betweenness' of each token in semantic space
    and adjusts its positional encoding accordingly, allowing the model to
    develop a more interconnected understanding of token relationships.
    """
    def __init__(self, dim, max_seq_len=2048, adjustment_scale=0.1):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.adjustment_scale = adjustment_scale
        
        self.content_proj = nn.Linear(dim, dim)
        
        self.betweenness_gate = nn.Parameter(torch.ones(1) * 0.5)
        
        self.register_buffer(
            "base_freqs", 
            1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        )

    def precompute_freqs_cis(self, seq_len, device):
        """Compute frequency bands for RoPE"""
        t = torch.arange(seq_len, device=device)
        freqs = torch.outer(t, self.base_freqs.to(device))
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return freqs_cos, freqs_sin

    def compute_betweenness(self, x):
        """Compute betweenness scores for each token in the sequence"""
        shape = x.shape
        orig_dim = x.dim()
        
        if orig_dim == 4:
            batch, seq_len, heads, dim = x.shape
            x = x.transpose(1, 2).reshape(batch * heads, seq_len, dim)
        else:
            batch, seq_len = x.shape[:2]
        
        content = self.content_proj(x)
        
        token_i = content.unsqueeze(2)
        token_j = content.unsqueeze(1)
        distances = torch.norm(token_i - token_j, dim=-1)
        
        betweenness = torch.zeros(x.shape[0], seq_len, device=x.device)
        
        for i in range(seq_len-2):
            direct = distances[:, i, i+2]
            path = distances[:, i, i+1] + distances[:, i+1, i+2]
            
            between_score = torch.relu(1.0 - (path - direct)/torch.clamp(direct, min=1e-6))
            betweenness[:, i+1] += between_score
        
        if seq_len > 2:
            betweenness = betweenness / (seq_len - 2)
        
        if orig_dim == 4:
            betweenness = betweenness.view(batch, heads, seq_len).transpose(1, 2)
        
        return betweenness

    def apply_rope(self, x, freqs_cos, freqs_sin, betweenness=None):
        """Apply RoPE rotation with betweenness-adjusted positions"""
        seq_len = x.shape[1]
        
        if betweenness is None:
            x_rope = x.float()
            
            x_out = torch.zeros_like(x_rope)
            for i in range(seq_len):
                x_out[:, i, :, 0::2] = x_rope[:, i, :, 0::2] * freqs_cos[i] - x_rope[:, i, :, 1::2] * freqs_sin[i]
                x_out[:, i, :, 1::2] = x_rope[:, i, :, 1::2] * freqs_cos[i] + x_rope[:, i, :, 0::2] * freqs_sin[i]
            
            return x_out.type_as(x)
        
        pos_adjustments = (betweenness - 0.5) * self.adjustment_scale
        
        x_out = torch.zeros_like(x, dtype=torch.float)
        for i in range(seq_len):
            adj_pos = torch.clamp(i + pos_adjustments[:, i:i+1], 0, seq_len-1) # Shape (B, 1, H) or (B, 1)
            
            idx_lo = torch.floor(adj_pos).long() # Shape (B, 1, H) or (B, 1)
            idx_hi = torch.ceil(adj_pos).long()  # Shape (B, 1, H) or (B, 1)
            frac = (adj_pos - idx_lo).unsqueeze(-1) # Shape (B, 1, H, 1) or (B, 1, 1)
            
            # Interpolate frequencies
            # freqs_cos[idx_lo] shape: (B, 1, H, D//2) or (B, 1, D//2)
            cos_interp = (1 - frac) * freqs_cos[idx_lo] + frac * freqs_cos[idx_hi]
            sin_interp = (1 - frac) * freqs_sin[idx_lo] + frac * freqs_sin[idx_hi]
            
            # Squeeze the singleton dimension to allow broadcasting with x[:, i, ...]
            # Shape becomes (B, H, D//2) or (B, D//2)
            cos_interp = cos_interp.squeeze(1) 
            sin_interp = sin_interp.squeeze(1)
            
            # Apply rotation: x[:, i, :, 0::2] has shape (B, H, D//2) or (B, D//2)
            x_out[:, i, :, 0::2] = x[:, i, :, 0::2] * cos_interp - x[:, i, :, 1::2] * sin_interp
            x_out[:, i, :, 1::2] = x[:, i, :, 1::2] * cos_interp + x[:, i, :, 0::2] * sin_interp
        
        return x_out.type_as(x)
        
    def forward(self, x):
        """Forward pass with betweenness-adjusted RoPE"""
        batch, seq_len = x.shape[:2]
        device = x.device
        
        betweenness = self.compute_betweenness(x)
        
        freqs_cos, freqs_sin = self.precompute_freqs_cis(seq_len, device)
        
        return self.apply_rope(x, freqs_cos, freqs_sin, betweenness)


class BetweennessAttention(nn.Module):
    """
    Attention mechanism using betweenness-adjusted RoPE embeddings
    """
    def __init__(self, dim, heads, dim_head=64):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        
        self.q_proj = nn.Linear(dim, inner_dim)
        self.k_proj = nn.Linear(dim, inner_dim)
        self.v_proj = nn.Linear(dim, inner_dim)
        self.output = nn.Linear(inner_dim, dim)
        
        self.rope = BetweennessRoPE(dim_head, adjustment_scale=0.1)
        
    def forward(self, x, mask=None):
        batch, seq_len = x.shape[:2]
        
        q = self.q_proj(x).reshape(batch, seq_len, self.heads, self.dim_head)
        k = self.k_proj(x).reshape(batch, seq_len, self.heads, self.dim_head)
        v = self.v_proj(x).reshape(batch, seq_len, self.heads, self.dim_head)
        
        q_rope = self.rope(q)
        k_rope = self.rope(k)
        
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q_rope, k_rope.transpose(-1, -2)) / math.sqrt(self.dim_head)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, v)
        
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.output(output)


def test_betweenness_rope():
    torch.manual_seed(42)
    
    dim = 64
    heads = 4
    dim_head = dim // heads
    seq_len = 10
    batch_size = 2
    
    x = torch.randn(batch_size, seq_len, dim)
    
    rope_module = BetweennessRoPE(dim=dim_head)
    betweenness = rope_module.compute_betweenness(x.reshape(batch_size, seq_len, heads, dim_head))
    
    print(f"Betweenness shape: {betweenness.shape}")
    print(f"Betweenness scores: {betweenness[0]}")
    
    attn = BetweennessAttention(dim=dim, heads=heads)
    output = attn(x)
    
    print(f"Output shape: {output.shape}")
    
    # Calculate the mean betweenness score across heads for the first batch item
    mean_betweenness_scores = betweenness[0].mean(dim=-1).detach().cpu().numpy() 
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(seq_len), mean_betweenness_scores) # Use the mean scores for plotting
    plt.title("Mean Betweenness Scores Across Heads for Tokens (Batch 0)")
    plt.xlabel("Token Position")
    plt.ylabel("Mean Betweenness Score")
    plt.show()


if __name__ == "__main__":
    test_betweenness_rope()

    dim = 64
    
    x = torch.zeros(1, 5, dim)
    x[0, 0] = torch.tensor([1.0, 0.0, 0.0] + [0.0] * (dim - 3))
    x[0, 1] = torch.tensor([0.5, 0.5, 0.0] + [0.0] * (dim - 3))
    x[0, 2] = torch.tensor([0.0, 1.0, 0.0] + [0.0] * (dim - 3))
    x[0, 3] = torch.tensor([0.0, 0.5, 0.5] + [0.0] * (dim - 3))
    x[0, 4] = torch.tensor([0.0, 0.0, 1.0] + [0.0] * (dim - 3))
    
    rope = BetweennessRoPE(dim)
    with torch.no_grad():
        rope.content_proj.weight.copy_(torch.eye(dim))
        rope.content_proj.bias.fill_(0.0)
    
    betweenness = rope.compute_betweenness(x)
    
    print("\nSynthetic example betweenness scores:")
    print(betweenness[0])
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(5), betweenness[0].detach().numpy())
    plt.title("Betweenness Scores - Synthetic Example")
    plt.xlabel("Token Position")
    plt.ylabel("Betweenness Score")
    plt.xticks(range(5))
    plt.show()
