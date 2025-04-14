import torch
import torch.nn as nn
import math
class RoPEWithBetweenness(nn.Module):
    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 2048, adjustment_scale: float = 0.1,
        enable_betweenness: bool = True):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.adjustment_scale = adjustment_scale
        self.enable_betweenness = enable_betweenness
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.content_proj = nn.Linear(dim, dim)
        self.betweenness_gate = nn.Parameter(torch.ones(1) * 0.5)
        self._seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def compute_betweenness(self, x):
        orig_shape = x.shape
        orig_dim = len(orig_shape)
        if orig_dim == 4:
            batch, seq_len, heads, dim = x.shape
            x_flat = x.transpose(1, 2).reshape(batch * heads, seq_len, dim)
        else:
            batch, seq_len, dim = x.shape
            x_flat = x
            heads = 1
        content = self.content_proj(x_flat)
        token_i = content.unsqueeze(2)
        token_j = content.unsqueeze(1)
        distances = torch.norm(token_i - token_j, dim=-1)
        betweenness = torch.zeros(x_flat.shape[0], seq_len, device=x.device)
        if seq_len > 2:
            for i in range(seq_len - 2):
                direct = distances[:, i, i+2]
                path = distances[:, i, i+1] + distances[:, i+1, i+2]
                between_score = torch.relu(1.0 - (path - direct) / torch.clamp(direct, min=1e-6))
                betweenness[:, i+1] += between_score
            betweenness = betweenness / (seq_len - 2)
        if orig_dim == 4:
            betweenness = betweenness.view(batch, heads, seq_len).transpose(1, 2)
        else:
            betweenness = betweenness.view(batch, seq_len, 1)
        return betweenness
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        freqs = self._compute_or_get_freqs(seq_len, x.device)
        if self.enable_betweenness:
            betweenness = self.compute_betweenness(x)
            pos_adjustments = self.betweenness_gate * (betweenness - 0.5) * self.adjustment_scale
            return self._apply_rotary_with_adjustments(x, freqs, pos_adjustments)
        else:
            return self._apply_rotary(x, freqs)
    def _compute_or_get_freqs(self, seq_len, device):

        if seq_len != self._seq_len_cached or self.cos_cached is None or self.cos_cached.device != device:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            self.freqs_cached = freqs
        return self.freqs_cached
    
    def _apply_rotary(self, x, freqs):
        batch, seq_len, heads, dim = x.shape
        cos = torch.cos(freqs).view(1, seq_len, 1, dim // 2)
        sin = torch.sin(freqs).view(1, seq_len, 1, dim // 2)
        x_2d = x.reshape(batch * seq_len, heads, dim)
        x_2d_rot = self._rotate_half(x_2d)
        cos = cos.repeat(batch, 1, heads, 1).reshape(batch * seq_len, heads, dim // 2)
        sin = sin.repeat(batch, 1, heads, 1).reshape(batch * seq_len, heads, dim // 2)
        x_out = torch.cat([
            x_2d[..., :dim//2] * cos - x_2d_rot[..., :dim//2] * sin,
            x_2d[..., dim//2:] * cos + x_2d_rot[..., dim//2-dim:] * sin
        ], dim=-1)
        return x_out.reshape(batch, seq_len, heads, dim)
    
    def _apply_rotary_with_adjustments(self, x, freqs, pos_adjustments):
        batch, seq_len, heads, dim = x.shape
        x_out = torch.zeros_like(x, dtype=torch.float)
        for i in range(seq_len):
            adj_pos = i + pos_adjustments[:, i, :]
            adj_pos = torch.clamp(adj_pos, 0, self.max_seq_len - 1)
            idx_lo = torch.floor(adj_pos).long()
            idx_hi = torch.ceil(adj_pos).long()
            frac = (adj_pos - idx_lo).unsqueeze(-1)
            freqs_lo = freqs[idx_lo]
            freqs_hi = freqs[idx_hi]
            interp_freqs = (1 - frac) * freqs_lo + frac * freqs_hi
            cos_interp = torch.cos(interp_freqs)
            sin_interp = torch.sin(interp_freqs)
            x_i = x[:, i]
            x_rot_i = self._rotate_half(x_i)
            x_out[:, i] = torch.cat([
                x_i[..., :dim//2] * cos_interp - x_rot_i[..., :dim//2] * sin_interp,
                x_i[..., dim//2:] * cos_interp + x_rot_i[..., dim//2-dim:] * sin_interp
            ], dim=-1)
        return x_out
    
    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
class SampleAttention(nn.Module):
    def __init__(
        self, 
        dim: int = 768, 
        n_heads: int = 12,
        use_betweenness: bool = True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.rope = RoPEWithBetweenness(
            dim=self.head_dim,
            max_seq_len=4096,
            adjustment_scale=0.1,
            enable_betweenness=use_betweenness)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = self.rope(q)
        k = self.rope(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.wo(out)

class BetweennessModule(nn.Module):
    def __init__(self, dim, adjustment_scale=0.1):
        super().__init__()
        self.dim = dim
        self.adjustment_scale = adjustment_scale
        self.content_proj = nn.Linear(dim, dim)
        self.betweenness_gate = nn.Parameter(torch.ones(1) * 0.5)

    def compute_betweenness(self, x):
        orig_dim = x.dim()
        if orig_dim == 4:
            batch, seq_len, heads, dim = x.shape
            x_flat = x.transpose(1, 2).reshape(batch * heads, seq_len, dim)
            has_heads = True
        else:
            batch, seq_len, dim = x.shape
            x_flat = x
            has_heads = False
            heads = 1
        content = self.content_proj(x_flat)
        token_i = content.unsqueeze(2)
        token_j = content.unsqueeze(1)
        distances = torch.norm(token_i - token_j, dim=-1)
        betweenness = torch.zeros(x_flat.shape[0], seq_len, device=x.device)
        if seq_len > 2:
            for i in range(seq_len - 2):
                direct = distances[:, i, i+2]
                path = distances[:, i, i+1] + distances[:, i+1, i+2]
                between_score = torch.relu(1.0 - (path - direct) / torch.clamp(direct, min=1e-6))
                betweenness[:, i+1] += between_score
            betweenness = betweenness / (seq_len - 2)
        if has_heads:
            betweenness = betweenness.view(batch, heads, seq_len).transpose(1, 2)
        elif orig_dim <= 3:
            betweenness = betweenness.view(batch, seq_len)
        return betweenness
    def get_position_adjustments(self, x):
        betweenness = self.compute_betweenness(x)
        if betweenness.dim() == 3:
            adj = self.betweenness_gate * (betweenness - 0.5) * self.adjustment_scale
        else:
            adj = self.betweenness_gate * (betweenness - 0.5) * self.adjustment_scale
        return adj
    
def apply_to_rope(rope_func, x, positions, betweenness_module):
    adjustments = betweenness_module.get_position_adjustments(x)
    adjusted_positions = positions + adjustments
    return rope_func(x, adjusted_positions)
