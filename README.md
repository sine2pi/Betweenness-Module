### For testing

```python


class BetweennessModule(nn.Module): 
    def __init__(self, dim, adjustment_scale=1.0, window_size=10):
        super().__init__()
        self.dim = dim
        self.adjustment_scale = adjustment_scale
        self.content_proj = nn.Linear(dim, dim)
        self.betweenness_gate = nn.Parameter(torch.ones(1) * 0.5)
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def compute_betweenness(self, x):
        batch, seq_len, dim = x.shape
        content = self.norm(self.content_proj(self.dropout(x)))
        device = x.device
        window = self.window_size

        betweenness = torch.zeros(batch, seq_len, device=device)

        for offset in range(1, window + 1):
            n_indices = seq_len - 2 * offset
            if n_indices <= 0:
                continue
            i = torch.arange(n_indices, device=device)
            j = i + offset
            k = i + 2 * offset

            c_i = content[:, i, :]
            c_j = content[:, j, :]
            c_k = content[:, k, :]

            def cos_dist(a, b):
                a = F.normalize(a, dim=-1)
                b = F.normalize(b, dim=-1)
                return 1 - (a * b).sum(dim=-1)

            direct = cos_dist(c_i, c_k)
            path = cos_dist(c_i, c_j) + cos_dist(c_j, c_k)
            safe_direct = torch.clamp(direct, min=1e-3)
            between_score = 1.0 - (path - direct) / safe_direct
            betweenness[:, j] += between_score

        betweenness = betweenness / max(window, 1)
        betweenness = betweenness - betweenness.mean(dim=1, keepdim=True)
        std = betweenness.std(dim=1, keepdim=True) + 1e-6
        betweenness = betweenness / std

        betweenness = self.betweenness_gate * self.adjustment_scale * betweenness
        betweenness = torch.clamp(betweenness, -2.0, 2.0)

        return betweenness


class AudioEncoder(nn.Module):
    def __init__(self, mels: int, ctx: int, dims: int, head: int, layer, act: str = "relu"):
        super().__init__()
        self.ctx = ctx
        self.dropout = 0.1
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(),
                   "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.blend_sw = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.blend = torch.sigmoid(self.blend_sw)
        self.ln_enc = RMSNorm(normalized_shape=dims)
        self.register_buffer("positional_embedding", sinusoids(ctx, dims))

        self.use_betweenness = True  

        if self.use_betweenness:
            self.betweenness = BetweennessModule(dim=dims, window_size=10)

        self.se = nn.Sequential(
            Conv1d(mels, dims, kernel_size=3, padding=1), self.act,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=2, dilation=2),
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims),
            Conv1d(dims, dims, kernel_size=1), SEBlock(dims, reduction=16), self.act,
            nn.Dropout(p=self.dropout), Conv1d(dims, dims, kernel_size=3, stride=1, padding=1)
        )
        self.we = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=11, stride=5, padding=5),
            nn.GELU(),
            nn.Conv1d(dims, dims, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(ctx),
        )

        self.blockA = (nn.ModuleList([Residual(dims=dims, head=head, cross_attention=False, act="relu")
                    for _ in range(layer)]) if layer > 0 else None)

    def forward(self, x, w) -> Tensor:
        if x is not None:
            if w is not None:
                x = self.se(x).permute(0, 2, 1)
                x = (x + self.positional_embedding).to(x.dtype)
                w = self.we(w).permute(0, 2, 1)
                x = self.blend * x + (1 - self.blend) * w
            else:
                x = self.se(x)
                x = x.permute(0, 2, 1)
                assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
                x = (x + self.positional_embedding).to(x.dtype)
        else:
            assert w is not None, "You have to provide either x or w"
            x = self.we(w).permute(0, 2, 1)
            assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            x = (x + self.positional_embedding).to(x.dtype)

        if self.use_betweenness:
            be = self.betweenness.compute_betweenness(x)
            # print(f"betweenness: {be}")
            x = x + be.unsqueeze(-1) 

        for block in chain(self.blockA or []):
            x = block(x)

        return self.ln_enc(x)

```

#### RoPE implementation with betweenness-based positional adjustments. 
#### Idea was inspired by the concepts of betweenness, realtive locations, hilbert spaces, and the letter O.

#### Proof of concept + tests and working embeddings 
  
  This approach measures the 'betweenness' of each token in continuous semantic space using geometric path comparisons to define betweennes
  and applying these relationships to adjust the underlying rotary embeddings, allowing the model to  develop a more interconnected understanding of token 
  relationships.

### Betweenness and Token Connectivity

The betweenness calculation creates an implicit form of connectivity between tokens, but in a specific way:

### Local vs. Global Connectivity

The implementation creates **local connectivity** rather than fully global connections:

```python
# Calculate betweenness for intermediate tokens
if seq_len > 2:
    for i in range(seq_len - 2):
        direct = distances[:, i, i+2]
        path = distances[:, i, i+1] + distances[:, i+1, i+2]
        
        between_score = torch.relu(1.0 - (path - direct) / torch.clamp(direct, min=1e-6))
        betweenness[:, i+1] += between_score
```

This creates a sliding window of triplets: abc, bcd, cde, etc. Each token's betweenness is calculated based on how it mediates the relationship between its neighbors.

### Implicit Global Effects

While the calculation only uses local triplets, the effects propagate more globally:

1. When a token's betweenness score affects its positional encoding
2. This changes how that token attends to and is attended by all other tokens
3. The position adjustments create subtle shifts in the attention patterns

So while the betweenness measure itself is calculated locally, its effect on the transformer's behavior is more global through the adjusted RoPE positions.

Content-aware positioning - Unlike standard positional encodings that are fixed regardless of content, this method adjusts positions based on semantic relationships between tokens

Semantic structure awareness - The model can potentially learn to recognize important "bridge" tokens that connect ideas, which could be valuable for:

Understanding complex semantic structures
Identifying key tokens in long contexts
Better preserving important relationships between distant tokens
Learnable behavior - The betweenness_gate parameter allows the model to learn how much to rely on this information

Minimal overhead - The implementation adds relatively little computational cost while potentially gaining meaningful semantic awareness

The approach essentially adds a form of "semantic topology" to the transformer's representation space. In domains where the logical/semantic flow matters (like reasoning, code, complex explanations), this could help the model better understand how concepts are connected.


