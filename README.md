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
