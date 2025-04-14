### RoPE implementation with betweenness-based positional adjustments. 
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

This creates a sliding window of triplets exactly as you described: abc, bcd, cde, etc. Each token's betweenness is calculated based on how it mediates the relationship between its neighbors.

### Implicit Global Effects

While the calculation only uses local triplets, the effects propagate more globally:

1. When a token's betweenness score affects its positional encoding
2. This changes how that token attends to and is attended by all other tokens
3. The position adjustments create subtle shifts in the attention patterns

So while the betweenness measure itself is calculated locally, its effect on the transformer's behavior is more global through the adjusted RoPE positions.

Semantic-Geometric Position Adjustment, Path Comparison Method, Fractional Position Interpolation.
