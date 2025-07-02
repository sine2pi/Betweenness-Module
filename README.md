
Usage Example

```python
# Initialize the different betweenness modules
dim = 64
triplet_btw = TripletBetweenness(dim=dim, adjustment_scale=0.1)
multigap_btw = MultiGapBetweenness(dim=dim, adjustment_scale=0.1, max_gap=8)

# Example usage in an attention mechanism
def forward(self, x):
    # Regular processing steps
    q = self.q_proj(x)
    k = self.k_proj(x)
    
    # Choose which betweenness approach to use
    use_triplet = True  # Toggle for experiments
    btw_module = triplet_btw if use_triplet else multigap_btw
    
    # Get position adjustments
    pos_adj = btw_module.get_position_adjustments(x)
    
    # Apply to positions for RoPE
    base_positions = torch.arange(x.shape[1], device=x.device)
    adjusted_positions = base_positions.unsqueeze(0) + pos_adj
    
    # Use adjusted positions with existing RoPE implementation
    q_rope = apply_rope(q, adjusted_positions)
    k_rope = apply_rope(k, adjusted_positions)
    
    # Continue with attention calculation...
```

## Key Differences Between Implementations

1. TripletBetweenness:
   - Only considers immediate triplets (i, i+1, i+2)
   - Computationally efficient
   - Focuses on local structure
   - Normalized by (seq_len - 2)

2. MultiGapBetweenness:
   - Considers paths with various gap sizes (2 to max_gap)
   - More computationally expensive
   - Captures both local and longer-range dependencies
   - Normalized by total valid paths considered
