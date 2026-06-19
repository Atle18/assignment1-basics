import numpy as np

def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    # Your code here
    seq_len, d = x.shape
    assert d % 2 == 0
    assert positions.shape == (seq_len,)

    k = d // 2

    # pair index: 0, 1, ..., d/2 - 1
    pair_idx = np.arange(k)

    # shape: (k,)
    inv_freq = base ** (-2 * pair_idx / d)

    # shape: (seq_len, k)
    angles = positions[:, None] * inv_freq[None, :]

    cos = np.cos(angles)
    sin = np.sin(angles)

    # 拿出每一对的第一个/第二个坐标
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    # 旋转
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    # 拼回原来的交错顺序
    y = np.empty_like(x)
    y[:, 0::2] = y_even
    y[:, 1::2] = y_odd

    return y


x = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
])
positions = np.arange(4)
result = apply_rope(x, positions)
print(np.round(result, 4).tolist())