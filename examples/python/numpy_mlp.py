"""Numpy arrays with a stdlib fallback — identical artifact either way.

If numpy is importable the globals are numpy arrays (harvested via tolist);
otherwise the plain lists below produce the exact same Safetensors file.
Same weights as mlp_lists.py: input [1.0, 2.0] → 1.3.
"""
try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    layer0_weight = np.array([[1.0, 0.5], [-0.5, 2.0]], dtype="f4")
    layer0_bias = np.array([0.1, -0.1], dtype="f4")
    layer1_weight = np.array([[2.0, -1.0]], dtype="f4")
    layer1_bias = np.array([0.5], dtype="f4")
else:
    layer0_weight = [[1.0, 0.5], [-0.5, 2.0]]
    layer0_bias = [0.1, -0.1]
    layer1_weight = [[2.0, -1.0]]
    layer1_bias = [0.5]
