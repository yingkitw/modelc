"""Scalars and strings are skipped; only tensors are harvested.

LR/NAME are hyperparameters, not weights — the artifact contains exactly
weight + bias (a singleton affine pair → runnable `mlp`). Input [1.0, 1.0] → 3.5.
"""
LR = 0.1
NAME = "tiny"
weight = [[1.0, 2.0]]
bias = [0.5]
