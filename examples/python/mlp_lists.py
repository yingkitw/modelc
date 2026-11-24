"""Pure-stdlib 2-layer MLP — compiles to a `mlp` artifact.

`modelc pack mlp_lists.py` → layer0.weight [2,2], layer0.bias [2],
layer1.weight [1,2], layer1.bias [1] (F32). Forward for input [1.0, 2.0]
is 1.3 (asserted by tests/python_samples_test.rs).
"""
layer0_weight = [[1.0, 0.5], [-0.5, 2.0]]
layer0_bias = [0.1, -0.1]
layer1_weight = [[2.0, -1.0]]
layer1_bias = [0.5]
