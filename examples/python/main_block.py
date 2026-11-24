"""Weights built inside `if __name__ == "__main__"` — second-pass extraction.

The bridge first runs the file as an import; when that yields no tensors it
re-runs it as __main__, so scripts that build their model in a main block
still compile. Input [1.0, 1.0] → [3.0, 3.0].
"""
def build():
    global layer0_weight, layer0_bias
    layer0_weight = [[2.0, 0.0], [0.0, 2.0]]
    layer0_bias = [1.0, 1.0]

if __name__ == "__main__":
    build()
