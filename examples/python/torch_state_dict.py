"""Torch-style nn.Module without torch — harvested via state_dict() duck-typing.

The driver calls any object's state_dict() and re-emits an alternating
2-D-weight/1-D-bias stack as layer{N}.weight / layer{N}.bias, so this compiles
to a runnable `mlp` artifact without torch installed. Input [1.0, 1.0] →
[11.5, 16.0].
"""
class FakeTensor:
    def __init__(self, data):
        self._data = data
    def tolist(self):
        return self._data

class Linear:
    def __init__(self, weight, bias):
        self.weight = FakeTensor(weight)
        self.bias = FakeTensor(bias)

class Net:
    def __init__(self):
        self.fc1 = Linear([[1.0, 2.0]], [0.5])
        self.fc2 = Linear([[3.0], [4.0]], [1.0, 2.0])
    def state_dict(self):
        return {
            "fc1.weight": self.fc1.weight,
            "fc1.bias": self.fc1.bias,
            "fc2.weight": self.fc2.weight,
            "fc2.bias": self.fc2.bias,
        }

model = Net()
