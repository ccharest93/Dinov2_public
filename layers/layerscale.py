from torch import ones, nn
class LayerScale(nn.Module):
    def __init__(self, dim, init_values = 1e-05, inplace = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma