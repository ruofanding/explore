import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.shape = shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))


    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)

        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return normalized * self.gamma + self.beta

if __name__ == '__main__':
    bs = 16
    num_feature = 4
    input = torch.randn(bs, num_feature)
    ln = LayerNorm(num_feature)
    output = ln(input)
    #print(input, output)