import torch
import torch.nn as nn
import torch.nn.functional as F

class BN(nn.Module):
    def __init__(self, dim, momentum=0.01, eps=1e-5):
        super(BN, self).__init__()
        self.dim = dim
        self.eps = eps

        self.momentum = momentum

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    
    def forward(self, input):
        bs, _ = input.shape
        
        if self.training:
            batch_mean = torch.mean(input, 0)
            batch_var = torch.var(input, dim=0, unbiased=False)

            self.running_mean = self.running_mean * (1-self.momentum) + self.momentum * batch_mean
            self.running_var = self.running_var * (1-self.momentum) + self.momentum * batch_var

            input = (input - batch_mean)  / torch.sqrt(batch_var + self.eps)
        else:
            input = (input - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return self.gamma * input + self.beta

if __name__ == '__main__':
    bs = 16
    num_feature = 4
    input = torch.randn(bs, num_feature)
    bn = BN(num_feature)
    output = bn(input)
    #print(input, output)


        
