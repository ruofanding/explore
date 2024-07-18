import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    # shape [bs, num_head, seq, d_k] or [bs, seq, d_k]
    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weight = F.softmax(scores, dim=-1)
        return torch.matmul(weight, v), weight

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_head):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_head == 0
        self.d_k = model_dim // num_head
        self.num_head = num_head

        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.out_linear = nn.Linear(model_dim, model_dim)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, q, k, v, mask=None):
        bs, seq, _ = q.shape

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = q.view(bs, seq, self.num_head, self.d_k).transpose(1, 2)
        k = k.view(bs, seq, self.num_head, self.d_k).transpose(1, 2)
        v = v.view(bs, seq, self.num_head, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # Make mask broadcastable with attention scores
            print(mask.shape)

        output, attn_weights = self.attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, seq, self.num_head * self.d_k)
        
        return self.out_linear(output), attn_weights

if __name__ == '__main__':
    # Example usage
    model_dim = 512
    num_head = 8
    multi_head_attn = MultiHeadAttention(model_dim, num_head)

    q = torch.randn(32, 10, model_dim)
    k = torch.randn(32, 10, model_dim)
    v = torch.randn(32, 10, model_dim)
    mask = torch.ones(32, 10, 10)  # Example mask

    output, attn_weights = multi_head_attn(q, k, v, mask)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)
