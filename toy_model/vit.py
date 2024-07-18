import torch
import torch.nn as nn
import torch.nn.functional as F
from mha import MultiHeadAttention
from torch.utils.tensorboard import SummaryWriter



class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = self.proj(x)  # shape: [batch_size, embed_dim, num_patches_h, num_patches_w]
        x = x.flatten(2)  # shape: [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # shape: [batch_size, num_patches, embed_dim]
        return x

class VitLayer(nn.Module):
    def __init__(self, emb_dim, num_head, ffn_dim=2048, dropout=0.1, layer_norm='pre-ln'):
        super(VitLayer, self).__init__()
        self.layer_norm = layer_norm
        print('using ' + self.layer_norm)
        self.emb_dim = emb_dim
        self.mha = MultiHeadAttention(emb_dim, num_head)
        self.ffn = self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, emb_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
    
    def forward_preln(self, x):
        norm_x = self.ln1(x)
        x = self.mha(norm_x, norm_x, norm_x)[0] + x

        norm_x = self.ln2(x)
        x = self.ffn(norm_x) + x
        return x
    
    def forward_postln(self, x):
        x = self.ln1(self.mha(x, x, x)[0] + x)
        x = self.ln2(self.ffn(x) + x)
        return x
    
    def forward_noln(self, x):
        x = self.mha(x, x, x)[0] + x
        x = self.ffn(x) + x
        return x


    def forward(self, x):
        if self.layer_norm == 'pre-ln':
            return self.forward_preln(x)
        elif self.layer_norm == 'post-ln':
            return self.forward_postln(x)
        else:
            return self.forward_noln(x)


class VIT(nn.Module):
    def __init__(self, img_size, patch_size, emb_dim=768, in_channels=3, num_head=12, num_layer=12, ffn_dim=2048, dropout=0.1, layer_norm='pre-ln'):
        super(VIT, self).__init__()
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.emb_dim = emb_dim
        self.in_channels = in_channels


        self.class_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, emb_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.emb_dim))

        self.layers = nn.ModuleList([VitLayer(emb_dim, num_head, ffn_dim, dropout, layer_norm) for _ in range(num_layer)])

    # x: [bs, H, W, C]
    def forward(self, x):
        bs, _, _, _ = x.shape
        emb = self.patch_embedding(x)

        class_emb = self.class_token.expand(bs, -1, -1)
        emb = torch.concat([class_emb, emb], dim=1)
        input = emb + self.position_embedding


        for layer in self.layers:
            input = layer(input)
        
        return input

if __name__ == '__main__':
    bs = 16
    img_size = 224
    patch_size = 16
    input = torch.randn(bs, img_size, img_size, 3)

    vit = VIT(img_size, patch_size)
    vit(input)

        