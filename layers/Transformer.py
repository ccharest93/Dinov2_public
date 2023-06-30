from torch import nn
from torch import Tensor
from layers.attention import MemEffAttention
from layers.layerscale import LayerScale
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_layer,
        norm_layer = nn.LayerNorm
    ):
        super().__init__()

        #Attention resBlock
        self.norm1 = norm_layer(dim)
        self.attn = MemEffAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop= 0.0,
            proj_drop= 0.0,
        )
        self.ls1 = LayerScale(dim, init_values=1e-05)

        #FFN resBlock
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * 4.0)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            bias= True,
        )
        self.ls2 = LayerScale(dim, init_values=1e-05)
    def forward(self, x):
        #Attention
        res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.ls1(x)
        x = res + x

        #FFN
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.ls2(x)
        x = res + x
        return x
