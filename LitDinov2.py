import pytorch_lightning as pl
from typing import Callable
from torch import Tensor, nn
import torch
import math
from functools import partial
from layers.patch_embed import PatchEmbedding
from layers.swiglu_ffn import SwiGLUFFN
from layers.layerscale import LayerScale
from layers.attention import Attention
from layers.mlp import Mlp
from torch.nn.init import trunc_normal_
import os
"""
Inspiration from https://github.com/facebookresearch/dinov2
Simplified EVAL version of the DINO v2 model, that does not require the XTRANSFORMER library
model weights are under facebookresearch/dinov2 CCC LICENSE, read before using
"""
VERSIONS = {
    "vit_giant2": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "ffn_layer": Mlp,
    },
    "vit_large": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "ffn_layer": Mlp,
    },
    "vit_base": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "ffn_layer": Mlp,
    },
    "vit_small": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_layer": Mlp,
    },
}
CHECKPOINTS = {"vit_giant2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
               "vit_large": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
               "vit_base": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
               "vit_small": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"}

#HELPER FUNCTIONS
def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

#MAIN MODEL LAYERS
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_layer: Callable[..., nn.Module],
        init_values= 1e-05,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm
    ) -> None:
        super().__init__()

        #Attention resBlock
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop= 0.0,
            proj_drop= 0.0,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        #FFN resBlock
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * 4.0)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            bias= True,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
    def forward(self, x: Tensor) -> Tensor:
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

class LitDinov2(pl.LightningModule):
    def __init__(self, version = "vit_giant2", init_values= 1e-05): 
        super().__init__()
        assert version in VERSIONS.keys(), f"version {version} not in {list(VERSIONS.keys())}"
        args = VERSIONS[version]

        self.version = version
        self.depth = args["depth"]
        self.patch_size = args["patch_size"]

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbedding(img_size= args["img_size"], patch_size= args["patch_size"], embed_dim= args["embed_dim"])
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args["embed_dim"]))
        self.mask_token = nn.Parameter(torch.zeros(1, args["embed_dim"]))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, args["embed_dim"]))
        blocks_list = [
            Block(
                dim= args["embed_dim"],
                num_heads=args["num_heads"],
                norm_layer=norm_layer,
                ffn_layer= args["ffn_layer"],
                init_values=init_values,
            )
            for i in range(self.depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(args["embed_dim"])
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)
    
    def load_weights(self, output_dir = "./pretrain_pth"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, f"{self.version}.pth")):
            print(f"Downloading weights for {self.version}...")
            url = CHECKPOINTS[self.version]
            state_dict = torch.hub.load_state_dict_from_url(url, output_dir, map_location="cpu",file_name=f"{self.version}.pth")
        else:
            state_dict = torch.load(os.path.join(output_dir, f"{self.version}.pth"), map_location="cpu")
        return state_dict


    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1 #remove cls token
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def forward(self,x):
        B, nc, w, h = x.shape #B C H W
        x = self.patch_embed(x) #B N D
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) #B N+1 D
        x = x + self.interpolate_pos_encoding(x, w, h)

        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)    
        return x

#usage
if __name__ == "__main__":
    version = 'vit_large'
    model = LitDinov2(version)
    state_dict = model.load_weights()
    print(state_dict.keys())
    model.load_state_dict(state_dict, strict=False)