
import torch
from torch import nn
import os
import math
from functools import partial
from layers.patch_embed import PatchEmbedding
from layers.swiglu_ffn import SwiGLUFFN
from layers.Transformer import Transformer
from layers.mlp import Mlp
"""
Inspiration from https://github.com/facebookresearch/dinov2
Simplified EVAL version of the DINO v2 model,
model weights are under facebookresearch/dinov2 CCC LICENSE, read before using
"""
CONFIGS = {
    "vit_giant2": {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "ffn_layer": SwiGLUFFN,
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

class Dinov2(nn.Module):
    def __init__(self, version = "vit_giant2"): 
        super().__init__()
        assert version in CONFIGS.keys(), f"version {version} not in {list(CONFIGS.keys())}"
        args = CONFIGS[version]

        self.version = version
        self.depth = args["depth"]
        self.patch_size = args["patch_size"]

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbedding(img_size= args["img_size"], patch_size= args["patch_size"], embed_dim= args["embed_dim"])
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args["embed_dim"]))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, args["embed_dim"]))
        blocks_list = [
            Transformer(
                dim= args["embed_dim"],
                num_heads=args["num_heads"],
                norm_layer=norm_layer,
                ffn_layer= args["ffn_layer"],
            )
            for _ in range(self.depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(args["embed_dim"])
    
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
    version = 'vit_giant2'
    model = Dinov2(version)
    state_dict = model.load_weights()
    model.load_state_dict(state_dict, strict=False)