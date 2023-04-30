from torch import nn
class PatchEmbedding(nn.Module):
    def __init__( self, img_size = 224, patch_size = 14, embed_dim = 768, norm_layer = None):
        super().__init__()

        image_HW = (img_size, img_size)
        self.patch_HW = (patch_size, patch_size)
        patch_grid_size = (
            image_HW[0] // self.patch_HW[0],
            image_HW[1] // self.patch_HW[1],
        )
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.proj = nn.Conv2d(3, embed_dim, kernel_size=self.patch_HW, stride=self.patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape #B C H W
        patch_H, patch_W = self.patch_HW

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B D NH NW
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B NHW D
        x = self.norm(x)
        return x
