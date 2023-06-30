from Dinov2 import Dinov2
import torch
from torchvision import transforms
from PIL import Image
"""
Inspiration from https://github.com/ShirAmir/dino-vit-features
"""
class Dinov2HOOKS:
    FACETS = ['attn','key', 'query', 'value', 'token']
    def __init__(self, version = "vit_giant2"):
        model = Dinov2(version)
        state_dict = model.load_weights()
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        self.model = model
        self.hook_handlers = []
    
    def preprocess(self, image_path):
        pil_image = Image.open(image_path).convert('RGB')
        pil_image = transforms.Compose([ transforms.Resize(518, interpolation=transforms.InterpolationMode.LANCZOS),
                                        transforms.CenterCrop(518)])(pil_image)
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        prep_img = torch.utils.data.default_collate([prep(pil_image)])
        return prep_img, pil_image
    
    def _get_hook(self, facet):
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook

    def _register_hooks(self, layers, facets):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if 'token' in facets:
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook('token')))
                if 'attn' in facets:
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook('attn')))
                for facet in ['key', 'query', 'value']:
                    if facet in facets:
                        self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch, layers = [39], facets = [FACETS[4]]):
        
        B, C, H, W = batch.shape
        self._feats = []

        self._register_hooks(layers, facets)
        _ = self.model(batch)
        self._unregister_hooks()

        return self._feats

#usage    
if __name__ == "__main__":
    img_path = "path/to/image"
    version = "vit_large"
    extractor = Dinov2HOOKS(version)
    prep_img, pil_img = extractor.preprocess(img_path)
    layers = [extractor.model.depth-1] #index of layers we want to extract from
    facets = [extractor.FACETS[4]] #list of FACETS we want to extract
    with torch.no_grad():
        outputs = extractor._extract_features(prep_img, layers, facets)
