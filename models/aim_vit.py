

import torch
import torchvision
from torch import nn

from lightly.models import utils
from lightly.models.modules import AIMPredictionHead, MaskedCausalVisionTransformer
#from lightly.transforms import AIMTransform

from timm.models.vision_transformer import vit_base_patch32_224
#import timm.models.vision_transformer #import vision_transformer
from torch import nn
import timm
def aim_vit(src_channels=202, mask_ratio=0.90, patch_size=16, vit_model='vit_base_patch16_224'):
    return AIM(src_channels=src_channels, mask_ratio=mask_ratio, patch_size=patch_size, vit_model=vit_model)
class AIM(nn.Module):
    def __init__(self, src_channels, mask_ratio, patch_size, vit_model):
        super().__init__()
        self.src_channels = src_channels
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        vit = MaskedCausalVisionTransformer(
            in_chans=self.src_channels,
            img_size=128,
            patch_size=self.patch_size,
            embed_dim=self._get_embed_dim(vit_model),
            depth=self._get_depth(vit_model),
            num_heads=self._get_num_heads(vit_model),
            qk_norm=False,
            class_token=False,
            no_embed_class=True,
            num_classes= 0,
        )
        self.backbone = vit
        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=self.backbone.pos_embed, has_class_token=self.backbone.has_class_token
        )
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_patches = self.backbone.patch_embed.num_patches

       
        self.projection_head = AIMPredictionHead(
            input_dim=self.backbone.embed_dim, output_dim=self.src_channels * self.patch_size**2, num_blocks=1
        )

    def forward(self, images):
        batch_size = images.shape[0]

        mask = utils.random_prefix_mask(
            size=(batch_size, self.num_patches),
#            min_prefix_length=0,#int(self.num_patches*self.mask_ratio),
            max_prefix_length=self.num_patches - 1,
            device=images.device,
        )
        features = self.backbone.forward_features(images, mask=mask)
        # Add positional embedding before head.
        features = self.backbone._pos_embed(features)
        predictions = self.projection_head(features)

        # Convert images to patches and normalize them.
        patches = utils.patchify(images, self.patch_size)
        #patches = utils.normalize_mean_var(patches, dim=-1)
        #pred_patches = utils.set_at_index(patches, mask, predictions)
        pred_img = utils.unpatchify(predictions, self.patch_size, self.src_channels)
        #masked_patches = utils.set_at_index(patches, mask, torch.zeros_like(predictions))
        #masked_img = utils.unpatchify(patches, self.patch_size, self.src_channels)
        masked_img = patches * mask.unsqueeze(2)
        masked_img = utils.unpatchify(masked_img, self.patch_size, self.src_channels)
        return predictions, patches, pred_img, masked_img
    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls()
        net.load_state_dict(state_dict)
        return net
    @staticmethod
    def _get_embed_dim(vit_model):
        """Get the embedding dimension based on the ViT model."""
        if 'small' in vit_model:
            return 384
        elif 'base' in vit_model:
            return 768
        elif 'large' in vit_model:
            return 1024
        elif 'huge' in vit_model:
            return 1280
        else:
            raise ValueError(f"Unknown vit_model: {vit_model}")

    @staticmethod
    def _get_depth(vit_model):
        """Get the depth (number of layers) based on the ViT model."""
        if 'small' in vit_model:
            return 12
        elif 'base' in vit_model:
            return 12
        elif 'large' in vit_model:
            return 24
        elif 'huge' in vit_model:
            return 32
        else:
            raise ValueError(f"Unknown vit_model: {vit_model}")

    @staticmethod
    def _get_num_heads(vit_model):
        """Get the number of attention heads based on the ViT model."""
        if 'small' in vit_model:
            return 6
        elif 'base' in vit_model:
            return 12
        elif 'large' in vit_model:
            return 16
        elif 'huge' in vit_model:
            return 16
        else:
            raise ValueError(f"Unknown vit_model: {vit_model}")


if __name__ == '__main__':
    import torch
    import torchsummary
    model = aim_vit(src_channels=202,mask_ratio = 0.90)
    print(model)

    torchsummary.summary(model, input_size=(202, 128, 128), batch_size=2, device='cpu')

    in_tensor = torch.randn(1, 202, 128, 128)
    print("in shape:\t\t", in_tensor.shape)