import torch
from timm.models.vision_transformer import vit_base_patch32_224
#import timm.models.vision_transformer #import vision_transformer
from torch import nn
import timm
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
#from lightly.transforms import MAETransform

def mim_vit(src_channels=202,mask_ratio = 0.90):
    return SimMIM(src_channels=src_channels,mask_ratio = mask_ratio )

class SimMIM(nn.Module):
    def __init__(self, src_channels,mask_ratio):
        super().__init__()
        self.src_channels = src_channels
        vit= timm.create_model(
            #'vit_base_patch16_224.augreg_in21k',
            'vit_base_patch32_224',
            in_chans=self.src_channels,
            img_size=128,
            patch_size=8,
            pretrained= False, 
            # embed_dim= 768
            # features_only=True,
            # embed_dim=embed_dim,
        )
        self.mask_ratio = mask_ratio
        self.patch_size = vit.patch_embed.patch_size[0]
        #print(self.patch_size)
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        # the decoder is a simple linear layer
        self.decoder = nn.Linear(vit.embed_dim, vit.patch_embed.patch_size[0]**2 * self.src_channels)
        #print(vit.patch_embed.num_patches)
    def forward_encoder(self, images, idx_mask):
        # pass all the tokens to the encoder, both masked and non masked ones
        return self.backbone.encode(images=images, idx_mask=idx_mask)

    def forward_decoder(self, x_encoded):
        return self.decoder(x_encoded)

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_mask)
        x_encoded_masked = utils.get_at_index(x_encoded, idx_mask)
        # Decoding...
        x_pred = self.forward_decoder(x_encoded_masked)
        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        #print(patches.shape)
        #print(patches.shape)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        #print(target.shape)
        # Unpatchifying 
        pred_patches = utils.set_at_index(patches, idx_mask - 1, x_pred)
        pred_img = utils.unpatchify(pred_patches, self.patch_size, self.src_channels)
        masked_patches = utils.set_at_index(patches, idx_mask - 1, torch.zeros_like(x_pred))
        masked_img = utils.unpatchify(masked_patches, self.patch_size, self.src_channels)
        # masked_patches = utils.set_at_index(patches, idx_mask - 1, patches)
        # masked_img = utils.unpatchify(masked_patches, self.patch_size, self.src_channels)
        return x_pred,target, pred_img, masked_img
    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls()
        net.load_state_dict(state_dict)
        return net



if __name__ == '__main__':
    import torch
    import torchsummary
    model = mim_vit(src_channels=202,mask_ratio = 0.90)
    print(model)

    torchsummary.summary(model, input_size=(202, 128, 128), batch_size=2, device='cpu')

    in_tensor = torch.randn(1, 202, 128, 128)
    print("in shape:\t\t", in_tensor.shape)