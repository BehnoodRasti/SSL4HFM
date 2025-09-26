import torch
from timm.models.vision_transformer import vit_base_patch32_224
#import timm.models.vision_transformer #import vision_transformer
from torch import nn
import timm
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
#from lightly.transforms import MAETransform

def mae_vit(src_channels=202, mask_ratio=0.90, patch_size=16, vit_model='vit_base_patch16_224'):
    return MAE(src_channels=src_channels, mask_ratio=mask_ratio, patch_size=patch_size, vit_model=vit_model)
class MAE(nn.Module):
    def __init__(self, src_channels,mask_ratio, patch_size, vit_model):
        super().__init__()
        self.src_channels = src_channels
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        vit= timm.create_model(
            vit_model,
            in_chans=self.src_channels,
            img_size=128,
            patch_size=self.patch_size,
            pretrained= False, 
        )

        decoder_dim = 512
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            in_chans=202,
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )
    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))
        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        # Unpatchifying 
        pred_patches = utils.set_at_index(patches, idx_mask - 1, x_pred)
        pred_img = utils.unpatchify(pred_patches, self.patch_size, self.src_channels)

        masked_patches = utils.set_at_index(patches, idx_mask - 1, torch.zeros_like(x_pred))
        masked_img = utils.unpatchify(masked_patches, self.patch_size, self.src_channels)
        return x_pred,target, pred_img, masked_img
    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls()
        net.load_state_dict(state_dict)
        return net

if __name__ == '__main__':
    import torch
    import torchsummary
    model = mae_vit(src_channels=202)
    print(model)

    torchsummary.summary(model, input_size=(202, 128, 128), batch_size=2, device='cpu')

    in_tensor = torch.randn(1, 202, 128, 128)
    print("in shape:\t\t", in_tensor.shape)