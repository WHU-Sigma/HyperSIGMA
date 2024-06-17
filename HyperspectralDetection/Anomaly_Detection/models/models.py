import torch
import torch.nn as nn
from typing import Optional, Union, List

from .SpatViT import SpatialVisionTransformer
from .SpecViT import SpectralVisionTransformer


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SpatialHADFramework(torch.nn.Module):
    
    def __init__(self, 
                  args, 
                  img_size = None,
                  in_channels = None):
        super(SpatialHADFramework, self).__init__()

        # encoder

        self.args = args

        print('################# Using SpatSIGMA as backbone! ###################')
        
        self.encoder =  SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=1,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval = 3,
            n_points=8
        )

        self.encoder.init_weights('spat-vit-b-checkpoint-1599.pth')
        print('################# Initing SpatViT pretrained weights for Finetuning! ###################')


        self.conv_features = nn.Conv2d(128, 128, kernel_size=1, bias=False)

        self.conv_fc = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, bias=False)
        )


    def forward(self, x):

        # x: (b, c, h, w)
        # ts:(b, c)

        b, _, h, w = x.shape

        features = self.encoder(x)

        img_feature = sum(features)

        img_feature = self.conv_features(img_feature) # b, 128, h, w

        output = self.conv_fc(img_feature)

        return output.squeeze(1)


class SSHADFramework(torch.nn.Module):
    
    def __init__(self, 
                  args, 
                  img_size = None,
                  in_channels = None):
        super(SSHADFramework, self).__init__()

        # encoder

        self.args = args

        print('################# Using HyperSIGMA as backbone! ###################')

        self.spat_encoder =  SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=1,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval = 3,
            n_points=8
        )

        NUM_TOKENS = 100

        self.spec_encoder =  SpectralVisionTransformer(
                NUM_TOKENS = NUM_TOKENS,
                img_size=img_size,
                in_chans=in_channels,
                drop_path_rate=0.1,
                out_indices=[11],
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                use_checkpoint=True,
                use_abs_pos_emb=True,
                interval = 3,
                n_points=8
        )

        # decoder

        self.spat_encoder.init_weights('spat-vit-b-checkpoint-1599.pth')

        self.spec_encoder.init_weights('spec-vit-b-checkpoint-1599.pth')

        print('################# Initing SpatViT and SpecViT pretrained weights for Finetuning! ###################')


        self.conv_features = nn.Conv2d(128, 128, kernel_size=1, bias=False)

        self.conv_fc = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, bias=False)
        )


        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc_spec = nn.Sequential(
            nn.Linear(NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 128, bias=False),
            nn.Sigmoid(),
        )


    def forward(self, x):

        # x: (b, c, h, w)
        # ts:(b, c)

        # spat

        b, _, h, w = x.shape

        img_features = self.spat_encoder(x)

        img_feature = sum(img_features)

        img_feature = self.conv_features(img_feature) # b, 128, h, w

        # spec

        spec_features = self.spec_encoder(x)

        spec_feature = sum(spec_features) # b, c, 128

        spec_feature = self.pool(spec_feature).view(b, -1) #b, c

        spec_weights = self.fc_spec(spec_feature).view(b, -1, 1, 1)
        
        ss_feature = (1 + spec_weights) * img_feature

        
        # ss

        output = self.conv_fc(ss_feature)

        return output.squeeze(1)




