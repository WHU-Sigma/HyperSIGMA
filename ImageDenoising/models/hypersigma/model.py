import torch
import torch.nn as nn

from .Spatial import SpatialVisionTransformer
from .Spectral import SpectralVisionTransformer
from .Spatial_route import Adpater
from .Spectral_route import Adapter_Spectral
class SSFusionFramework(torch.nn.Module):
    def __init__(self, img_size = None, in_channels = None):
        super(SSFusionFramework, self).__init__()

        # encoder
        self.spat_encoder =  SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=2,
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
            n_points=8,
            original_channels=191
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

        self.adapter =  Adpater(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=1,
            drop_path_rate=0.1,
            out_indices=[3],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_abs_pos_emb=True,
            interval = 3,
            n_points=8,
            original_channels=191
        )

        self.adapter_spec =  Adapter_Spectral(
            NUM_TOKENS = NUM_TOKENS,
            img_size=img_size,
            in_chans=in_channels,
            drop_path_rate=0.1,
            out_indices=[3],
            embed_dim=128,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=False,
            use_abs_pos_emb=True,
            interval = 3,
            n_points=8
        )
        # decoder

        self.spat_encoder.init_weights('/mnt/code/users/yuchunmiao/SST-master/pre_train/spat-vit-base-ultra-checkpoint-1599.pth')

        self.spec_encoder.init_weights('/mnt/code/users/yuchunmiao/SST-master/pre_train/spec-vit-base-ultra-checkpoint-1599.pth')

        # print('################# Initing pretrained weights for Finetuning! ###################')

        # self.conv_features = nn.Conv2d(100, 100, kernel_size=1, bias=False)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc_spec = nn.Sequential(
            nn.Linear(NUM_TOKENS, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 191, bias=False),
            nn.Sigmoid(),
        )

        self.conv3_reconstruct = nn.Conv2d(382, 191, kernel_size=3, padding=1)
        self.conv_tail = nn.Conv2d(191, 191, kernel_size=3, padding=1)

    def forward(self, x):
        # spat
        b, _, h, w = x.shape
        x0, img_feature = self.spat_encoder(x)
        img_feature = self.adapter(img_feature)

        # spec

        spec_feature, deform_inputs, H, W= self.spec_encoder(x)
        spec_feature = self.adapter_spec(spec_feature, deform_inputs, H, W)
        spec_feature = self.pool(spec_feature).view(b, -1) #b, c
        spec_weights = self.fc_spec(spec_feature).view(b, -1, 1, 1)


        # ss
        ss_feature = (1 + spec_weights) * img_feature
 
        x = torch.cat([x0, ss_feature], dim=1) 
        x = self.conv3_reconstruct(x) 
        x = self.conv_tail(x)

        return x


def spat_vit_b_rvsa(args=None):
    model = SSFusionFramework(
        img_size=64,
        in_channels=191,
    )
    return model

if __name__ == "__main__":

    backbone = spat_vit_b_rvsa()
    backbone.cuda()
    backbone.eval()
    input = torch.Tensor(2, 191, 64, 64).cuda()
    out = backbone(input)
    print(out.shape)

    
    


