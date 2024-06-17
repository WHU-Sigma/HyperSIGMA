import torch, math
import torch.nn as nn

from .Spatial import SpatialVisionTransformer
from .Spectral import SpectralVisionTransformer
from .Spatial_route import Adpater
from .Spectral_route import Adapter_Spectral

def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True, dilation=1, groups=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias, groups=groups)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation, groups=groups)

    else:
       padding = int((kernel_size - 1) / 2) * dilation
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           stride, padding=padding, bias=bias, dilation=dilation, groups=groups)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class SSFusionFramework(torch.nn.Module):
    
    def __init__(self, img_size = None, in_channels = None, scale=4):
        super(SSFusionFramework, self).__init__()

        # encoder
        self.spat_encoder =  SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=1,
            drop_path_rate=0.0,
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
            use_abs_pos_emb=False,
            interval = 3,
            n_points=8,
            original_channels=in_channels
        )

        NUM_TOKENS = 36

        self.spec_encoder =  SpectralVisionTransformer(
                NUM_TOKENS = NUM_TOKENS,
                img_size=img_size,
                in_chans=in_channels,
                drop_path_rate=0.0,
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
                use_abs_pos_emb=False,
                interval = 3,
                n_points=8
        )

        self.adapter =  Adpater(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=1,
            drop_path_rate=0.0,
            out_indices=[3],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_abs_pos_emb=False,
            interval = 3,
            n_points=8,
            original_channels=in_channels
        )

        self.adapter_spec =  Adapter_Spectral(
            NUM_TOKENS = NUM_TOKENS,
            img_size=img_size,
            in_chans=in_channels,
            drop_path_rate=0.0,
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
            use_abs_pos_emb=False,
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
            nn.Linear(128, in_channels, bias=False),
            nn.Sigmoid(),
        )

   
        self.conv3_reconstruct = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.conv_tail = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.upsample = Upsampler(default_conv, scale, in_channels)
        self.skip_conv = default_conv(in_channels, in_channels, 3)

    def forward(self, x, lms):

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

        x = self.upsample(x)

        x = torch.cat([x, self.skip_conv(lms)], dim=1)

        x = self.conv_tail(x)

        return x


def spat_vit_b_rvsa(inchannels=100, original_channels=191, img_size=64, args=None):

    model = SSFusionFramework(
        img_size=32,
        in_channels=original_channels,
        scale = args.n_scale
    )
    return model

if __name__ == "__main__":

    backbone = spat_vit_b_rvsa(original_channels=48)

    backbone.eval()

    input = torch.Tensor(1, 48, 32, 32)
    lms = torch.Tensor(1, 48, 128, 128)
    out = backbone(input, lms)
    print(out.shape)

    
    


