import torch.nn as nn
from deep_networks.SpatViT import ALter_SpatViT

import torch.nn.functional as F
import torch
from deep_networks.SpecViT_new import altered_SpecViT_0428

class MSCDNet(nn.Module):
    def __init__(self, use_checkpoint=True):
        super(MSCDNet, self).__init__()
        self.dim_red = nn.Conv2d(kernel_size=1, in_channels=13, out_channels=100)
        self.spat_encoder  = ALter_SpatViT(img_size=128,
                                            in_chans=100,
                                            use_checkpoint=use_checkpoint,
                                            patch_size=8,
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
                                            use_abs_pos_emb=False,
                                            interval=3, pretrained='/home/songjian/project/HSIFM/pretrained_weight/HSI_spatial_checkpoint-1600.pth')
        NUM_TOKENS = 100
        self.spec_encoder = altered_SpecViT_0428(
            NUM_TOKENS=NUM_TOKENS,
            img_size=128,
            in_chans=100,
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
            use_checkpoint=False,
            use_abs_pos_emb=False,
            interval=3
        )

        self.conv_features = nn.Conv2d(128, 128, kernel_size=1, bias=False)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc_spec = nn.Sequential(
            nn.Linear(NUM_TOKENS, 768, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(768, 768, bias=False),
            nn.Sigmoid(),
        )

        self.fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=768 * 2, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=768 * 2, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=768 * 2, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=768 * 2, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())

        self.smooth_layer_3 = nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1)
        self.smooth_layer_2 = nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1)
        self.smooth_layer_1 = nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1)

        self.main_clf_1 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y
    

    def forward(self, x, y):
        b, _, h, w = x.shape

        x = self.dim_red(x)
        y = self.dim_red(y)

        _, x_1, x_2, x_3, x_4 = self.spat_encoder(x)
        _, y_1, y_2, y_3, y_4 = self.spat_encoder(y)

        x_spe = self.spec_encoder(x)[0]
        y_spe = self.spec_encoder(y)[0]

        x_spe = self.pool(x_spe).view(b, -1) #b, c
        y_spe = self.pool(y_spe).view(b, -1) #b, c

        x_spe_weights = self.fc_spec(x_spe).view(b, -1, 1, 1)
        y_spe_weights = self.fc_spec(y_spe).view(b, -1, 1, 1)
        
        x_4 = (1 + x_spe_weights) * x_4
        y_4 = (1 + y_spe_weights) * y_4

        # print(x_spe.size())
       
        feature_4 = torch.cat([x_4, y_4], dim=1)
        feature_4 =  self.fuse_layer_4(feature_4)

        x_3 = (1 + x_spe_weights) * x_3
        y_3 = (1 + y_spe_weights) * y_3

        feature_3 = torch.cat([x_3, y_3], dim=1)
        feature_3 =  self.fuse_layer_3(feature_3)

        feature_3 = self._upsample_add(feature_4, feature_3)

        x_2 = (1 + x_spe_weights) * x_2
        y_2 = (1 + y_spe_weights) * y_2

        feature_2 = torch.cat([x_2, y_2], dim=1)
        feature_2 =  self.fuse_layer_2(feature_2)

        feature_2 = self._upsample_add(feature_3, feature_2)

        x_1 = (1 + x_spe_weights) * x_1
        y_1 = (1 + y_spe_weights) * y_1
        feature_1 = torch.cat([x_1, y_1], dim=1)
        feature_1 =  self.fuse_layer_1(feature_1)

        feature_1 = self._upsample_add(feature_2, feature_1)

        output = self.main_clf_1(feature_1)
        output = F.interpolate(output, size=x.size()[2:], mode='bilinear')
        return output
    

if __name__ == '__main__':
    input_data = torch.rand(1, 13, 256, 256)
    cd_network = MSCDNet()

    output = cd_network(input)
    print(output)