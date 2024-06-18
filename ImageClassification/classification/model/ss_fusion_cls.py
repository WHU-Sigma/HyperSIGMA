import torch
import torch.nn as nn
from typing import Optional, Union, List

from model import SpatViT_fusion_patch 
from model import SpecViT_fusion 
import torch.nn.functional as F
class SSFusionFramework(torch.nn.Module):
    
    def __init__(self, 
                  img_size = None,
                  in_channels = None,
                  patch_size=None,
                  classes: int = 1):
        super(SSFusionFramework, self).__init__()

        # encoder

        self.spat_encoder =  SpatViT_fusion_patch.SpatViT(
            img_size=img_size,
            num_classes = classes,
            in_chans=in_channels,
            patch_size=patch_size,
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

        self.spec_encoder =  SpecViT_fusion.SpectralVisionTransformer(
                NUM_TOKENS = NUM_TOKENS,
                img_size=img_size,
                in_chans=in_channels,
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

        # decoder

        #self.spat_encoder.init_weights(r"spat-fina.pth")

#        self.spec_encoder.init_weights(r"spec-base.pth")
        #print('################# Initing pretrained weights for Finetuning! ###################')

        self.conv_features = nn.Conv2d(768, 64, kernel_size=1, bias=False)
        self.DR1 = nn.Conv2d(768, 128, kernel_size=1, bias=False)
        self.DR2= nn.Conv2d(768, 128, kernel_size=1, bias=False)
        self.DR3 = nn.Conv2d(768, 128, kernel_size=1, bias=False)
        self.DR4= nn.Conv2d(768, 128, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(128*4,128,kernel_size=1,stride=1,padding=0,bias=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=512,
                out_features=256
            ),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=256,
                out_features=128
            ),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=128,
                out_features=classes
            )
        )
        self.fc_spec1 = nn.Sequential(
            nn.Linear(NUM_TOKENS, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec2 = nn.Sequential(
            nn.Linear(NUM_TOKENS, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec3 = nn.Sequential(
            nn.Linear(NUM_TOKENS, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec4 = nn.Sequential(
            nn.Linear(NUM_TOKENS, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False),
            nn.Sigmoid(),
        )

    # def freeze_attn(self):
        
    #     attn_layers = ['attn.qkv', 'attn.proj']

    #     for name, param in self.spec_encoder.named_parameters():
    #         for attn_name in attn_layers:
    #             if attn_name in name:
    #                 #print('$$$$$$$$$$',name)
    #                 param.requires_grad = False

    def forward(self, x):
        b, _, h, w = x.shape

        img_feature1,img_feature2,img_feature3,img_feature4= self.spat_encoder(x)


        # spec

        spec_feature = self.spec_encoder(x)
        spec_feature = spec_feature[0]


        spec_feature = self.pool(spec_feature).view(b, -1) #b, c

        spec_weights1 = self.fc_spec1(spec_feature).view(b, -1, 1, 1)
        spec_weights2 = self.fc_spec2(spec_feature).view(b, -1, 1, 1)
        spec_weights3 = self.fc_spec3(spec_feature).view(b, -1, 1, 1)
        spec_weights4 = self.fc_spec4(spec_feature).view(b, -1, 1, 1)

        img_feature1 = self.DR1(img_feature1)
        img_feature2=  self.DR2(img_feature2)
        img_feature3 = self.DR3(img_feature3)
        img_feature4 = self.DR4(img_feature4)


        ss_feature1 = (1 + spec_weights1) * img_feature1
        ss_feature2 = (1 + spec_weights2) * img_feature2
        ss_feature3 = (1 + spec_weights3) * img_feature3
        ss_feature4 = (1 + spec_weights4) * img_feature4
 



        ss_feature1 = F.avg_pool2d(ss_feature1, ss_feature1.size()[2:])
        ss_feature2 = F.avg_pool2d(ss_feature2, ss_feature2.size()[2:])
        ss_feature3 = F.avg_pool2d(ss_feature3, ss_feature3.size()[2:])
        ss_feature4 = F.avg_pool2d(ss_feature4, ss_feature4.size()[2:])



        ss_feature1 = ss_feature1.view(b, 128,-1)
        ss_feature2 = ss_feature2.view(b, 128,-1)
        ss_feature3 = ss_feature3.view(b, 128,-1)
        ss_feature4 = ss_feature4.view(b, 128,-1)

        
        ss_feature= torch.concat((ss_feature1,ss_feature2,ss_feature3,ss_feature4),1)
        
        ss_feature = torch.unsqueeze(ss_feature,2)




        ss_feature = ss_feature.view(b, -1)


        output = self.classifier(ss_feature)


        return output




