import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from HyperspectralUnmixing.func import SumToOne,Decoder

from .SpatVit import SpatViT
from .SpecVit import SpecViT

class SpatSIGMA_Unmix(torch.nn.Module):
    def __init__(self, args ):
        super(SpatSIGMA_Unmix, self).__init__()
        self.patch_size = args.patch_size
        self.encoder = SpatViT(img_size=args.patch_size,
                                            in_chans=args.channels,
                                            use_checkpoint=True,
                                            patch_size=args.seg_patches,
                                            drop_path_rate=0.1, out_indices=[3, 5, 7, 11], embed_dim=768,
                                            depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                            drop_rate=0., attn_drop_rate=0., use_abs_pos_emb=False, n_points=8)



        self.conv_features1 = nn.Conv2d(args.embed_dim, args.NUM_TOKENS, kernel_size=1, bias=False)
        self.conv_features2 = nn.Conv2d(args.embed_dim, args.NUM_TOKENS, kernel_size=1, bias=False)
        self.conv_features3 = nn.Conv2d(args.embed_dim, args.NUM_TOKENS, kernel_size=1, bias=False)
        self.conv_features4 = nn.Conv2d(args.embed_dim, args.NUM_TOKENS, kernel_size=1, bias=False)


        self.conv1 =nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS, args.NUM_TOKENS, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.NUM_TOKENS),
            nn.Dropout(0.2),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS, args.NUM_TOKENS, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS, args.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS, args.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.smooth = nn.Conv2d(args.NUM_TOKENS*4, args.NUM_TOKENS, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS, args.num_em, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.num_em),
            nn.Dropout(0.2),
        )

        self.sumtoone = SumToOne(args.scale)
        self.decoder = Decoder(args)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.normal_(m.weight.data, mean=0.0, std=0.3)

    def getAbundances(self, x):
        H, W = x.shape[2], x.shape[3]

        img_fea = self.encoder(x)  # x: list : 5

        x = []
        x.append(img_fea[0])
        ops = [self.conv_features1, self.conv_features2, self.conv_features3, self.conv_features4]
        for i in range(len(ops)):
            x.append(ops[i](img_fea[i+1]))
        p4 = self.conv1(x[4]) #
        p3 = self.conv2(x[3])
        p2 = self.conv3(x[2])
        p1 = self.conv4(x[1])
        p1 = torch.cat([p1,p2,p3,p4], dim=1)

        p1 = F.interpolate(p1, size=(H, W), mode='bilinear', align_corners=True)
        p1 = self.smooth(p1)
        x = self.conv5(p1)
        abunds = self.sumtoone(x)
        abunds = abunds
        return abunds


    def forward(self, patch):
        abunds = self.getAbundances(patch)
        output = self.decoder(abunds)
        return abunds,output

    def getEndmembers(self):
        endmembers = self.decoder.getEndmembers()
        if endmembers.shape[2] > 1:
            endmembers = np.squeeze(endmembers).mean(axis=2).mean(axis=2)
        else:
            endmembers = np.squeeze(endmembers)
        return endmembers
class HyperSIGMA_Unmix(torch.nn.Module):
    def __init__(self, args ):
        super(HyperSIGMA_Unmix, self).__init__()
        self.patch_size = args.patch_size
        self.spat_encoder = SpatViT(img_size=args.patch_size,
                                            in_chans=args.channels,
                                            use_checkpoint=True,
                                            patch_size=args.seg_patches,
                                            drop_path_rate=0.1, out_indices=[3, 5, 7, 11], embed_dim=768,
                                            depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                            drop_rate=0., attn_drop_rate=0., use_abs_pos_emb=False, n_points=8)
        self.spec_encoder = SpecViT(
            NUM_TOKENS=args.NUM_TOKENS,
            img_size=args.patch_size,
            in_chans=args.channels,
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

        self.conv_features1 = nn.Conv2d(args.embed_dim, args.NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec1 = nn.Sequential(
            nn.Linear(args.NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, args.NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )
        self.conv_features2 = nn.Conv2d(args.embed_dim, args.NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec2 = nn.Sequential(
            nn.Linear(args.NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, args.NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )
        self.conv_features3 = nn.Conv2d(args.embed_dim, args.NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec3 = nn.Sequential(
            nn.Linear(args.NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, args.NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )
        self.conv_features4 = nn.Conv2d(args.embed_dim, args.NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec4 = nn.Sequential(
            nn.Linear(args.NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, args.NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.conv1 =nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS, args.NUM_TOKENS, kernel_size=1),
            nn.LeakyReLU(0.02),# nn.ReLU(),
            nn.BatchNorm2d(args.NUM_TOKENS),
            nn.Dropout(0.2),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS, args.NUM_TOKENS, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS, args.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS, args.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.NUM_TOKENS),
            nn.Dropout(0.2),
        )

        self.conv2_ = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS*2, args.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv3_ = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS*2, args.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv4_ = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS*2, args.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS*4, args.NUM_TOKENS*2, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.NUM_TOKENS*2),
            nn.Dropout(0.2),
            nn.Conv2d(args.NUM_TOKENS*2,  args.NUM_TOKENS, kernel_size=(1, 1)) )


        self.conv5 = nn.Sequential(
            nn.Conv2d(args.NUM_TOKENS, args.num_em, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(args.num_em),
            nn.Dropout(0.2),
        )

        self.sumtoone = SumToOne(args.scale)
        self.decoder = Decoder(args)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.normal_(m.weight.data, mean=0.0, std=0.3)


    def forward_fusion(self, x):
        # x: (b, c, h, w)
        # ts:(b, c)
        # spat
        b, _, h, w = x.shape
        img_features = self.spat_encoder(x)

        img_fea  = []
        ops = [self.conv_features1, self.conv_features2, self.conv_features3, self.conv_features4]
        for i in range(len(ops)):
            img_fea.append(ops[i](img_features[i+1]))

        spec_features = self.spec_encoder(x)
        spec_feature = spec_features[-1]
        spec_feature = self.pool(spec_feature).view(b, -1)  # b, c

        spec_weights = []
        ops_ = [self.fc_spec1, self.fc_spec2, self.fc_spec3, self.fc_spec4]
        for i in range(len(ops_)):
            spec_weights.append((ops_[i](spec_feature)).view(b, -1, 1, 1))
        ss_feature = []
        ss_feature.append(x)
        for i in range(4):
            ss_feature.append((1 + spec_weights[i]) * img_fea[i])
        return ss_feature

    def getAbundances(self, x):
        H, W = x.shape[2], x.shape[3]
        x = self.forward_fusion(x)  # x: list : 5
        p4 = self.conv1(x[4])
        p3 = self.conv2(x[3])
        p2 = self.conv3(x[2])
        p1 = self.conv4(x[1])
        p1 = torch.cat([p1,p2,p3,p4], dim=1)

        p1 = F.interpolate(p1, size=(H, W), mode='bilinear', align_corners=True)
        p1 = self.smooth(p1)
        x = self.conv5(p1)
        abunds = self.sumtoone(x)
        abunds = abunds
        return abunds


    def forward(self, patch):
        abunds = self.getAbundances(patch)
        output = self.decoder(abunds)
        return abunds,output

    def getEndmembers(self):
        endmembers = self.decoder.getEndmembers()
        if endmembers.shape[2] > 1:
            endmembers = np.squeeze(endmembers).mean(axis=2).mean(axis=2)
        else:
            endmembers = np.squeeze(endmembers)
        return endmembers


# if __name__ == "__main__":
#     from HyperspectralUnmixing.func import get_args
#
#     img_size = 64
#     in_chans = 162
#     seed, Dataset_name = 1, 'Urban4'
#     args = get_args(Dataset_name,seed,
#                     use_checkpoint=True,
#                     mode='Spat_Spec_Pretraining')
#     model = HyperSIGMA_Unmix(args)
#     model.eval()
#     input = torch.Tensor(2, in_chans, img_size, img_size)
#     output = model(input)
#     for x in output:
#         print(x.shape)
