# -------------HyperSIGMA for hyperspectral change detection---------------------------
import torch
import torch.nn as nn
from ChangeDetection.func import DoubleConv_pad
from .SpatVit import SpatViT
from .SpecVit import SpecViT



class SpatSIGMA_CD(nn.Module):
    def __init__(self, args, use_checkpoint=False):
        super(SpatSIGMA_CD, self).__init__()
        self.patch_size = args.patch_size
        self.encoder = SpatViT(img_size=args.patch_size,
                               in_chans=args.channels,
                               use_checkpoint=use_checkpoint,
                               patch_size=args.seg_patches,
                               drop_path_rate=0.1, out_indices=[3, 5, 7, 11], embed_dim=args.embed_dim,
                               depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                               drop_rate=0., attn_drop_rate=0., use_abs_pos_emb=False, n_points=8)

        num_dim = 64
        self.conv_features1 = nn.Conv2d(args.embed_dim, num_dim, kernel_size=1, bias=False)
        self.conv_features2 = nn.Conv2d(args.embed_dim, num_dim, kernel_size=1, bias=False)
        self.conv_features3 = nn.Conv2d(args.embed_dim, num_dim, kernel_size=1, bias=False)
        self.conv_features4 = nn.Conv2d(args.embed_dim, num_dim, kernel_size=1, bias=False)

        self.conv1 = DoubleConv_pad(num_dim, 64, kernel_sz=1)
        self.conv2 = DoubleConv_pad(num_dim, 64, kernel_sz=1)
        self.conv3 = DoubleConv_pad(num_dim, 64, kernel_sz=3)
        self.conv4 = DoubleConv_pad(num_dim, 64, kernel_sz=3)

        if self.patch_size == 5:
            in_planes = num_dim*4
            self.BCD_classifier = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes, in_planes // 2, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 2, 2, kernel_size=1))
        elif self.patch_size == 15:
            in_planes = num_dim * 4
            self.BCD_classifier = nn.Sequential(
                nn.Conv2d(in_planes, in_planes // 2, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 2, in_planes // 4, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 4, in_planes // 4, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 4, 2, kernel_size=1))

    def forward(self,x,y):
        x0 = self.encoder(x)  # x: list : 5
        y0 = self.encoder(y)
        x, y = [], []
        x.append(x0[0])
        y.append(y0[0])
        ops = [self.conv_features1, self.conv_features2, self.conv_features3, self.conv_features4]
        for i in range(len(ops)):
            x.append(ops[i](x0[i + 1]))
            y.append(ops[i](y0[i + 1]))

        f1 = torch.abs(x[4]-y[4])
        f1 = self.conv1(f1)

        f2 = torch.abs(x[3] - y[3])
        f2 = self.conv2(f2)

        f3 = torch.abs(x[2] - y[2])
        f3 = self.conv3(f3)

        f4 = torch.abs(x[1] - y[1])
        f4 = self.conv4(f4)

        f4 = torch.cat([f1, f2, f3, f4], dim=1)
        output = self.BCD_classifier(f4.squeeze())
        return output.squeeze()

class HyperSIGMA_CD(torch.nn.Module):
    def __init__(self, args ):
        super(HyperSIGMA_CD, self).__init__()
        self.patch_size = args.patch_size
        self.spat_encoder = SpatViT(
            img_size=args.patch_size,
            in_chans=args.channels,
            use_checkpoint=True,
            patch_size=args.seg_patches,
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
            n_points=8)

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

        num_dim = 64
        self.conv1 = DoubleConv_pad(args.NUM_TOKENS, 64, kernel_sz=1)
        self.conv2 = DoubleConv_pad(args.NUM_TOKENS, 64, kernel_sz=1)
        self.conv3 = DoubleConv_pad(args.NUM_TOKENS, 64, kernel_sz=3)
        self.conv4 = DoubleConv_pad(args.NUM_TOKENS, 64, kernel_sz=3) # downsample x0.25


        if self.patch_size == 5:
            in_planes = num_dim*4
            self.BCD_classifier = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes, in_planes // 2, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 2, 2, kernel_size=1))
        elif self.patch_size == 15:
            in_planes = num_dim * 4
            self.BCD_classifier = nn.Sequential(
                nn.Conv2d(in_planes, in_planes // 2, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 2, in_planes // 4, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 4, in_planes // 4, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 4, 2, kernel_size=1))

    def forward(self, x, y):
        x = self.forward_fusion(x)  # x: list : 5
        y = self.forward_fusion(y)

        f1 = torch.abs(x[4] - y[4])
        f1 = self.conv1(f1)

        f2 = torch.abs(x[3] - y[3])
        f2 = self.conv2(f2)

        f3 = torch.abs(x[2] - y[2])
        f3 = self.conv3(f3)

        f4 = torch.abs(x[1] - y[1])
        f4 = self.conv4(f4)

        f4 = torch.cat([f1, f2, f3, f4], dim=1)
        output = self.BCD_classifier(f4.squeeze())
        return output.squeeze()

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
        spec_feature = sum(spec_features)  # b, c, 128
        spec_feature = self.pool(spec_feature).view(b, -1)  # b, c

        spec_weights = []
        ops_ = [self.fc_spec1, self.fc_spec2, self.fc_spec3, self.fc_spec4]
        for i in range(len(ops_)):
            spec_weights.append((ops_[i](spec_feature)).view(b, -1, 1, 1))
        ss_feature = []
        ss_feature.append(img_features[0])
        for i in range(4):
            ss_feature.append((1 + spec_weights[i]) * img_fea[i])
        return ss_feature


# if __name__ == "__main__":
#     from ChangeDetection.func import get_args
#
#     seed, Dataset_name = 1, 'Hermiston'
#     patch_size = 5
#     in_chans = 154
#     args = get_args(Dataset_name, seed,
#                     use_checkpoint=True,
#                     mode='Spat_Spec_Pretraining')
#
#     model = HyperSIGMA_CD(args)
#     model.eval()
#     input = torch.Tensor(2, in_chans, patch_size, patch_size)
#     output = model(input,input)
#     for x in output:
#         print(x.shape)

