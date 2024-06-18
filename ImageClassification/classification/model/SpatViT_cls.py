# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import warnings
import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from mmengine.dist import get_dist_info
from torch.nn.init import constant_, xavier_uniform_

# from mmcv_custom import load_checkpoint
# from mmdet.utils import get_root_logger
# from ..builder import BACKBONES


def get_reference_points(spatial_shapes, device):
    #reference_points_list = []
    #for lvl, (H_, W_) in enumerate(spatial_shapes):
    H_, W_ =  spatial_shapes[0], spatial_shapes[1]
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_
    ref = torch.stack((ref_x, ref_y), -1) # 每个点归一化后的坐标，尺寸(1，H_*W_，2)
    #reference_points_list.append(ref)
    #reference_points = torch.cat(reference_points_list, 1) # 连接各特征图的坐标
    #reference_points = reference_points[:, :, None] #(1，H_*W_，1，2)
    return ref


def deform_inputs_func(x,patch_size):
    B, c, h, w = x.shape
    b = B // 3

    spatial_shapes = torch.as_tensor([h // patch_size, w // patch_size],
                                    dtype=torch.long, device=x.device) # 3*2的tensor
    # level_start_index = torch.cat((spatial_shapes.new_zeros(
    #     (1,)), spatial_shapes.prod(1).cumsum(0)[:-1])) # 第一个数为0，后边的是每行累乘后的依次累加
    reference_points = get_reference_points([h // patch_size, w // patch_size], x.device)
    deform_inputs = [reference_points, spatial_shapes]
    # inputs1三个shape, 一种ref points, ref points对应query
    # 即，injector query单尺度，value多尺度
    
    return deform_inputs


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class SampleAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, n_points=4):
        super().__init__()
        self.n_points = n_points
        print(n_points)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        #self.window_size = window_size
        # q_size = window_size[0]
        # kv_size = q_size
        # rel_sp_dim = 2 * q_size - 1
        # self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim)) # 2ws-1,C'
        # self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim)) # 2ws-1,C'

        self.sampling_offsets = nn.Linear(all_head_dim, self.num_heads  * n_points * 2) # 通过n point设置每个level采样几个点

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, deform_inputs):

        B, N, C = x.shape
        # qkv_bias = None
        # if self.q_bias is not None:
        #     qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, -1).permute(2, 0, 1, 3) # 3，B，N，c
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # B，N，L

        reference_points, input_spatial_shapes = deform_inputs

        sampling_offsets = self.sampling_offsets(q).reshape(
            B,  N, self.num_heads, self.n_points, 2).transpose(1, 2) # B, N, H*P*2 -> B, H, N, P, 2
        
        _, _, L = q.shape
        
        q = q.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2) # B, N, H, c -> B, H, N, c

        offset_normalizer =  torch.stack([input_spatial_shapes[1], input_spatial_shapes[0]]) # 倒转坐标顺序为xy

        sampling_locations = reference_points[:, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, None, :] # 每个level对xy分别归一化

        sampling_locations = 2 * sampling_locations - 1 # [0, 1] -> [-1, 1]

        k = k.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2) # B, N, H, c -> B, H, N, c
        v = v.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2)

        # B*H, c, H, W
        k = k.flatten(0,1).transpose(1,2).reshape(B*self.num_heads, L // self.num_heads, input_spatial_shapes[0], input_spatial_shapes[1])
        v = v.flatten(0,1).transpose(1,2).reshape(B*self.num_heads, L // self.num_heads, input_spatial_shapes[0], input_spatial_shapes[1])

        # B*H, N, P, 2
        sampling_locations = sampling_locations.flatten(0,1).reshape(B*self.num_heads, N, self.n_points, 2)

        q = q[:,:,:,None,:] # B, H, N, 1, C

        # B*H, c, N, P
        sampled_k = F.grid_sample(k, sampling_locations, mode='bilinear',
                                          padding_mode='zeros', align_corners=False).reshape(B, self.num_heads, L // self.num_heads, N, self.n_points).permute(0,1,3,4,2) # 根据位置从每个level的特征图中采样出点作为value
        
        # B*H, c, N, P
        sampled_v = F.grid_sample(v, sampling_locations, mode='bilinear',
                                          padding_mode='zeros', align_corners=False).reshape(B, self.num_heads, L // self.num_heads, N, self.n_points).permute(0,1,3,4,2) # 根据位置从每个level的特征图中采样出点作为value
        
        attn = (q * sampled_k).sum(-1) * self.scale # B, H, N, P, c -> B, H, N, P 

        attn = attn.softmax(dim=-1)[:, :, :, :, None] # B, H, N, P, 1

        x = (attn * sampled_v).sum(-2).transpose(1, 2).reshape(B, N, -1) # B, H, N, c 

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        # self.window_size = window_size
        # q_size = window_size[0]
        # kv_size = q_size
        # rel_sp_dim = 2 * q_size - 1
        # self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim)) # 2ws-1,C'
        # self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim)) # 2ws-1,C'

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, rel_pos_bias=None):
        B, N, C = x.shape
        # qkv_bias = None
        # if self.q_bias is not None:
        #     qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # 3，B，H，N，C
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # B，H，N，C

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # B,H,N,N

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, sample=False, restart_regression=True, n_points=None):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.sample = sample

        if not sample:
            self.attn = Attention(
             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
             attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
             
        else:
            self.attn = SampleAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim, n_points=n_points)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

#    def forward(self, x, H, W, deform_inputs):
#    def forward(self, x, H, W, deform_inputs):
#        if self.gamma_1 is None:
#            x = x + self.drop_path(self.attn(self.norm1(x), H, W, deform_inputs))
#            x = x + self.drop_path(self.mlp(self.norm2(x)))
#        else:
#            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W, deform_inputs))
#            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
#        return x
        
    def forward(self, x, H, W, deform_inputs):
        if self.gamma_1 is None:
            if not self.sample:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W, deform_inputs))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if not self.sample:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W, deform_inputs))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

#@BACKBONES.register_module()
class SpatViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=1, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, 
                 use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 out_indices=[11], interval=3, pretrained=None, restart_regression=True, n_points=4):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_chans = in_chans
        self.out_channels = (3, embed_dim, embed_dim, embed_dim, embed_dim)
        self.patch_size = patch_size
        self.DR = nn.Conv1d(embed_dim*4, 128, kernel_size=1, bias=False)
        self.cls = nn.Conv2d(128,num_classes,kernel_size=1,stride=1,padding = 0,bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=embed_dim*4,#196608
                out_features=128
            ),
            nn.Linear(
                in_features=128,
                out_features=64
            ),
            nn.Linear(
                in_features=64,
                out_features=num_classes
            ))
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.out_indices = out_indices

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint

        # MHSA after interval layers
        # WMHSA in other layers
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, sample=((i + 1) % interval != 0), 
                restart_regression=restart_regression, n_points=n_points)
            for i in range(depth)])
         
        self.interval = interval

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self.norm = norm_layer(embed_dim)
        if patch_size == 11:
            self.fpn1 = nn.Identity()

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.Identity()
        if patch_size == 1:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
        if patch_size == 2:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
        if patch_size == 3:
            self.fpn1 = nn.Identity()

            self.fpn2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=1, stride=1),
            )

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
        elif patch_size == 8:
            self.fpn1 = nn.Identity()

            self.fpn2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=1, stride=1),
            )

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )

        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.pretrained = pretrained


    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pretrained = pretrained or self.pretrained
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)

            checkpoint = torch.load(pretrained, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # for MoBY, load model of online branch
            if sorted(list(state_dict.keys()))[0].startswith('encoder'):
                state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

            # remove patch embed when inchan != 3

            if self.in_chans != 3:
                for k in list(state_dict.keys()):
                    if 'patch_embed.proj' in k:
                        del state_dict[k]

            # print('$$$$$$$$$$$$$$$$$')
            # print(state_dict.keys())

            # print('#################')
            # print(self.state_dict().keys())

            rank, _ = get_dist_info()
            if 'pos_embed' in state_dict:
                pos_embed_checkpoint = state_dict['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                H, W = self.patch_embed.patch_shape
                num_patches = self.patch_embed.num_patches
                num_extra_tokens = 0
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    if rank == 0:
                        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, H, W))
                    # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
                    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    state_dict['pos_embed'] = new_pos_embed
                else:
                    state_dict['pos_embed'] = pos_embed_checkpoint[:, num_extra_tokens:]

            msg = self.load_state_dict(state_dict, False)
            print(msg)

        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x,patch_size):

        img = [x]

        deform_inputs = deform_inputs_func(x,patch_size)

        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x) 
        batch_size, seq_len, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, Hp, Wp, deform_inputs)
            else:
                x = blk(x, Hp, Wp, deform_inputs)
                
            if i in self.out_indices:
                features.append(x)
        features = list(map(lambda x: x.permute(0, 2, 1).reshape(B, -1, Hp, Wp), features))

        # ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        # for i in range(len(ops)):
        #     features[i] = ops[i](features[i])

        return img + features

    def forward(self, x):
        (a,c, h, w) = x.shape
        upsampling = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        feture = self.forward_features(x,self.patch_size)
        feture1 = feture[4]
        feture2 = feture[3]
        feture3 = feture[2]
        feture4 = feture[1]
        y1 = F.avg_pool2d(feture1, feture1.size()[2:])
        y2 = F.avg_pool2d(feture2, feture2.size()[2:])
        y3 = F.avg_pool2d(feture3, feture3.size()[2:])
        y4 = F.avg_pool2d(feture4, feture4.size()[2:])
        y1 = y1.view(feture1.size(0), -1)
        y2 = y2.view(feture2.size(0), -1)
        y3 = y3.view(feture3.size(0), -1)
        y4 = y4.view(feture4.size(0), -1)
        
        
        
        output = torch.concat((y1,y2,y3,y4),1)

        output = self.classifier(output)
        return output
    
def spat_vit_b_rvsa(args, inchannels=3):

    backbone = SpatViT(
            img_size=args.image_size,
            in_chans=inchannels,
            patch_size=16,
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
            use_checkpoint=(args.use_ckpt=='True'),
            use_abs_pos_emb=False,
            interval = 3
    )

    return backbone

def spat_vit_l_rvsa(args, inchannels=3):

    backbone = SpatViT(
            img_size=args.image_size,
            in_chans=inchannels,
            patch_size=16,
            drop_path_rate=0.1,
            out_indices=[7, 11, 15, 23],
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=(args.use_ckpt=='True'),
            use_abs_pos_emb=False,
            interval = 6
    )

    return backbone

def spat_vit_h_rvsa(args, inchannels=3):

    backbone = SpatViT(
            img_size=args.image_size,
            in_chans=inchannels,
            patch_size=16,
            drop_path_rate=0.1,
            out_indices=[10, 15, 20, 31],
            embed_dim=1280,
            depth=32,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=(args.use_ckpt=='True'),
            use_abs_pos_emb=False,
            interval = 8
    )

    return backbone
    
    
if __name__ == "__main__":

    backbone = SpatViT(
        img_size=224,
        in_chans=3,
        patch_size=16,
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
        interval = 3
    )

    backbone.cuda()

    backbone.eval()

    input = torch.Tensor(2, 3, 224, 224).cuda()

    out = backbone(input)


    for x in out:

        print(x.shape)

    
    