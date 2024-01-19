from timm.models.vision_transformer import _create_vision_transformer, checkpoint_filter_fn, _init_vit_weights
from timm.models.vision_transformer_hybrid import _resnetv2, HybridEmbed, default_cfgs
from timm.models.layers import PatchEmbed, trunc_normal_, Mlp, DropPath
from timm.models.helpers import build_model_with_cfg
from functools import partial
import torch.nn as nn
import torch
import numpy as np

from collections import OrderedDict


class VisionTransformerMultiView(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=784, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', GTA=True, so3=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            RTA: RTA or not
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        #self.pos_embed_second = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) no longer used, commented out
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.GTA = GTA 
        self.so3 = so3
        # Classifier head(s)
        if not self.GTA:
            self.pose_embed = nn.Linear(16, embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
            print(img_size, num_patches, self.num_tokens, patch_size)
            print(self.pos_embed.shape)
            exit
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if not self.GTA:
            trunc_normal_(self.pos_embed, std=.02)
        #trunc_normal_(self.pos_embed_second, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, **kwargs):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        import pdb
        pdb.set_trace()
        print(x.size())
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        import pdb
        pdb.set_trace()
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformerMultiView, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


def vit_base_r50_s16_384(pretrained=False, **kwargs):
    """ R50+ViT-B/16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    backbone = _resnetv2((3, 4, 9), **kwargs)
    model_kwargs = dict(embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer_hybrid(
        'vit_base_r50_s16_384', backbone=backbone, pretrained=pretrained, **model_kwargs)
    return model


def vit_base_resnet50_384(pretrained=False, **kwargs):
    # DEPRECATED this is forwarding to model def above for backwards compatibility
    return vit_base_r50_s16_384(pretrained=pretrained, **kwargs)


def _create_vision_transformer_hybrid(variant, backbone, pretrained=False, **kwargs):
    embed_layer = partial(HybridEmbed, backbone=backbone)
    kwargs.setdefault('patch_size', 1)  # default patch size for hybrid models if not set
    return _create_vision_transformer(
        variant, pretrained=pretrained, embed_layer=embed_layer, default_cfg=default_cfgs[variant], **kwargs)


def geometric_transform_attention(q, k, v, attn_drop, scale, **kwargs):

    B, H, C = q.shape[0], q.shape[1],  q.shape[3]
    NqTq, NkTk = q.shape[2], k.shape[2]
    
    enable_so3 = 'so3' in kwargs and kwargs['so3']
    if enable_so3:
        C_se3_st, C_se3_ed = 0, C//2
        C_so3_st, C_so3_ed = C//2, 3*C//4
        C_so2_st, C_so2_ed = 3*C//4, C
    else:
        C_se3_st, C_se3_ed = 0, C//2
        C_so2_st, C_so2_ed = C//2, C
        
    qs = OrderedDict()
    ks = OrderedDict()
    vs = OrderedDict()

    # SE(3)
    rep_se3_q, rep_se3_k = kwargs['se3rep_q'], kwargs['se3rep_k']
    inv_rep_se3_q = torch.linalg.inv(rep_se3_q)
    Nq = rep_se3_q.shape[1]
    Nk = rep_se3_k.shape[1]
    Tq, Tk = NqTq//Nq, NkTk//Nk
    (q_se3, k_se3, v_se3) =  map(lambda x: x[...,  C_se3_st:C_se3_ed], (q, k, v))
    q_se3_shape, k_se3_shape, v_se3_shape = q_se3.shape, k_se3.shape, v_se3.shape
    q_se3 = q_se3.reshape(B, H, Nq, Tq, -1, 4) # [B, H, Nq*Tq, 4, C/4]
    k_se3 = k_se3.reshape(B, H, Nk, Tk, -1, 4) # [B, H, Nq*Tk, 4, C/4]
    v_se3 = v_se3.reshape(B, H, Nk, Tk, -1, 4) # [B, H, Nq*Tk, 4, C/4]

    fn_se3 = lambda A, x: torch.einsum('bnij,bhntkj->bhntki', A, x)
    qs['se3'] = fn_se3(inv_rep_se3_q.transpose(-2, -1), q_se3).reshape(q_se3_shape)
    ks['se3'] = fn_se3(rep_se3_k, k_se3).reshape(k_se3_shape)
    vs['se3'] = fn_se3(rep_se3_k, v_se3).reshape(v_se3_shape)

    if enable_so3:
        D_q = kwargs['so3rep_q']
        D_k = kwargs['so3rep_k']
        dims = [_D.shape[-1] for _D in D_q]

        total_dim = np.sum(dims)
        # Use deg1 and deg2
        (q_so3, k_so3, v_so3) = map(lambda x: x[..., C_so3_st:C_so3_ed], (q, k, v))
        q_so3_shape, k_so3_shape, v_so3_shape = q_so3.shape, k_so3.shape, v_so3.shape
        q_so3 = q_so3.reshape(B, H, Nq, Tq, -1, total_dim)  # [B, H, Nk, Tquery, C/8, 8]
        k_so3 = k_so3.reshape(B, H, Nk, Tk, -1, total_dim)  # [B, H, Nk, Tkey, C/8, 8]
        v_so3 = v_so3.reshape(B, H, Nk, Tk, -1, total_dim)  # [B, H, Nk, Tkey, C/8, 8]

        inv_D_q = [D.transpose(-2,-1) for D in D_q]
        q_so3s,k_so3s,v_so3s = [], [], []
        fn_so3 = lambda A, x: torch.einsum('bnij,bhntkj->bhntki', A, x)
        for i in range(len(dims)):
            end_dim = np.sum(dims[:i+1])
            dim = dims[i]
            q_so3s.append(fn_so3(D_q[i].detach(), q_so3[..., end_dim-dim:end_dim]))
            k_so3s.append(fn_so3(D_k[i].detach(), k_so3[..., end_dim-dim:end_dim]))
            v_so3s.append(fn_so3(D_k[i].detach(), v_so3[..., end_dim-dim:end_dim]))
        qs['so3'] = torch.cat(q_so3s, -1).reshape(*q_so3_shape)
        ks['so3'] = torch.cat(k_so3s, -1).reshape(*k_so3_shape)
        vs['so3'] = torch.cat(v_so3s, -1).reshape(*v_so3_shape)

    # SO(2) x SO(2)
    rep_so2_q = kwargs['so2rep_q']
    rep_so2_k = kwargs['so2rep_k']
    (q_so2, k_so2, v_so2) =  map(lambda x: x[..., C_so2_st:C_so2_ed], (q, k, v))
    q_so2_shape, k_so2_shape, v_so2_shape = q_so2.shape, k_so2.shape, v_so2.shape
    inv_rep_so2_q = rep_so2_q.transpose(-2, -1)
    q_so2 = q_so2.reshape(B, H, NqTq, -1, 2)
    k_so2 = k_so2.reshape(B, H, NkTk, -1, 2)
    v_so2 = v_so2.reshape(B, H, NkTk, -1, 2)
    def fn_so2(A, x): # Einsum is too slow
        x = torch.sum(A[:,None] * x[..., None, :], -1)
        return x
    qs['so2'] = fn_so2(rep_so2_q, q_so2).reshape(q_so2_shape)
    ks['so2'] = fn_so2(rep_so2_k, k_so2).reshape(k_so2_shape)
    vs['so2'] = fn_so2(rep_so2_k, v_so2).reshape(v_so2_shape)

    qt = torch.cat([x for _,x in qs.items()], -1)
    kt = torch.cat([x for _,x in ks.items()], -1)
    vt = torch.cat([x for _,x in vs.items()], -1)

    attn = (qt @ kt.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    attn = attn_drop(attn)

    out = (attn @ vt)
    outs = OrderedDict()

    outs['se3'] = fn_se3(
       inv_rep_se3_q, 
        out[...,  C_se3_st:C_se3_ed].reshape(B, H, Nq, Tq, -1, 4)
        ).reshape(q_se3_shape)
    if enable_so3:
        out_so3 = out[..., C_so3_st:C_so3_ed].reshape(B, H, Nq, Tq, -1, total_dim)
        out_so3s = []
        for i in range(len(dims)):
            dim = dims[i]
            end_dim = np.sum(dims[:i+1])
            out_so3s.append(fn_so3(inv_D_q[i].detach(), out_so3[..., end_dim-dim:end_dim]))
        outs['so3'] = torch.cat(out_so3s, -1).reshape(q_so3_shape)
    outs['so2'] = fn_so2(inv_rep_so2_q, out[..., C_so2_st:C_so2_ed].reshape(B, H, NqTq, -1, 2)).reshape(q_so2_shape)
        
    out_t = torch.cat([x for _,x in outs.items()], -1)

    return out_t


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [3, B, H, T, C]
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) # [B, H, T, C]
        
        if kwargs['gta']:
            x = geometric_transform_attention(q,k,v, self.attn_drop, self.scale, **kwargs)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, **kwargs):
        x = x + self.drop_path(self.attn(self.norm1(x), **kwargs))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
