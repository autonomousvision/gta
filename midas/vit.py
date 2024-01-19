import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F
from vit_models import vit_base_resnet50_384
from einops import rearrange
from wigner_d import rotmat_to_wigner_d_matrices

class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def forward_vit(pretrained, x, rel_transform, nviews):
    b, c, h, w = x.shape

    glob = pretrained.model.forward_flex(x, rel_transform, nviews)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]# [:, :257]
    layer_4 = pretrained.activations["4"]# [:, :257]

    s = layer_3.size()
    layer_3 = layer_3.view(s[0] * nviews, s[1] // nviews, s[2])

    s = layer_4.size()
    layer_4 = layer_4.view(s[0] * nviews, s[1] // nviews, s[2])

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)
    return layer_1, layer_2, layer_3, layer_4


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def make_2dcoord(H, W):
    """
    Return 2d coord values of shape [H*W+1, 2]. The last coord is the cls token's coord.
    """
    x = torch.arange(H, dtype=torch.float32)/H   # [-0., 1.)
    y = torch.arange(W, dtype=torch.float32)/W   # [-0., 1.)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([x_grid.flatten(), y_grid.flatten()], -1).reshape(H*W, 2)
    grid = torch.cat([grid, 0.5 * torch.ones(size=(1,2), dtype=torch.float32)], axis=0)
    return grid


def make_SO2mats(coord, nfreqs, max_freqs=[1, 1], shared_freqs=False):
    """
    Args:
      coord: [..., 2 or 3]
      freqs: [n_freqs, 2 or 3]
      max_freqs: [2 or 3]
    Return:
      mats of shape [..., (2 or 3)*n_freqs, 2, 2]
    """
    dim = coord.shape[-1]
    if shared_freqs:
        freqs = torch.ones(size=(nfreqs,)).to(coord.device)
    else:
        freqs = (2 ** torch.arange(1.0, nfreqs+1.0).to(coord.device)) / (2 ** float(nfreqs))
    grid_ths = [
        max_freqs[d] * 2 * math.pi * torch.einsum('...i,j->...ij', coord[..., d:d+1], freqs).flatten(-2, -1) for d in range(dim)]
    _mats = [[torch.cos(grid_ths[d]), -torch.sin(grid_ths[d]),
              torch.sin(grid_ths[d]), torch.cos(grid_ths[d])] for d in range(dim)]
    mats = [rearrange(torch.stack(_mats[d], -1), '... (h w)->... h w', h=2, w=2) for d in range(dim)]
    mat = torch.stack(mats, -3)
    return mat

def eye_like(tensor):
    """
    Args:
        tensor: [B, d, d]

    Returns:
        Batch of identity matrices of shape [B, d, d]
    """
    # Determine the batch size and the size of each square matrix
    batch_size, matrix_size, _ = tensor.shape
    
    # Create a batch of identity matrices with the same shape
    eye_tensor = torch.eye(matrix_size, device=tensor.device, dtype=tensor.dtype).unsqueeze(0)
    batch_eye_tensor = eye_tensor.repeat(batch_size, 1, 1)
    
    return batch_eye_tensor

# Memo: This is the main encoder function
def forward_flex(self, x, pose, nviews):
    b, c, h, w = x.shape
    # Seems this is no longer used and commented out
    #pos_embed_second = self._resize_pos_embed(
    #    self.pos_embed_second, h // self.patch_size[1], w // self.patch_size[0]
    #)

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    # s = x.size()
    # x_unpack = x.view(s[0] // nviews, nviews, *s[1:])

    # s = pose_embed.size()
    # pose_embed_unpack = pose_embed.view(s[0] // nviews, nviews, *s[1:])
    # x_first = x_unpack[:, 0]
    # x_second = x_unpack[:, 1]

    # x_first_concat = torch.cat([x_first, x_second], dim=1)
    # x_second_concat = torch.cat([x_second, x_first], dim=1)

    # x_concat = torch.stack([x_first_concat, x_second_concat], dim=1)
    # x = torch.flatten(x_concat, 0, 1)

    # x_first_image, x_second_image = torch.chunk(x, 2, 1)
    # pose_embed_ex = pose_embed[:, None, :].expand(-1, x_first_image.size(1), -1)

    # x_second_image = x_second_image + pose_embed_ex

    #x = torch.cat([x_first_image, x_second_image], dim=1)

    # x = torch.cat([x, x], dim=1)
    # pos_embed = torch.cat([pos_embed, pos_embed_second], dim=1)

    if not self.GTA:
        pos_embed = self._resize_pos_embed(
            self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
        )
        # Pose embedding is used here
        pose_embed = self.pose_embed(pose) 
        x = x + pos_embed + pose_embed[:, None, :]
    x = self.pos_drop(x)

    os = h*w//256 + 1

    bn, t, c = x.shape[0], x.shape[1], x.shape[2:]
    x = x.view(bn // nviews, nviews * t, *c)
    if self.GTA:
        # Make SO2 representations
        nfreqs = 8 if not self.so3 else 4 
        if hasattr(self, 'coord_cache') and self.coord_cache.shape[0] == bn//nviews:
            coord = self.coord_cache
        else:
            coord = make_2dcoord(h//16, w//16)
            coord = make_SO2mats(coord, nfreqs)
            coord = torch.stack([coord]*bn, 0).reshape(bn // nviews, nviews*t, nfreqs*2, 2, 2) # [B, N*T, nfreqs*2, 2, 2]
            coord = coord.to(x.device).detach()
            self.coord_cache = coord
        pose = pose.reshape(bn//nviews, nviews, 4, 4) # the first view's pose is set to the origin

        kwargs = {'se3rep_q':pose, 'se3rep_k':pose, 'so2rep_q': coord, 'so2rep_k': coord, 'gta':True}
        if self.so3:
            R_q = pose[..., :3, :3]
            B, Nq = R_q.shape[0], R_q.shape[1]
            D_q = rotmat_to_wigner_d_matrices(2, R_q.flatten(0,1))[1:]
            for i, D in enumerate(D_q):
                D_q[i] = D.reshape(B, Nq, D.shape[-2], D.shape[-1])
            kwargs['so3rep_q'] = kwargs['so3rep_k'] = D_q
            kwargs['so3'] = True
    else:
        kwargs = {'gta': False}
    for i, blk in enumerate(self.blocks):
        x = blk(x, **kwargs)

        # if i == 4:
        #     x = x[:, :os, :]

        # print(x.size())

    x = self.norm(x)
    s = x.size()

    x = torch.flatten(x.view(s[0], nviews, os, *s[2:]), 0, 1)

    return x


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper


def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    # 32, 48, 136, 384
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_vitl16_384(pretrained, use_readout="ignore", hooks=None):
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
    )


def _make_pretrained_vitb16_384(pretrained, use_readout="ignore", hooks=None):
    model = timm.create_model("vit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model, features=[96, 192, 384, 768], hooks=hooks, use_readout=use_readout
    )


def _make_pretrained_deitb16_384(pretrained, use_readout="ignore", hooks=None):
    model = timm.create_model("vit_deit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model, features=[96, 192, 384, 768], hooks=hooks, use_readout=use_readout
    )


def _make_pretrained_deitb16_distil_384(pretrained, use_readout="ignore", hooks=None):
    model = timm.create_model(
        "vit_deit_base_distilled_patch16_384", pretrained=pretrained
    )

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        start_index=2,
    )


def _make_vit_b_rn50_backbone(
    model,
    features=[256, 512, 768, 768],
    size=[384, 384],
    hooks=[0, 1, 8, 11],
    vit_features=768,
    use_vit_only=False,
    use_readout="ignore",
    start_index=1,
):
    pretrained = nn.Module()

    pretrained.model = model
    print("""
          length of model blocks:
          """, len(model.blocks))
    if use_vit_only == True:
        pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
        pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    else:
        pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(
            get_activation("1")
        )
        pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
            get_activation("2")
        )

    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    if use_vit_only == True:
        pretrained.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        pretrained.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
    else:
        pretrained.act_postprocess1 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )
        pretrained.act_postprocess2 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_vitb_rn50_384(
    pretrained, use_readout="ignore", hooks=None, use_vit_only=False, **model_kwargs):
    # model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)
    model = vit_base_resnet50_384(pretrained=False, **model_kwargs)

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_vit_b_rn50_backbone(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
    )
