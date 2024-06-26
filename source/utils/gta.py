import math
import numpy as np
import torch
from einops import rearrange
from collections import OrderedDict
import torch.nn.functional as F


def make_2dcoord(H, W):
    """
    Return 2d coord values of shape [H, W, 2] 
    """
    x = np.arange(H, dtype=np.float32)/H   # [-0.5, 0.5)
    y = np.arange(W, dtype=np.float32)/W   # [-0.5, 0.5)
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    return np.stack([x_grid.flatten(), y_grid.flatten()], -1).reshape(H, W, 2)


def make_2dimgcoord(H, W):
    """
    Return 2d coord values of shape [H, W, 2] 
    """
    x = np.arange(W, dtype=np.float32)/W  # [-0.5, 0.5)
    y = np.arange(H, dtype=np.float32)/H  # [-0.5, 0.5)
    y = y[::-1]
    x = x[::-1]
    x_grid, y_grid = np.meshgrid(x, y, indexing='xy')
    return np.stack([x_grid.flatten(), y_grid.flatten()], -1).reshape(H, W, 2)


def homogenisation(V, trans_coeff=1.0):
    shape = V.shape
    _vec = np.ones([1, ]*len(shape))
    _vec = torch.Tensor(_vec).to(V.device)
    _vec = trans_coeff * _vec
    ret = torch.cat([V, _vec.repeat(*shape[:-1], 1)], -1)
    return ret


def scale_mask(trans_coeff, device):
    msk = trans_coeff * torch.ones(size=(3, 1)).to(device)
    msk = torch.cat([torch.ones(size=(3, 3)).to(device), msk], -1)
    msk = torch.cat([msk, torch.Tensor([[0, 0, 0, 1]]).to(device)], -2)
    return msk


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
        freqs = (2 ** torch.arange(1.0, nfreqs+1.0).to(coord.device)
                 ) / (2 ** float(nfreqs))
    grid_ths = [
        max_freqs[d] * 2 * math.pi * torch.einsum('...i,j->...ij', coord[..., d:d+1], freqs).flatten(-2, -1) for d in range(dim)]
    _mats = [[torch.cos(grid_ths[d]), -torch.sin(grid_ths[d]),
              torch.sin(grid_ths[d]), torch.cos(grid_ths[d])] for d in range(dim)]
    mats = [rearrange(torch.stack(_mats[d], -1),
                      '... (h w)->... h w', h=2, w=2) for d in range(dim)]
    mat = torch.stack(mats, -3)
    return mat


def make_T2mats(coord):
    """
    Args:
      coord: [..., 2]
    Return:
      mats of shape [..., 3, 3]
    """
    device = coord.device
    shape = coord.shape[:-1]
    eyes = torch.eye(2).to(device)  # [2, 2]
    zzo = torch.Tensor([[0], [0], [1]]).to(device)  # [3, 1]
    for _ in range(len(shape)):
        eyes = eyes.unsqueeze(0)
        zzo = zzo.unsqueeze(0)

    eyes = eyes.repeat(*shape, 1, 1)
    zzo = zzo.repeat(*shape, 1, 1)
    return torch.cat([torch.cat([eyes, coord[..., None, :]], -2), zzo], axis=-1)


def multihead_geometric_transform_attention(
        q, k, v, attn_fn, f_dims, reps, trans_coeff=1.0,
        v_transform=True, euclid=False, **kwargs):
    """
    Args:
        q: Tensor of shape [B, H, Nq*Tq, C] where Nq and Tq is the num of query views and image patch tokens 
        k: Tensor of shape [B, H, Nk*Tk, C] where Nk and Tk is the num of key-value views and image patch tokens 
        v: Tensor of shape [B, H, Nk*Tk, Cv] 
        attn_fn: Attention function over QKV-vectors. Typically it is Softmax(QK')V.
        f_dims: Dict which specifies dimensions of each geometric type
        reps: Dict contains pre-computed representation matrices ρ
        trans_coeff: Scaler that adjusts scale coeffs 
        v_transform: apply transforms to V or not 
        euclid: If True, use the squared euclid distance for QK-similarity instead of the dot-product similarity.

    Return:
        Tensor of shape [B, H, Nk*Tk, C]
    """
    B, H, C = q.shape[0], q.shape[1], q.shape[3]
    Tq, Tk = q.shape[2], k.shape[2]
    curr_dim = 0
    st_dims = {}
    ed_dims = {}
    for key in ['triv', 'se3', 'so3', 'so2', 't2']:
        if not key in f_dims:
            continue
        else:
            dim = f_dims[key]
            st_dims[key] = curr_dim
            ed_dims[key] = curr_dim + dim
            curr_dim += dim
    qs = OrderedDict()
    ks = OrderedDict()
    vs = OrderedDict()

    if 'triv' in f_dims and f_dims['triv'] > 0:
        (q_triv, k_triv, v_triv) = map(
            lambda x: x[..., st_dims['triv']:ed_dims['triv']], (q, k, v))
        qs['triv'] = q_triv
        ks['triv'] = k_triv
        vs['triv'] = v_triv

    if 'se3' in f_dims and f_dims['se3'] > 0:
        msk = scale_mask(trans_coeff, reps['se3rep_q'].device)
        c_q, c_k = reps['se3rep_q'], reps['se3rep_k']
        inv_c_q = reps['inv_se3rep_q']
        for i in range(len(c_q.shape[:-2])):
            msk.unsqueeze(0)
        c_q, c_k = c_q*msk, c_k*msk # [B, Nq, 4, 4]
        inv_c_q = inv_c_q*msk # [B, Nq, 4, 4]
        Nq, Nk = c_q.shape[1], c_k.shape[1]
        (q_se3, k_se3, v_se3) = map(
                lambda x: x[..., st_dims['se3']:ed_dims['se3']], (q, k, v))
        q_se3_shape, k_se3_shape, v_se3_shape = q_se3.shape, k_se3.shape, v_se3.shape
        if euclid:
            q_se3 = q_se3.reshape(B, H, Nq, Tq//Nq, -1, 3) # [B, H, Nq, Tq, C/3, 3]
            k_se3 = k_se3.reshape(B, H, Nk, Tk//Nk, -1, 3) # [B, H, Nk, Tk, C/3, 3]
            v_se3 = v_se3.reshape(B, H, Nk, Tk//Nk, -1, 3) # [B, H, Nk, Tk, C/3, 3]
            (q_se3, k_se3, v_se3) = map(
                lambda x: homogenisation(x), (q_se3, k_se3, v_se3))
            fn_se3 = reps['se3fn']
            qs['se3'] = fn_se3(c_q, q_se3)[..., :-1].reshape(q_se3_shape)
            ks['se3'] = fn_se3(c_k, k_se3)[..., :-1].reshape(k_se3_shape)
            vs['se3'] = (fn_se3(c_k, v_se3)[..., :-1]
                        if v_transform else v_se3[..., :-1]).reshape(v_se3_shape)
            
        else:
            q_se3_shape, k_se3_shape, v_se3_shape = q_se3.shape, k_se3.shape, v_se3.shape
            q_se3 = q_se3.reshape(B, H, Nq, Tq//Nq, -1, 4) # [B, H, Nq, Tq, C/4, 4]
            k_se3 = k_se3.reshape(B, H, Nk, Tk//Nk, -1, 4) # [B, H, Nk, Tk, C/4, 4]
            v_se3 = v_se3.reshape(B, H, Nk, Tk//Nk, -1, 4) # [B, H, Nk, Tk, C/4, 4]

            fn_se3 = reps['se3fn']
            qs['se3'] = fn_se3(inv_c_q.transpose(-2, -1), q_se3).reshape(q_se3_shape)
            ks['se3'] = fn_se3(c_k, k_se3).reshape(k_se3_shape)
            vs['se3'] = (fn_se3(c_k, v_se3)
                        if v_transform else v_se3).reshape(v_se3_shape)

    if 'so3' in f_dims and f_dims['so3'] > 0:
        D_q = reps['so3rep_q']
        D_k = reps['so3rep_k']
        fn_so3 = reps['so3fn']
        dims = [_D.shape[-1] for _D in D_q]

        total_dim = np.sum(dims)
        # Use deg1 and deg2
        (q_so3, k_so3, v_so3) = map(
            lambda x: x[..., st_dims['so3']:ed_dims['so3']], (q, k, v))
        q_so3_shape, k_so3_shape, v_so3_shape = q_so3.shape, k_so3.shape, v_so3.shape
        # [B, H, Nk, Tquery*C/8, 8]
        q_so3 = q_so3.reshape(B, H, Nq, -1, total_dim)
        # [B, H, Nk, Tkey*C/8, 8]
        k_so3 = k_so3.reshape(B, H, Nk, -1, total_dim)
        # [B, H, Nk, Tkey*C/8, 8]
        v_so3 = v_so3.reshape(B, H, Nk, -1, total_dim)

        inv_D_q = [D.transpose(-2, -1) for D in D_q]
        q_so3s, k_so3s, v_so3s = [], [], []
        for i in range(len(dims)):
            end_dim = np.sum(dims[:i+1])
            dim = dims[i]
            q_so3s.append(
                fn_so3(D_q[i].detach(), q_so3[..., end_dim-dim:end_dim]))
            k_so3s.append(
                fn_so3(D_k[i].detach(), k_so3[..., end_dim-dim:end_dim]))
            v_so3s.append(fn_so3(D_k[i].detach(), v_so3[..., end_dim-dim:end_dim])
                          if v_transform else v_so3[..., end_dim-dim:end_dim])
        qs['so3'] = torch.cat(q_so3s, -1).reshape(*q_so3_shape)
        ks['so3'] = torch.cat(k_so3s, -1).reshape(*k_so3_shape)
        vs['so3'] = torch.cat(v_so3s, -1).reshape(*v_so3_shape)

    if 'so2' in f_dims and f_dims['so2'] > 0:
        hw_q = reps['so2rep_q']
        hw_k = reps['so2rep_k']
        fn_so2 = reps['so2fn']

        (q_so2, k_so2, v_so2) = map(
            lambda x: x[..., st_dims['so2']:ed_dims['so2']], (q, k, v))
        q_so2_shape, k_so2_shape, v_so2_shape = q_so2.shape, k_so2.shape, v_so2.shape

        inv_hw_q = hw_q.transpose(-2, -1)
        q_so2 = q_so2.reshape(B, H, -1, hw_q.shape[2], 2)
        k_so2 = k_so2.reshape(B, H, -1, hw_k.shape[2], 2)
        v_so2 = v_so2.reshape(B, H, -1, hw_k.shape[2], 2)
        qs['so2'] = fn_so2(hw_q, q_so2).reshape(q_so2_shape)
        ks['so2'] = fn_so2(hw_k, k_so2).reshape(k_so2_shape)
        vs['so2'] = (fn_so2(hw_k, v_so2)
                     if v_transform else v_so2).reshape(v_so2_shape)

    if 't2' in f_dims and f_dims['t2'] > 0:
        rept2_q = reps['t2rep_q']  # [B, Nq, T, 3, 3]
        rept2_k = reps['t2rep_k']  # [B, Nq, T, 3, 3]
        fn_t2 = reps['t2fn']

        (q_t2, k_t2, v_t2) = map(
            lambda x: x[..., st_dims['t2']:ed_dims['t2']], (q, k, v))
        q_t2_shape, k_t2_shape, v_t2_shape = q_t2.shape, k_t2.shape, v_t2.shape

        inv_rept2_q = reps['inv_t2rep_q']
        q_t2 = q_t2.reshape(B, H, rept2_q.shape[1], -1, 3)
        k_t2 = k_t2.reshape(B, H, rept2_k.shape[1], -1, 3)
        v_t2 = v_t2.reshape(B, H, rept2_k.shape[1], -1, 3)
        qs['t2'] = fn_t2(inv_rept2_q.transpose(-2, -1),
                         q_t2).reshape(q_t2_shape)
        ks['t2'] = fn_t2(rept2_k, k_t2).reshape(k_t2_shape)
        vs['t2'] = (fn_t2(rept2_k, v_t2)
                    if v_transform else v_t2).reshape(v_t2_shape)

    qt = torch.cat([x for _, x in qs.items()], -1)
    kt = torch.cat([x for _, x in ks.items()], -1)
    vt = torch.cat([x for _, x in vs.items()], -1)

    out, attn = attn_fn(qt, kt, vt)

    if v_transform:
        outs = OrderedDict()
        if 'triv' in f_dims and f_dims['triv'] > 0:
            outs['triv'] = out[..., st_dims['triv']:ed_dims['triv']]
        if 'se3' in f_dims and f_dims['se3'] > 0:
            if euclid:
                out_se3 = homogenisation(out[..., st_dims['se3']:ed_dims['se3']].reshape(B, H, Nq, Tq//Nq, -1, 3))
                outs['se3'] = fn_se3(inv_c_q, out_se3)[..., :-1].reshape(q_se3_shape)
            else:
                outs['se3'] = fn_se3(
                    inv_c_q, out[..., st_dims['se3']:ed_dims['se3']].reshape(B, H, Nq, Tq//Nq, -1, 4)
                    ).reshape(q_se3_shape)

        if 'so3' in f_dims and f_dims['so3'] > 0:
            out_so3 = out[..., st_dims['so3']:ed_dims['so3']
                          ].reshape(B, H, Nq, -1, total_dim)
            out_so3s = []
            for i in range(len(dims)):
                dim = dims[i]
                end_dim = np.sum(dims[:i+1])
                out_so3s.append(
                    fn_so3(inv_D_q[i].detach(), out_so3[..., end_dim-dim:end_dim]))
            outs['so3'] = torch.cat(out_so3s, -1).reshape(q_so3_shape)
        if 'so2' in f_dims and f_dims['so2'] > 0:
            outs['so2'] = fn_so2(inv_hw_q, out[..., st_dims['so2']:ed_dims['so2']].reshape(
                B, H, -1, hw_q.shape[2], 2)).reshape(q_so2_shape)
        if 't2' in f_dims and f_dims['t2'] > 0:
            outs['t2'] = fn_t2(inv_rept2_q, out[..., st_dims['t2']:ed_dims['t2']].reshape(
                B, H, rept2_q.shape[1], -1, 3)).reshape(q_t2_shape)

        out_t = torch.cat([x for _, x in outs.items()], -1)
    else:
        out_t = out
    return out_t, attn


def multihead_vecrep_attention(q, k, v, attn_fn, extras, **kwargs):
    """
    Args:
        q: Tensor of shape [B, H, Nq*Tq, C]
        k: Tensor of shape [B, H, Nk*Tk, C]
        v: Tensor of shape [B, H, Nk*Tk, C]
    Return:
        softmax(Q(c_rel K))(c_rel V)
    """

    q = extras['vecrep_q'][:, None] * q
    k = extras['vecrep_k'][:, None] * k
    v = extras['vecrep_k'][:, None] * v

    out, attn = attn_fn(q, k, v)
    out = extras['vecinvrep_q'][:, None] * out
    return out, attn
