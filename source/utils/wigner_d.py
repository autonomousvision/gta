# The wigner D mats implementation (originally from https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/irr_repr.py, https://github.com/AMLab-Amsterdam/lie_learn/blob/master/lie_learn/representations/SO3/wigner_d.py
import torch
import torch.nn as nn
import numpy as np
import math


path = 'J_dense.pt'
Jd = torch.load(str(path))


def to_order(degree):
    return 2 * degree + 1


def z_rot_mat(angle, l):
    N = angle.shape[0]
    order = to_order(l)
    m = torch.zeros((N, order, order)).type(angle.dtype).to(angle.device)
    inds = torch.arange(0, order, 1)
    reversed_inds = torch.arange(2 * l, -1, -1)
    frequencies = torch.arange(l, -l - 1, -1)[None].type(angle.dtype).to(angle.device)
    m[:, inds, reversed_inds] = torch.sin(frequencies * angle[:, None])
    m[:, inds, inds] = torch.cos(frequencies * angle[:, None])
    return m


def wigner_d_matrix(degree, g1, g2, g3):
    """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
    J = Jd[degree].type(g1.dtype).to(g1.device)
    J = J[None].repeat(g1.shape[0], 1, 1)
    x_1 = z_rot_mat(g1, degree)
    x_2 = z_rot_mat(g2, degree)
    x_3 = z_rot_mat(g3, degree)
    return x_3 @ J @ x_2 @ J @ x_1

EPS = 1e-5

def rotmat2ZYZeuler(R):
    # The inverse of ZYZ euler to rotmat.
    g1 = torch.atan2(R[..., 2, 1], -R[..., 2, 0])
    g2 = torch.atan2(torch.sqrt(R[..., 0, 2]**2+R[..., 1, 2]**2), R[..., 2, 2])
    g3 = torch.atan2(R[..., 1, 2], R[..., 0, 2])
    mask1 = (torch.abs(R[..., 2, 2] - 1) < EPS).type(torch.float32)
    mask2 = (torch.abs(R[..., 2, 2] + 1) < EPS).type(torch.float32)
    g1 = (1-mask1)*(1-mask2)*g1 + mask1*torch.atan2(
        R[..., 1, 0], R[..., 0, 0]) + mask2*torch.atan2(-R[..., 1, 0], -R[..., 0, 0])
    g3 = (1-mask1)*(1-mask2)*g3
    return g1, g2, g3


def rotmat_to_wigner_d_matrices(max_degree, R):
    device = R.device
    g1, g2, g3 = rotmat2ZYZeuler(R)
    mats = []
    for d in range(0, max_degree+1, 1):
        mats.append(wigner_d_matrix(d, g1, g2, g3).to(device))
    return mats


def ZYZeuler2rotmat(g1, g2, g3, device, dtype):
    N = g1.shape[0]
    one = torch.ones(N, dtype=dtype).to(device)
    zero = torch.zeros(N, dtype=dtype).to(device)
    Rz1 = torch.Tensor([
        [torch.cos(g1), -torch.sin(g1), zero],
        [torch.sin(g1), torch.cos(g1), zero],
        [zero, zero, one]
    ]).transpose(2, 0, 1)
    Ry = torch.Tensor([
        [torch.cos(g2), zero, torch.sin(g2)],
        [zero, one, zero],
        [-torch.sin(g2), zero, torch.cos(g2)]
    ]).transpose(2, 0, 1)
    Rz2 = torch.array([
        [torch.cos(g3), -torch.sin(g3), zero],
        [torch.sin(g3), torch.cos(g3), zero],
        [zero, zero, one]
    ]).transpose(2, 0, 1)

    return Rz2 @ Ry @ Rz1
