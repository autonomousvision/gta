#https://github.com/mayankgrwl97/gbt/tree/main

import torch
import torch.nn as nn
import numpy as np

def positional_encoding(ray, n_freqs=15, parameterize='plucker', start_freq=-6):
    """
    Positional Embeddings. For more details see Section 5.1 of
    NeRFs: https://arxiv.org/pdf/2003.08934.pdf

    Args:
        ray: (B,num_input_views,P,6)
        n_freqs: num of frequency bands
        parameterize(str|None): Parameterization used for rays. Recommended: use 'plucker'. Default=None.

    Returns:
        pos_embeddings: Mapping input ray from R to R^{2*n_freqs}.
    """

    if parameterize is None:
        pass
    elif parameterize == 'plucker':
        # direction unit-normalized, (o+nd, d) has same representation as (o+md, d) [4 DOF]
        # ray_origins = ray[..., :3]
        # ray_directions = ray[..., 3:]
        # ray_directions = ray_directions / ray_directions.norm(dim=-1).unsqueeze(-1)  # Normalize ray directions to unit vectors
        # plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
        # plucker_parameterization = torch.cat([ray_directions, plucker_normal], dim=-1)
        ray = get_plucker_parameterization(ray)
    else:
        raise NotImplementedError(f'parameterize={parameterize} not implemented.')

    freq_bands = 2. ** torch.arange(start_freq, start_freq+n_freqs) * np.pi
    sin_encodings = [torch.sin(ray * freq) for freq in freq_bands]
    cos_encodings = [torch.cos(ray * freq) for freq in freq_bands]

    pos_embeddings = torch.cat(sin_encodings + cos_encodings, dim=-1)  # B, num_input_views, P, 6 * 2n_freqs
    return pos_embeddings


def get_plucker_parameterization(ray):
    """Returns the plucker representation of the rays given the (origin, direction) representation

    Args:
        ray(torch.Tensor): Tensor of shape (..., 6) with the (origin, direction) representation

    Returns:
        torch.Tensor: Tensor of shape (..., 6) with the plucker (D, OxD) representation
    """
    ray = ray.clone()  # Create a clone
    ray_origins = ray[..., :3]
    ray_directions = ray[..., 3:]
    ray_directions = ray_directions / ray_directions.norm(dim=-1).unsqueeze(-1)  # Normalize ray directions to unit vectors
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    plucker_parameterization = torch.cat([ray_directions, plucker_normal], dim=-1)

    return plucker_parameterization


def plucker_dist(ray1, ray2, eps=1e-6):
    # Plucker ray is represented as (l, m),
    # l is direction unit norm, m = (oxl)

    # ray1 (l1, m1): (B, Q, 6)
    # ray2 (l2, m2): (B, P, 6)

    Q = ray1.shape[1]
    P = ray2.shape[1]

    ray1 = ray1.unsqueeze(2).repeat(1, 1, P, 1)  # (B, Q, P, 6)
    ray2 = ray2.unsqueeze(1).repeat(1, Q, 1, 1)  # (B, Q, P, 6)

    # (l1, m1) * (l2, m2) = l1.m2 + l2.m1
    reci_prod = ((ray1[..., :3] * ray2[..., 3:]).sum(-1) + \
                (ray1[..., 3:] * ray2[..., :3]).sum(-1)).abs()  # (B, Q, P)

    # || l1 x l2 ||
    l1_cross_l2 = torch.cross(ray1[..., :3], ray2[..., :3], dim=-1)  # (B, Q, P, 3)
    l1_cross_l2_norm = l1_cross_l2.norm(dim=-1) # (B, Q, P)

    # || l1 x (m1-m2)/s ||
    # s = ray2[..., :3] / ray1[..., :3]  # (B, Q, P, 3)
    # s = s.mean(dim=-1).unsqueeze(-1)  # (B, Q, P, 1)
    s = 1
    l1_cross_m1_minus_m2 = torch.cross(ray1[..., :3], (ray1[..., 3:] - ray2[..., 3:])/s)
    l1_cross_m1_minus_m2_norm = l1_cross_m1_minus_m2.norm(dim=-1) # (B, Q, P)

    # ||l1||^2
    l1_norm_sq = torch.norm(ray1[..., :3], dim=-1) ** 2 # (B, Q, P)

    distance = l1_cross_m1_minus_m2_norm / (l1_norm_sq + eps) # (B, Q, P)
    mask = (l1_cross_l2_norm > eps)
    distance[mask] = reci_prod[mask] / (l1_cross_l2_norm[mask] + eps)

    return distance


