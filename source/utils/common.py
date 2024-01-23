import math
import os
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from source.utils.wigner_d import rotmat_to_wigner_d_matrices
from collections import OrderedDict
import torch.nn.functional as F

__LOG10 = math.log(10)


def mse2psnr(x):
    return -10.*torch.log(x)/__LOG10


def init_ddp():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        return 0, 1  # Single GPU run

    dist.init_process_group(backend="nccl")
    print(f'Initialized process {local_rank} / {world_size}')
    torch.cuda.set_device(local_rank)

    setup_dist_print(local_rank == 0)
    return local_rank, world_size


def setup_dist_print(is_main):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)


def using_dist():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not using_dist():
        return 1
    return dist.get_world_size()


def get_rank():
    if not using_dist():
        return 0
    return dist.get_rank()


def gather_all(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]

    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    return tensor_list


def reduce_dict(input_dict, average=True):
    """
    Reduces the values in input_dict across processes, when distributed computation is used.
    In all processes, the dicts should have the same keys mapping to tensors of identical shape.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    keys = sorted(input_dict.keys())
    values = [input_dict[k] for k in keys]

    if average:
        op = dist.ReduceOp.AVG
    else:
        op = dist.ReduceOp.SUM

    for value in values:
        dist.all_reduce(value, op=op)

    reduced_dict = {k: v for k, v in zip(keys, values)}

    return reduced_dict


def downsample(x, num_steps=1):
    # x: [..., H, W, C]
    if num_steps is None or num_steps < 1:
        return x
    stride = 2**num_steps
    return x[..., stride//2::stride, stride//2::stride, :]



# https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :,
        :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe


def positionalencoding2d_given_coord(d_model, coord, scale=[1, 1]):
    """
    Args:
        d_model: dimension of the model
        coord: Tensor of shape [..., 2] assumed to be given in [0, 1.0]
        scale: [2]
    :return: d_model* len(coord)
    """
    shape = [1]*len(coord.shape[:-1])
    scale = torch.Tensor(scale).to(device=coord.device)
    coord = coord * scale.reshape(shape + [-1])
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model)).to(coord.device)  # [D/4]

    h = torch.einsum('i,...j->...i', div_term, coord[..., 0:1])  # [..., D/4]
    w = torch.einsum('i,...j->...i', div_term, coord[..., 1:2])  # [..., D/4]
    pe_w = torch.stack([torch.sin(w), torch.cos(w)], -
                       1).reshape(list(coord.shape[:-1]) + [-1])
    pe_h = torch.stack([torch.sin(h), torch.cos(h)], -
                       1).reshape(list(coord.shape[:-1]) + [-1])
    return torch.cat([pe_w, pe_h], -1)


def apply_batch_matmul(M, z):
    if len(M.shape) == 4:
        if len(z.shape) == 4:
            z = torch.einsum('nmij, nmkj->nmki', M, z)
        else:
            z = torch.einsum('nmij, nkj->nmki', M, z)
    else:
        z = torch.einsum('nij, nkj->nki', M, z)
    return z


def rigid_transform(M, z, trans_coeff):
    """
    Args:
        M: element of SE(3) of shape [..., 4,4] or [..., 3, 4]
        z: Tensor of shape [..., 3] or [..., K, 3]
        trans_coeff: scaler 
    Return:
        transformed Tensor of shape [..., 3]
    """
    z = torch.concat([z, trans_coeff * torch.ones(size=(*
                     z.shape[:-1], 1), dtype=z.dtype, device=z.device)], axis=-1)
    z = apply_batch_matmul(M, z)
    if z.shape[-1] == 4:
        z = z[..., :3]
    return z