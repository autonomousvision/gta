import numpy as np
import torch

from torch.nn import functional as F
from utils import util
import random


def get_ray_origin(cam2world):
    return cam2world[..., :3, 3]


def get_coord_transform(r1, r2):
    # Following http://www.cse.psu.edu/~rtc12/CSE486/lecture19.pdf
    # cross_prod = torch.cross(r1, r2, dim=-1)
    # dot_prod = (r1 * r2).sum(dim=-1)[..., None]
    # matrix_feature = torch.cat([cross_prod, dot_prod], dim=-1)
    matrix_feature = torch.cat([r1, r2], dim=-1)
    return matrix_feature


def get_essential_matrix(c2w_1, c2w_2):
    # Following http://www.cse.psu.edu/~rtc12/CSE486/lecture19.pdf
    R = torch.einsum('b...ij,b...jk->b...ik', c2w_2[..., :3, :3].transpose(-1, -2), c2w_1[..., :3, :3])
    T = get_ray_origin(c2w_2) - get_ray_origin(c2w_1)

    zero = torch.zeros_like(T[..., :1])
    x = T[..., :1]
    y = T[..., 1:2]
    z = T[..., -1:]
    s1 = torch.stack((zero, -z, y), dim=-1)
    s2 = torch.stack((z, zero, -x), dim=-1)
    s3 = torch.stack((-y, x, zero), dim=-1)
    S = torch.cat((s1, s2, s3), dim=-2)
    E = torch.einsum('b...ij,b...jk->b...ik', R, S) # essential matrix
    return E


def get_fundamental_matrix(c2w_1, c2w_2, intrinsics_1, intrinsics_2):
    '''Gets fundamental matrix, which, given a uv coordinate in camera_1, yields the equation of the epipolar line
    in cam2'''
    E = get_essential_matrix(c2w_1, c2w_2)
    left = torch.einsum('b...ij,b...jk->b...ik', torch.inverse(intrinsics_1[..., :3, :3]).transpose(-1, -2), E)
    F = torch.einsum('b...ij,b...jk->b...ik', left, torch.inverse(intrinsics_2[..., :3, :3]))
    return F


def homogenize_vec(vec):
    return torch.cat((vec, torch.ones_like(vec[..., :1])), dim=-1)


def get_depth_epipolar(ray_dir, pixel_val, query_cam2world, H, W, intrinsics):
    # An alternative depth computation method that is a bit more interpretable but less numerically stable
    # Computes depth along each value in pixel_val with respect to a ray in 
    # direction ray_dir and at starting location ray_orig
    pixel_y = (pixel_val[..., 1] + 1) / 2 * (H - 1)
    pixel_x = (pixel_val[..., 0] + 1) / 2 * (W - 1)

    ray_orig = get_ray_origin(query_cam2world).flatten(0, 1)

    fx, fy, cx, cy = parse_intrinsics(intrinsics.flatten(0, 1))
    fx, fy, cx, cy = fx[:, None], fy[:, None], cx[:, None], cy[:, None]
    ray_orig = ray_orig[:, None, None]
    ray_dir = ray_dir[:, :, None]

    # TODO Refactor into its own function in geometry.py and document.

    # Compute depth of the epipolar line
    y_num = pixel_y * ray_orig[..., 2] - cy * ray_orig[..., 2] - fy * ray_orig[..., 1]
    y_denom = fy * ray_dir[..., 1] + cy * ray_dir[..., 2] - pixel_y * ray_dir[..., 2] + 1e-12

    depth_y = y_num / (y_denom)
    # depth_log_y = torch.log(y_num) - torch.log(y_denom)

    x_num = pixel_x * ray_orig[..., 2] - cx * ray_orig[..., 2] - fx * ray_orig[..., 0]
    x_denom = fx * ray_dir[..., 0] + cx * ray_dir[..., 2] - pixel_x * ray_dir[..., 2] + 1e-12

    # depth_log_x = torch.log(x_num) - torch.log(x_denom)

    depth_x = (x_num) / (x_denom)

    depth = torch.max(depth_y, depth_x)

    return depth_x, depth_y, depth


def plucker_isparallel(line_1, line_2):
    return torch.cross(line_1[..., :3].double(), line_2[..., :3].double()).norm(dim=-1) < 1e-12

def plucker_isequivalent(line_1, line_2):
    unit_self = F.normalize(line_1.double(), dim=-1)
    unit_other = F.normalize(line_2.double(), dim=-1)
    return torch.abs(1 - torch.einsum('...i,...i', unit_self, unit_other)) < 1e-12

def plucker_closest_to_origin(line_1):
    return torch.cross(line_1[..., :3], line_1[..., 3:], dim=-1)

def get_3d_point_epipolar(query_ray, pixel_val, context_cam2world, H, W, intrinsics):
    b, n_qry = query_ray.shape[:2]
    pixel_y = (pixel_val[..., 1:2] + 1) / 2 * (H - 1)
    pixel_x = (pixel_val[..., 0:1] + 1) / 2 * (W - 1)

    pixel_coord = torch.cat((pixel_x, pixel_y), dim=-1)
    num_epi_points = pixel_coord.shape[-2]
    pixel_coord = pixel_coord.view(b, n_qry*num_epi_points, 2)

    # compute context plucker
    context_plucker = plucker_embedding(context_cam2world, pixel_coord, intrinsics)
    context_plucker = context_plucker.view(b, n_qry, num_epi_points, 6)

    # Get ray intersection
    p1, p2, = get_intersection(query_ray[..., None, :], context_plucker)

    line_1 = query_ray[..., None, :].double()
    line_2 = context_plucker.double()
    line_1, line_2 = torch.broadcast_tensors(line_1, line_2)
    parallel = plucker_isparallel(line_1, line_2)

    # identical lines
    equivalent = plucker_isequivalent(line_1, line_2)
    # p1[equivalent] = plucker_closest_to_origin(line_1)[equivalent]
    # p2[equivalent] = plucker_closest_to_origin(line_2)[equivalent]

    dist = torch.norm(p2 - p1, p=2, dim=-1)[..., None]

    p1[torch.isnan(p1)] = 0.0
    p1[torch.isinf(p1)] = 0.0

    return p1.float(), dist, parallel, equivalent


def get_intersection(line_1, line_2):
    line_1 = line_1.broadcast_to(line_2.shape)
    line_1 = line_1.double()
    line_2 = line_2.double()

    l1 = line_1[..., :3]
    l2 = line_2[..., :3]
    m1 = line_1[..., 3:]
    m2 = line_2[..., 3:]

    l1_cross_l2 = torch.cross(l1, l2, dim=-1)
    l2_cross_l1_cross_l2 = torch.cross(l2, l1_cross_l2, dim=-1)

    m1_cross_l2_cross_l1_cross_l2 = -torch.cross(m1, l2_cross_l1_cross_l2, dim=-1)

    second_term = (m2 * l1_cross_l2).sum(dim=-1)[..., None] * l1

    cross_dir = torch.norm(torch.cross(l1, l2, dim=-1), p=2, dim=-1)[..., None].pow(2) + 1e-12

    p1 = (m1_cross_l2_cross_l1_cross_l2 + second_term) / cross_dir
    p1_inverse = cross_dir / (m1_cross_l2_cross_l1_cross_l2 + second_term)

    l1_cross_l1_cross_l2 = torch.cross(l1, l1_cross_l2, dim=-1)
    m2_cross_l1_cross_l1_cross_l2 = torch.cross(m2, l1_cross_l1_cross_l2, dim=-1)

    second_term = (m1 * l1_cross_l2).sum(dim=-1)[..., None] * l2

    p2 = (m2_cross_l1_cross_l1_cross_l2 - second_term) / cross_dir
    p2_inverse = cross_dir / (m2_cross_l1_cross_l1_cross_l2 - second_term)

    return p1, p2


def get_epipolar_lines_volumetric(plucker_cam, cam2world, intrinsics, H, W, npoints, debug=False):
    '''Get epipolar lines for plucker coordinates in camera frame. Epi line is returned as homogenous line.'''
    camera_origin = p1 = get_ray_origin(cam2world)[:, :, None]

    p1 = p1
    near = 0.1
    far = 10.0

    interval = torch.linspace(0.1, 10., npoints).to(cam2world.device)

    interp_points = p1[..., None, :] + interval[None, None, None, :, None] * plucker_cam[..., None, :3]


    # get start and end points of epipolar line
    points = project(interp_points[..., 0], interp_points[..., 1], interp_points[..., 2], intrinsics)[..., :2]
    points = util.normalize_for_grid_sample(points, H, W)
    start = points[..., 0, :]
    end = points[..., -1, :]
    diff = end - start

    no_intersect = ((points < 1) & (points > -1)).all(dim=-1).any(dim=-1)

    return start, end, diff, no_intersect, points


def intersect_line_image_border(line_hom):
    '''A new approach to line and line segment clipping in homogeneous coordinates, Skala 2005'''
    b, n, _ = line_hom.shape

    # Intersection algorithm following Skala
    ones = torch.ones_like(line_hom[..., :1], device=line_hom.device)
    top_left = homogenize_vec(-1 * torch.cat((ones, ones), dim=-1))
    bottom_right = homogenize_vec(torch.cat((ones, ones), dim=-1))
    bottom_left = homogenize_vec(torch.cat((-1 * ones, ones), dim=-1))
    top_right = homogenize_vec(torch.cat((ones, -1*ones), dim=-1))

    # Index "4" (e4) is for invalid!
    tab_1 = torch.Tensor([4, 0, 0, 1, 1, 4, 0, 2, 2, 0, 4, 1, 1, 0, 0, 4])[None, None, :].long().to(line_hom.device)
    tab_2 = torch.Tensor([4, 3, 1, 3, 2, 4, 2, 3, 3, 2, 4, 2, 3, 1, 3, 4])[None, None, :].long().to(line_hom.device)

    e0 = torch.cross(bottom_left, bottom_right, dim=-1)
    e1 = torch.cross(top_right, bottom_right, dim=-1)
    e2 = torch.cross(top_right, top_left, dim=-1)
    e3 = torch.cross(top_left, bottom_left, dim=-1)
    e4 = torch.ones_like(top_left)
    all_es = torch.stack((e0, e1, e2, e3, e4), dim=-2)

    all_points = torch.stack((bottom_left, bottom_right, top_right, top_left), dim=-2)
    all_dot = torch.einsum('...i,...i', line_hom[:, :, None, :], all_points)
    c = (all_dot >= 0).int()

    dec = (c[..., 0] + c[..., 1] * 2 + c[..., 2] * 4 + c[..., 3] * 8).long()[..., None]
    tab_1 = tab_1.repeat(b, n, 1)
    tab_2 = tab_2.repeat(b, n, 1)

    i = torch.gather(input=tab_1, dim=-1, index=dec)[..., None].repeat(1, 1, 1, 3)
    j = torch.gather(input=tab_2, dim=-1, index=dec)[..., None].repeat(1, 1, 1, 3)

    e_i = torch.gather(input=all_es, dim=-2, index=i).squeeze(-2)
    e_j = torch.gather(input=all_es, dim=-2, index=j).squeeze(-2)

    x_a = torch.cross(line_hom, e_i)
    x_b = torch.cross(line_hom, e_j)

    x_a = x_a / x_a[..., -1:]
    x_b = x_b / x_b[..., -1:]

    no_intersection = torch.logical_or(dec==0, dec==15)
    return x_a, x_b, no_intersection


def plucker_embedding(cam2world, uv, intrinsics):
    ray_dirs = get_ray_directions(uv, cam2world=cam2world, intrinsics=intrinsics)
    cam_pos = get_ray_origin(cam2world)
    cam_pos = cam_pos[..., None, :].expand(list(uv.shape[:-1]) + [3])

    # https://www.euclideanspace.com/maths/geometry/elements/line/plucker/index.htm
    # https://web.cs.iastate.edu/~cs577/handouts/plucker-coordinates.pdf
    cross = torch.cross(cam_pos, ray_dirs, dim=-1)
    plucker = torch.cat((ray_dirs, cross), dim=-1)
    return plucker


def homogenize_mat(mat):
    hom = torch.Tensor([0.0, 0.0, 0.0, 1.0])

    while len(hom.shape) < len(mat.shape):
        hom = hom.unsqueeze(0)

    hom = hom.expand(mat.shape)
    return torch.cat((mat, hom), dim=-2)


def closest_to_origin(plucker_coord):
    direction = plucker_coord[..., :3]
    moment = plucker_coord[..., 3:]
    return torch.cross(direction, moment, dim=-1)


def plucker_sd(plucker_coord, point_coord):
    # Get closest point to origin along plucker line.
    plucker_origin = closest_to_origin(plucker_coord)

    # Compute signed distance: offset times dot product.
    direction = plucker_coord[..., :3]
    diff = point_coord - plucker_origin
    signed_distance = torch.einsum('...j,...j', diff, direction)
    return signed_distance[..., None]


def get_relative_rotation_matrix(vector_1, vector_2):
    "https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d"
    a_plus_b = vector_1 + vector_2
    outer = a_plus_b.unsqueeze(-2) * a_plus_b.unsqueeze(-1)
    dot = torch.einsum('...j,...j', a_plus_b, a_plus_b)[..., None, None]
    R = 2 * outer/dot - torch.eye(3)[None, None, None].to(vector_1.device)
    return R


def plucker_reciprocal_product(line_1, line_2):
    return torch.einsum('...j,...j', line_1[..., :3], line_2[..., 3:]) + \
           torch.einsum('...j,...j', line_2[..., :3], line_1[..., 3:])


def plucker_distance(line_1, line_2):
    line_1_dir, line_2_dir = torch.broadcast_tensors(line_1[..., :3], line_2[..., :3])
    direction_cross = torch.cross(line_1_dir, line_2_dir, dim=-1)
    # https://web.cs.iastate.edu/~cs577/handouts/plucker-coordinates.pdf
    return torch.abs(plucker_reciprocal_product(line_1, line_2))/direction_cross.norm(dim=-1)


def compute_normal_map(x_img, y_img, z, intrinsics):
    cam_coords = lift(x_img, y_img, z, intrinsics)
    cam_coords = util.lin2img(cam_coords)

    shift_left = cam_coords[:, :, 2:, :]
    shift_right = cam_coords[:, :, :-2, :]

    shift_up = cam_coords[:, :, :, 2:]
    shift_down = cam_coords[:, :, :, :-2]

    diff_hor = F.normalize(shift_right - shift_left, dim=1)[:, :, :, 1:-1]
    diff_ver = F.normalize(shift_up - shift_down, dim=1)[:, :, 1:-1, :]

    cross = torch.cross(diff_hor, diff_ver, dim=1)
    return cross


def get_ray_directions_cam(uv, intrinsics, H, W):
    '''Translates meshgrid of uv pixel coordinates to normalized directions of rays through these pixels,
    in camera coordinates.
    '''
    y_cam = (uv[..., 1] + 1) / 2 * (H - 1)
    x_cam = (uv[..., 0] + 1) / 2 * (W - 1)

    z_cam = torch.ones_like(x_cam)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=False)  # (batch_size, -1, 4)
    ray_dirs = F.normalize(pixel_points_cam, dim=-1)
    return ray_dirs


def reflect_vector_on_vector(vector_to_reflect, reflection_axis):
    refl = F.normalize(vector_to_reflect.cuda())
    ax = F.normalize(reflection_axis.cuda())

    r = 2 * (ax * refl).sum(dim=1, keepdim=True) * ax - refl
    return r


def parse_intrinsics(intrinsics):
    fx = intrinsics[..., 0, :1]
    fy = intrinsics[..., 1, 1:2]
    cx = intrinsics[..., 0, 2:3]
    cy = intrinsics[..., 1, 2:3]
    return fx, fy, cx, cy


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(x.device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)


def project(x, y, z, intrinsics, debug=False):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_proj = expand_as(fx, x) * x / (z + 1e-12) + expand_as(cx, x)
    y_proj = expand_as(fy, y) * y / (z + 1e-12) + expand_as(cy, y)

    coord = torch.stack((x_proj, y_proj, z), dim=-1)
    coord[torch.isnan(coord)] = 1e10
    coord[torch.isinf(coord)] = 1e10

    return coord

def project_cam2world(world_coords, cam2world):
    batch_size, num_samples, _ = world_coords.shape

    points_hom = torch.cat((world_coords, torch.ones((batch_size, num_samples, 1)).to(world_coords.device)),
                           dim=2)  # (batch, num_samples, 4)

    # permute for bmm
    points_hom = points_hom.permute(0, 2, 1)

    points_cam = torch.inverse(cam2world).bmm(points_hom)  # (batch, 4, num_samples)
    points_cam = points_cam.permute(0, 2, 1)[..., :3]
    return points_cam


def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    batch_size, *_ = cam2world.shape

    x_cam = xy[..., 0]
    y_cam = xy[..., 1]
    z_cam = depth

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)
    world_coords = torch.einsum('b...ij,b...kj->b...ki', cam2world, pixel_points_cam)[..., :3]

    return world_coords


def project_point_on_line(projection_point, line_direction, point_on_line):
    dot = torch.einsum('...j,...j', projection_point-point_on_line, line_direction)
    return point_on_line + dot[..., None] * line_direction

def get_ray_directions(xy, cam2world, intrinsics):
    z_cam = torch.ones(xy.shape[:-1]).to(xy.device)
    pixel_points = world_from_xy_depth(xy, z_cam, intrinsics=intrinsics, cam2world=cam2world)  # (batch, num_samples, 3)

    cam_pos = cam2world[..., :3, 3]
    ray_dirs = pixel_points - cam_pos[..., None, :]  # (batch, num_samples, 3)
    ray_dirs = F.normalize(ray_dirs, dim=-1)
    return ray_dirs


def depth_from_world(world_coords, cam2world):
    batch_size, num_samples, npoints, _ = world_coords.shape

    points_hom = torch.cat((world_coords, torch.ones((batch_size, num_samples, npoints, 1)).to(world_coords.device)),
                           dim=-1)  # (batch, num_samples, 4)

    # permute for bmm
    cam2world_inv = torch.inverse(cam2world)
    cam2world_inv = torch.flatten(cam2world_inv, 0, 1)[:, None]

    points_cam = torch.sum(cam2world_inv[:, :, None, :, :] * points_hom[:, :, :, None, :], dim=-1)

    depth = points_cam[..., 2]

    return depth


def ray_sphere_intersect(ray_origin, ray_dir, sphere_center=None, radius=1):
    if sphere_center is None:
        sphere_center = torch.zeros_like(ray_origin)

    ray_dir_dot_origin = torch.einsum('b...jd,b...id->b...ji', ray_dir, ray_origin - sphere_center)
    discrim = torch.sqrt( ray_dir_dot_origin**2 - (torch.einsum('b...id,b...id->b...i', ray_origin-sphere_center, ray_origin - sphere_center)[..., None] - radius**2) )

    t0 = - ray_dir_dot_origin + discrim
    t1 = - ray_dir_dot_origin - discrim
    return ray_origin + t0*ray_dir, ray_origin + t1*ray_dir


def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = torch.acos(1 - 2 * v)
    cx = torch.sin(phi) * torch.cos(theta)
    cy = torch.sin(phi) * torch.sin(theta)
    cz = torch.cos(phi)
    s = torch.stack([cx, cy, cz], dim=-1)
    return s


def polar_to_cartesian(r, theta, phi, deg=True):
    if deg:
        phi = phi * np.pi / 180
        theta = theta * np.pi / 180
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return r * np.stack([cx, cy, cz])


def to_uv(loc):
    # normalize to unit sphere
    loc = loc / loc.norm(dim=1, keepdim=True)

    cx, cy, cz = loc.t()
    v = (1 - cz) / 2

    phi = torch.acos(cz)
    sin_phi = torch.sin(phi)

    # ensure we do not divide by zero
    eps = 1e-8
    sin_phi[sin_phi.abs() < eps] = eps

    theta = torch.acos(cx / sin_phi)

    # check for sign of phi
    cx_rec = sin_phi * torch.cos(theta)
    if not np.isclose(cx.numpy(), cx_rec.numpy(), atol=1e-5).all():
        sin_phi = -sin_phi

    # check for sign of theta
    cy_rec = sin_phi * torch.sin(theta)
    if not np.isclose(cy.numpy(), cy_rec.numpy(), atol=1e-5).all():
        theta = -theta

    u = theta / (2 * np.pi)
    assert np.isclose(to_sphere(u, v).detach().cpu().numpy(), loc.t().detach().cpu().numpy(), atol=1e-5).all()

    return u, v


def to_phi(u):
    return 360 * u  # 2*pi*u*180/pi


def to_theta(v):
    return np.arccos(1 - 2 * v) * 180. / np.pi


def sample_on_sphere(size, range_u=(0, 1), range_v=(0, 1)):
    u = torch.zeros(size).uniform_(*range_u)
    v = torch.zeros(size).uniform_(*range_v)
    return to_sphere(u, v)


def look_at(eye, at=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 0, 1]), eps=1e-5):
    at = at.unsqueeze(0).unsqueeze(0)
    up = up.unsqueeze(0).unsqueeze(0)

    z_axis = eye - at
    z_axis /= z_axis.norm(dim=-1, keepdim=True) + eps

    up = up.expand(z_axis.shape)
    x_axis = torch.cross(up, z_axis)
    x_axis /= x_axis.norm(dim=-1, keepdim=True) + eps

    y_axis = torch.cross(z_axis, x_axis)
    y_axis /= y_axis.norm(dim=-1, keepdim=True) + eps

    r_mat = torch.stack((x_axis, y_axis, z_axis), axis=-1)
    return r_mat

