
import numpy as np
import torch

# default parameters for Clevr
def normalized_K(height=240, width=320, focal_length=35, sensor_width=32):
    fx = focal_length / sensor_width
    fy = focal_length*(width/height) / sensor_width
    K = torch.Tensor([[fx, 0, 1.0/2], [0, fy, 1.0/2], [0, 0, 1]])
    return K


def generate_frustum_pixel_points(coords, cam_to_ref, D, intrinsic_matrix=normalized_K(),
                                  dmin=0.1, dmax=10):
    # coords: [B, N, T, 2], in the range of [0,1]
    inv_intrinsics = torch.linalg.inv(intrinsic_matrix).to(coords.device)

    # Initialize arrays to store the 3D points for each pixel
    frustum_points = []
    coords = torch.cat(
        [coords, torch.ones(*coords.shape[:-1], 1).to(coords.device)], -1)  # [N, 3]
    for d_i in range(1, D+1):
        # Transform pixel coordinates to camera space
        camera_space_coords = coords @ inv_intrinsics.transpose(-2, -1)

        # Scale the camera space coordinates to the near and far planes
        d = dmin + ((dmax-dmin)/(D*(D+1))) * d_i*(d_i+1)
        point = camera_space_coords * d
        point = torch.cat(
            [point, torch.ones(*point.shape[:-1], 1).to(point.device)], -1)  # [B, N, T, 4]
        frustum_points.append(point)

    p3d = torch.stack(frustum_points, -2)  # [, ..., D, 4]
    p3d = torch.einsum('bnij,bntdj->bntdi', cam_to_ref,
                       p3d).flatten(-2, -1)  # [B, N, T, D*4]
    return p3d


if __name__ == "__main__":
    # Reducing image resolution for computational feasibility
    W = 20
    H = 15

    xs = torch.arange(0, W)/(W-1)
    ys = torch.arange(0, H)/(H-1)
    coords = torch.meshgrid(xs, ys, indexing='xy')
    coords = torch.stack([torch.reshape(coords[0], (-1,)),
                         torch.reshape(coords[1], (-1,))], -1)

    def normalized_K(height=240, width=320, focal_length=35, sensor_width=32):
        fx = focal_length / sensor_width
        fy = focal_length*(width/height) / sensor_width
        K = torch.Tensor([[fx, 0, 1.0/2], [0, fy, 1.0/2], [0, 0, 1]])
        return K

    K = normalized_K()

    # Generate frustum pixel points
    points = generate_frustum_pixel_points(coords, K, 10)
