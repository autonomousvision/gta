import matplotlib
import torch.nn.functional as F
matplotlib.use('Agg')
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
from utils import util
import torchvision

import geometry


def img_summaries(model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix="", img_shape=(98, 144), n_view=1):
    predictions = model_output['rgb']

    predictions = predictions.view(*predictions.size()[:-2], img_shape[0], img_shape[1], 3)
    predictions = util.flatten_first_two(predictions)
    predictions = predictions.permute(0, 3, 1, 2)
    predictions = torch.clamp(predictions, -1, 1)

    if 'at_wt' in model_output:
        at_wt = model_output['at_wt']
        ent = -(at_wt * torch.log(at_wt + 1e-5)).sum(dim=-1)
        ent = ent.mean()
        writer.add_scalar(prefix + "ent", ent, iter)
        print(at_wt[0, 2065])
        print("entropy: ", ent)

    writer.add_image(prefix + "predictions",
                     torchvision.utils.make_grid(predictions, scale_each=False, normalize=True).cpu().numpy(),
                     iter)

    depth_img = model_output['depth_ray'].view(-1, img_shape[0], img_shape[1])
    depth_img = depth_img.detach().cpu().numpy() / 10.
    cmap = plt.get_cmap("jet")
    depth_img = cmap(depth_img)[..., :3]
    depth_img = depth_img.transpose((0, 3, 1, 2))

    depth_img = torch.Tensor(depth_img)
    writer.add_image(prefix + "depth_images",
                     torchvision.utils.make_grid(depth_img, scale_each=True, normalize=True).cpu().numpy(),
                     iter)

    context_images = util.flatten_first_two(model_input['context']['rgb'])
    context_images = context_images.permute(0, 3, 1, 2)

    writer.add_image(prefix + "context_images",
                     torchvision.utils.make_grid(context_images, scale_each=False, normalize=True).cpu().numpy(),
                     iter)

    query_images = model_input['query']['rgb']
    query_images = query_images.view(*query_images.size()[:-2], img_shape[0], img_shape[1], 3)

    query_images = util.flatten_first_two(query_images)
    query_images = query_images.permute(0, 3, 1, 2)
    writer.add_image(prefix + "query_images",
                     torchvision.utils.make_grid(query_images, scale_each=False, normalize=True).cpu().numpy(),
                     iter)

    epi_summary(model_output, query_images, context_images, writer, iter, prefix=prefix, n_view=n_view)

    writer.add_scalar(prefix + "out_min", predictions.min(), iter)
    writer.add_scalar(prefix + "out_max", predictions.max(), iter)

    writer.add_scalar(prefix + "trgt_min", query_images.min(), iter)
    writer.add_scalar(prefix + "trgt_max", query_images.max(), iter)



def epi_summary(model_output, trgt_imgs_tile, ctxt_imgs_tile, writer, iter, prefix="", n_view=1):
    pixel_val = model_output['pixel_val']
    at_wt_max = model_output['at_wt_max']

    uv = model_output['uv']

    trgt_imgs_tile = trgt_imgs_tile.clone()
    ctxt_imgs_tile = ctxt_imgs_tile.clone()

    B, _, H, W = trgt_imgs_tile.size()
    uv = uv

    pixel_val = pixel_val.cpu().numpy()
    s = pixel_val.shape
    pixel_val = pixel_val.reshape((s[0]//n_view, n_view, *s[1:]))
    s = at_wt_max.shape
    at_wt_max = at_wt_max.reshape((s[0]//n_view, n_view, *s[1:]))

    pix_size = H // 64 + 1

    counter = 0

    for i in range(B):
        six = random.randint(0, uv.shape[2] - 1)
        six = 2065

        coord = uv[i, 0, six]
        x, y = int(coord[0]), int(coord[1])
        xmin, xmax = max(x - pix_size, 0), min(x + pix_size, trgt_imgs_tile.size(3) - 1)
        ymin, ymax = max(y - pix_size, 0), min(y + pix_size, trgt_imgs_tile.size(2) - 1)
        trgt_imgs_tile[i, :, ymin:ymax, xmin:xmax] = -1.0


        for k in range(n_view):
            for j in range(pixel_val.shape[3]):
                val = pixel_val[i, k, six, j]
                val = (val + 1) / 2
                val = np.clip(val, 0, 1)
                x = int(val[0] * (W - 1))
                y = int(val[1] * (H - 1))

                xmin, xmax = max(x - pix_size, 0), min(x + pix_size, trgt_imgs_tile.size(3) - 1)
                ymin, ymax = max(y - pix_size, 0), min(y + pix_size, trgt_imgs_tile.size(2) - 1)

                ctxt_imgs_tile[counter, :, ymin:ymax, xmin:xmax] = 0.0


            # Plot the maximum attention point on the epipolar
            max_idx = at_wt_max[i, k, six].item()
            val = pixel_val[i, k, six, max_idx]
            val = (val + 1) / 2
            val = np.clip(val, 0, 1)
            x = int(val[0] * (W - 1))
            y = int(val[1] * (H - 1))

            xmin, xmax = max(x - pix_size, 0), min(x + pix_size, trgt_imgs_tile.size(3) - 1)
            ymin, ymax = max(y - pix_size, 0), min(y + pix_size, trgt_imgs_tile.size(2) - 1)
            # print(x, y)

            ctxt_imgs_tile[counter, :, ymin:ymax, xmin:xmax] = -1.0
            counter = counter + 1

    s = ctxt_imgs_tile.size()
    ctxt_imgs_tile = ctxt_imgs_tile.view(-1, n_view, *s[1:]).permute(1, 0, 2, 3, 4).reshape(*s)
    panel = torch.cat((trgt_imgs_tile, ctxt_imgs_tile), dim=0)

    writer.add_image(prefix + "epipolar_line",
                     torchvision.utils.make_grid(panel, scale_each=False,
                                                 normalize=True).cpu().detach().numpy(),
                     iter)
