"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import utils.pixel_util as util
import torch.autograd.profiler as profiler


class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """
    def __init__(
        self,
        backbone="resnet34",
        pretrained=False,
        num_layers=5,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)

        print("Using torchvision", backbone, "encoder")
        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained
        )
        # Following 2 lines need to be uncommented for older configs
        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        # self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp

    def forward(self, x, cam2world, n_view):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        x = x
        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            # self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latents = latents[::-1]
            latents_large = latents

            self.latent = latents
        return self.latent



class UNetEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """
    def __init__(
        self,
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        self.down1 = nn.Conv2d(3, 32, 7, padding=3)
        self.down1a = nn.Conv2d(32, 32, 7,  padding=3)

        self.down2 = nn.Conv2d(32, 64, 5, padding=2)
        self.down2a = nn.Conv2d(64, 64, 5, padding=2)

        self.down3 = nn.Conv2d(64, 128, 3, padding=1)
        self.down3a = nn.Conv2d(128, 128, 3, padding=1)

        self.down4 = nn.Conv2d(128, 256, 3, padding=1)
        self.down4a = nn.Conv2d(256, 256, 3, padding=1)

        self.down5 = nn.Conv2d(256, 512, 3, padding=1)
        self.down5a = nn.Conv2d(512, 512, 3, padding=1)

        self.down6 = nn.Conv2d(512, 512, 3, padding=1)
        self.down6a = nn.Conv2d(512, 512, 3, padding=1)

        self.down7 = nn.Conv2d(512, 512, 3, padding=1)
        self.down7a = nn.Conv2d(512, 512, 3, padding=1)

        self.mid1 = nn.Conv2d(512, 512, 3, padding=1)
        self.mid2 = nn.Conv2d(512, 512, 3, padding=1)

        self.up7 = nn.Conv2d(1024, 512, 3, padding=1)
        self.up7b = nn.Conv2d(512, 512, 3, padding=1)

        self.up6 = nn.Conv2d(1024, 512, 3, padding=1)
        self.up6b = nn.Conv2d(512, 512, 3, padding=1)

        self.up5 = nn.Conv2d(1024, 256, 3, padding=1)
        self.up5b = nn.Conv2d(256, 256, 3, padding=1)

        self.up4 = nn.Conv2d(512, 128, 3, padding=1)
        self.up4b = nn.Conv2d(128, 128, 3, padding=1)

        self.up3 = nn.Conv2d(256, 64, 3, padding=1)
        self.up3b = nn.Conv2d(64, 64, 3, padding=1)

        self.up2 = nn.Conv2d(128, 32, 3, padding=1)
        self.up2b = nn.Conv2d(32, 32, 3, padding=1)

        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)
        # self.latent (B, L, H, W)

    def forward(self, x, camera_encode, n_view):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """

        downs = []

        x = (F.relu(self.down1(x)))
        # downs.append(x)
        x = self.downsample(x)

        x = (F.relu(self.down2(x)))
        downs.append(x)
        x = self.downsample(x)

        x =(F.relu(self.down3(x)))
        downs.append(x)
        x = self.downsample(x)

        x = (F.relu(self.down4(x)))
        downs.append(x)
        x = self.downsample(x)

        x = (F.relu(self.down5(x)))
        downs.append(x)

        return downs
        # x = self.downsample(x)

        # x = (F.relu(self.down6(x)))
        # downs.append(x)
        # x = self.downsample(x)

        # x = (F.relu(self.down7(x)))
        # downs.append(x)

        # x = (F.relu(self.mid1(x)))

        # x = torch.cat([x, downs[-1]], dim=1)
        # x = (F.relu(self.up7(x)))
        # x = F.interpolate(x, (downs[-2].size(2), downs[-2].size(3)))

        # # x = x + downs[-2]
        # x = torch.cat([x, downs[-2]], dim=1)
        # x = (F.relu(self.up6(x)))
        # x = F.interpolate(x, (downs[-3].size(2), downs[-3].size(3)))

        # # x = x + downs[-3]
        # x = torch.cat([x, downs[-3]], dim=1)
        # x = (F.relu(self.up5(x)))
        # x = F.interpolate(x, (downs[-4].size(2), downs[-4].size(3)))


        # x = torch.cat([x, downs[-4]], dim=1)
        # x = (F.relu(self.up4(x)))
        # x = F.interpolate(x, (downs[-5].size(2), downs[-5].size(3)))

        # x = x + downs[-5]
        # x = torch.cat([x, downs[-5]], dim=1)
        # x = (F.relu(self.up3(x)))
        # x = F.interpolate(x, (downs[-6].size(2), downs[-6].size(3)))

        # x = x + downs[-6]
        # x = torch.cat([x, downs[-6]], dim=1)
        # x = self.up2b(F.relu(self.up2(x)))

        return x


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1))
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )
