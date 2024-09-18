import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stylegan2.model import EqualLinear
from models.psp.psp_encoders import Inverter, Encoder4Editing
from utils.torch_utils import get_keys, requires_grad
from models.deca.deca import DECA
from models.stylegan2.model import Generator
from face_parsing.face_parsing_demo import FaceParser, faceParsing_demo
from utils.morphology import dilation
from torchvision.transforms import GaussianBlur as blur
import torch.nn.init as init
from StyleFeatureEditor.models.methods import FSEInverter32
from models.psp.model_irse import Backbone


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0)
    elif isinstance(m, nn.Conv2d):
        init.constant_(m.weight, 0)


class Mapper(nn.Module):
    def __init__(self):
        super(Mapper, self).__init__()
        self.backbone = nn.ModuleList()
        self.backbone.append(EqualLinear(100, 512))
        for i in range(4):
            self.backbone.append(EqualLinear(512, 512, activation=True))
        self.backbone = nn.Sequential(*self.backbone)

    def forward(self, alpha):
        """alpha -- B x 100
            return B x 512"""
        return self.backbone(alpha)


def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05, )
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class Fuser(nn.Module):
    def __init__(self, blocks, inplanes=1024, init_zeros=True):
        super().__init__()

        self.res_blocks = {}

        for n, block in enumerate(blocks, start=1):
            planes, num_blocks = block

            for k in range(1, num_blocks + 1):
                downsample = None
                if inplanes != planes:
                    downsample = nn.Sequential(conv1x1(inplanes, planes, 1), nn.BatchNorm2d(planes, eps=1e-05, ), )

                self.res_blocks[f'res_block_{n}_{k}'] = IBasicBlock(inplanes, planes, 1, downsample, 1, 64, 1)
                inplanes = planes

        self.res_blocks = nn.ModuleDict(self.res_blocks)
        if init_zeros:
            self.apply(_weights_init)

    def forward(self, x):
        for module in self.res_blocks.values():
            x = module(x)
        return x


class AdaIN_param(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512 * 7 * 7, 512 * 7)
        self.fc2 = nn.Linear(512 * 7, 512)
        self.norm = nn.LayerNorm(512 * 7)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.fc2(self.relu(self.norm(self.fc1(x))))


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_net = AdaIN_param()
        self.beta_net = AdaIN_param()
        self.conv = nn.Conv2d(512, 512, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        self.norm = nn.InstanceNorm2d(512)

    def forward(self, input, source_latent):
        gamma = self.gamma_net(source_latent)
        beta = self.beta_net(source_latent)
        x = gamma[..., None, None] * self.norm(input) + beta[..., None, None]  # B x 512 x 32 x 32
        x = self.relu(x)
        x = self.conv(x)
        return x


class Net(nn.Module):
    def __init__(self, opts):
        super(Net, self).__init__()

        self.target_encoder = FSEInverter32(checkpoint_path='pretrained_ckpts/iteration_135000.pt').eval()

        self.arcface = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.arcface.load_state_dict(torch.load(opts.ir_se50_path))
        self.arcface.output_layer = nn.Flatten()
        self.arcface.eval()

        self.source_identity = Encoder4Editing(50, 'ir_se', opts).eval()
        ckpt = torch.load('./pretrained_ckpts/e4e_ffhq_encode.pt')
        self.source_identity.load_state_dict(get_keys(ckpt, "encoder"))

        self.source_shape = DECA('./pretrained_ckpts/deca/deca_model.tar')

        self.mapping = Mapper()

        self.G = Generator(1024, 512, 8)

        self.adain1 = AdaIN()
        self.adain2 = AdaIN()

        self.fuser = Fuser([[528, 2], [512, 2]], inplanes=528, init_zeros=False)
        self.arcface_fuser = Fuser([[448, 2], [232, 2], [16, 2]], inplanes=448, init_zeros=False)

        requires_grad(self.arcface, False)
        requires_grad(self.G, False)
        requires_grad(self.source_shape, False)
        requires_grad(self.source_identity, False)
        requires_grad(self.target_encoder, False)
        requires_grad(self.mapping, True)
        requires_grad(self.adain1, True)
        requires_grad(self.adain2, True)
        requires_grad(self.arcface_fuser, True)
        requires_grad(self.fuser, True)

    def shift_tensor(self, feat, mask):
        mask = mask.squeeze()
        hor = mask.sum(0)
        ver = mask.sum(1)
        non_zero_hor = torch.nonzero(hor, as_tuple=True)[0]
        non_zero_ver = torch.nonzero(ver, as_tuple=True)[0]

        first_hor = non_zero_hor[0]
        last_hor = mask.shape[1] - non_zero_hor[-1] - 1

        first_ver = non_zero_ver[0]
        last_ver = mask.shape[0] - non_zero_ver[-1] - 1

        if torch.rand(1).cuda() < first_hor / (first_hor + last_hor):
            hor_ind = first_hor
            hor_type = -1
        else:
            hor_ind = last_hor
            hor_type = 1

        if torch.rand(1).cuda() < first_ver / (first_ver + last_ver):
            ver_ind = first_ver
            ver_type = -1
        else:
            ver_ind = last_ver
            ver_type = 1
        hor_shift = torch.randint(0, hor_ind.item() + 1, (1,))
        ver_shift = torch.randint(0, ver_ind.item() + 1, (1,))
        return torch.roll(feat, (hor_shift * hor_type, ver_shift * ver_type), dims=(2, 1))

    def get_mask(self, img, mode=None, verbose=False):

        mask = faceParsing_demo(self.face_parser, img, convert_to_seg12=True, model_name='default').long()

        if mode == 'target':
            mask_ = logical_or_reduce(*[mask == item for item in [0, 4, 6, 7, 8, 10, 11]]).float()
        else:
            mask_ = logical_or_reduce(*[mask == item for item in [1, 2, 3, 5, 9]]).float()

        mask_32 = torch.round(F.interpolate(mask_.unsqueeze(1), 32, mode='bilinear', align_corners=False))
        mask_dil = dilation(mask_32, torch.ones(3, 3), engine='convolution')
        mask_blur = blur(kernel_size=3, sigma=1)(mask_dil)
        if verbose:
            return mask_blur, mask
        return mask_blur, None

    def forward(self, source, target, verbose=False, return_feat=False, step=None, max_step=2.5e4, train=False):
        """source -- B x 3 x 1024 x 1024
           target -- B x 3 x 1024 x 1024
           skip -- B x 3 x 32 x 32"""
        s_256 = F.interpolate(source, (256, 256), mode='bilinear')
        t_256 = F.interpolate(target, (256, 256), mode='bilinear')
        s_w_id = self.source_identity(s_256)
        t_w_id = self.source_identity(t_256)
        s_w_id += self.latent_avg[None, ...]
        t_w_id += self.latent_avg[None, ...]

        alpha = self.source_shape(source)['shape']
        s_w_shape = self.mapping(alpha)

        s_style = s_w_id + s_w_shape[:, None, :]

        _, t_feat = self.target_encoder(t_256)

        s_112 = F.interpolate(s_256[:, :, 35:223, 32:220], (112, 112), mode='bilinear')
        source_latents = self.arcface(s_112, True)
        source_latent = source_latents[-1].view(s_112.shape[0], -1)

        t_adain1 = self.adain1(t_feat, source_latent)
        t_adain2 = self.adain2(t_adain1, source_latent)

        c1 = F.interpolate(source_latents[0], (32, 32), mode='bilinear')
        c2 = F.interpolate(source_latents[1], (32, 32), mode='bilinear')
        c3 = F.interpolate(source_latents[2], (32, 32), mode='bilinear')

        arcface_feat = self.arcface_fuser(torch.cat([c1, c2, c3], dim=1))  # B x 16 x 32 x 32

        feat = t_adain2 + self.fuser(torch.cat([t_adain2, arcface_feat], dim=1))  # B x 512 x 32 x 32

        s_style[:, :7] = t_w_id[:, :7]
        img, _ = self.G([s_style], new_features=[None] * 7 + [feat] + [None] * (17 - 7))
        if return_feat:
            return img, None, None, None
        if verbose:
            return img, self.G([s_style])[0]
        return img
