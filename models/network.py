import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.stylegan2.model import EqualLinear
from models.psp.psp_encoders import Inverter, Encoder4Editing
from utils.torch_utils import get_keys, requires_grad
from models.deca.deca import DECA
from models.stylegan2.model import Generator
# from models.psp.fse import fs_encoder_v2
# from my_models.FeatureStyleEncoder import FSencoder
from face_parsing.face_parsing_demo import FaceParser, faceParsing_demo, vis_parsing_maps
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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = nn.ModuleList()
        self.backbone.append(nn.Conv2d(3, 64, 3, padding=1))
        self.backbone.append(nn.LeakyReLU(0.2))
        self.backbone.append(nn.Conv2d(64, 128, 3, padding=1))
        self.backbone.append(nn.LeakyReLU(0.2))
        in_c = 128
        out_c = 256
        for i in range(3):
            self.backbone.append(nn.Conv2d(in_c, out_c, 3, padding=1, stride=2))
            self.backbone.append(nn.LeakyReLU(0.2))
            self.backbone.append(nn.Conv2d(out_c, out_c, 3, padding=1))
            if i < 2:
                self.backbone.append(nn.LeakyReLU(0.2))
            in_c = out_c
            if i == 1:
                out_c *= 2
        self.backbone = nn.Sequential(*self.backbone)
        self.img_conv = nn.Conv2d(512, 3, 1, padding=0)

    def forward(self, img, return_img=True):
        """img -- batch x 3 x 256 x 256
            return -- batch x 512 x 32 x 32"""
        feat = self.backbone(img)
        if return_img:
            return feat, self.img_conv(feat)
        return feat, None


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
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.norm = nn.LayerNorm(512)
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
        self.arcface.eval()

        ########################### E4E #######################
        self.source_identity = Encoder4Editing(50, 'ir_se', opts).eval()
        ckpt = torch.load(opts.e4e_path)
        self.source_identity.load_state_dict(get_keys(ckpt, "encoder"))
        #######################################################

        self.source_shape = DECA(opts.deca_path)

        self.mapping = Mapper()

        self.G = Generator(1024, 512, 8)

        self.adain1 = AdaIN()
        self.adain2 = AdaIN()
        self.adain3 = AdaIN()
        self.adain4 = AdaIN()
        self.adain5 = AdaIN()
        self.adain6 = AdaIN()

        requires_grad(self.mapping, True)
        requires_grad(self.G, False)
        requires_grad(self.source_shape, False)
        requires_grad(self.source_identity, False)
        requires_grad(self.target_encoder, False)
        requires_grad(self.arcface, False)
        requires_grad(self.adain1, True)
        requires_grad(self.adain2, True)
        requires_grad(self.adain3, True)
        requires_grad(self.adain4, True)
        requires_grad(self.adain5, True)
        requires_grad(self.adain6, True)

    def get_mask(self, img, mode=None, verbose=False):

        mask = faceParsing_demo(self.face_parser, img, convert_to_seg12=True, model_name='default').long()

        if mode == 'target':
            mask_ = logical_or_reduce(*[mask == item for item in [0, 4, 6, 7, 8, 10, 11]]).float()
        else:
            mask_ = logical_or_reduce(*[mask == item for item in [1, 2, 3, 5, 9]]).float()

        mask_32 = F.interpolate(mask_.unsqueeze(1), 32, mode='bilinear', align_corners=False)
        mask_dil = dilation(mask_32, torch.ones(3, 3), engine='convolution')
        mask_blur = blur(kernel_size=3, sigma=1)(mask_dil)
        if verbose:
            return mask_blur, mask
        return mask_blur, None

    def forward(self, source, target, verbose=False, return_feat=False, step=0):
        """source -- B x 3 x 1024 x 1024
           target -- B x 3 x 1024 x 1024
           skip -- B x 3 x 32 x 32"""
        s_256 = F.interpolate(source, (256, 256), mode='bilinear')
        t_256 = F.interpolate(target, (256, 256), mode='bilinear')

        s_w_id, _ = self.source_identity(s_256, True)
        t_w_id, _ = self.source_identity(t_256, True)
        t_w_id += self.latent_avg[None, ...]

        alpha = self.source_shape(source)['shape']
        s_w_shape = self.mapping(alpha)

        s_style = s_w_id + s_w_shape[:, None, :] + self.latent_avg[None, ...]

        t_style, t_feat = self.target_encoder(t_256)

        s_112 = F.interpolate(s_256[:, :, 35:223, 32:220], (112, 112), mode='bilinear')
        source_latent = self.arcface(s_112)[0]

        delta_feat_1 = self.adain1(t_feat, source_latent)
        delta_feat_2 = self.adain2(delta_feat_1, source_latent)
        delta_feat_3 = self.adain3(delta_feat_2, source_latent)
        delta_feat_4 = self.adain4(delta_feat_3, source_latent)
        delta_feat_5 = self.adain5(delta_feat_4, source_latent)
        delta_feat = self.adain6(delta_feat_5, source_latent)
        a = 1.0
        feat = t_feat + delta_feat

        s_style[:, :7] = t_w_id[:, :7]
        img, _ = self.G([s_style], new_features=[None] * 7 + [feat] + [None] * (17 - 7))
        if return_feat:
            return img, t_feat, delta_feat, a
        if verbose:
            return img, None, None
        return img

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as T
# from models.stylegan2.model import EqualLinear
# from models.psp.psp_encoders import Inverter, Encoder4Editing
# from utils.torch_utils import get_keys, requires_grad
# from models.deca.deca import DECA
# from models.stylegan2.model import Generator
# # from models.psp.fse import fs_encoder_v2
# # from my_models.FeatureStyleEncoder import FSencoder
# from face_parsing.face_parsing_demo import FaceParser, faceParsing_demo, vis_parsing_maps
# from utils.morphology import dilation
# from torchvision.transforms import GaussianBlur as blur
# import torch.nn.init as init
# from StyleFeatureEditor.models.methods import FSEInverter32


# def _weights_init(m):
#     classname = m.__class__.__name__
#     if isinstance(m, nn.Linear):
#         init.constant_(m.weight, 0)
#     elif isinstance(m, nn.Conv2d):
#         init.constant_(m.weight, 0)

# class Mapper(nn.Module):
#     def __init__(self):
#         super(Mapper, self).__init__()
#         self.backbone = nn.ModuleList()
#         self.backbone.append(EqualLinear(100, 512))
#         for i in range(4):
#             self.backbone.append(EqualLinear(512, 512, activation=True))
#         self.backbone = nn.Sequential(*self.backbone)

#     def forward(self, alpha):
#         """alpha -- B x 100
#             return B x 512"""
#         return self.backbone(alpha)


# def logical_or_reduce(*tensors):
#     return torch.stack(tensors, dim=0).any(dim=0)


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.backbone = nn.ModuleList()
#         self.backbone.append(nn.Conv2d(3, 64, 3, padding=1))
#         self.backbone.append(nn.LeakyReLU(0.2))
#         self.backbone.append(nn.Conv2d(64, 128, 3, padding=1))
#         self.backbone.append(nn.LeakyReLU(0.2))
#         in_c = 128
#         out_c = 256
#         for i in range(3):
#             self.backbone.append(nn.Conv2d(in_c, out_c, 3, padding=1, stride=2))
#             self.backbone.append(nn.LeakyReLU(0.2))
#             self.backbone.append(nn.Conv2d(out_c, out_c, 3, padding=1))
#             if i < 2:
#                 self.backbone.append(nn.LeakyReLU(0.2))
#             in_c = out_c
#             if i == 1:
#                 out_c *= 2
#         self.backbone = nn.Sequential(*self.backbone)
#         self.img_conv = nn.Conv2d(512, 3, 1, padding=0)

#     def forward(self, img, return_img=True):
#         """img -- batch x 3 x 256 x 256
#             return -- batch x 512 x 32 x 32"""
#         feat = self.backbone(img)
#         if return_img:
#             return feat, self.img_conv(feat)
#         return feat, None

# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes,
#                      out_planes,
#                      kernel_size=3,
#                      stride=stride,
#                      padding=dilation,
#                      groups=groups,
#                      bias=False,
#                      dilation=dilation)

# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes,
#                      out_planes,
#                      kernel_size=1,
#                      stride=stride,
#                      bias=False)

# class IBasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None,
#                  groups=1, base_width=64, dilation=1):
#         super(IBasicBlock, self).__init__()
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05, )
#         self.conv1 = conv3x3(inplanes, planes)
#         self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, )
#         self.prelu = nn.PReLU(planes)
#         self.conv2 = conv3x3(planes, planes, stride)
#         self.bn3 = nn.BatchNorm2d(planes, eps=1e-05, )
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x
#         out = self.bn1(x)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out = self.prelu(out)
#         out = self.conv2(out)
#         out = self.bn3(out)
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         return out


# class Fuser(nn.Module):
#     def __init__(self, blocks, inplanes=1024, init_zeros=True):
#         super().__init__()

#         self.res_blocks = {}

#         for n, block in enumerate(blocks, start=1):
#             planes, num_blocks = block

#             for k in range(1, num_blocks + 1):
#                 downsample = None
#                 if inplanes != planes:
#                     downsample = nn.Sequential(conv1x1(inplanes, planes, 1), nn.BatchNorm2d(planes, eps=1e-05, ), )

#                 self.res_blocks[f'res_block_{n}_{k}'] = IBasicBlock(inplanes, planes, 1, downsample, 1, 64, 1)
#                 inplanes = planes

#         self.res_blocks = nn.ModuleDict(self.res_blocks)
#         if init_zeros:
#             self.apply(_weights_init)

#     def forward(self, x):
#         for module in self.res_blocks.values():
#             x = module(x)
#         return x


# class Net(nn.Module):
#     def __init__(self, opts):
#         super(Net, self).__init__()

#         # ########################### FSE #######################
#         # self.target_encoder = FSencoder.get_trainer(opts.device)
#         # #######################################################

#         self.target_encoder = FSEInverter32(checkpoint_path='pretrained_ckpts/iteration_135000.pt').eval()


#         ########################### E4E #######################
#         self.source_identity = Encoder4Editing(50, 'ir_se', opts).eval()
#         ckpt = torch.load(opts.e4e_path)
#         self.source_identity.load_state_dict(get_keys(ckpt, "encoder"))
#         #######################################################

#         self.source_shape = DECA(opts.deca_path)

#         self.mapping = Mapper()

#         self.G = Generator(1024, 512, 8)

#         self.face_parser = FaceParser(seg_ckpt='./pretrained_ckpts/79999_iter.pth', device='cuda:0').eval()

#         self.fuser = Fuser([[1024, 2], [768, 2], [512, 2]])
#         self.shifter = Fuser([[512, 2]], 512)

#         requires_grad(self.mapping, True)
#         requires_grad(self.shifter, True)
#         requires_grad(self.fuser, True)
#         requires_grad(self.G, False)
#         requires_grad(self.source_shape, False)
#         requires_grad(self.source_identity, False)
#         requires_grad(self.target_encoder, False)
#         requires_grad(self.face_parser, False)

#     def get_mask(self, img, mode=None, verbose=False):

#         mask = faceParsing_demo(self.face_parser, img, convert_to_seg12=True, model_name='default').long()

#         if mode == 'target':
#             mask_ = logical_or_reduce(*[mask == item for item in [0, 4, 6, 8, 10, 11]]).float()
#         else:
#             mask_ = logical_or_reduce(*[mask == item for item in [1, 2, 3, 5, 7, 9]]).float()


#         mask_32 = F.interpolate(mask_.unsqueeze(1), 32, mode='bilinear', align_corners=False)
#         mask_dil = dilation(mask_32, torch.ones(3, 3), engine = 'convolution')
#         mask_blur = blur(kernel_size=3, sigma=1)(mask_dil)
#         if verbose:
#             return mask_blur, mask
#         return mask_blur, None

#     def forward(self, source, target, verbose=False):
#         """source -- B x 3 x 1024 x 1024
#            target -- B x 3 x 1024 x 1024
#            skip -- B x 3 x 32 x 32"""
#         s_256 = F.interpolate(source, (256, 256), mode='bilinear')
#         t_256 = F.interpolate(target, (256, 256), mode='bilinear')

#         if verbose:
#             recon = []
#             masks = []

#         s_w_id, _ = self.source_identity(s_256, True)
#         t_w_id, _ = self.source_identity(t_256, True)
#         t_w_id += self.latent_avg[None, ...]
#         # s_w_id_fse, s_feat = self.target_encoder.test(img=source, return_latent=True)[-2:]
#         s_w_id_sfe, s_feat  = self.target_encoder(s_256)
#         if verbose:
#             # recon.append(self.G([s_w_id_sfe], s_feat, start=7)[0])
#             recon.append(self.G([s_w_id_sfe], new_features=[None] * 7 + [s_feat] + [None] * (17 - 7))[0])
#             # recon.append(im_s)

#         # s_feat = self.G([s_w_id_fse], s_feat, early_stop=32)


#         alpha = self.source_shape(source)['shape']
#         s_w_shape = self.mapping(alpha)


#         s_style = s_w_id + s_w_shape[:, None, :] + self.latent_avg[None, ...]


#         # t_style, t_feat_ = self.target_encoder.test(img=target, return_latent=True)[-2:]
#         t_style, t_feat = self.target_encoder(t_256)
#         if verbose:
#             # recon.append(self.G([t_style], t_feat, start=7)[0])
#             recon.append(self.G([t_style], new_features=[None] * 7 + [t_feat] + [None] * (17 - 7))[0])
#             # recon.append(im_t)
#         # t_feat = self.G([t_style], t_feat_, early_stop=32)


#         s_mask, s_mask_vis = self.get_mask(source, 'source', verbose)
#         s_mask = s_mask.cuda()
#         t_mask, t_mask_vis = self.get_mask(target, 'target', verbose)
#         t_mask = t_mask.cuda()

#         if verbose:
#             masks.append(s_mask_vis)
#             masks.append(t_mask_vis)

#         # #################################
#         # x = s_mask1[0].cpu() * (F.interpolate(target[0].cpu().unsqueeze(0), 32, mode='bilinear')[0] + 1) / 2
#         # T.ToPILImage()(F.interpolate(x.unsqueeze(0), 1024, mode='bilinear')[0]).save('image.jpg')
#         # #################################


#         s_feat_masked = s_feat * s_mask
#         s_feat_shifted = self.shifter(s_feat_masked)
#         # s_feat_shifted = self.shifter(s_feat)

#         t_feat_masked = t_feat * t_mask

#         delta_feat = self.fuser(torch.cat([s_feat_shifted, t_feat_masked], dim=1))
#         # delta_feat = self.fuser(torch.cat([s_feat_shifted, t_feat], dim=1))
#         feat = t_feat + delta_feat

#         s_style[:, :7] = t_w_id[:, :7]
#         img, _ = self.G([s_style], new_features=[None] * 7 + [feat] + [None] * (17 - 7))
#         # t_style, t_feat = self.target_encoder.test(img=img, return_latent=True)[-2:]# here
#         # return img, t_feat_, t_feat # here
#         if verbose:
#             return img, masks, recon
#         return img


# class Net(nn.Module):
#     def __init__(self, opts):
#         super(Net, self).__init__()

#         self.target_encoder = FSencoder.get_trainer(opts.device)
#         requires_grad(self.target_encoder, False)  # TODO потом удалить
#         #######################################################


#         self.source_identity = Encoder4Editing(50, 'ir_se', opts)
#         ckpt = torch.load(opts.e4e_path)
#         self.source_identity.load_state_dict(get_keys(ckpt, "encoder"))
#         requires_grad(self.source_identity, False)  # TODO потом удалить
#         #######################################################

#         self.source_shape = DECA(opts.deca_path)
#         requires_grad(self.source_shape, False)

#         self.mapping = Mapper()


#         self.G = Generator(1024, 512, 8)

#         if not opts.train_G:
#             requires_grad(self.G, False)
#         # notice that the 8-layer fully connected module is always fixed
#         else:
#             requires_grad(self.G.style, False)
#             requires_grad(self.G.input, False)
#             requires_grad(self.G.conv1, False)
#             requires_grad(self.G.to_rgb1, False)
#             requires_grad(self.G.convs[:6], False)
#             requires_grad(self.G.to_rgbs[:3], False)

#     def get_mask(self, img, mode=None):

#         mask = faceParsing_demo(self.face_parser, img, convert_to_seg12=True, model_name='default').long()

#         if mode == 'target':
#             mask = logical_or_reduce(*[mask == item for item in [0, 4, 6, 8, 10, 11]]).float()
#         else:
#             mask = logical_or_reduce(*[mask == item for item in [1, 2, 3, 5, 7, 9]]).float()


#         mask = F.interpolate(mask.unsqueeze(1), 32, mode='bilinear', align_corners=False)
#         # mask = dilation(mask, torch.ones(3, 3), engine = 'convolution')
#         mask = blur(kernel_size=3, sigma=1)(mask)

#         return mask

#     def forward(self, source, target):
#         """source -- B x 3 x 1024 x 1024
#            target -- B x 3 x 1024 x 1024
#            skip -- B x 3 x 32 x 32"""
#         s_256 = F.interpolate(source, (256, 256), mode='bilinear')
#         t_256 = F.interpolate(target, (256, 256), mode='bilinear')

#         s_w_id, _ = self.source_identity(s_256, True)


#         alpha = self.source_shape(source)['shape']
#         s_w_shape = self.mapping(alpha)


#         s_style = s_w_id + s_w_shape[:, None, :] + self.latent_avg[None, ...]


#         t_style, t_feat_ = self.target_encoder.test(img=target, return_latent=True)[-2:]

#         s_style[:, :7] = t_style[:, :7]

#         img, _ = self.G([s_style], t_feat_)
#         return img


# class Net(nn.Module):
#     def __init__(self, opts):
#         super(Net, self).__init__()
#         # self.target_encoder = Encoder()

#         ########################### new #######################
#         self.target_encoder = FSencoder.get_trainer(opts.device)  # TODO потом удалить
#         # ckpt = torch.load(opts.fse_path)
#         # self.target_encoder.load_state_dict(ckpt)
#         requires_grad(self.target_encoder, False)  # TODO потом удалить
#         #######################################################

#         # self.source_identity = Inverter(opts)
#         # ckpt = torch.load(opts.sfe_inverter_path)
#         # self.source_identity.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
#         # requires_grad(self.source_identity, False)

#         ########################### new #######################
#         self.source_identity = Encoder4Editing(50, 'ir_se', opts)
#         ckpt = torch.load(opts.e4e_path)
#         self.source_identity.load_state_dict(get_keys(ckpt, "encoder"))
#         requires_grad(self.source_identity, False)  # TODO потом удалить
#         #######################################################

#         self.source_shape = DECA(opts.deca_path)
#         requires_grad(self.source_shape, False)

#         self.mapping = Mapper()
#         # requires_grad(self.mapping, False) # here

#         self.G = Generator(1024, 512, 8)

#         if not opts.train_G:
#             requires_grad(self.G, False)
#         # notice that the 8-layer fully connected module is always fixed
#         else:
#             requires_grad(self.G.style, False)
#             requires_grad(self.G.input, False)
#             requires_grad(self.G.conv1, False)
#             requires_grad(self.G.to_rgb1, False)
#             requires_grad(self.G.convs[:6], False)
#             requires_grad(self.G.to_rgbs[:3], False)

#         self.face_parser = FaceParser(seg_ckpt='./pretrained_ckpts/79999_iter.pth', device='cuda:0').eval()
#         requires_grad(self.face_parser, False)

#         self.fuser = Fuser([[1024, 2], [768, 2], [512, 2]])
#         # self.fuser = nn.Conv2d(1024, 512, 3, padding=1, bias=False)
#         self.shifter = Fuser([[512, 2]], 512)
#         # requires_grad(self.shifter, False) # here

#     def get_mask(self, img, mode=None):

#         mask = faceParsing_demo(self.face_parser, img, convert_to_seg12=True, model_name='default').long()

#         if mode == 'target':
#             mask = logical_or_reduce(*[mask == item for item in [0, 4, 6, 8, 10, 11]]).float()
#         else:
#             mask = logical_or_reduce(*[mask == item for item in [1, 2, 3, 5, 7, 9]]).float()


#         mask = F.interpolate(mask.unsqueeze(1), 32, mode='bilinear', align_corners=False)
#         # mask = dilation(mask, torch.ones(3, 3), engine = 'convolution')
#         mask = blur(kernel_size=3, sigma=1)(mask)

#         return mask

#     def forward(self, source, target):
#         """source -- B x 3 x 1024 x 1024
#            target -- B x 3 x 1024 x 1024
#            skip -- B x 3 x 32 x 32"""
#         s_256 = F.interpolate(source, (256, 256), mode='bilinear')
#         t_256 = F.interpolate(target, (256, 256), mode='bilinear')

#         s_w_id, _ = self.source_identity(s_256, True)
#         s_w_id_fse, s_feat = self.target_encoder.test(img=source, return_latent=True)[-2:]
#         s_feat = self.G([s_w_id_fse], s_feat, early_stop=32)


#         ## s_w_id, s_feats = self.target_encoder.test(img=source, return_latent=True)[-2:]


#         alpha = self.source_shape(source)['shape']
#         s_w_shape = self.mapping(alpha)


#         s_style = s_w_id + s_w_shape[:, None, :] + self.latent_avg[None, ...]


#         ## s_feat = s_w_id + s_w_shape[:, None, :]
#         ## t_feat, rgb_image = self.target_encoder(t_256)


#         t_style, t_feat_ = self.target_encoder.test(img=target, return_latent=True)[-2:]
#         t_feat = self.G([t_style], t_feat_, early_stop=32)

#         s_style[:, :7] = t_style[:, :7]

#         s_mask = self.get_mask(source, 'source') # here
#         s_mask = s_mask.cuda()
#         t_mask = self.get_mask(target, 'target')
#         t_mask = t_mask.cuda()

#         # #################################
#         # x = s_mask1[0].cpu() * (F.interpolate(target[0].cpu().unsqueeze(0), 32, mode='bilinear')[0] + 1) / 2
#         # T.ToPILImage()(F.interpolate(x.unsqueeze(0), 1024, mode='bilinear')[0]).save('image.jpg')
#         # #################################


#         s_feat = s_feat * s_mask #here
#         s_feat = self.shifter(s_feat)

#         t_feat = t_feat * t_mask

#         feat = self.fuser(torch.cat([s_feat, t_feat], dim=1)) # here


#         img, _ = self.G([s_style], feat, start=7) # here
#         # t_style, t_feat = self.target_encoder.test(img=img, return_latent=True)[-2:]# here
#         # return img, t_feat_, t_feat # here
#         return img
