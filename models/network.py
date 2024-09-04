import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.stylegan2.model import EqualLinear
from models.psp.psp_encoders import Inverter, Encoder4Editing, Encoder4Editing_backbone, StyleHead
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
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module


def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)


class Net(nn.Module):
    def __init__(self, opts):
        super(Net, self).__init__()

        ########################### E4E #######################
        self.identity_backbone = Encoder4Editing_backbone(50, 'ir_se', opts)
        ckpt = torch.load('./pretrained_ckpts/e4e_ffhq_encode.pt')
        self.identity_backbone.body.load_state_dict(get_keys(ckpt, "encoder.body"))
        self.identity_backbone.input_layer.load_state_dict(get_keys(ckpt, "encoder.input_layer"))

        #######################################################

        self.source_identity = StyleHead()
        ckpt = torch.load('./pretrained_ckpts/e4e_ffhq_encode.pt')
        self.source_identity.styles.load_state_dict(get_keys(ckpt, "encoder.styles"))
        self.source_identity.latlayer1.load_state_dict(get_keys(ckpt, "encoder.latlayer1"))
        self.source_identity.latlayer2.load_state_dict(get_keys(ckpt, "encoder.latlayer2"))

        self.target_identity = StyleHead()
        ckpt = torch.load('./pretrained_ckpts/e4e_ffhq_encode.pt')
        self.target_identity.styles.load_state_dict(get_keys(ckpt, "encoder.styles"))
        self.target_identity.latlayer1.load_state_dict(get_keys(ckpt, "encoder.latlayer1"))
        self.target_identity.latlayer2.load_state_dict(get_keys(ckpt, "encoder.latlayer2"))

        self.encoder = Encoder4Editing(50, 'ir_se', opts)
        ckpt = torch.load('./pretrained_ckpts/e4e_ffhq_encode.pt')
        self.encoder.load_state_dict(get_keys(ckpt, "encoder"))

        self.encoder.input_layer = Sequential(
            Conv2d(6, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )

        self.G = Generator(1024, 512, 8)

        self.face_parser = FaceParser(seg_ckpt='./pretrained_ckpts/79999_iter.pth', device='cuda:0').eval()

        requires_grad(self.G, False)
        requires_grad(self.identity_backbone, True)
        requires_grad(self.source_identity, True)
        requires_grad(self.target_identity, True)
        requires_grad(self.encoder, True)
        requires_grad(self.face_parser, False)

    def get_mask(self, img, mode=None, verbose=False):

        mask = faceParsing_demo(self.face_parser, img, convert_to_seg12=True, model_name='default').long()

        if mode == 'target':
            mask_ = logical_or_reduce(*[mask == item for item in [0, 4, 8, 10, 11]]).float()
        else:
            mask_ = logical_or_reduce(*[mask == item for item in [1, 2, 3, 5, 6, 7, 9]]).float()

        mask_32 = torch.round(F.interpolate(mask_.unsqueeze(1), 1024, mode='bilinear', align_corners=False))
        mask_dil = dilation(mask_32, torch.ones(3, 3), engine='convolution')
        mask_blur = blur(kernel_size=3, sigma=1)(mask_dil)
        if verbose:
            return mask_blur, mask
        return mask_blur, None

    def forward(self, source, target, verbose=False, return_feat=False, step=5000, is_stylefusion=False):
        """source -- B x 3 x 1024 x 1024
           target -- B x 3 x 1024 x 1024
           skip -- B x 3 x 32 x 32"""
        s_256 = F.interpolate(source, (256, 256), mode='bilinear')
        t_256 = F.interpolate(target, (256, 256), mode='bilinear')

        x, c1, c2, c3 = self.identity_backbone(s_256)
        s_w_id = self.source_identity(x, c1, c2, c3)

        x, c1, c2, c3 = self.identity_backbone(t_256)
        t_w_id = self.target_identity(x, c1, c2, c3)

        s_w_id += self.latent_avg[None, ...]
        t_w_id += self.latent_avg[None, ...]

        s_mask, s_mask_vis = self.get_mask(source, 'source', verbose)
        s_mask = s_mask.cuda()
        t_mask, t_mask_vis = self.get_mask(target, 'target', verbose)
        t_mask = t_mask.cuda()

        m = self.encoder(torch.cat([s_256, t_256], dim=1))
        m = torch.sigmoid(m)

        s_style = m * s_w_id + (1 - m) * t_w_id

        img, _ = self.G([s_style])

        if return_feat:
            return img, s_mask, t_mask
        if verbose:
            return img, None, None
        return img
