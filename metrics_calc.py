import argparse
import os
import warnings

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy.linalg import sqrtm
from torch.autograd import Variable

from metrics import utils, hopenet
from metrics.arcface import iresnet100
from metrics.detect import detect_landmarks
from metrics.FAN import FAN
from utils.torch_utils import tensor2im
import torchvision.transforms as T
import torch.nn.functional as F
from criteria.lpips.lpips import LPIPS

warnings.filterwarnings("ignore")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class EvalMetric(object):
    def __init__(self):
        self._metric = 0
        self._len = 0

    def update(self, **kwargs):
        pass

    def reset(self):
        self._metric = 0
        self._len = 0

    def get(self):
        return self._metric / self._len


class Identity(EvalMetric):
    def __init__(self, device):
        super().__init__()
        self.model = iresnet100(fp16=False)
        self.model.load_state_dict(torch.load('pretrained_ckpts/arcface.pt', map_location=device))
        self.model.to(device).eval()

    def update(self, source, swap):
        source = self.model(
            torch.nn.functional.interpolate(source, [112, 112], mode='bilinear', align_corners=False))
        swap = self.model(
            torch.nn.functional.interpolate(swap, [112, 112], mode='bilinear', align_corners=False))
        self._metric += torch.cosine_similarity(source, swap, dim=1)
        self._len += 1


class PoseMetric(EvalMetric):
    def __init__(self, device):
        super().__init__()
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        self.model.load_state_dict(torch.load('pretrained_ckpts/hopenet_robust_alpha1.pkl', map_location=device))
        self.model.to(device).eval()

    def get_face_points(self, img):
        transformations = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224)])

        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)

        img = transformations(img)

        images = Variable(img)

        yaw, pitch, roll = self.model(images)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

        pitch = pitch_predicted[0]
        yaw = -yaw_predicted[0]
        roll = roll_predicted[0]
        return torch.tensor([pitch.item(), yaw.item(), roll.item()])

    def update(self, target, swap):
        target = self.get_face_points(target)
        swap = self.get_face_points(swap)
        self._metric += (target - swap).pow(2).sum().pow(0.5)
        self._len += 1


class Expression(EvalMetric):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = FAN(4, "False", "False", 98)
        self.setup_model('pretrained_ckpts/WFLW_4HG.pth')
        self._mse = torch.nn.MSELoss(reduction='none')

    def setup_model(self, path_to_model: str):
        checkpoint = torch.load(path_to_model, map_location='cpu')
        if 'state_dict' not in checkpoint:
            self.model.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = self.model.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            self.model.load_state_dict(model_weights)
        self.model.eval().to(self.device)

    def update(self, target, swap):
        swap = torch.nn.functional.interpolate(swap, [256, 256], mode='bilinear', align_corners=False)
        target = torch.nn.functional.interpolate(target, [256, 256], mode='bilinear', align_corners=False)

        swap_lmk, _ = detect_landmarks(swap, self.model, normalize=True)
        target_lmk, _ = detect_landmarks(target, self.model, normalize=True)

        self._metric += torch.norm(swap_lmk - target_lmk, 2)
        self._len += 1


class LPIPS_(EvalMetric):
    def __init__(self, device):
        super().__init__()
        self.model = LPIPS(net_type='alex').to(device).eval()

    def update(self, target, swap):
        for i in range(1):
            loss_lpips_ = self.model(
                    F.adaptive_avg_pool2d(swap, (1024 // 2 ** i, 1024 // 2 ** i)),
                    F.adaptive_avg_pool2d(target, (1024 // 2 ** i, 1024 // 2 ** i))
                )
            self._metric += loss_lpips_.sum()
        self._len += 1
    


@torch.no_grad()
def calc_metrics(net, device):
    identity = Identity(device)
    recon_id = Identity(device)
    pose = PoseMetric(device)
    expression = Expression(device)
    lpips = LPIPS_(device)
    invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5, ]),
                                T.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    # fid = os.popen(f"export PYTHONPATH='.' && python3 -m pytorch_fid {args.target} {args.swap}").read()[6:-2]
    for i, (source_path, target_path) in enumerate(
            zip(os.listdir('celeba/source'), os.listdir('celeba/target'))):
        print(i)
        source = Image.open(os.path.join('celeba/source', source_path)).convert('RGB').resize((1024, 1024))
        target = Image.open(os.path.join('celeba/target', target_path)).convert('RGB').resize((1024, 1024))
        
        source = transforms.ToTensor()(source).to(device)
        source1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(source).unsqueeze(0)
        
        target = transforms.ToTensor()(target).to(device)
        target1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target).unsqueeze(0)
        
        swap1 = net(source1, target1)
        recon1 = net(target1, target1)

        # swap = tensor2im(swap1)
        # swap = transforms.ToTensor()(swap).to(device)

        # recon = tensor2im(recon1)
        # recon = transforms.ToTensor()(recon).to(device)
        
        identity.update(source1, swap1)
        recon_id.update(target1, recon1)
        pose.update(target1, swap1)
        expression.update(target1, swap1)
        lpips.update(target1, swap1)
    return {'cos_id':np.round(identity.get().item(), 2), 'cos_id_recon':np.round(recon_id.get().item(), 2), 'pose':np.round(pose.get().item(), 2), 'expression':np.round(expression.get().item(), 2), 'lpips': np.round(lpips.get().item(), 2)}


# if __name__ == '__main__':
#     args = argparse.ArgumentParser()
#     args.add_argument('--source', type=str)
#     args.add_argument('--target', type=str)
#     args.add_argument('--swap', type=str)
#     args.add_argument('--output', type=str, default='output_metrics.txt')
#     args = args.parse_args()
#     main(args)
