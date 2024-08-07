from argparse import Namespace
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, utils

from trainer import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None

opts = Namespace(config='001', pretrained_model_path='pretrained_ckpts//inverter/143_enc.pth',
                 stylegan_model_path=f'pretrained_ckpts/stylegan2/stylegan2-ffhq-config-f.pt',
                 arcface_model_path=f'pretrained_ckpts/auxiliary/iresnet50-7f187506.pth',
                 parsing_model_path=f'pretrained_ckpts/79999_iter.pth', log_path='./logs/',
                 resume=False, checkpoint='', checkpoint_noiser='', multigpu=False, input_path='./test/',
                 save_path='./')

config = yaml.load(open(f'{current_dir}/configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)


def get_trainer(device):
    # Initialize trainer
    trainer = Trainer(config, opts, device)
    trainer.initialize(opts.stylegan_model_path, opts.arcface_model_path, opts.parsing_model_path)

    # state_dict = torch.load(opts.pretrained_model_path)#os.path.join(opts.log_path, opts.config + '/checkpoint.pth'))
    trainer.enc.load_state_dict(torch.load(opts.pretrained_model_path))
    trainer.enc.eval()

    return trainer
