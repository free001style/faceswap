import json
import sys
from options.train import TrainOptions
import warnings
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
import wandb
import torch.distributed as dist
import math
from collections import OrderedDict
import glob
from datasets.dataset import CelebAHQDataset, TO_TENSOR, NORMALIZE, FFHQDataset
from criteria.id_loss import IDLoss
from criteria.lpips.lpips import LPIPS
from criteria.adverserial_loss import AdvDLoss, AdvGLoss, DR1Loss
from criteria.pl_loss.pl_loss import PLLoss
from models.network import Net
from models.stylegan2.model import Discriminator
from utils import torch_utils
from PIL import Image
from metrics_calc import calc_metrics
from face_parsing.face_parsing_demo import vis_parsing_maps

sys.path.append(".")
sys.path.append("..")

warnings.filterwarnings("ignore")
matplotlib.use('Agg')

KEY = "5bcf0f78a0fb42c29b9763fab4cb6dad62918270"

ACCUM = 0.5 ** (32 / (100 * 1000))  # 0.9977843871238888


class Trainer:

    def __init__(self, opts):
        self.opts = opts
        self.global_step = 0

        # distributed training
        if self.opts.dist_train:
            self.num_gpus = torch.cuda.device_count()
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])

            torch.cuda.set_device(self.rank % self.num_gpus)

            dist.init_process_group(
                backend='nccl',
                world_size=self.world_size,
                rank=self.rank,
            )
            self.device = torch.device("cuda", self.local_rank)

        else:
            self.rank = 0  # dummy rank
            self.device = torch.device("cuda", 0)

        self.opts.device = self.device

        # ==== Initialize network ====
        self.net = Net(self.opts)
        # print(self.device)
        self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net = self.net.to(self.device)

        self.net_ema = Net(self.opts).to(self.device).eval()
        torch_utils.accumulate(self.net_ema, self.net, 0)

        if self.opts.train_D:
            self.D = Discriminator(self.opts.out_size).to(self.device).eval()

        if self.opts.dist_train:
            # Wrap the model
            self.net = nn.parallel.DistributedDataParallel(self.net,
                                                           device_ids=[self.local_rank], output_device=self.local_rank,
                                                           broadcast_buffers=False,
                                                           find_unused_parameters=True
                                                           )

            if self.opts.train_D:
                self.D = nn.parallel.DistributedDataParallel(self.D,
                                                             device_ids=[self.local_rank],
                                                             output_device=self.local_rank,
                                                             broadcast_buffers=False,
                                                             find_unused_parameters=True
                                                             )

        # resume
        if self.opts.checkpoint_path is not None:
            ckpt_dict = torch.load(self.opts.checkpoint_path)
            self.global_step = ckpt_dict['current_step'] + 1

            if self.opts.dist_train:
                self.net.module.latent_avg = ckpt_dict['latent_avg'].to(self.device)
                self.net.load_state_dict(ckpt_dict["state_dict"])
                if self.opts.train_D:
                    ## self.D.module.load_state_dict(ckpt_dict["D_state_dict"])
                    # self.D.load_state_dict(ckpt_dict["D_state_dict"])
                    styleGAN2_ckpt = torch.load(self.opts.stylegan_weights)
                    self.D.module.load_state_dict(styleGAN2_ckpt['d'], strict=False)
            else:
                self.net.latent_avg = ckpt_dict['latent_avg'].to(self.device)
                self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"], prefix="module."))
                if self.opts.train_D:
                    self.D.load_state_dict(
                        torch_utils.remove_module_prefix(ckpt_dict["D_state_dict"], prefix="module."))

            print("Resume training at step %d..." % self.global_step)

            # load StyleGAN weights
        else:
            styleGAN2_ckpt = torch.load(self.opts.stylegan_weights)

            if self.opts.dist_train:
                self.net.module.G.load_state_dict(styleGAN2_ckpt['g_ema'], strict=False)
                if self.opts.train_D:
                    if self.opts.out_size == 1024:
                        self.D.module.load_state_dict(styleGAN2_ckpt['d'], strict=False)  # 1024 resolution
                    else:
                        self.custom_load_D_state_dict(self.D.module, styleGAN2_ckpt['d'])  # load partial D
                # avg latent code
                self.net.module.latent_avg = styleGAN2_ckpt['latent_avg'].to(self.device)
                if self.opts.learn_in_w:
                    self.net.module.latent_avg = self.net.module.latent_avg.repeat(1, 1)
                else:
                    self.net.module.latent_avg = self.net.module.latent_avg.repeat(
                        2 * int(math.log(self.opts.out_size, 2)) - 2, 1)
            else:
                self.net.G.load_state_dict(styleGAN2_ckpt['g_ema'], strict=False)
                if self.opts.train_D:
                    if self.opts.out_size == 1024:
                        self.D_swap.load_state_dict(styleGAN2_ckpt['d'], strict=False)  # 1024 resolution
                        self.D_recon.load_state_dict(styleGAN2_ckpt['d'], strict=False)  # 1024 resolution
                    else:
                        self.custom_load_D_state_dict(self.D, styleGAN2_ckpt['d'])  # load partial D
                # avg latent code
                self.net.latent_avg = styleGAN2_ckpt['latent_avg'].to(self.device)
                if self.opts.learn_in_w:
                    self.net.latent_avg = self.net.latent_avg.repeat(1, 1)
                else:
                    self.net.latent_avg = self.net.latent_avg.repeat(2 * int(math.log(self.opts.out_size, 2)) - 2, 1)

            print('Loading pretrained styleGAN2 weights!')

        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.opts.dist_train:
            if self.net.module.latent_avg is None:
                self.net.module.latent_avg = self.net.module.G.mean_latent(int(1e5))[0].detach()
        else:
            if self.net.latent_avg is None:
                self.net.latent_avg = self.net.G.mean_latent(int(1e5))[0].detach()

        if self.opts.recon_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = IDLoss(self.opts).to(self.device).eval()
        if self.opts.pl_lambda > 0:
            self.pl_loss = PLLoss(self.opts).to(self.device).eval()

        self.adv_d_loss = AdvDLoss().to(self.device).eval()
        self.adv_g_loss = AdvGLoss().to(self.device).eval()
        self.d_r1_reg_loss = DR1Loss().to(self.device).eval()

        # Initialize optimizer
        self.optimizer, self.optimizer_D, = self.configure_optimizers()
        # self.optimizer, _ = self.configure_optimizers()  # TODO потом удалить

        # Initialize datasets
        self.train_dataset, self.test_dataset = self.configure_datasets()
        if self.opts.dist_train:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            self.train_dataloader = DataLoader(self.train_dataset,
                                               batch_size=self.opts.batch_size,
                                               num_workers=int(self.opts.workers),
                                               drop_last=True,
                                               pin_memory=True,
                                               sampler=self.train_sampler)
        else:
            self.train_dataloader = DataLoader(self.train_dataset,
                                               batch_size=self.opts.batch_size,
                                               shuffle=True,
                                               num_workers=int(self.opts.workers),
                                               drop_last=True)
        # test set
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(
                                              self.opts.test_workers),
                                          drop_last=False)
        self.log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        # Initialize wandb
        if self.rank == 0:
            wandb.login(key=KEY, relogin=True)
            # if self.opts.checkpoint_path is not None:
            #     wandb.init(project='RobustSwap', entity='free001style', name=opts.exp_dir.strip('exp/'),
            #                config=self.opts,
            #                resume='allow')
            #     code = wandb.Artifact("project-source", type="code")
            #     for path in glob.glob("**/*.py", recursive=True):
            #         if not path.startswith("wandb"):
            #             if os.path.basename(path) != path:
            #                 code.add_dir(
            #                     os.path.dirname(path), name=os.path.dirname(path)
            #                 )
            #             else:
            #                 code.add_file(os.path.basename(path), name=path)
            #     wandb.run.log_artifact(code)
            # else:
            wandb.init(project='RobustSwap', entity='free001style', name=opts.exp_dir.strip('exp/'),
                           config=self.opts)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def custom_load_D_state_dict(self, module, state_dict):
        """Load partial StyleGAN discriminator weights
        Args:
            module (nn.Module): the module to be updated
            state_dict (): styleGAN weights, convs.0 corresponds to 1024 resolution
        """
        local_state = {k: v for k, v in module.named_parameters() if v is not None}

        #
        del local_state["convs.0.0.weight"]
        del local_state["convs.0.1.bias"]

        idx_gap = int(math.log(1024, 2)) - int(math.log(self.opts.out_size, 2))

        new_state_dict = OrderedDict()
        for name, param in local_state.items():
            if name[:5] == "convs":
                layer_idx = int(name[6])
                name_in_pretrained = name[:6] + str(layer_idx + idx_gap) + name[7:]
                new_state_dict[name] = state_dict[name_in_pretrained]
            else:
                new_state_dict[name] = state_dict[name]  # FC

        module.load_state_dict(new_state_dict, strict=False)

    def configure_optimizers(self):
        self.params = list(filter(lambda p: p.requires_grad, list(self.net.parameters())))
        self.params_D = list(
            filter(lambda p: p.requires_grad, list(self.D.parameters()))) if self.opts.train_D else None

        d_reg_ratio = self.opts.d_reg_every / (self.opts.d_reg_every + 1) if self.opts.d_reg_every > 0 else 1

        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(self.params, lr=self.opts.learning_rate)
            optimizer_D = torch.optim.Adam(self.params_D,
                                           lr=self.opts.learning_rate * d_reg_ratio) if self.opts.train_D else None
        else:
            raise ValueError('Only Adam can be used')
        return optimizer, optimizer_D

    def configure_datasets(self):
        if self.opts.dataset_name == "ffhq":
            train_ds = FFHQDataset(dataset_root=self.opts.ffhq_dataset_root,
                                   img_transform=transforms.Compose(
                                       [TO_TENSOR, NORMALIZE]),
                                   fraction=self.opts.ds_frac,
                                   flip_p=self.opts.flip_p)
        else:
            train_ds = CelebAHQDataset(dataset_root=self.opts.celeba_dataset_root, mode="train",
                                       img_transform=transforms.Compose(
                                           [TO_TENSOR, NORMALIZE]),
                                       fraction=self.opts.ds_frac,
                                       flip_p=self.opts.flip_p)

        test_ds = CelebAHQDataset(dataset_root=self.opts.celeba_dataset_root, mode="test",
                                  img_transform=transforms.Compose(
                                      [TO_TENSOR, NORMALIZE]),
                                  fraction=self.opts.ds_frac)
        print(f"Number of training samples: {len(train_ds)}")
        print(f"Number of test samples: {len(test_ds)}")
        return train_ds, test_ds

    def train(self):
        self.net.train()
        if self.opts.train_D:
            self.D.train()

        while self.global_step <= self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                source, target, flag = batch
                if flag.sum() == 0:
                    target[-1, ...] = source[-1, ...]
                    flag[-1] = 1
                # if flag.sum() == source.shape[0]:
                #     target[-1, ...] = source[-2, ...]
                #     flag[-1] = 0
                # if self.global_step % self.opts.same_image_interval == 0:
                #     target = torch.clone(source)

                source = source.to(self.device).float()
                target = target.to(self.device).float()
                flag = flag.to(self.device)
                
                # ============ update D ===============
                if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
                    torch_utils.requires_grad(self.net, False)
                    torch_utils.requires_grad(self.D, True)
                    self.net.module.face_parser.eval()
                    self.net.module.source_identity.eval()
                    self.net.module.target_encoder.eval()
                    swap = self.net(source, target)
                    fake_pred = self.D(swap)
                    real_pred = self.D(target)

                    d_loss = self.adv_d_loss(real_pred, fake_pred)

                    d_loss_dict = {}
                    d_loss_dict["d_loss"] = float(d_loss)
                    d_loss_dict["real_score"] = float(real_pred.mean())
                    d_loss_dict["fake_score"] = float(fake_pred.mean())

                    self.D.zero_grad()
                    d_loss.backward()
                    self.optimizer_D.step()

                    r1_loss = torch.tensor(0.0, device=self.device)
                    # R1 regularization
                
                    if self.opts.d_reg_every != -1 and self.global_step % self.opts.d_reg_every == 0:
                        source.requires_grad = True

                        real_pred = self.D(source)
                        r1_loss = self.d_r1_reg_loss(real_pred, source)

                        self.D.zero_grad()
                        (self.opts.r1_lambda / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]).backward()
                        self.optimizer_D.step()

                    d_loss_dict["r1_loss"] = float(r1_loss)
                    

                # ============ update G ===============
                # self.opts.train_G and self.opts.train_D should be both true or false
#                 if self.opts.train_G and self.opts.train_D:
#                     torch_utils.requires_grad(self.net, True)
#                     torch_utils.requires_grad(self.net.module.G.style, False)  # fix z-to-W mapping of original StyleGAN
#                     torch_utils.requires_grad(self.net.module.G.input, False)
#                     torch_utils.requires_grad(self.net.module.G.conv1, False)
#                     torch_utils.requires_grad(self.net.module.G.to_rgb1, False)
#                     torch_utils.requires_grad(self.net.module.G.convs[:6], False)
#                     torch_utils.requires_grad(self.net.module.G.to_rgbs[:3], False)
#                     torch_utils.requires_grad(self.net.module.source_shape, False)
#                     torch_utils.requires_grad(self.net.module.source_identity, False)
#                     torch_utils.requires_grad(self.net.module.source_identity, False)

#                 # only training Mapping
#                 elif not self.opts.train_G and not self.opts.train_D:
#                     torch_utils.requires_grad(self.net.module.G, False)
#                     torch_utils.requires_grad(self.net.module.source_shape, False)
#                     torch_utils.requires_grad(self.net.module.source_identity, False)
#                     torch_utils.requires_grad(self.net.module.target_encoder, False)  # TODO потом удалить
                
                if self.opts.train_D:
                    torch_utils.requires_grad(self.D, False)

                torch_utils.requires_grad(self.net.module.mapping, True)
                torch_utils.requires_grad(self.net.module.shifter, True)
                torch_utils.requires_grad(self.net.module.fuser, True)
                torch_utils.requires_grad(self.net.module.G, False)
                torch_utils.requires_grad(self.net.module.source_shape, False)
                torch_utils.requires_grad(self.net.module.source_identity, False)
                torch_utils.requires_grad(self.net.module.target_encoder, False)
                torch_utils.requires_grad(self.net.module.face_parser, False)
                self.net.module.face_parser.eval()
                self.net.module.source_identity.eval()
                self.net.module.target_encoder.eval()
                
                
                swap, gt_feat, t_feat, a = self.net(source, target, return_feat=True, step=self.global_step)

                g_loss = torch.tensor(0.0, device=self.device)
#                 if self.opts.train_G:
#                     fake_pred = self.D(swap)

#                     g_loss = self.adv_g_loss(fake_pred)

                fake_pred = self.D(swap)
                g_loss = self.adv_g_loss(fake_pred)
    

                loss_, loss_dict = self.calc_loss(source, target, swap, flag, gt_feat, t_feat, a)
                loss_dict["g_loss"] = float(g_loss)
                # if self.opts.train_G:
                #     overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
                # else:
                #     overall_loss = loss_
                overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
                loss_dict["loss"] = float(overall_loss)

                self.net.zero_grad()
                # overall_loss = torch.autograd.Variable(overall_loss, requires_grad = True)
                overall_loss.backward()
                self.optimizer.step()

                # Logging related
                if self.rank == 0 and (self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 100 == 0)):
                    img_transform = transforms.Compose([TO_TENSOR, NORMALIZE])
                    source11 = Image.open('photos/cil.png').convert('RGB').resize((1024, 1024))
                    source11 = img_transform(source11).unsqueeze(0)
                    source12 = Image.open('photos/harry.png').convert('RGB').resize((1024, 1024))
                    source12 = img_transform(source12).unsqueeze(0)
                    source13 = Image.open('photos/liza.png').convert('RGB').resize((1024, 1024))
                    source13 = img_transform(source13).unsqueeze(0)
                    source14 = Image.open('photos/rose.png').convert('RGB').resize((1024, 1024))
                    source14 = img_transform(source14).unsqueeze(0)
                    source15 = Image.open('photos/mask.png').convert('RGB').resize((1024, 1024))
                    source15 = img_transform(source15).unsqueeze(0)
                    source1 = torch.cat([source11, source12, source13, source14, source15])

                    
                    target11 = Image.open('photos/dua.png').convert('RGB').resize((1024, 1024))
                    target11 = img_transform(target11).unsqueeze(0)
                    target12 = Image.open('photos/timon.png').convert('RGB').resize((1024, 1024))
                    target12 = img_transform(target12).unsqueeze(0)
                    target13 = Image.open('photos/tent.png').convert('RGB').resize((1024, 1024))
                    target13 = img_transform(target13).unsqueeze(0)
                    target14 = Image.open('photos/rose.png').convert('RGB').resize((1024, 1024))
                    target14 = img_transform(target14).unsqueeze(0)
                    target15 = Image.open('photos/girl.png').convert('RGB').resize((1024, 1024))
                    target15 = img_transform(target15).unsqueeze(0)
                    target1 = torch.cat([target11, target12, target13, target14, target15])

                    source1 = source1.to(self.device).float()
                    target1 = target1.to(self.device).float()
                    
                    swap1, mask, recon = self.net(source1, target1, verbose=True)
                    imgs = self.parse_images(source1, target1, swap1, None, None)
                    self.log_images('images/train/faces', imgs1_data=imgs)
                    if self.global_step < 100:
                        imgs = self.parse_images(source1, target1, swap1, mask, recon)
                        self.log_images('images/train/mask', imgs1_data=imgs, mask=True)
                    

                if self.rank == 0 and (self.global_step % self.opts.board_interval == 0):
                    self.print_metrics(loss_dict, prefix='train')
                    if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
                        self.print_metrics(d_loss_dict, prefix='train')

                    self.log_metrics(loss_dict, prefix='train')
                    if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
                        self.log_metrics(d_loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                # # save model
                if self.rank == 0 and (
                        self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps):
                    val_loss_dict = self.validate()
                # if self.rank == 0 and (
                #         val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss)):
                #     self.best_val_loss = val_loss_dict['loss']
                #     self.checkpoint_me(val_loss_dict)

                if self.rank == 0 and (
                        self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps):
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict)
                    else:
                        self.checkpoint_me(loss_dict)

                self.global_step += 1
                # ema
                if self.opts.dist_train:
                    torch_utils.accumulate(self.net_ema, self.net.module, ACCUM)
                else:
                    torch_utils.accumulate(self.net_ema, self.net, ACCUM)

        if self.rank == 0:
            print('OMG, finished training!')
            wandb.finish()

    def calc_loss(self, source, target, swap, flag, gt_feat, t_feat, a):
        print(self.global_step)
        loss_dict = {}
        # loss = 0.0
        loss = torch.tensor(0.0, device=self.device)
        
        feat_loss = nn.MSELoss()(gt_feat, t_feat)
        loss_dict['feat_loss'] = float(feat_loss)
        loss += (1 - a) * feat_loss

        if self.opts.id_lambda > 0:
            loss_id = self.id_loss(swap, source)
            
            loss_id = loss_id.to(self.device)
            
            loss_id = torch.sum(loss_id * (1 - flag)) / torch.sum(1 - flag)
            

            loss_dict['loss_id'] = float(loss_id)
            loss += loss_id * self.opts.id_lambda
        if self.opts.recon_lambda > 0:
            loss_lpips = 0
            for i in range(3):
                loss_lpips_ = self.lpips_loss(
                    F.adaptive_avg_pool2d(swap * flag[:, None, None, None], (1024 // 2 ** i, 1024 // 2 ** i)),
                    F.adaptive_avg_pool2d(target * flag[:, None, None, None], (1024 // 2 ** i, 1024 // 2 ** i))
                )
                loss_lpips += loss_lpips_.sum()
            loss_l2 = ((swap - target) ** 2).mean(dim=(1, 2, 3)) ** 0.5
            loss_l2 = (loss_l2 * flag).sum()
            recon = (loss_lpips + loss_l2) / torch.sum(flag)
            loss_dict['recon'] = float(recon)
            loss += recon * self.opts.recon_lambda
        if self.opts.pl_lambda > 0:
            loss_pl = self.pl_loss(source, target, swap)
            loss_dict['pl_loss'] = float(loss_pl)
            loss += loss_pl * self.opts.pl_lambda
        print(loss_dict)
        return loss, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            # continue
            wandb.log({f'{prefix}/{key}': value}, step=self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    @staticmethod
    def parse_images(source, target, swap, mask, recon, display_count=10):
        im_data = []

        display_count = min(display_count, len(source))
        for i in range(display_count):
            if mask is not None:
                source_ = torch_utils.tensor2im(source[i])
                target_ = torch_utils.tensor2im(target[i])
                cur_im_data = {
                    'source_recon' : torch_utils.tensor2im(recon[0][i]),
                    'source_mask' : vis_parsing_maps(source_, mask[0][i].detach().numpy()),
                    'target_recon' : torch_utils.tensor2im(recon[1][i]),
                    'target_mask' : vis_parsing_maps(target_, mask[1][i].detach().numpy()),
                }
            else:
                cur_im_data = {
                    'source': torch_utils.tensor2im(source[i]),
                    'target': torch_utils.tensor2im(target[i]),
                    'swap': torch_utils.tensor2im(swap[i]),}
            im_data.append(cur_im_data)
        return im_data

    def log_images(self, name, imgs1_data, mask=False):
        fig = torch_utils.vis_faces(imgs1_data, mask)
        step = self.global_step
        path = os.path.join(self.log_dir, name, f'{step:06d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        if 'train' in name:
            # self.opts.exp_dir.strip("exp/")}/train/{step:06d}.jpg
            if mask:
                wandb.log({"mask and reconstruction": wandb.Image(path)},
                          step=self.global_step)
            else:
                wandb.log({"image_test": wandb.Image(path)},
                          step=self.global_step)
        elif 'test' in name:
            # f'{self.opts.exp_dir.strip("exp/")}/test/{step:06d}.jpg'
            wandb.log({'image_val': wandb.Image(path)},
                      step=self.global_step)

    def checkpoint_me(self, loss_dict):
        save_name = f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts),
            'state_dict_ema': self.net_ema.state_dict(),
            'current_step': self.global_step
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.module.latent_avg if self.opts.dist_train else self.net.latent_avg

        if self.opts.train_D:
            save_dict['D_state_dict'] = self.D.state_dict()
            save_dict['D_state_dict'] = self.D.state_dict()
        return save_dict

    def validate(self):
        self.net.eval()
        # if self.opts.train_D:
        #     self.D.eval()
        # agg_loss_dict = []
        # os.makedirs(os.path.dirname(f'validate/{self.global_step}'))
        # os.makedirs(os.path.dirname(f'validate/{self.global_step}/source'))
        # os.makedirs(os.path.dirname(f'validate/{self.global_step}/target'))
        # os.makedirs(os.path.dirname(f'validate/{self.global_step}/swap'))
        # os.makedirs(f'validate/{self.global_step}', exist_ok=True)
        # os.makedirs(f'validate/{self.global_step}/source', exist_ok=True)
        # os.makedirs(f'validate/{self.global_step}/target', exist_ok=True)
        # os.makedirs(f'validate/{self.global_step}/swap', exist_ok=True)
        # for batch_idx, batch in enumerate(self.test_dataloader):
        #     source, target = batch
        #     # if batch_idx % self.opts.same_image_interval == 0:
        #     #     target = torch.clone(source)

        #     with torch.no_grad():
        #         source = source.to(self.device).float()
        #         target = target.to(self.device).float()
        #         swap = self.net(source, target)
        #         loss_dict = {}
        #         loss_ = 0.0
        #         g_loss = torch.tensor(0.0, device=self.device)
        #         if self.opts.train_G:
        #             fake_pred = self.D(swap)
        #             g_loss = self.adv_g_loss(fake_pred)

        #         # loss_, loss_dict = self.calc_loss(source, target, swap, flag, batch_idx)
        #         loss_dict["g_loss"] = float(g_loss)

        #         if self.opts.train_G:
        #             overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
        #         else:
        #             overall_loss = loss_
        #         loss_dict["loss"] = float(overall_loss)

        #     agg_loss_dict.append(loss_dict)

        #     # For first step just do sanity test on small amount of data
        #     if self.global_step == 0 and batch_idx >= 4:
        #         self.net.train()
        #         if self.opts.train_D:
        #             self.D.train()
        #         return None  # Do not log, inaccurate in first batch
            
        #     for i in range(swap.shape[0]):
        #         torch_utils.tensor2im(source[i]).save(f'validate/{self.global_step}/source/{(batch_idx + 1) * i}.jpg')
        #         torch_utils.tensor2im(target[i]).save(f'validate/{self.global_step}/target/{(batch_idx + 1) * i}.jpg')
        #         torch_utils.tensor2im(swap[i]).save(f'validate/{self.global_step}/swap/{(batch_idx + 1) * i}.jpg')
                

        # loss_dict = torch_utils.aggregate_loss_dict(agg_loss_dict)
        loss_dict = calc_metrics(self.net, self.device)
        self.log_metrics(loss_dict, prefix='metrics')
        self.print_metrics(loss_dict, prefix='metrics')

        self.net.train()
        if self.opts.train_D:
            self.D.train()
        return loss_dict


def main():
    opts = TrainOptions().parse()
    os.makedirs(opts.exp_dir, exist_ok=True)

    opts_dict = vars(opts)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    trainer = Trainer(opts)
    trainer.train()


if __name__ == '__main__':
    main()


# import json
# import sys
# from options.train import TrainOptions
# import warnings
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch import nn
# import torch
# import os
# import matplotlib
# import matplotlib.pyplot as plt
# import wandb
# import torch.distributed as dist
# import math
# from collections import OrderedDict
# import glob
# from datasets.dataset import CelebAHQDataset, TO_TENSOR, NORMALIZE, FFHQDataset
# from criteria.id_loss import IDLoss
# from criteria.lpips.lpips import LPIPS
# from criteria.adverserial_loss import AdvDLoss, AdvGLoss, DR1Loss
# from criteria.pl_loss.pl_loss import PLLoss
# from models.network import Net
# from models.stylegan2.model import Discriminator
# from utils import torch_utils
# from PIL import Image
# from metrics_calc import calc_metrics
# from face_parsing.face_parsing_demo import vis_parsing_maps

# sys.path.append(".")
# sys.path.append("..")

# warnings.filterwarnings("ignore")
# matplotlib.use('Agg')

# KEY = "5bcf0f78a0fb42c29b9763fab4cb6dad62918270"

# ACCUM = 0.5 ** (32 / (100 * 1000))  # 0.9977843871238888


# class Trainer:

#     def __init__(self, opts):
#         self.opts = opts
#         self.global_step = 0

#         # distributed training
#         if self.opts.dist_train:
#             self.num_gpus = torch.cuda.device_count()
#             self.rank = int(os.environ["RANK"])
#             self.world_size = int(os.environ["WORLD_SIZE"])
#             self.local_rank = int(os.environ["LOCAL_RANK"])

#             torch.cuda.set_device(self.rank % self.num_gpus)

#             dist.init_process_group(
#                 backend='nccl',
#                 world_size=self.world_size,
#                 rank=self.rank,
#             )
#             self.device = torch.device("cuda", self.local_rank)

#         else:
#             self.rank = 0  # dummy rank
#             self.device = torch.device("cuda", 0)

#         self.opts.device = self.device

#         # ==== Initialize network ====
#         self.net = Net(self.opts)
#         # print(self.device)
#         self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
#         self.net = self.net.to(self.device)

#         self.net_ema = Net(self.opts).to(self.device).eval()
#         torch_utils.accumulate(self.net_ema, self.net, 0)

#         if self.opts.train_D:
#             self.D = Discriminator(self.opts.out_size).to(self.device).eval()

#         if self.opts.dist_train:
#             # Wrap the model
#             self.net = nn.parallel.DistributedDataParallel(self.net,
#                                                            device_ids=[self.local_rank], output_device=self.local_rank,
#                                                            broadcast_buffers=False,
#                                                            find_unused_parameters=True
#                                                            )

#             if self.opts.train_D:
#                 self.D = nn.parallel.DistributedDataParallel(self.D,
#                                                              device_ids=[self.local_rank],
#                                                              output_device=self.local_rank,
#                                                              broadcast_buffers=False,
#                                                              find_unused_parameters=True
#                                                              )

#         # resume
#         if self.opts.checkpoint_path is not None:
#             ckpt_dict = torch.load(self.opts.checkpoint_path)
#             self.global_step = ckpt_dict['current_step'] + 1

#             if self.opts.dist_train:
#                 self.net.module.latent_avg = ckpt_dict['latent_avg'].to(self.device)
#                 self.net.load_state_dict(ckpt_dict["state_dict"])
#                 if self.opts.train_D:
#                     # self.D.module.load_state_dict(ckpt_dict["D_state_dict"])
#                     self.D.load_state_dict(ckpt_dict["D_state_dict"])
#             else:
#                 self.net.latent_avg = ckpt_dict['latent_avg'].to(self.device)
#                 self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"], prefix="module."))
#                 if self.opts.train_D:
#                     self.D.load_state_dict(
#                         torch_utils.remove_module_prefix(ckpt_dict["D_state_dict"], prefix="module."))

#             print("Resume training at step %d..." % self.global_step)

#             # load StyleGAN weights
#         else:
#             styleGAN2_ckpt = torch.load(self.opts.stylegan_weights)

#             if self.opts.dist_train:
#                 self.net.module.G.load_state_dict(styleGAN2_ckpt['g_ema'], strict=False)
#                 if self.opts.train_D:
#                     if self.opts.out_size == 1024:
#                         self.D.module.load_state_dict(styleGAN2_ckpt['d'], strict=False)  # 1024 resolution
#                     else:
#                         self.custom_load_D_state_dict(self.D.module, styleGAN2_ckpt['d'])  # load partial D
#                 # avg latent code
#                 self.net.module.latent_avg = styleGAN2_ckpt['latent_avg'].to(self.device)
#                 if self.opts.learn_in_w:
#                     self.net.module.latent_avg = self.net.module.latent_avg.repeat(1, 1)
#                 else:
#                     self.net.module.latent_avg = self.net.module.latent_avg.repeat(
#                         2 * int(math.log(self.opts.out_size, 2)) - 2, 1)
#             else:
#                 self.net.G.load_state_dict(styleGAN2_ckpt['g_ema'], strict=False)
#                 if self.opts.train_D:
#                     if self.opts.out_size == 1024:
#                         self.D_swap.load_state_dict(styleGAN2_ckpt['d'], strict=False)  # 1024 resolution
#                         self.D_recon.load_state_dict(styleGAN2_ckpt['d'], strict=False)  # 1024 resolution
#                     else:
#                         self.custom_load_D_state_dict(self.D, styleGAN2_ckpt['d'])  # load partial D
#                 # avg latent code
#                 self.net.latent_avg = styleGAN2_ckpt['latent_avg'].to(self.device)
#                 if self.opts.learn_in_w:
#                     self.net.latent_avg = self.net.latent_avg.repeat(1, 1)
#                 else:
#                     self.net.latent_avg = self.net.latent_avg.repeat(2 * int(math.log(self.opts.out_size, 2)) - 2, 1)

#             print('Loading pretrained styleGAN2 weights!')

#         # Estimate latent_avg via dense sampling if latent_avg is not available
#         if self.opts.dist_train:
#             if self.net.module.latent_avg is None:
#                 self.net.module.latent_avg = self.net.module.G.mean_latent(int(1e5))[0].detach()
#         else:
#             if self.net.latent_avg is None:
#                 self.net.latent_avg = self.net.G.mean_latent(int(1e5))[0].detach()

#         if self.opts.recon_lambda > 0:
#             self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
#         if self.opts.id_lambda > 0:
#             self.id_loss = IDLoss(self.opts).to(self.device).eval()
#         if self.opts.pl_lambda > 0:
#             self.pl_loss = PLLoss(self.opts).to(self.device).eval()

#         self.adv_d_loss = AdvDLoss().to(self.device).eval()
#         self.adv_g_loss = AdvGLoss().to(self.device).eval()
#         self.d_r1_reg_loss = DR1Loss().to(self.device).eval()

#         # Initialize optimizer
#         self.optimizer, self.optimizer_D, = self.configure_optimizers()
#         # self.optimizer, _ = self.configure_optimizers()  # TODO потом удалить

#         # Initialize datasets
#         self.train_dataset, self.test_dataset = self.configure_datasets()
#         if self.opts.dist_train:
#             self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
#             self.train_dataloader = DataLoader(self.train_dataset,
#                                                batch_size=self.opts.batch_size,
#                                                num_workers=int(self.opts.workers),
#                                                drop_last=True,
#                                                pin_memory=True,
#                                                sampler=self.train_sampler)
#         else:
#             self.train_dataloader = DataLoader(self.train_dataset,
#                                                batch_size=self.opts.batch_size,
#                                                shuffle=True,
#                                                num_workers=int(self.opts.workers),
#                                                drop_last=True)
#         # test set
#         self.test_dataloader = DataLoader(self.test_dataset,
#                                           batch_size=self.opts.test_batch_size,
#                                           shuffle=False,
#                                           num_workers=int(
#                                               self.opts.test_workers),
#                                           drop_last=False)
#         self.log_dir = os.path.join(opts.exp_dir, 'logs')
#         os.makedirs(self.log_dir, exist_ok=True)
#         # Initialize wandb
#         if self.rank == 0:
#             wandb.login(key=KEY, relogin=True)
#             # if self.opts.checkpoint_path is not None:
#             #     wandb.init(project='RobustSwap', entity='free001style', name=opts.exp_dir.strip('exp/'),
#             #                config=self.opts,
#             #                resume='allow')
#             #     code = wandb.Artifact("project-source", type="code")
#             #     for path in glob.glob("**/*.py", recursive=True):
#             #         if not path.startswith("wandb"):
#             #             if os.path.basename(path) != path:
#             #                 code.add_dir(
#             #                     os.path.dirname(path), name=os.path.dirname(path)
#             #                 )
#             #             else:
#             #                 code.add_file(os.path.basename(path), name=path)
#             #     wandb.run.log_artifact(code)
#             # else:
#             wandb.init(project='RobustSwap', entity='free001style', name=opts.exp_dir.strip('exp/'),
#                            config=self.opts)

#         # Initialize checkpoint dir
#         self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
#         os.makedirs(self.checkpoint_dir, exist_ok=True)
#         self.best_val_loss = None
#         if self.opts.save_interval is None:
#             self.opts.save_interval = self.opts.max_steps

#     def custom_load_D_state_dict(self, module, state_dict):
#         """Load partial StyleGAN discriminator weights
#         Args:
#             module (nn.Module): the module to be updated
#             state_dict (): styleGAN weights, convs.0 corresponds to 1024 resolution
#         """
#         local_state = {k: v for k, v in module.named_parameters() if v is not None}

#         #
#         del local_state["convs.0.0.weight"]
#         del local_state["convs.0.1.bias"]

#         idx_gap = int(math.log(1024, 2)) - int(math.log(self.opts.out_size, 2))

#         new_state_dict = OrderedDict()
#         for name, param in local_state.items():
#             if name[:5] == "convs":
#                 layer_idx = int(name[6])
#                 name_in_pretrained = name[:6] + str(layer_idx + idx_gap) + name[7:]
#                 new_state_dict[name] = state_dict[name_in_pretrained]
#             else:
#                 new_state_dict[name] = state_dict[name]  # FC

#         module.load_state_dict(new_state_dict, strict=False)

#     def configure_optimizers(self):
#         self.params = list(filter(lambda p: p.requires_grad, list(self.net.parameters())))
#         self.params_D = list(
#             filter(lambda p: p.requires_grad, list(self.D.parameters()))) if self.opts.train_D else None

#         d_reg_ratio = self.opts.d_reg_every / (self.opts.d_reg_every + 1) if self.opts.d_reg_every > 0 else 1

#         if self.opts.optim_name == 'adam':
#             optimizer = torch.optim.Adam(self.params, lr=self.opts.learning_rate)
#             optimizer_D = torch.optim.Adam(self.params_D,
#                                            lr=self.opts.learning_rate * d_reg_ratio) if self.opts.train_D else None
#         else:
#             raise ValueError('Only Adam can be used')
#         return optimizer, optimizer_D

#     def configure_datasets(self):
#         if self.opts.dataset_name == "ffhq":
#             train_ds = FFHQDataset(dataset_root=self.opts.ffhq_dataset_root,
#                                    img_transform=transforms.Compose(
#                                        [TO_TENSOR, NORMALIZE]),
#                                    fraction=self.opts.ds_frac,
#                                    flip_p=self.opts.flip_p)
#         else:
#             train_ds = CelebAHQDataset(dataset_root=self.opts.celeba_dataset_root, mode="train",
#                                        img_transform=transforms.Compose(
#                                            [TO_TENSOR, NORMALIZE]),
#                                        fraction=self.opts.ds_frac,
#                                        flip_p=self.opts.flip_p)

#         test_ds = CelebAHQDataset(dataset_root=self.opts.celeba_dataset_root, mode="test",
#                                   img_transform=transforms.Compose(
#                                       [TO_TENSOR, NORMALIZE]),
#                                   fraction=self.opts.ds_frac)
#         print(f"Number of training samples: {len(train_ds)}")
#         print(f"Number of test samples: {len(test_ds)}")
#         return train_ds, test_ds

#     def train(self):
#         self.net.train()
#         if self.opts.train_D:
#             self.D.train()

#         while self.global_step <= self.opts.max_steps:
#             for batch_idx, batch in enumerate(self.train_dataloader):
#                 source, target, flag = batch
#                 if flag.sum() == 0:
#                     target[-1, ...] = source[-1, ...]
#                     flag[-1] = 1
#                 # if flag.sum() == source.shape[0]:
#                 #     target[-1, ...] = source[-2, ...]
#                 #     flag[-1] = 0
#                 # if self.global_step % self.opts.same_image_interval == 0:
#                 #     target = torch.clone(source)

#                 source = source.to(self.device).float()
#                 target = target.to(self.device).float()
#                 flag = flag.to(self.device)
                
#                 # ============ update D ===============
#                 # if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
#                 #     torch_utils.requires_grad(self.net, False)
#                 #     torch_utils.requires_grad(self.D, True)
#                 #     self.net.module.face_parser.eval()
#                 #     self.net.module.source_identity.eval()
#                 #     self.net.module.target_encoder.eval()
#                 #     swap = self.net(source, target)
#                 #     fake_pred = self.D(swap)
#                 #     real_pred = self.D(target)

#                 #     d_loss = self.adv_d_loss(real_pred, fake_pred)

#                 #     d_loss_dict = {}
#                 #     d_loss_dict["d_loss"] = float(d_loss)
#                 #     d_loss_dict["real_score"] = float(real_pred.mean())
#                 #     d_loss_dict["fake_score"] = float(fake_pred.mean())

#                 #     self.D.zero_grad()
#                 #     d_loss.backward()
#                 #     self.optimizer_D.step()

#                 #     r1_loss = torch.tensor(0.0, device=self.device)
#                 #     # R1 regularization
                
#                 #     if self.opts.d_reg_every != -1 and self.global_step % self.opts.d_reg_every == 0:
#                 #         source.requires_grad = True

#                 #         real_pred = self.D(source)
#                 #         r1_loss = self.d_r1_reg_loss(real_pred, source)

#                 #         self.D.zero_grad()
#                 #         (self.opts.r1_lambda / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]).backward()
#                 #         self.optimizer_D.step()

#                 #     d_loss_dict["r1_loss"] = float(r1_loss)
                    

#                 # ============ update G ===============
#                 # self.opts.train_G and self.opts.train_D should be both true or false
# #                 if self.opts.train_G and self.opts.train_D:
# #                     torch_utils.requires_grad(self.net, True)
# #                     torch_utils.requires_grad(self.net.module.G.style, False)  # fix z-to-W mapping of original StyleGAN
# #                     torch_utils.requires_grad(self.net.module.G.input, False)
# #                     torch_utils.requires_grad(self.net.module.G.conv1, False)
# #                     torch_utils.requires_grad(self.net.module.G.to_rgb1, False)
# #                     torch_utils.requires_grad(self.net.module.G.convs[:6], False)
# #                     torch_utils.requires_grad(self.net.module.G.to_rgbs[:3], False)
# #                     torch_utils.requires_grad(self.net.module.source_shape, False)
# #                     torch_utils.requires_grad(self.net.module.source_identity, False)
# #                     torch_utils.requires_grad(self.net.module.source_identity, False)

# #                 # only training Mapping
# #                 elif not self.opts.train_G and not self.opts.train_D:
# #                     torch_utils.requires_grad(self.net.module.G, False)
# #                     torch_utils.requires_grad(self.net.module.source_shape, False)
# #                     torch_utils.requires_grad(self.net.module.source_identity, False)
# #                     torch_utils.requires_grad(self.net.module.target_encoder, False)  # TODO потом удалить
                
#                 # if self.opts.train_D:
#                 #     torch_utils.requires_grad(self.D, False)
#                 torch_utils.requires_grad(self.D, False)

#                 torch_utils.requires_grad(self.net.module.mapping, True)
#                 torch_utils.requires_grad(self.net.module.shifter, True)
#                 torch_utils.requires_grad(self.net.module.fuser, True)
#                 torch_utils.requires_grad(self.net.module.G, False)
#                 torch_utils.requires_grad(self.net.module.source_shape, False)
#                 torch_utils.requires_grad(self.net.module.source_identity, False)
#                 torch_utils.requires_grad(self.net.module.target_encoder, False)
#                 torch_utils.requires_grad(self.net.module.face_parser, False)
#                 self.net.module.face_parser.eval()
#                 self.net.module.source_identity.eval()
#                 self.net.module.target_encoder.eval()
                
                
#                 swap = self.net(source, target)

#                 g_loss = torch.tensor(0.0, device=self.device)
# #                 if self.opts.train_G:
# #                     fake_pred = self.D(swap)

# #                     g_loss = self.adv_g_loss(fake_pred)

#                 fake_pred = self.D(swap)
#                 g_loss = self.adv_g_loss(fake_pred)
    

#                 loss_, loss_dict = self.calc_loss(source, target, swap, flag)
#                 loss_dict["g_loss"] = float(g_loss)
#                 # if self.opts.train_G:
#                 #     overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
#                 # else:
#                 #     overall_loss = loss_
#                 overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
#                 loss_dict["loss"] = float(overall_loss)

#                 self.net.zero_grad()
#                 # overall_loss = torch.autograd.Variable(overall_loss, requires_grad = True)
#                 overall_loss.backward()
#                 self.optimizer.step()

#                 # Logging related
#                 if self.rank == 0 and (self.global_step % self.opts.image_interval == 0 or (
#                         self.global_step < 1000 and self.global_step % 100 == 0)):
#                     img_transform = transforms.Compose([TO_TENSOR, NORMALIZE])
#                     source11 = Image.open('photos/cil.png').convert('RGB').resize((1024, 1024))
#                     source11 = img_transform(source11).unsqueeze(0)
#                     source12 = Image.open('photos/harry.png').convert('RGB').resize((1024, 1024))
#                     source12 = img_transform(source12).unsqueeze(0)
#                     source13 = Image.open('photos/liza.png').convert('RGB').resize((1024, 1024))
#                     source13 = img_transform(source13).unsqueeze(0)
#                     source14 = Image.open('photos/rose.png').convert('RGB').resize((1024, 1024))
#                     source14 = img_transform(source14).unsqueeze(0)
#                     source1 = torch.cat([source11, source12, source13, source14])

                    
#                     target11 = Image.open('photos/dua.png').convert('RGB').resize((1024, 1024))
#                     target11 = img_transform(target11).unsqueeze(0)
#                     target12 = Image.open('photos/timon.png').convert('RGB').resize((1024, 1024))
#                     target12 = img_transform(target12).unsqueeze(0)
#                     target13 = Image.open('photos/tent.png').convert('RGB').resize((1024, 1024))
#                     target13 = img_transform(target13).unsqueeze(0)
#                     target14 = Image.open('photos/rose.png').convert('RGB').resize((1024, 1024))
#                     target14 = img_transform(target14).unsqueeze(0)
#                     target1 = torch.cat([target11, target12, target13, target14])

#                     source1 = source1.to(self.device).float()
#                     target1 = target1.to(self.device).float()
                    
#                     swap1, mask, recon = self.net(source1, target1, True)
#                     imgs = self.parse_images(source1, target1, swap1, None, None)
#                     self.log_images('images/train/faces', imgs1_data=imgs)
#                     if self.global_step < 100:
#                         imgs = self.parse_images(source1, target1, swap1, mask, recon)
#                         self.log_images('images/train/mask', imgs1_data=imgs, mask=True)
                    

#                 if self.rank == 0 and (self.global_step % self.opts.board_interval == 0):
#                     self.print_metrics(loss_dict, prefix='train')
#                     # if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
#                     #     self.print_metrics(d_loss_dict, prefix='train')

#                     self.log_metrics(loss_dict, prefix='train')
#                     # if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
#                     #     self.log_metrics(d_loss_dict, prefix='train')

#                 # Validation related
#                 val_loss_dict = None
#                 # # save model
#                 if self.rank == 0 and (
#                         self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps):
#                     val_loss_dict = self.validate()
#                 # if self.rank == 0 and (
#                 #         val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss)):
#                 #     self.best_val_loss = val_loss_dict['loss']
#                 #     self.checkpoint_me(val_loss_dict)

#                 if self.rank == 0 and (
#                         self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps):
#                     if val_loss_dict is not None:
#                         self.checkpoint_me(val_loss_dict)
#                     else:
#                         self.checkpoint_me(loss_dict)

#                 self.global_step += 1
#                 # ema
#                 if self.opts.dist_train:
#                     torch_utils.accumulate(self.net_ema, self.net.module, ACCUM)
#                 else:
#                     torch_utils.accumulate(self.net_ema, self.net, ACCUM)

#         if self.rank == 0:
#             print('OMG, finished training!')
#             wandb.finish()

#     def calc_loss(self, source, target, swap, flag):
#         print(self.global_step)
#         loss_dict = {}
#         # loss = 0.0
#         loss = torch.tensor(0.0, device=self.device)

#         if self.opts.id_lambda > 0:
#             loss_id = self.id_loss(swap, source)
            
#             loss_id = loss_id.to(self.device)
            
#             loss_id = torch.sum(loss_id * (1 - flag)) / torch.sum(1 - flag)
            

#             loss_dict['loss_id'] = float(loss_id)
#             loss += loss_id * self.opts.id_lambda
#         if self.opts.recon_lambda > 0:
#             loss_lpips = 0
#             for i in range(3):
#                 loss_lpips_ = self.lpips_loss(
#                     F.adaptive_avg_pool2d(swap * flag[:, None, None, None], (1024 // 2 ** i, 1024 // 2 ** i)),
#                     F.adaptive_avg_pool2d(target * flag[:, None, None, None], (1024 // 2 ** i, 1024 // 2 ** i))
#                 )
#                 loss_lpips += loss_lpips_.sum()
#             loss_l2 = ((swap - target) ** 2).mean(dim=(1, 2, 3)) ** 0.5
#             loss_l2 = (loss_l2 * flag).sum()
#             recon = (loss_lpips + loss_l2) / torch.sum(flag)
#             loss_dict['recon'] = float(recon)
#             loss += recon * self.opts.recon_lambda
#         if self.opts.pl_lambda > 0:
#             loss_pl = self.pl_loss(source, target, swap)
#             loss_dict['pl_loss'] = float(loss_pl)
#             loss += loss_pl * self.opts.pl_lambda
#         print(loss_dict)
#         return loss, loss_dict

#     def log_metrics(self, metrics_dict, prefix):
#         for key, value in metrics_dict.items():
#             # continue
#             wandb.log({f'{prefix}/{key}': value}, step=self.global_step)

#     def print_metrics(self, metrics_dict, prefix):
#         print(f'Metrics for {prefix}, step {self.global_step}')
#         for key, value in metrics_dict.items():
#             print(f'\t{key} = ', value)

#     @staticmethod
#     def parse_images(source, target, swap, mask, recon, display_count=4):
#         im_data = []

#         display_count = min(display_count, len(source))
#         for i in range(display_count):
#             if mask is not None:
#                 source_ = torch_utils.tensor2im(source[i])
#                 target_ = torch_utils.tensor2im(target[i])
#                 cur_im_data = {
#                     'source_recon' : torch_utils.tensor2im(recon[0][i]),
#                     'source_mask' : vis_parsing_maps(source_, mask[0][i].detach().numpy()),
#                     'target_recon' : torch_utils.tensor2im(recon[1][i]),
#                     'target_mask' : vis_parsing_maps(target_, mask[1][i].detach().numpy()),
#                 }
#             else:
#                 cur_im_data = {
#                     'source': torch_utils.tensor2im(source[i]),
#                     'target': torch_utils.tensor2im(target[i]),
#                     'swap': torch_utils.tensor2im(swap[i]),}
#             im_data.append(cur_im_data)
#         return im_data

#     def log_images(self, name, imgs1_data, mask=False):
#         fig = torch_utils.vis_faces(imgs1_data, mask)
#         step = self.global_step
#         path = os.path.join(self.log_dir, name, f'{step:06d}.jpg')
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         fig.savefig(path)
#         plt.close(fig)
#         if 'train' in name:
#             # self.opts.exp_dir.strip("exp/")}/train/{step:06d}.jpg
#             if mask:
#                 wandb.log({"mask and reconstruction": wandb.Image(path)},
#                           step=self.global_step)
#             else:
#                 wandb.log({"image_test": wandb.Image(path)},
#                           step=self.global_step)
#         elif 'test' in name:
#             # f'{self.opts.exp_dir.strip("exp/")}/test/{step:06d}.jpg'
#             wandb.log({'image_val': wandb.Image(path)},
#                       step=self.global_step)

#     def checkpoint_me(self, loss_dict):
#         save_name = f'iteration_{self.global_step}.pt'
#         save_dict = self.__get_save_dict()
#         checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
#         torch.save(save_dict, checkpoint_path)
#         with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
#             f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

#     def __get_save_dict(self):
#         save_dict = {
#             'state_dict': self.net.state_dict(),
#             'opts': vars(self.opts),
#             'state_dict_ema': self.net_ema.state_dict(),
#             'current_step': self.global_step
#         }
#         # save the latent avg in state_dict for inference if truncation of w was used during training
#         if self.opts.start_from_latent_avg:
#             save_dict['latent_avg'] = self.net.module.latent_avg if self.opts.dist_train else self.net.latent_avg

#         if self.opts.train_D:
#             save_dict['D_state_dict'] = self.D.state_dict()
#             save_dict['D_state_dict'] = self.D.state_dict()
#         return save_dict

#     def validate(self):
#         self.net.eval()
#         # if self.opts.train_D:
#         #     self.D.eval()
#         # agg_loss_dict = []
#         # os.makedirs(os.path.dirname(f'validate/{self.global_step}'))
#         # os.makedirs(os.path.dirname(f'validate/{self.global_step}/source'))
#         # os.makedirs(os.path.dirname(f'validate/{self.global_step}/target'))
#         # os.makedirs(os.path.dirname(f'validate/{self.global_step}/swap'))
#         # os.makedirs(f'validate/{self.global_step}', exist_ok=True)
#         # os.makedirs(f'validate/{self.global_step}/source', exist_ok=True)
#         # os.makedirs(f'validate/{self.global_step}/target', exist_ok=True)
#         # os.makedirs(f'validate/{self.global_step}/swap', exist_ok=True)
#         # for batch_idx, batch in enumerate(self.test_dataloader):
#         #     source, target = batch
#         #     # if batch_idx % self.opts.same_image_interval == 0:
#         #     #     target = torch.clone(source)

#         #     with torch.no_grad():
#         #         source = source.to(self.device).float()
#         #         target = target.to(self.device).float()
#         #         swap = self.net(source, target)
#         #         loss_dict = {}
#         #         loss_ = 0.0
#         #         g_loss = torch.tensor(0.0, device=self.device)
#         #         if self.opts.train_G:
#         #             fake_pred = self.D(swap)
#         #             g_loss = self.adv_g_loss(fake_pred)

#         #         # loss_, loss_dict = self.calc_loss(source, target, swap, flag, batch_idx)
#         #         loss_dict["g_loss"] = float(g_loss)

#         #         if self.opts.train_G:
#         #             overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
#         #         else:
#         #             overall_loss = loss_
#         #         loss_dict["loss"] = float(overall_loss)

#         #     agg_loss_dict.append(loss_dict)

#         #     # For first step just do sanity test on small amount of data
#         #     if self.global_step == 0 and batch_idx >= 4:
#         #         self.net.train()
#         #         if self.opts.train_D:
#         #             self.D.train()
#         #         return None  # Do not log, inaccurate in first batch
            
#         #     for i in range(swap.shape[0]):
#         #         torch_utils.tensor2im(source[i]).save(f'validate/{self.global_step}/source/{(batch_idx + 1) * i}.jpg')
#         #         torch_utils.tensor2im(target[i]).save(f'validate/{self.global_step}/target/{(batch_idx + 1) * i}.jpg')
#         #         torch_utils.tensor2im(swap[i]).save(f'validate/{self.global_step}/swap/{(batch_idx + 1) * i}.jpg')
                

#         # loss_dict = torch_utils.aggregate_loss_dict(agg_loss_dict)
#         loss_dict = calc_metrics(self.net, self.device)
#         self.log_metrics(loss_dict, prefix='metrics')
#         self.print_metrics(loss_dict, prefix='metrics')

#         self.net.train()
#         if self.opts.train_D:
#             self.D.train()
#         return loss_dict


# def main():
#     opts = TrainOptions().parse()
#     os.makedirs(opts.exp_dir, exist_ok=True)

#     opts_dict = vars(opts)
#     with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
#         json.dump(opts_dict, f, indent=4, sort_keys=True)

#     trainer = Trainer(opts)
#     trainer.train()


# if __name__ == '__main__':
#     main()
# import json
# import sys
# from options.train import TrainOptions
# import warnings
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch import nn
# import torch
# import os
# import matplotlib
# import matplotlib.pyplot as plt
# import wandb
# import torch.distributed as dist
# import math
# from collections import OrderedDict
# import glob
# from datasets.dataset import CelebAHQDataset, TO_TENSOR, NORMALIZE, FFHQDataset
# from criteria.id_loss import IDLoss
# from criteria.lpips.lpips import LPIPS
# from criteria.adverserial_loss import AdvDLoss, AdvGLoss, DR1Loss
# from criteria.pl_loss.pl_loss import PLLoss
# from models.network import Net
# from models.stylegan2.model import Discriminator
# from utils import torch_utils
# from PIL import Image
# from metrics_calc import calc_metrics
# from face_parsing.face_parsing_demo import vis_parsing_maps

# sys.path.append(".")
# sys.path.append("..")

# warnings.filterwarnings("ignore")
# matplotlib.use('Agg')

# KEY = "5bcf0f78a0fb42c29b9763fab4cb6dad62918270"

# ACCUM = 0.5 ** (32 / (100 * 1000))  # 0.9977843871238888


# class Trainer:

#     def __init__(self, opts):
#         self.opts = opts
#         self.global_step = 0

#         # distributed training
#         if self.opts.dist_train:
#             self.num_gpus = torch.cuda.device_count()
#             self.rank = int(os.environ["RANK"])
#             self.world_size = int(os.environ["WORLD_SIZE"])
#             self.local_rank = int(os.environ["LOCAL_RANK"])

#             torch.cuda.set_device(self.rank % self.num_gpus)

#             dist.init_process_group(
#                 backend='nccl',
#                 world_size=self.world_size,
#                 rank=self.rank,
#             )
#             self.device = torch.device("cuda", self.local_rank)

#         else:
#             self.rank = 0  # dummy rank
#             self.device = torch.device("cuda", 0)

#         self.opts.device = self.device

#         # ==== Initialize network ====
#         self.net = Net(self.opts)
#         # print(self.device)
#         self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
#         self.net = self.net.to(self.device)

#         self.net_ema = Net(self.opts).to(self.device).eval()
#         torch_utils.accumulate(self.net_ema, self.net, 0)

#         if self.opts.train_D:
#             self.D_swap = Discriminator(self.opts.out_size).to(self.device).eval()
#             self.D_recon = Discriminator(self.opts.out_size).to(self.device).eval()

#         if self.opts.dist_train:
#             # Wrap the model
#             self.net = nn.parallel.DistributedDataParallel(self.net,
#                                                            device_ids=[self.local_rank], output_device=self.local_rank,
#                                                            broadcast_buffers=False,
#                                                            find_unused_parameters=True
#                                                            )

#             if self.opts.train_D:
#                 self.D_swap = nn.parallel.DistributedDataParallel(self.D_swap,
#                                                              device_ids=[self.local_rank],
#                                                              output_device=self.local_rank,
#                                                              broadcast_buffers=False,
#                                                              find_unused_parameters=True
#                                                              )
#                 self.D_recon = nn.parallel.DistributedDataParallel(self.D_recon,
#                                                              device_ids=[self.local_rank],
#                                                              output_device=self.local_rank,
#                                                              broadcast_buffers=False,
#                                                              find_unused_parameters=True
#                                                              )

#         # resume
#         if self.opts.checkpoint_path is not None:
#             ckpt_dict = torch.load(self.opts.checkpoint_path)
#             self.global_step = ckpt_dict['current_step'] + 1

#             if self.opts.dist_train:
#                 self.net.module.latent_avg = ckpt_dict['latent_avg'].to(self.device)
#                 self.net.load_state_dict(ckpt_dict["state_dict"])
#                 if self.opts.train_D:
#                     # self.D.module.load_state_dict(ckpt_dict["D_state_dict"])
#                     self.D_swap.load_state_dict(ckpt_dict["D_swap_state_dict"])
#                     self.D_recon.load_state_dict(ckpt_dict["D_recon_state_dict"])
#             else:
#                 self.net.latent_avg = ckpt_dict['latent_avg'].to(self.device)
#                 self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"], prefix="module."))
#                 if self.opts.train_D:
#                     self.D.load_state_dict(
#                         torch_utils.remove_module_prefix(ckpt_dict["D_state_dict"], prefix="module."))

#             print("Resume training at step %d..." % self.global_step)

#             # load StyleGAN weights
#         else:
#             styleGAN2_ckpt = torch.load(self.opts.stylegan_weights)

#             if self.opts.dist_train:
#                 self.net.module.G.load_state_dict(styleGAN2_ckpt['g_ema'], strict=False)
#                 if self.opts.train_D:
#                     if self.opts.out_size == 1024:
#                         self.D_swap.module.load_state_dict(styleGAN2_ckpt['d'], strict=False)  # 1024 resolution
#                         self.D_recon.module.load_state_dict(styleGAN2_ckpt['d'], strict=False)
#                     else:
#                         self.custom_load_D_state_dict(self.D.module, styleGAN2_ckpt['d'])  # load partial D
#                 # avg latent code
#                 self.net.module.latent_avg = styleGAN2_ckpt['latent_avg'].to(self.device)
#                 if self.opts.learn_in_w:
#                     self.net.module.latent_avg = self.net.module.latent_avg.repeat(1, 1)
#                 else:
#                     self.net.module.latent_avg = self.net.module.latent_avg.repeat(
#                         2 * int(math.log(self.opts.out_size, 2)) - 2, 1)
#             else:
#                 self.net.G.load_state_dict(styleGAN2_ckpt['g_ema'], strict=False)
#                 if self.opts.train_D:
#                     if self.opts.out_size == 1024:
#                         self.D_swap.load_state_dict(styleGAN2_ckpt['d'], strict=False)  # 1024 resolution
#                         self.D_recon.load_state_dict(styleGAN2_ckpt['d'], strict=False)  # 1024 resolution
#                     else:
#                         self.custom_load_D_state_dict(self.D, styleGAN2_ckpt['d'])  # load partial D
#                 # avg latent code
#                 self.net.latent_avg = styleGAN2_ckpt['latent_avg'].to(self.device)
#                 if self.opts.learn_in_w:
#                     self.net.latent_avg = self.net.latent_avg.repeat(1, 1)
#                 else:
#                     self.net.latent_avg = self.net.latent_avg.repeat(2 * int(math.log(self.opts.out_size, 2)) - 2, 1)

#             print('Loading pretrained styleGAN2 weights!')

#         # Estimate latent_avg via dense sampling if latent_avg is not available
#         if self.opts.dist_train:
#             if self.net.module.latent_avg is None:
#                 self.net.module.latent_avg = self.net.module.G.mean_latent(int(1e5))[0].detach()
#         else:
#             if self.net.latent_avg is None:
#                 self.net.latent_avg = self.net.G.mean_latent(int(1e5))[0].detach()

#         if self.opts.recon_lambda > 0:
#             self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
#         if self.opts.id_lambda > 0:
#             self.id_loss = IDLoss(self.opts).to(self.device).eval()
#         if self.opts.pl_lambda > 0:
#             self.pl_loss = PLLoss(self.opts).to(self.device).eval()

#         self.adv_d_loss = AdvDLoss().to(self.device).eval()
#         self.adv_g_loss = AdvGLoss().to(self.device).eval()
#         self.d_r1_reg_loss = DR1Loss().to(self.device).eval()

#         # Initialize optimizer
#         self.optimizer, self.optimizer_D_swap, self.optimizer_D_recon = self.configure_optimizers()
#         # self.optimizer, _ = self.configure_optimizers()  # TODO потом удалить

#         # Initialize datasets
#         self.train_dataset, self.test_dataset = self.configure_datasets()
#         if self.opts.dist_train:
#             self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
#             self.train_dataloader = DataLoader(self.train_dataset,
#                                                batch_size=self.opts.batch_size,
#                                                num_workers=int(self.opts.workers),
#                                                drop_last=True,
#                                                pin_memory=True,
#                                                sampler=self.train_sampler)
#         else:
#             self.train_dataloader = DataLoader(self.train_dataset,
#                                                batch_size=self.opts.batch_size,
#                                                shuffle=True,
#                                                num_workers=int(self.opts.workers),
#                                                drop_last=True)
#         # test set
#         self.test_dataloader = DataLoader(self.test_dataset,
#                                           batch_size=self.opts.test_batch_size,
#                                           shuffle=False,
#                                           num_workers=int(
#                                               self.opts.test_workers),
#                                           drop_last=False)
#         self.log_dir = os.path.join(opts.exp_dir, 'logs')
#         os.makedirs(self.log_dir, exist_ok=True)
#         # Initialize wandb
#         if self.rank == 0:
#             wandb.login(key=KEY, relogin=True)
#             # if self.opts.checkpoint_path is not None:
#             #     wandb.init(project='RobustSwap', entity='free001style', name=opts.exp_dir.strip('exp/'),
#             #                config=self.opts,
#             #                resume='allow')
#             #     code = wandb.Artifact("project-source", type="code")
#             #     for path in glob.glob("**/*.py", recursive=True):
#             #         if not path.startswith("wandb"):
#             #             if os.path.basename(path) != path:
#             #                 code.add_dir(
#             #                     os.path.dirname(path), name=os.path.dirname(path)
#             #                 )
#             #             else:
#             #                 code.add_file(os.path.basename(path), name=path)
#             #     wandb.run.log_artifact(code)
#             # else:
#             wandb.init(project='RobustSwap', entity='free001style', name=opts.exp_dir.strip('exp/'),
#                            config=self.opts)

#         # Initialize checkpoint dir
#         self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
#         os.makedirs(self.checkpoint_dir, exist_ok=True)
#         self.best_val_loss = None
#         if self.opts.save_interval is None:
#             self.opts.save_interval = self.opts.max_steps

#     def custom_load_D_state_dict(self, module, state_dict):
#         """Load partial StyleGAN discriminator weights
#         Args:
#             module (nn.Module): the module to be updated
#             state_dict (): styleGAN weights, convs.0 corresponds to 1024 resolution
#         """
#         local_state = {k: v for k, v in module.named_parameters() if v is not None}

#         #
#         del local_state["convs.0.0.weight"]
#         del local_state["convs.0.1.bias"]

#         idx_gap = int(math.log(1024, 2)) - int(math.log(self.opts.out_size, 2))

#         new_state_dict = OrderedDict()
#         for name, param in local_state.items():
#             if name[:5] == "convs":
#                 layer_idx = int(name[6])
#                 name_in_pretrained = name[:6] + str(layer_idx + idx_gap) + name[7:]
#                 new_state_dict[name] = state_dict[name_in_pretrained]
#             else:
#                 new_state_dict[name] = state_dict[name]  # FC

#         module.load_state_dict(new_state_dict, strict=False)

#     def configure_optimizers(self):
#         self.params = list(filter(lambda p: p.requires_grad, list(self.net.parameters())))
#         self.params_D = list(
#             filter(lambda p: p.requires_grad, list(self.D_swap.parameters()))) if self.opts.train_D else None

#         d_reg_ratio = self.opts.d_reg_every / (self.opts.d_reg_every + 1) if self.opts.d_reg_every > 0 else 1

#         if self.opts.optim_name == 'adam':
#             optimizer = torch.optim.Adam(self.params, lr=self.opts.learning_rate)
#             optimizer_D_swap = torch.optim.Adam(self.params_D,
#                                            lr=self.opts.learning_rate * d_reg_ratio) if self.opts.train_D else None
#             optimizer_D_recon = torch.optim.Adam(self.params_D,
#                                            lr=self.opts.learning_rate * d_reg_ratio) if self.opts.train_D else None
#         else:
#             raise ValueError('Only Adam can be used')
#         return optimizer, optimizer_D_swap, optimizer_D_recon

#     def configure_datasets(self):
#         if self.opts.dataset_name == "ffhq":
#             train_ds = FFHQDataset(dataset_root=self.opts.ffhq_dataset_root,
#                                    img_transform=transforms.Compose(
#                                        [TO_TENSOR, NORMALIZE]),
#                                    fraction=self.opts.ds_frac,
#                                    flip_p=self.opts.flip_p)
#         else:
#             train_ds = CelebAHQDataset(dataset_root=self.opts.celeba_dataset_root, mode="train",
#                                        img_transform=transforms.Compose(
#                                            [TO_TENSOR, NORMALIZE]),
#                                        fraction=self.opts.ds_frac,
#                                        flip_p=self.opts.flip_p)

#         test_ds = CelebAHQDataset(dataset_root=self.opts.celeba_dataset_root, mode="test",
#                                   img_transform=transforms.Compose(
#                                       [TO_TENSOR, NORMALIZE]),
#                                   fraction=self.opts.ds_frac)
#         print(f"Number of training samples: {len(train_ds)}")
#         print(f"Number of test samples: {len(test_ds)}")
#         return train_ds, test_ds

#     def train(self):
#         self.net.train()
#         if self.opts.train_D:
#             self.D_swap.train()
#             self.D_recon.train()

#         while self.global_step <= self.opts.max_steps:
#             for batch_idx, batch in enumerate(self.train_dataloader):
#                 source, target, flag = batch
#                 if flag.sum() == 0:
#                     target[-1, ...] = source[-1, ...]
#                     flag[-1] = 1
#                 # if flag.sum() == source.shape[0]:
#                 #     target[-1, ...] = source[-2, ...]
#                 #     flag[-1] = 0
#                 # if self.global_step % self.opts.same_image_interval == 0:
#                 #     target = torch.clone(source)

#                 source = source.to(self.device).float()
#                 target = target.to(self.device).float()
#                 flag = flag.to(self.device)
                
#                 # ============ update D ===============
#                 if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
#                     torch_utils.requires_grad(self.net, False)
#                     torch_utils.requires_grad(self.D_swap, True)
#                     torch_utils.requires_grad(self.D_recon, True)
#                     self.net.module.face_parser.eval()
#                     self.net.module.target_encoder.eval()
#                     self.net.module.source_identity.eval()
#                     swap = self.net(source, target)
#                     fake_pred_swap = self.D_swap(swap[:-1])
#                     real_pred_swap = self.D_swap(target[:-1])
                    
#                     fake_pred_recon = self.D_recon(swap[-1].unsqueeze(0))
#                     real_pred_recon = self.D_recon(target[-1].unsqueeze(0))

#                     d_loss_swap = self.adv_d_loss(real_pred_swap, fake_pred_swap)
#                     d_loss_recon = self.adv_d_loss(real_pred_recon, fake_pred_recon)

#                     d_loss_dict = {}
#                     d_loss_dict["d_loss_swap"] = float(d_loss_swap)
#                     d_loss_dict["real_score_swap"] = float(real_pred_swap.mean())
#                     d_loss_dict["fake_score_swap"] = float(fake_pred_swap.mean())

#                     d_loss_dict["d_loss_recon"] = float(d_loss_recon)
#                     d_loss_dict["real_score_recon"] = float(real_pred_recon.mean())
#                     d_loss_dict["fake_score_recon"] = float(fake_pred_recon.mean())

#                     self.D_swap.zero_grad()
#                     d_loss_swap.backward()
#                     self.optimizer_D_swap.step()

#                     self.D_recon.zero_grad()
#                     d_loss_recon.backward()
#                     self.optimizer_D_recon.step()

#                     r1_loss_swap = torch.tensor(0.0, device=self.device)
#                     r1_loss_recon = torch.tensor(0.0, device=self.device)
#                     # R1 regularization
                
#                     if self.opts.d_reg_every != -1 and self.global_step % self.opts.d_reg_every == 0:
#                         source_swap = source[:-1]
#                         source_recon = source[-1].unsqueeze(0)
#                         source_swap.requires_grad = True
#                         source_recon.requires_grad = True

#                         real_pred_swap = self.D_swap(source_swap)
#                         real_pred_recon = self.D_recon(source_recon)
#                         r1_loss_swap = self.d_r1_reg_loss(real_pred_swap, source_swap)
#                         r1_loss_recon = self.d_r1_reg_loss(real_pred_recon, source_recon)

#                         self.D_swap.zero_grad()
#                         (self.opts.r1_lambda / 2 * r1_loss_swap * self.opts.d_reg_every + 0 * real_pred_swap[0]).backward()
#                         self.optimizer_D_swap.step()
                        
#                         self.D_recon.zero_grad()
#                         (self.opts.r1_lambda / 2 * r1_loss_recon * self.opts.d_reg_every + 0 * real_pred_recon[0]).backward()
#                         self.optimizer_D_recon.step()

#                     d_loss_dict["r1_loss_swap"] = float(r1_loss_swap)
#                     d_loss_dict["r1_loss_recon"] = float(r1_loss_recon)
                    

#                 # ============ update G ===============
#                 # self.opts.train_G and self.opts.train_D should be both true or false
# #                 if self.opts.train_G and self.opts.train_D:
# #                     torch_utils.requires_grad(self.net, True)
# #                     torch_utils.requires_grad(self.net.module.G.style, False)  # fix z-to-W mapping of original StyleGAN
# #                     torch_utils.requires_grad(self.net.module.G.input, False)
# #                     torch_utils.requires_grad(self.net.module.G.conv1, False)
# #                     torch_utils.requires_grad(self.net.module.G.to_rgb1, False)
# #                     torch_utils.requires_grad(self.net.module.G.convs[:6], False)
# #                     torch_utils.requires_grad(self.net.module.G.to_rgbs[:3], False)
# #                     torch_utils.requires_grad(self.net.module.source_shape, False)
# #                     torch_utils.requires_grad(self.net.module.source_identity, False)
# #                     torch_utils.requires_grad(self.net.module.source_identity, False)

# #                 # only training Mapping
# #                 elif not self.opts.train_G and not self.opts.train_D:
# #                     torch_utils.requires_grad(self.net.module.G, False)
# #                     torch_utils.requires_grad(self.net.module.source_shape, False)
# #                     torch_utils.requires_grad(self.net.module.source_identity, False)
# #                     torch_utils.requires_grad(self.net.module.target_encoder, False)  # TODO потом удалить
                
#                 if self.opts.train_D:
#                     torch_utils.requires_grad(self.D_swap, False)
#                     torch_utils.requires_grad(self.D_recon, False)

#                 torch_utils.requires_grad(self.net.module.mapping, True)
#                 torch_utils.requires_grad(self.net.module.shifter, True)
#                 torch_utils.requires_grad(self.net.module.fuser, True)
#                 torch_utils.requires_grad(self.net.module.G, False)
#                 torch_utils.requires_grad(self.net.module.source_shape, False)
#                 torch_utils.requires_grad(self.net.module.source_identity, False)
#                 torch_utils.requires_grad(self.net.module.target_encoder, False)
#                 torch_utils.requires_grad(self.net.module.face_parser, False)
#                 self.net.module.face_parser.eval()
#                 self.net.module.target_encoder.eval()
#                 self.net.module.source_identity.eval()
                
#                 swap = self.net(source, target)

#                 g_loss = torch.tensor(0.0, device=self.device)
# #                 if self.opts.train_G:
# #                     fake_pred = self.D(swap)

# #                     g_loss = self.adv_g_loss(fake_pred)

#                 fake_pred = self.D_swap(swap)
#                 fake_pred[-1] = self.D_recon(swap[-1].unsqueeze(0))[0]
#                 g_loss = self.adv_g_loss(fake_pred)
    

#                 loss_, loss_dict = self.calc_loss(source, target, swap, flag, self.global_step)
#                 loss_dict["g_loss"] = float(g_loss)
#                 # if self.opts.train_G:
#                 #     overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
#                 # else:
#                 #     overall_loss = loss_
#                 overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
#                 loss_dict["loss"] = float(overall_loss)

#                 self.net.zero_grad()
#                 # overall_loss = torch.autograd.Variable(overall_loss, requires_grad = True)
#                 overall_loss.backward()
#                 self.optimizer.step()

#                 # Logging related
#                 if self.rank == 0 and (self.global_step % self.opts.image_interval == 0 or (
#                         self.global_step < 1000 and self.global_step % 100 == 0)):
#                     img_transform = transforms.Compose([TO_TENSOR, NORMALIZE])
#                     source11 = Image.open('photos/cil.png').convert('RGB').resize((1024, 1024))
#                     source11 = img_transform(source11).unsqueeze(0)
#                     source12 = Image.open('photos/harry.png').convert('RGB').resize((1024, 1024))
#                     source12 = img_transform(source12).unsqueeze(0)
#                     source13 = Image.open('photos/liza.png').convert('RGB').resize((1024, 1024))
#                     source13 = img_transform(source13).unsqueeze(0)
#                     source14 = Image.open('photos/rose.png').convert('RGB').resize((1024, 1024))
#                     source14 = img_transform(source14).unsqueeze(0)

                    
#                     target11 = Image.open('photos/dua.png').convert('RGB').resize((1024, 1024))
#                     target11 = img_transform(target11).unsqueeze(0)
#                     target12 = Image.open('photos/timon.png').convert('RGB').resize((1024, 1024))
#                     target12 = img_transform(target12).unsqueeze(0)
#                     target13 = Image.open('photos/tent.png').convert('RGB').resize((1024, 1024))
#                     target13 = img_transform(target13).unsqueeze(0)
#                     target14 = Image.open('photos/rose.png').convert('RGB').resize((1024, 1024))
#                     target14 = img_transform(target14).unsqueeze(0)

                    
#                     # target11 = Image.open('photos/cil.png').convert('RGB').resize((1024, 1024))
#                     # target11 = img_transform(target11).unsqueeze(0)
#                     # target12 = Image.open('photos/harry.png').convert('RGB').resize((1024, 1024))
#                     # target12 = img_transform(target12).unsqueeze(0)
#                     # target13 = Image.open('photos/liza.png').convert('RGB').resize((1024, 1024))
#                     # target13 = img_transform(target13).unsqueeze(0)
#                     # target14 = Image.open('photos/rose.png').convert('RGB').resize((1024, 1024))
#                     # target14 = img_transform(target14).unsqueeze(0)
#                     source1 = torch.cat([source11, source12, source13, source14])
#                     target1 = torch.cat([target11, target12, target13, target14])

#                     source1 = source1.to(self.device).float()
#                     target1 = target1.to(self.device).float()

                    
#                     swap1, mask, recon = self.net(source1, target1, True)
#                     imgs = self.parse_images(source1, target1, swap1, None, None)
#                     self.log_images('images/train/faces', imgs1_data=imgs)
#                     if self.global_step < 100:
#                         imgs = self.parse_images(source1, target1, swap1, mask, recon)
#                         self.log_images('images/train/mask', imgs1_data=imgs, mask=True)
                    

#                 if self.rank == 0 and (self.global_step % self.opts.board_interval == 0):
#                     self.print_metrics(loss_dict, prefix='train')
#                     if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
#                         self.print_metrics(d_loss_dict, prefix='train')

#                     self.log_metrics(loss_dict, prefix='train')
#                     if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
#                         self.log_metrics(d_loss_dict, prefix='train')

#                 # Validation related
#                 val_loss_dict = None
#                 # # save model
#                 if self.rank == 0 and (
#                         self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps):
#                     val_loss_dict = self.validate()
#                 # if self.rank == 0 and (
#                 #         val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss)):
#                 #     self.best_val_loss = val_loss_dict['loss']
#                 #     self.checkpoint_me(val_loss_dict)

#                 if self.rank == 0 and (
#                         self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps):
#                     if val_loss_dict is not None:
#                         self.checkpoint_me(val_loss_dict)
#                     else:
#                         self.checkpoint_me(loss_dict)

#                 self.global_step += 1
#                 # ema
#                 if self.opts.dist_train:
#                     torch_utils.accumulate(self.net_ema, self.net.module, ACCUM)
#                 else:
#                     torch_utils.accumulate(self.net_ema, self.net, ACCUM)

#         if self.rank == 0:
#             print('OMG, finished training!')
#             wandb.finish()

#     def calc_loss(self, source, target, swap, flag, step):
#         print(step)
#         loss_dict = {}
#         # loss = 0.0
#         loss = torch.tensor(0.0, device=self.device)

#         if self.opts.id_lambda > 0:
#             loss_id = self.id_loss(swap, source)
            
#             loss_id = loss_id.to(self.device)
            
#             loss_id = torch.sum(loss_id * (1 - flag)) / torch.sum(1 - flag)
            

#             loss_dict['loss_id'] = float(loss_id)
#             loss += loss_id * self.opts.id_lambda
#         if self.opts.recon_lambda > 0:
#             loss_lpips = 0
#             for i in range(3):
#                 loss_lpips_ = self.lpips_loss(
#                     F.adaptive_avg_pool2d(swap * flag[:, None, None, None], (1024 // 2 ** i, 1024 // 2 ** i)),
#                     F.adaptive_avg_pool2d(target * flag[:, None, None, None], (1024 // 2 ** i, 1024 // 2 ** i))
#                 )
#                 loss_lpips += loss_lpips_.sum()
#             loss_l2 = ((swap - target) ** 2).mean(dim=(1, 2, 3)) ** 0.5
#             loss_l2 = (loss_l2 * flag).sum()
#             recon = (loss_lpips + loss_l2) / torch.sum(flag)
#             loss_dict['recon'] = float(recon)
#             loss += recon * self.opts.recon_lambda
#         if self.opts.pl_lambda > 0:
#             loss_pl = self.pl_loss(source, target, swap)
#             loss_dict['pl_loss'] = float(loss_pl)
#             loss += loss_pl * self.opts.pl_lambda
#         print(loss_dict)
#         return loss, loss_dict

#     def log_metrics(self, metrics_dict, prefix):
#         for key, value in metrics_dict.items():
#             # continue
#             wandb.log({f'{prefix}/{key}': value}, step=self.global_step)

#     def print_metrics(self, metrics_dict, prefix):
#         print(f'Metrics for {prefix}, step {self.global_step}')
#         for key, value in metrics_dict.items():
#             print(f'\t{key} = ', value)

#     @staticmethod
#     def parse_images(source, target, swap, mask, recon, display_count=4):
#         im_data = []

#         display_count = min(display_count, len(source))
#         for i in range(display_count):
#             if mask is not None:
#                 source_ = torch_utils.tensor2im(source[i])
#                 target_ = torch_utils.tensor2im(target[i])
#                 cur_im_data = {
#                     'source_recon' : torch_utils.tensor2im(recon[0][i]),
#                     'source_mask' : vis_parsing_maps(source_, mask[0][i].detach().numpy()),
#                     'target_recon' : torch_utils.tensor2im(recon[1][i]),
#                     'target_mask' : vis_parsing_maps(target_, mask[1][i].detach().numpy()),
#                 }
#             else:
#                 cur_im_data = {
#                     'source': torch_utils.tensor2im(source[i]),
#                     'target': torch_utils.tensor2im(target[i]),
#                     'swap': torch_utils.tensor2im(swap[i]),}
#             im_data.append(cur_im_data)
#         return im_data

#     def log_images(self, name, imgs1_data, mask=False):
#         fig = torch_utils.vis_faces(imgs1_data, mask)
#         step = self.global_step
#         path = os.path.join(self.log_dir, name, f'{step:06d}.jpg')
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         fig.savefig(path)
#         plt.close(fig)
#         if 'train' in name:
#             # self.opts.exp_dir.strip("exp/")}/train/{step:06d}.jpg
#             if mask:
#                 wandb.log({"mask and reconstruction": wandb.Image(path)},
#                           step=self.global_step)
#             else:
#                 wandb.log({"image_test": wandb.Image(path)},
#                           step=self.global_step)
#         elif 'test' in name:
#             # f'{self.opts.exp_dir.strip("exp/")}/test/{step:06d}.jpg'
#             wandb.log({'image_val': wandb.Image(path)},
#                       step=self.global_step)

#     def checkpoint_me(self, loss_dict):
#         save_name = f'iteration_{self.global_step}.pt'
#         save_dict = self.__get_save_dict()
#         checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
#         torch.save(save_dict, checkpoint_path)
#         with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
#             f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

#     def __get_save_dict(self):
#         save_dict = {
#             'state_dict': self.net.state_dict(),
#             'opts': vars(self.opts),
#             'state_dict_ema': self.net_ema.state_dict(),
#             'current_step': self.global_step
#         }
#         # save the latent avg in state_dict for inference if truncation of w was used during training
#         if self.opts.start_from_latent_avg:
#             save_dict['latent_avg'] = self.net.module.latent_avg if self.opts.dist_train else self.net.latent_avg

#         if self.opts.train_D:
#             save_dict['D_swap_state_dict'] = self.D_swap.state_dict()
#             save_dict['D_recon_state_dict'] = self.D_recon.state_dict()
#         return save_dict

#     def validate(self):
#         self.net.eval()
#         # if self.opts.train_D:
#         #     self.D.eval()
#         # agg_loss_dict = []
#         # os.makedirs(os.path.dirname(f'validate/{self.global_step}'))
#         # os.makedirs(os.path.dirname(f'validate/{self.global_step}/source'))
#         # os.makedirs(os.path.dirname(f'validate/{self.global_step}/target'))
#         # os.makedirs(os.path.dirname(f'validate/{self.global_step}/swap'))
#         # os.makedirs(f'validate/{self.global_step}', exist_ok=True)
#         # os.makedirs(f'validate/{self.global_step}/source', exist_ok=True)
#         # os.makedirs(f'validate/{self.global_step}/target', exist_ok=True)
#         # os.makedirs(f'validate/{self.global_step}/swap', exist_ok=True)
#         # for batch_idx, batch in enumerate(self.test_dataloader):
#         #     source, target = batch
#         #     # if batch_idx % self.opts.same_image_interval == 0:
#         #     #     target = torch.clone(source)

#         #     with torch.no_grad():
#         #         source = source.to(self.device).float()
#         #         target = target.to(self.device).float()
#         #         swap = self.net(source, target)
#         #         loss_dict = {}
#         #         loss_ = 0.0
#         #         g_loss = torch.tensor(0.0, device=self.device)
#         #         if self.opts.train_G:
#         #             fake_pred = self.D(swap)
#         #             g_loss = self.adv_g_loss(fake_pred)

#         #         # loss_, loss_dict = self.calc_loss(source, target, swap, flag, batch_idx)
#         #         loss_dict["g_loss"] = float(g_loss)

#         #         if self.opts.train_G:
#         #             overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
#         #         else:
#         #             overall_loss = loss_
#         #         loss_dict["loss"] = float(overall_loss)

#         #     agg_loss_dict.append(loss_dict)

#         #     # For first step just do sanity test on small amount of data
#         #     if self.global_step == 0 and batch_idx >= 4:
#         #         self.net.train()
#         #         if self.opts.train_D:
#         #             self.D.train()
#         #         return None  # Do not log, inaccurate in first batch
            
#         #     for i in range(swap.shape[0]):
#         #         torch_utils.tensor2im(source[i]).save(f'validate/{self.global_step}/source/{(batch_idx + 1) * i}.jpg')
#         #         torch_utils.tensor2im(target[i]).save(f'validate/{self.global_step}/target/{(batch_idx + 1) * i}.jpg')
#         #         torch_utils.tensor2im(swap[i]).save(f'validate/{self.global_step}/swap/{(batch_idx + 1) * i}.jpg')
                

#         # loss_dict = torch_utils.aggregate_loss_dict(agg_loss_dict)
#         loss_dict = calc_metrics(self.net, self.device)
#         self.log_metrics(loss_dict, prefix='metrics')
#         self.print_metrics(loss_dict, prefix='metrics')

#         self.net.train()
#         if self.opts.train_D:
#             self.D_swap.train()
#             self.D_recon.train()
#         return loss_dict


# def main():
#     opts = TrainOptions().parse()
#     os.makedirs(opts.exp_dir, exist_ok=True)

#     opts_dict = vars(opts)
#     with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
#         json.dump(opts_dict, f, indent=4, sort_keys=True)

#     trainer = Trainer(opts)
#     trainer.train()


# if __name__ == '__main__':
#     main()
