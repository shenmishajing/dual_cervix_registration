import os
import shutil
import cv2
import copy
import torch
from torch import nn
from .base import LightningModule


class Pix2PixModel(LightningModule):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    ModalDict = {'src': 'acid', 'trg': 'iodine'}

    def __init__(self,
                 netG: nn.Module,
                 netD: nn.Module = None,
                 lambda_L1: float = 100.0,
                 lambda_GAN: float = 100.0,
                 lambda_D: float = 100.0,
                 *args, **kwargs):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

        self.netG = netG
        self.netD = netD

        self.lambda_L1 = lambda_L1
        self.lambda_GAN = lambda_GAN
        self.lambda_D = lambda_D

    def _construct_optimizers(self, optimizers):
        """
        Constructs all optimizers.

        Args:
            optimizers: list of dictionary containing optimizer configuration.
        """
        params = [self.netG.parameters(), self.netD.parameters()]
        if len(optimizers) == 1:
            optimizers = [copy.deepcopy(optimizers[0]) for _ in range(len(params))]
        for i in range(len(optimizers)):
            optimizers[i] = self._construct_optimizer(optimizers[i], set_lr = i == 0, params = params[i])

        return optimizers

    def forward(self, batch):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        real_A = batch[self.ModalDict['src']]['img']
        real_B = batch[self.ModalDict['trg']]['img']
        fake_B = self.netG(real_A)  # G(A)
        return {'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B}

    def backward_D(self, res):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((res['real_A'], res['fake_B']), 1)
        pred_fake = self.netD(fake_AB.detach())
        loss_D_fake = self.loss_gan(pred_fake, False) * self.lambda_D
        # Real
        real_AB = torch.cat((res['real_A'], res['real_B']), 1)
        pred_real = self.netD(real_AB)
        loss_D_real = self.loss_gan(pred_real, True) * self.lambda_D
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return {'loss_D': loss_D, 'loss_D_real': loss_D_real, 'loss_D_fake': loss_D_fake}

    def backward_G(self, res):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((res['real_A'], res['fake_B']), 1)
        pred_fake = self.netD(fake_AB)
        loss_G_GAN = self.loss_gan(pred_fake, True) * self.lambda_GAN
        # Second, G(A) = B
        loss_G_L1 = self.loss_L1(res['fake_B'], res['real_B']) * self.lambda_L1
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        return {'loss_G': loss_G, 'loss_G_GAN': loss_G_GAN, 'loss_G_L1': loss_G_L1}

    def loss_step(self, batch, res, prefix = 'train', use_loss_weight = True, loss_use_loss_weight = True, detach = None):
        loss_G = self.backward_G(res)
        loss_D = self.backward_D(res)
        loss = {**loss_G, **loss_D}
        loss['loss'] = loss['loss_G'] + loss['loss_D']
        # add prefix
        if detach is None:
            detach = prefix != 'train'
        loss = {(f'{prefix}/' if prefix is not None else '') + ('loss_' if 'loss' not in k else '') + k: (v.detach() if detach else v) for
                k, v in loss.items()}
        return loss

    def training_step(self, batch, batch_idx):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        res = self(batch)
        loss = self.loss_step(batch, res, 'train')
        optimizer_G, optimizer_D = self.optimizers()
        # G_A and G_B
        self.toggle_optimizer(optimizer_G, 0)
        optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.manual_backward(loss['train/loss_G'])  # calculate gradients for G_A and G_B
        optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.untoggle_optimizer(0)
        optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.manual_backward(loss['train/loss_D'])  # calculate gradients for D_A and D_B
        optimizer_D.step()  # update D_A and D_B's weights

        self.log_dict(loss)
        self.manual_lr_schedulers_step('step', batch_idx = batch_idx)
        return loss['train/loss']

    def on_predict_start(self) -> None:
        for part in self.normalize_config:
            for k in self.normalize_config[part]:
                self.normalize_config[part][k] = torch.tensor(self.normalize_config[part][k]).to(self.device)[None, :, None, None]

        log_dir = os.path.dirname(os.path.dirname(self.trainer.predicted_ckpt_path))
        self.output_path = os.path.join(log_dir, 'visualization')
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

    def predict_step(self, batch, *args, **kwargs):
        res = self(batch)
        res_img = {}
        for part in self.ModalDict.values():
            res_img[part] = batch[part]['img'] * self.normalize_config[part]['std'] + self.normalize_config[part]['mean']
        res_img['res'] = res['fake_B'] * self.normalize_config[self.ModalDict['trg']]['std'] + self.normalize_config[self.ModalDict['trg']][
            'mean']
        res_img = torch.cat([res_img[self.ModalDict['src']], res_img['res'], res_img[self.ModalDict['trg']]], dim = -1)
        res_img = res_img.add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for i in range(res_img.shape[0]):
            cur_name = batch['acid']['img_metas'][i]['ori_filename'].removesuffix('_2.jpg') + '.png'
            cv2.imwrite(os.path.join(self.output_path, cur_name), cv2.cvtColor(res_img[i], cv2.COLOR_RGB2BGR))
        return res_img
