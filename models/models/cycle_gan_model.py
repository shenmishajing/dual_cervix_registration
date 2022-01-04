import torch
from torch import nn
import random
import itertools
from .base import LightningModule


class ImagePool:
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images


class CycleGANModel(LightningModule):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    ModalDict = {'src': 'acid', 'trg': 'iodine'}

    def __init__(self,
                 netG_A: nn.Module,
                 netG_B: nn.Module,
                 netD_A: nn.Module = None,
                 netD_B: nn.Module = None,
                 is_train: bool = True,
                 pool_size: int = 50,
                 *args, **kwargs):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.automatic_optimization = False
        super().__init__(*args, **kwargs)
        self.netG_A = netG_A
        self.netG_B = netG_B

        if is_train:
            self.netD_A = netD_A
            self.netD_B = netD_B

            self.fake_A_pool = ImagePool(pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(pool_size)  # create image buffer to store previously generated images

    def _construct_optimizers(self, optimizers):
        """
        Constructs all optimizers.

        Args:
            optimizers: list of dictionary containing optimizer configuration.
        """
        params = [itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                  itertools.chain(self.netD_A.parameters(), self.netD_B.parameters())]
        if len(optimizers) == 1:
            optimizers = [optimizers[0].copy() for _ in range(len(params))]
        for i in range(len(optimizers)):
            optimizers[i] = self._construct_optimizer(optimizers[i], set_lr = i == 0, params = params[i])

        return optimizers

    def forward(self, batch):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        real_A = batch[self.ModalDict['src']]['img']
        real_B = batch[self.ModalDict['trg']]['img']
        fake_B = self.netG_A(real_A)  # G_A(A)
        rec_A = self.netG_B(fake_B)  # G_B(G_A(A))
        fake_A = self.netG_B(real_B)  # G_B(B)
        rec_B = self.netG_A(fake_A)  # G_A(G_B(B))
        return {'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B, 'rec_A': rec_A, 'rec_B': rec_B}

    def backward_D(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.loss_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.loss_gan(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return {'loss_D': loss_D, 'loss_D_real': loss_D_real, 'loss_D_fake': loss_D_fake}

    def backward_G(self, res):
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss
        if self.lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = self.netG_A(res['real_B'])
            loss_idt_A = self.loss_idt(idt_A, res['real_B']) * self.lambda_B * self.lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = self.netG_B(res['real_A'])
            loss_idt_B = self.loss_idt(idt_B, res['real_A']) * self.lambda_A * self.lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        loss_G_A = self.loss_gan(self.netD_A(res['fake_B']), True)
        # GAN loss D_B(G_B(B))
        loss_G_B = self.loss_gan(self.netD_B(res['fake_A']), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = self.loss_cycle(res['rec_A'], res['real_A']) * self.lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = self.loss_cycle(res['rec_B'], res['real_B']) * self.lambda_B
        # combined loss and calculate gradients
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        return {'loss_G': loss_G, 'loss_G_A': loss_G_A, 'loss_G_B': loss_G_B, 'loss_cycle_A': loss_cycle_A, 'loss_cycle_B': loss_cycle_B,
                'loss_idt_A': loss_idt_A, 'loss_idt_B': loss_idt_B}

    def loss_step(self, batch, res, prefix = 'train', use_loss_weight = True, loss_use_loss_weight = True):
        loss_G = self.backward_G(res)
        loss_D_A = self.backward_D(self.netD_A, res['real_B'], self.fake_B_pool.query(res['fake_B']))  # calculate gradients for D_A
        loss_D_B = self.backward_D(self.netD_B, res['real_A'], self.fake_A_pool.query(res['fake_A']))  # calculate graidents for D_B
        loss = {**loss_G, **{k + '_A': v for k, v in loss_D_A.items()}, **{k + '_B': v for k, v in loss_D_B.items()}}
        loss['loss_D'] = loss['loss_D_A'] + loss['loss_D_B']
        loss['loss'] = loss['loss_G'] + loss['loss_D']
        # add prefix
        loss = {(f'{prefix}/' if prefix is not None else '') + ('loss_' if 'loss' not in k else '') + k: v for k, v in loss.items()}
        return loss

    def train_step(self, batch, batch_idx):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        res = self(batch)
        loss = self.loss_step(batch, res, 'train')
        optimizer_G, optimizer_D = self.optimizers()
        # G_A and G_B
        self.toggle_optimizer(optimizer_G, 0)
        optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.manual_backward(loss['train/loss_G'])  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.untoggle_optimizer(0)
        optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.manual_backward(loss['train/loss_D'])  # calculate gradients for D_A and D_B
        optimizer_D.step()  # update D_A and D_B's weights

        self.log_dict(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        res = self(batch)
        loss = self.loss_step(batch, res, 'val')
        self.log_dict(loss)
        return loss

    def test_step(self, batch, batch_idx):
        res = self(batch)
        loss = self.loss_step(batch, res, 'test')
        self.log_dict(loss)
        return loss

    # def on_predict_start(self) -> None:
    #     self.norm_cfg = {
    #         'acid': {'mean': [122.95729064941406], 'std': [15.282942771911621]},
    #         'iodine': {'mean': [83.96733856201172], 'std': [20.51559829711914]},
    #     }
    #     for part in self.norm_cfg:
    #         for k in self.norm_cfg[part]:
    #             self.norm_cfg[part][k] = torch.tensor(self.norm_cfg[part][k]).to(self.device)[None, :, None, None]
    #
    #     log_dir = os.path.dirname(os.path.dirname(self.trainer.predicted_ckpt_path))
    #     self.output_path = os.path.join(log_dir, 'visualization')
    #     if os.path.exists(self.output_path):
    #         shutil.rmtree(self.output_path)
    #     os.makedirs(self.output_path)
    #
    # def predict_step(self, batch: Any, batch_idx: int, **kwargs) -> Any:
    #     res = self(batch)
    #     res_img = {}
    #     for part in self.ModalDict.values():
    #         res_img[part] = batch[part]['img'] * self.norm_cfg[part]['std'] + self.norm_cfg[part]['mean']
    #     res_img['res'] = res['y_source'] * self.norm_cfg[self.ModalDict['trg']]['std'] + self.norm_cfg[self.ModalDict['trg']]['mean']
    #     res_img = torch.cat([res_img[self.ModalDict['src']], res_img['res'], res_img[self.ModalDict['trg']]], dim = -1)
    #     res_img = res_img.add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    #     for i in range(res_img.shape[0]):
    #         cur_name = batch['acid']['img_metas'][i]['ori_filename'].removesuffix('_2.jpg') + '.png'
    #         cv2.imwrite(os.path.join(self.output_path, cur_name), res_img[i])
    #     return res_img
